#!/usr/bin/env python
"""Dual-read OCR driver for a city's printed renumbering books.

Renders each PDF page at 4x (pypdfium2), splits column strips on the printed
vertical rules, and produces TWO independent reads per strip — rapidocr on the
rendered pixels, and the scan's own embedded OCR text layer mapped into the
same pixel frame — then merges them row by row
(renumber_transcribe.py).

Output is one JSON per page under --out; the downstream chain-stitching,
validation, and compression read those files, so OCR never runs twice. No
network. Run under nice for full-book batches.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from renumber_transcribe import (
    MergedRow,
    Token,
    detect_rules,
    merge_reads,
    rows_from_tokens,
    strip_bounds,
)


def textlayer_tokens(page: object, scale: float, img_h: int) -> list[Token]:
    """Embedded OCR tokens mapped into rendered-image pixel coordinates."""
    tp = page.get_textpage()  # type: ignore[attr-defined]
    page_h = img_h / scale
    out: list[Token] = []
    for i in range(tp.count_rects()):
        left, bottom, right, top = tp.get_rect(i)
        text = (tp.get_text_bounded(left, bottom, right, top) or "").strip()
        if not text:
            continue
        # A rect may span several printed lines (embedded newlines); split it
        # and interpolate each line's y across the rect height, else those
        # rows all collapse into one garbled multi-row and read B loses them.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        height = top - bottom
        for j, line in enumerate(lines):
            frac = (j + 0.5) / len(lines)
            y = (page_h - (top - frac * height)) * scale
            out.append(Token(x0=left * scale, x1=right * scale, y=y, text=line))
    return out


def engine_tokens(ocr: object, arr: np.ndarray, x_offset: int) -> list[Token]:
    """rapidocr tokens for one strip, shifted into page coordinates."""
    if arr.shape[0] < 16 or arr.shape[1] < 60:
        return []
    rgb = np.stack([arr] * 3, axis=-1)
    try:
        result, _ = ocr(rgb)  # type: ignore[operator]
    except Exception as exc:  # a bad strip must not kill the batch
        print(f"  OCR failed on strip at x={x_offset}: {exc!r}", flush=True)
        return []
    out: list[Token] = []
    for box, text, _score in result or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append(
            Token(
                x0=min(xs) + x_offset,
                x1=max(xs) + x_offset,
                y=(min(ys) + max(ys)) / 2,
                text=str(text),
            )
        )
    return out


def easyocr_tokens(reader: object, arr: np.ndarray, x_offset: int) -> list[Token]:
    """easyocr tokens for one strip (read B for scans without a text layer)."""
    if arr.shape[0] < 16 or arr.shape[1] < 60:
        return []
    try:
        result = reader.readtext(arr)  # type: ignore[attr-defined]
    except Exception as exc:  # a bad strip must not kill the batch
        print(f"  easyocr failed on strip at x={x_offset}: {exc!r}", flush=True)
        return []
    out: list[Token] = []
    for box, text, _conf in result or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append(
            Token(
                x0=float(min(xs)) + x_offset,
                x1=float(max(xs)) + x_offset,
                y=(float(min(ys)) + float(max(ys))) / 2,
                text=str(text),
            )
        )
    return out


def row_to_json(m: MergedRow) -> dict[str, object]:
    d: dict[str, object] = {"y": round(m.y, 1), "st": m.status}
    for key, rc in (("a", m.a), ("b", m.b)):
        if rc is not None:
            rd = {k: v for k, v in asdict(rc).items() if v not in (None, "", False)}
            d[key] = rd
    if m.resolved is not None:
        d["v"] = {k: v for k, v in asdict(m.resolved).items() if v not in (None, "", False)}
    return d


# The shape written to one page JSON, so callers index it without casting.
class PageStrip(TypedDict):
    x0: int
    x1: int
    rows: list[dict[str, object]]


class PageJSON(TypedDict):
    pdf_page: int
    image_size: list[int]
    scale: float
    rules: list[int]
    headers: list[dict[str, object]]
    strips: list[PageStrip]


def process_page(
    doc: object,
    ocr: object,
    page_index: int,
    scale: float,
    layout: str = "1909",
    ocr_b: object = None,
) -> PageJSON:
    from renumber_transcribe import classify_row, classify_row_leading, cluster_rows

    page = doc[page_index]  # type: ignore[index]
    bitmap = page.render(scale=scale)
    img = np.asarray(bitmap.to_pil().convert("L"))
    h, w = img.shape
    headers: list[dict[str, object]] = []
    if layout == "1911":
        # typewritten Loop register: no vertical rules; two fixed half-page
        # column groups (Odd | Even), the number pair leading each row
        bounds = [(int(0.06 * w), int(0.52 * w)), (int(0.52 * w), int(0.97 * w))]
        classifier = classify_row_leading
        all_b = None
        rules: list[int] = []
    else:
        classifier = classify_row
        rules = detect_rules(img < 128)
        bounds = strip_bounds(rules)
        if not 2 <= len(bounds) <= 20:
            # preface/plate page, not a ruled table: one whole-page strip
            bounds = [(0, w)]
        all_b = textlayer_tokens(page, scale, h)
        for row in cluster_rows(all_b, gap=12.0):
            rc = classify_row([t.text for t in row])
            if rc.kind in ("header", "label"):
                headers.append(
                    {
                        "y": round(sum(t.y for t in row) / len(row), 1),
                        "x0": round(min(t.x0 for t in row), 1),
                        "x1": round(max(t.x1 for t in row), 1),
                        "kind": rc.kind,
                        "parity": rc.parity,
                        "cont": rc.cont,
                        "text": rc.text,
                    }
                )

    strips: list[PageStrip] = []
    for x0, x1 in bounds:
        pad = 3
        arr = img[:, max(x0 + pad, 0) : max(x1 - pad, x0 + pad + 1)]
        toks_a = engine_tokens(ocr, arr, x0 + pad) if ocr is not None else []
        if all_b is not None:
            toks_b = [t for t in all_b if x0 <= (t.x0 + t.x1) / 2 < x1]
        else:
            toks_b = easyocr_tokens(ocr_b, arr, x0 + pad) if ocr_b is not None else []
        rows_a = rows_from_tokens(toks_a, classifier=classifier)
        rows_b = rows_from_tokens(toks_b, classifier=classifier)
        merged = merge_reads(rows_a, rows_b)
        strips.append({"x0": int(x0), "x1": int(x1), "rows": [row_to_json(m) for m in merged]})
    return {
        "pdf_page": page_index + 1,
        "image_size": [int(w), int(h)],
        "scale": scale,
        "rules": [int(r) for r in rules],
        "headers": headers,
        "strips": strips,
    }


def run_tokens(args: argparse.Namespace) -> None:
    """One engine, one process, raw tokens per page (crash-isolated pass).

    onnxruntime (rapidocr) and torch (easyocr) destabilize each other when
    loaded together (measured: WSL-crashing/hanging dual-engine runs), so
    the 1911 register is read in two single-engine passes; ``merge1911``
    joins them. Per-page checkpointing: existing outputs are skipped, so a
    killed pass resumes where it died.
    """
    import pypdfium2 as pdfium

    if args.engine == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()

        def read(arr: np.ndarray, off: int) -> list[Token]:
            return engine_tokens(ocr, arr, off)
    else:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        def read(arr: np.ndarray, off: int) -> list[Token]:
            return easyocr_tokens(reader, arr, off)

    doc = pdfium.PdfDocument(str(args.pdf))
    args.out.mkdir(parents=True, exist_ok=True)
    for p in parse_pages(args.pages, len(doc)):
        out_path = args.out / f"page_{p:03d}.json"
        if out_path.exists():
            continue
        page = doc[p - 1]
        img = np.asarray(page.render(scale=args.scale).to_pil().convert("L"))
        h, w = img.shape
        bounds = [(int(0.06 * w), int(0.52 * w)), (int(0.52 * w), int(0.97 * w))]
        strips: list[dict[str, Any]] = []
        for x0, x1 in bounds:
            toks = read(img[:, x0 + 3 : x1 - 3], x0 + 3)
            strips.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "tokens": [
                        [round(t.x0, 1), round(t.x1, 1), round(t.y, 1), t.text] for t in toks
                    ],
                }
            )
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"pdf_page": p, "image_size": [int(w), int(h)], "strips": strips})
        )
        tmp.rename(out_path)
        print(f"page {p}: {sum(len(s['tokens']) for s in strips)} tokens", flush=True)


def run_merge1911(args: argparse.Namespace) -> None:
    """Join the two single-engine token passes into merged page JSONs."""
    from renumber_transcribe import classify_row_leading

    args.out.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for path_a in sorted(args.tokens_a.glob("page_*.json")):
        path_b = args.tokens_b / path_a.name
        if not path_b.exists():
            continue
        da = json.loads(path_a.read_text())
        db = json.loads(path_b.read_text())
        strips = []
        for sa, sb in zip(da["strips"], db["strips"], strict=True):
            toks_a = [Token(x0=t[0], x1=t[1], y=t[2], text=t[3]) for t in sa["tokens"]]
            toks_b = [Token(x0=t[0], x1=t[1], y=t[2], text=t[3]) for t in sb["tokens"]]
            rows_a = rows_from_tokens(toks_a, classifier=classify_row_leading)
            rows_b = rows_from_tokens(toks_b, classifier=classify_row_leading)
            merged = merge_reads(rows_a, rows_b)
            strips.append(
                {"x0": sa["x0"], "x1": sa["x1"], "rows": [row_to_json(m) for m in merged]}
            )
        out_path = args.out / path_a.name
        out_path.write_text(
            json.dumps(
                {
                    "pdf_page": da["pdf_page"],
                    "image_size": da["image_size"],
                    "rules": [],
                    "headers": [],
                    "strips": strips,
                }
            )
        )
        n_done += 1
    print(f"merged {n_done} pages -> {args.out}")


def run_chains(args: argparse.Namespace) -> None:
    from dataclasses import asdict as dc_asdict

    from renumber_chains import (
        Stitcher,
        compress_chain,
        repair_street_names,
        validate_chain,
    )

    paths = sorted(args.pages_dir.glob("page_*.json"))
    stitcher = Stitcher()
    for path in paths:
        data = json.loads(path.read_text())
        stitcher.feed_page(int(data["pdf_page"]), data)
    repairs: dict[str, str] = {}
    if args.centerlines is not None:
        from autogeoref.names import normalize

        geo = json.loads(args.centerlines.read_text())
        vocab = set()
        for feat in geo.get("features", []):
            props = feat.get("properties", {})
            name = str(props.get(args.name_property) or "").strip()
            typ = str(props.get(args.type_property) or "").strip()
            if name:
                core = normalize(f"{name} {typ}".strip())
                if core:
                    vocab.add(core)
        from renumber_chains import repair_names_by_alpha_bracket

        repairs = repair_street_names(stitcher.chains, vocab)
        bracket_repairs = repair_names_by_alpha_bracket(stitcher.chains, vocab)
        repairs.update(bracket_repairs)
        from renumber_chains import adopt_orphans_by_sibling

        adopted = adopt_orphans_by_sibling(stitcher.chains)
        print(
            f"name repairs: {len(repairs)} ({len(bracket_repairs)} bracket-pass; "
            f"{adopted} sibling adoptions; vocabulary: {len(vocab)} streets)"
        )
    entries: list[dict[str, object]] = []
    queue: list[dict[str, object]] = []
    for ch in stitcher.chains:
        validate_chain(ch)
        entries.extend(compress_chain(ch))
        for p in ch.pairs:
            if p.status not in ("agreed", "tiebreak", "concat") or any(
                f.startswith("uncertain") for f in p.flags
            ):
                q = dc_asdict(p)
                q["chain_id"] = ch.chain_id
                q["street"] = ch.street
                queue.append(q)
    from renumber_chains import select_shipped

    shipped, conflicts = select_shipped(entries)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "name_repairs.json").write_text(json.dumps(repairs, indent=1))
    (args.out_dir / "entries.json").write_text(json.dumps(entries, indent=1))
    (args.out_dir / "entries_shipped.json").write_text(json.dumps(shipped, indent=1))
    (args.out_dir / "entries_conflicts.json").write_text(json.dumps(conflicts, indent=1))
    (args.out_dir / "queue.json").write_text(json.dumps(queue, indent=1))
    chains_dump = [
        {
            "chain_id": c.chain_id,
            "street": c.street,
            "street_raw": c.street_raw,
            "parity": c.parity,
            "page": c.page,
            "flags": sorted(set(c.flags)),
            "n_pairs": len(c.pairs),
            "n_agreed": len(c.agreed()),
            "first": next(((p.new, p.old) for p in c.pairs if p.status == "agreed"), None),
            "last": ((lambda lp: (lp.new, lp.old) if lp else None)(c.last_agreed())),
        }
        for c in stitcher.chains
    ]
    (args.out_dir / "chains.json").write_text(json.dumps(chains_dump, indent=1))
    n_agreed = sum(len(c.agreed()) for c in stitcher.chains)
    named = sum(1 for c in stitcher.chains if c.street)
    print(
        f"{len(stitcher.chains)} chains ({named} named), {n_agreed} agreed pairs, "
        f"{len(entries)} entries ({len(shipped)} shipped, {len(conflicts)} conflict/unnamed), "
        f"{len(queue)} queued rows"
    )


def run_tiebreak(args: argparse.Namespace) -> None:
    """Third-read (CHM imaging) 2-of-3 settlement of queued rows, in place."""
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR
    from renumber_chains import tiebreak_strip

    ocr = RapidOCR()
    doc = pdfium.PdfDocument(str(args.chm_pdf))
    total_upgrades = total_queued = 0
    for path in sorted(args.pages_dir.glob("page_*.json")):
        data = json.loads(path.read_text())
        pdf_page = int(data["pdf_page"])
        needs = [
            si
            for si, s in enumerate(data["strips"])
            if any(r["st"] in ("disagree", "only_a", "only_b") for r in s["rows"])
        ]
        total_queued += sum(
            1
            for s in data["strips"]
            for r in s["rows"]
            if r["st"] in ("disagree", "only_a", "only_b")
        )
        if not needs or pdf_page > len(doc):
            continue
        page = doc[pdf_page - 1]
        img = np.asarray(page.render(scale=args.scale).to_pil().convert("L"))
        rules = detect_rules(img < 128)
        bounds = strip_bounds(rules)
        if not bounds:
            continue
        orig_bounds = [(s["x0"], s["x1"]) for s in data["strips"]]
        orig_w = data["image_size"][0]
        upgrades = 0
        for si in needs:
            # map by relative center: the two scans crop differently
            center = (orig_bounds[si][0] + orig_bounds[si][1]) / 2 / orig_w
            cx = center * img.shape[1]
            x0, x1 = min(bounds, key=lambda b: abs((b[0] + b[1]) / 2 - cx))
            toks = engine_tokens(ocr, img[:, x0 + 3 : x1 - 3], x0 + 3)
            chm_pairs = [
                (rc.new, rc.old, rc.new_suffix, rc.old_suffix)
                for _y, rc in rows_from_tokens(toks)
                if rc.kind == "pair" and rc.new is not None and rc.old is not None
            ]
            upgrades += tiebreak_strip(data["strips"][si]["rows"], chm_pairs)
        if upgrades:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data))
            tmp.rename(path)
        total_upgrades += upgrades
        print(f"page {pdf_page}: +{upgrades} tiebroken", flush=True)
    print(f"TOTAL: {total_upgrades} upgraded of {total_queued} queued", flush=True)


def parse_pages(spec: str, n_pages: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [p for p in out if 1 <= p <= n_pages]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("pages", help="render + dual-read a page range to JSON")
    pg.add_argument("--pdf", required=True, type=Path)
    pg.add_argument("--out", required=True, type=Path)
    pg.add_argument("--pages", default="1-99999")
    pg.add_argument("--scale", type=float, default=4.0)
    pg.add_argument(
        "--engine",
        choices=["rapidocr", "none"],
        default="rapidocr",
        help="'none' = text-layer-only (read B), for text-layer sanity passes",
    )
    pg.add_argument(
        "--layout",
        choices=["1909", "1911"],
        default="1909",
        help="1911 = typewritten Loop register: fixed halves, leading-pair rows, easyocr read B",
    )
    pg.add_argument("--force", action="store_true", help="reprocess existing pages")
    ch = sub.add_parser("chains", help="stitch page JSONs into chains + entries")
    ch.add_argument("--pages-dir", required=True, type=Path)
    ch.add_argument("--out-dir", required=True, type=Path)
    ch.add_argument(
        "--centerlines",
        type=Path,
        default=None,
        help="official centerlines GeoJSON: vocabulary for street-name repair",
    )
    ch.add_argument(
        "--name-property",
        default="street_nam",
        help="centerline property holding the street name (matches the city "
        "config's centerline_name_property; default: Chicago's schema)",
    )
    ch.add_argument(
        "--type-property",
        default="street_typ",
        help="centerline property holding the street type/suffix",
    )
    tk = sub.add_parser("tokens", help="single-engine raw-token pass (crash-isolated)")
    tk.add_argument("--pdf", required=True, type=Path)
    tk.add_argument("--out", required=True, type=Path)
    tk.add_argument("--pages", default="1-99999")
    tk.add_argument("--scale", type=float, default=3.0)
    tk.add_argument("--engine", choices=["rapidocr", "easyocr"], required=True)
    mg = sub.add_parser("merge1911", help="merge two token passes into page JSONs")
    mg.add_argument("--tokens-a", required=True, type=Path)
    mg.add_argument("--tokens-b", required=True, type=Path)
    mg.add_argument("--out", required=True, type=Path)
    tb = sub.add_parser("tiebreak", help="settle queued rows 2-of-3 via the CHM scan")
    tb.add_argument("--pages-dir", required=True, type=Path)
    tb.add_argument("--chm-pdf", required=True, type=Path)
    tb.add_argument("--scale", type=float, default=4.0)
    args = ap.parse_args()

    if args.cmd == "chains":
        run_chains(args)
        return
    if args.cmd == "tiebreak":
        run_tiebreak(args)
        return
    if args.cmd == "tokens":
        run_tokens(args)
        return
    if args.cmd == "merge1911":
        run_merge1911(args)
        return

    import pypdfium2 as pdfium

    ocr = None
    if args.engine == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
    ocr_b = None
    if args.layout == "1911":
        import easyocr

        ocr_b = easyocr.Reader(["en"], gpu=False, verbose=False)

    doc = pdfium.PdfDocument(str(args.pdf))
    pages = parse_pages(args.pages, len(doc))
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, p in enumerate(pages):
        out_path = args.out / f"page_{p:03d}.json"
        if out_path.exists() and not args.force:
            continue
        data = process_page(doc, ocr, p - 1, args.scale, layout=args.layout, ocr_b=ocr_b)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data))
        tmp.rename(out_path)
        n_pairs = sum(1 for s in data["strips"] for r in s["rows"] if r["st"] == "agreed")
        rate = (time.time() - t0) / (k + 1)
        print(
            f"page {p}: {len(data['strips'])} strips, {n_pairs} agreed pairs "
            f"({rate:.1f}s/page avg)",
            flush=True,
        )


if __name__ == "__main__":
    main()
