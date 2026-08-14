"""Renumbering-transcription parsing, stitching, and validation."""

from __future__ import annotations

import numpy as np

from conftest import load_script

# by path, then by name: these are `scripts/` libraries rather than an
# installed package, and registering them is what makes the imports below —
# including the ones inside test bodies — resolve
load_script("renumber_chains.py")
load_script("renumber_transcribe.py")

from renumber_chains import (  # noqa: E402
    Chain,
    PairRow,
    Stitcher,
    _split_sustained,
    clean_street,
    compress_chain,
    repair_street_names,
    validate_chain,
)
from renumber_transcribe import (  # noqa: E402
    RowClass,
    Token,
    classify_row,
    detect_rules,
    merge_reads,
    rows_from_tokens,
    strip_bounds,
)

# ---------------------------------------------------------------- classify


def test_classify_pair() -> None:
    rc = classify_row(["4211", "2410"])
    assert (rc.kind, rc.new, rc.old) == ("pair", 4211, 2410)


def test_classify_pair_single_token() -> None:
    rc = classify_row(["4211 2410"])
    assert (rc.kind, rc.new, rc.old) == ("pair", 4211, 2410)


def test_classify_suffixed_old_attached_and_detached() -> None:
    for texts in (["7", "90S"], ["7", "90", "S"], ["7", "90", "s"]):
        rc = classify_row(texts)
        assert (rc.kind, rc.new, rc.old, rc.old_suffix) == ("pair", 7, 90, "S")


def test_classify_no_digit_repair() -> None:
    # 0/O style repair would correlate the two reads' errors: '24lO' must
    # never be read as a numeral, so this row can never become a pair
    assert classify_row(["4211", "24lO"]).kind != "pair"


def test_classify_to_row() -> None:
    assert classify_row(["to"]).kind == "to"
    assert classify_row(["to", "to"]).kind == "to"


def test_classify_ditto() -> None:
    assert classify_row(["''"]).kind == "ditto"
    assert classify_row(['"']).kind == "ditto"


def test_classify_labels() -> None:
    odd = classify_row(["Odd", "Nos."])
    assert (odd.kind, odd.parity, odd.cont) == ("label", "odd", False)
    glued = classify_row(["EvenNos."])
    assert (glued.kind, glued.parity) == ("label", "even")
    cont = classify_row(["Odd", "Cont"])
    assert (cont.kind, cont.parity, cont.cont) == ("label", "odd", True)
    assert classify_row(["New", "Old"]).kind == "label"
    assert classify_row(["New", "0ld"]).kind == "label"


def test_classify_street_headers_not_labels() -> None:
    # 'No.' legitimately starts street names and must stay a header
    assert classify_row(["No.", "Hermitage", "Av."]).kind == "header"
    assert classify_row(["No."]).kind == "header"
    assert classify_row(["Henderson", "Street"]).kind == "header"


def test_classify_bounds() -> None:
    assert classify_row(["24211", "2410"]).kind == "junk"  # new > 13999
    assert classify_row(["4211", "24105"]).kind == "junk"  # old > 9999


def test_classify_lone_and_multi() -> None:
    assert classify_row(["4211"]).kind == "lone"
    assert classify_row(["4211", "2410", "4213"]).kind == "multi"


# ------------------------------------------------------- 1911 leading-pair


def test_leading_pair_with_building_name() -> None:
    from renumber_transcribe import classify_row_leading

    rc = classify_row_leading(["431", "299", "Elk", "Hotel", "s"])
    assert (rc.kind, rc.new, rc.old) == ("pair", 431, 299)


def test_leading_pair_building_digits_never_pair() -> None:
    from renumber_transcribe import classify_row_leading

    # numerals after a word (a building name's digits) are not data
    assert classify_row_leading(["Room", "431", "299"]).kind == "junk"


def test_leading_lone_number_not_pair() -> None:
    from renumber_transcribe import classify_row_leading

    assert classify_row_leading(["367", "Parking", "Space"]).kind != "pair"


def test_1911_street_header_kept_buildings_junked() -> None:
    from renumber_transcribe import classify_row_leading

    assert classify_row_leading(["S.", "CLARK", "ST."]).kind == "header"
    assert classify_row_leading(["S.", "CLARK", "ST.", "continued"]).kind == "header"
    assert classify_row_leading(["Elk", "Hotel"]).kind == "junk"
    assert classify_row_leading(["Rand", "McNally"]).kind == "junk"
    # a building named after a street must not hijack attribution
    assert classify_row_leading(["La", "Salle", "St.", "Station"]).kind == "junk"


# ---------------------------------------------------------------- merging


def _rows(*pairs: tuple[float, str, int, int]) -> list[tuple[float, RowClass]]:
    return [
        (y, RowClass(kind=kind, new=new if kind == "pair" else None, old=old))
        for y, kind, new, old in pairs
    ]


def test_merge_agreed_and_disagree() -> None:
    a = _rows((100.0, "pair", 4211, 2410), (127.0, "pair", 4213, 2412))
    b = _rows((101.0, "pair", 4211, 2410), (128.0, "pair", 4218, 2412))
    merged = merge_reads(a, b)
    assert [m.status for m in merged] == ["agreed", "disagree"]


def test_merge_single_read_rows() -> None:
    a = _rows((100.0, "pair", 4211, 2410))
    b = _rows((100.0, "pair", 4211, 2410), (127.0, "pair", 4213, 2412))
    merged = merge_reads(a, b)
    assert [m.status for m in merged] == ["agreed", "only_b"]


def test_merge_concat_agreement() -> None:
    # one engine glues the typewritten pair; equal digit strings = concat
    a = [(100.0, RowClass(kind="junk", text="411291 S"))]
    b = [(101.0, RowClass(kind="pair", new=411, old=291, text="411 291 S"))]
    merged = merge_reads(a, b)
    assert merged[0].status == "concat"
    assert merged[0].resolved is not None and merged[0].resolved.new == 411
    # different digit strings stay queued
    a2 = [(100.0, RowClass(kind="junk", text="411231 S"))]
    assert merge_reads(a2, b)[0].status == "disagree"


def test_merge_y_tolerance() -> None:
    a = _rows((100.0, "pair", 4211, 2410))
    b = _rows((130.0, "pair", 4211, 2410))
    merged = merge_reads(a, b)  # 30 px apart: different rows
    assert [m.status for m in merged] == ["only_a", "only_b"]


def test_rows_from_tokens_clusters_by_y() -> None:
    toks = [
        Token(10, 40, 100.0, "4211"),
        Token(60, 90, 102.0, "2410"),
        Token(10, 40, 127.0, "4213 2412"),
    ]
    rows = rows_from_tokens(toks)
    assert len(rows) == 2
    assert all(rc.kind == "pair" for _, rc in rows)


# ---------------------------------------------------------------- geometry


def test_detect_rules_and_infill() -> None:
    img = np.zeros((1000, 900), dtype=bool)
    for x in (100, 268, 436, 772):  # rule at ~604 missing (weak ink)
        img[:, x] = True
    rules = detect_rules(img)
    assert rules == [100, 268, 436, 772]
    bounds = strip_bounds(rules)
    # infill splits the double-width gap; no leading strip (first gap small)
    assert (100, 268) in bounds and (268, 436) in bounds
    assert (436, 604) in bounds and (604, 772) in bounds


def test_detect_rules_ignores_text() -> None:
    img = np.zeros((1000, 300), dtype=bool)
    img[:, 50] = True
    img[400:520, 120] = True  # a digit stroke: 12% of height, not a rule
    assert detect_rules(img) == [50]


# ---------------------------------------------------------------- chains


def _pair(
    y: float, new: int, old: int, page: int = 1, strip: int = 0, status: str = "agreed"
) -> PairRow:
    return PairRow(page=page, strip=strip, y=y, status=status, new=new, old=old)


def test_split_sustained_keeps_single_spike() -> None:
    run = [_pair(1, 3620, 2005), _pair(2, 2622, 2011), _pair(3, 3628, 2015)]
    assert len(_split_sustained(run)) == 1


def test_split_sustained_breaks_real_section() -> None:
    run = [_pair(1, 6427, 3928), _pair(2, 6429, 3930), _pair(3, 311, 220), _pair(4, 315, 224)]
    parts = _split_sustained(run)
    assert [len(p) for p in parts] == [2, 2]


def test_validate_flags_spike_not_successor() -> None:
    ch = Chain(1, "X ST", "X ST", "even", 1)
    ch.pairs = [_pair(1, 3620, 2005), _pair(2, 2622, 2011), _pair(3, 3628, 2015)]
    validate_chain(ch)
    assert ch.pairs[1].flags == ["uncertain:new_spike"]
    assert not ch.pairs[0].flags and not ch.pairs[2].flags


def test_validate_flags_old_spike_misprint() -> None:
    # An isolated old-number outlier is uncertain.
    ch = Chain(1, "X AV", "X AV", "even", 1)
    ch.pairs = [_pair(1, 4446, 2577), _pair(2, 4450, 2281), _pair(3, 4454, 2585)]
    validate_chain(ch)
    assert ch.pairs[1].flags == ["uncertain:old_spike"]


def test_compress_blocks_and_suffix_separation() -> None:
    ch = Chain(7, "N HERMITAGE AV", "N HERMITAGE AV", "odd", 1)
    ch.pairs = [
        _pair(1, 4211, 2410),
        _pair(2, 4257, 2454),
        _pair(3, 4303, 2464),
        _pair(4, 4357, 2520),
    ]
    ch.pairs.append(PairRow(page=1, strip=0, y=5, status="agreed", new=7, old=90, old_suffix="S"))
    entries = compress_chain(ch)
    ranges = {(tuple(e["old_range"]), tuple(e["new_range"])) for e in entries}
    assert ((2410, 2454), (4211, 4257)) in ranges
    assert ((2464, 2520), (4303, 4357)) in ranges
    suffixed = [e for e in entries if e.get("old_suffix")]
    assert len(suffixed) == 1 and suffixed[0]["old_suffix"] == "S"


def test_compress_excludes_uncertain() -> None:
    ch = Chain(1, "X AV", "X AV", "even", 1)
    ch.pairs = [_pair(1, 4406, 2535), _pair(2, 4450, 2281), _pair(3, 4454, 2585)]
    validate_chain(ch)
    entries = compress_chain(ch)
    assert len(entries) == 1
    assert entries[0]["old_range"] == [2535, 2585]
    assert entries[0]["new_range"] == [4406, 4454]


def test_clean_street() -> None:
    assert clean_street("No. Hermitage Av.") == "N HERMITAGE AV"
    assert clean_street("So. Hermitage Av.") == "S HERMITAGE AV"
    assert clean_street("Hein Place CONTINUED") == "HEIN PLACE"


def test_repair_street_names_unique_match_only() -> None:
    damaged = Chain(1, "No. Hern itage Av.", "N HERN ITAGE AV", "odd", 1)
    unknown = Chain(2, "Hein Place", "HEIN PLACE", "odd", 1)
    repairs = repair_street_names([damaged, unknown], {"HERMITAGE", "PAULINA"})
    assert repairs == {"N HERN ITAGE AV": "N HERMITAGE"}
    assert damaged.street == "N HERMITAGE" and "name_repaired" in damaged.flags
    assert unknown.street == "HEIN PLACE" and "name_unmatched" in unknown.flags


def test_repair_names_by_alpha_bracket() -> None:
    from renumber_chains import repair_names_by_alpha_bracket

    before = Chain(1, "Rascher Avenue", "RASCHER AVENUE", "odd", 1)
    garbled = Chain(2, "E. Raven pooms Pk.", "E RAVEN POOMS PK", "odd", 1)
    garbled.flags.append("name_unmatched")
    after = Chain(3, "Read Court", "READ COURT", "odd", 1)
    vocab = {"RASCHER", "RAVENSWOOD", "READ", "PAULINA", "WOOD"}
    repairs = repair_names_by_alpha_bracket([before, garbled, after], vocab)
    assert repairs == {"E RAVEN POOMS PK": "E RAVENSWOOD"}
    assert garbled.street == "E RAVENSWOOD"
    assert "name_repaired_bracket" in garbled.flags
    assert "name_unmatched" not in garbled.flags


def test_adopt_orphans_by_sibling() -> None:
    from renumber_chains import adopt_orphans_by_sibling

    named = Chain(1, "Doe Street", "DOE STREET", "even", 5)
    named.pairs = [
        _pair(float(i), 1000 + 2 * i, 400 + i, page=5, strip=(i // 20) * 2) for i in range(80)
    ]
    orphan = Chain(2, "", "", "odd", 5)
    orphan.flags.append("orphan_run")
    orphan.pairs = [
        _pair(float(i), 1001 + 2 * i, 401 + i, page=5, strip=(i // 20) * 2 + 1) for i in range(80)
    ]
    assert adopt_orphans_by_sibling([named, orphan]) == 1
    assert orphan.street == "DOE STREET"
    assert "name_from_sibling" in orphan.flags and "orphan_run" not in orphan.flags


def test_adopt_orphans_requires_unique_twin() -> None:
    from renumber_chains import adopt_orphans_by_sibling

    a = Chain(1, "Doe Street", "DOE STREET", "even", 5)
    a.pairs = [
        _pair(float(i), 1000 + 2 * i, 400 + i, page=5, strip=(i // 20) * 2) for i in range(80)
    ]
    b = Chain(2, "Roe Street", "ROE STREET", "even", 5)
    b.pairs = [
        _pair(float(i), 1000 + 2 * i, 400 + i, page=5, strip=(i // 20) * 2) for i in range(80)
    ]
    orphan = Chain(3, "", "", "odd", 5)
    orphan.pairs = [
        _pair(float(i), 1001 + 2 * i, 401 + i, page=5, strip=(i // 20) * 2 + 1) for i in range(80)
    ]
    assert adopt_orphans_by_sibling([a, b, orphan]) == 0
    assert orphan.street == ""


def test_select_shipped_drops_conflicts_both_ways() -> None:
    from renumber_chains import select_shipped

    good = {
        "street": "HERMITAGE",
        "side": "odd(new)",
        "old_range": [2410, 2454],
        "new_range": [4211, 4257],
    }
    a = {"street": "DOE", "side": "odd(new)", "old_range": [100, 200], "new_range": [1101, 1201]}
    b = {"street": "DOE", "side": "odd(new)", "old_range": [150, 250], "new_range": [2151, 2251]}
    unnamed = {"street": "", "side": "odd(new)", "old_range": [1, 9], "new_range": [1, 9]}
    shipped, conflicts = select_shipped([good, a, b, unnamed])
    assert shipped == [good]
    assert len(conflicts) == 3  # both DOE entries and the unnamed one


def test_select_shipped_keeps_agreeing_overlap() -> None:
    from renumber_chains import select_shipped

    a = {"street": "DOE", "side": "odd(new)", "old_range": [100, 200], "new_range": [1101, 1201]}
    b = {"street": "DOE", "side": "odd(new)", "old_range": [150, 250], "new_range": [1151, 1251]}
    shipped, conflicts = select_shipped([a, b])
    assert shipped == [a, b] and conflicts == []


# ------------------------------------------------------- tiebreak cases


def _cell(new: int, old: int) -> dict[str, object]:
    return {"kind": "pair", "new": new, "old": old, "text": ""}


def test_tiebreak_upgrades_only_matching_read() -> None:
    from renumber_chains import tiebreak_strip

    rows: list[dict[str, object]] = [
        {"y": 1, "st": "agreed", "a": _cell(4211, 2410), "b": _cell(4211, 2410)},
        {"y": 2, "st": "disagree", "a": _cell(4213, 2412), "b": _cell(4218, 2412)},
        {"y": 3, "st": "agreed", "a": _cell(4215, 2414), "b": _cell(4215, 2414)},
        # third value differs from both reads: must stay queued
        {"y": 4, "st": "disagree", "a": _cell(4217, 2416), "b": _cell(4219, 2416)},
        {"y": 5, "st": "agreed", "a": _cell(4221, 2420), "b": _cell(4221, 2420)},
    ]
    chm = [
        (4211, 2410, "", ""),
        (4213, 2412, "", ""),  # matches read a of row 2
        (4215, 2414, "", ""),
        (4227, 2416, "", ""),  # matches neither read of row 4
        (4221, 2420, "", ""),
    ]
    assert tiebreak_strip(rows, chm) == 1
    assert rows[1]["st"] == "tiebreak"
    assert rows[1]["v"]["new"] == 4213  # type: ignore[index]
    assert rows[3]["st"] == "disagree"


def test_tiebreak_requires_positional_match() -> None:
    from renumber_chains import tiebreak_strip

    rows: list[dict[str, object]] = [
        {"y": 1, "st": "agreed", "a": _cell(4211, 2410), "b": _cell(4211, 2410)},
        {"y": 2, "st": "disagree", "a": _cell(4213, 2412), "b": _cell(4218, 2412)},
        {"y": 3, "st": "agreed", "a": _cell(4215, 2414), "b": _cell(4215, 2414)},
    ]
    # CHM read dropped the middle row: counts differ, nothing upgrades
    chm = [(4211, 2410, "", ""), (4215, 2414, "", "")]
    assert tiebreak_strip(rows, chm) == 0
    assert rows[1]["st"] == "disagree"


# ------------------------------------------------------------- stitcher


def _page(strips: list[list[dict[str, object]]]) -> dict[str, object]:
    return {
        "pdf_page": 1,
        "strips": [
            {"x0": 100 + 170 * i, "x1": 270 + 170 * i, "rows": rows}
            for i, rows in enumerate(strips)
        ],
    }


def _jrow(y: float, kind: str, **kw: object) -> dict[str, object]:
    if kind == "pair":
        cell = {"kind": "pair", "new": kw["new"], "old": kw["old"], "text": ""}
        return {"y": y, "st": "agreed", "a": cell, "b": cell}
    cell = {"kind": kind, **kw}
    return {"y": y, "st": "context", "a": cell, "b": cell}


def test_stitcher_header_and_snake() -> None:
    # the printed header is centered over the strip pair, so each strip
    # catches its share of the words
    strip0 = [
        _jrow(100, "header", text="Doe"),
        _jrow(130, "label", parity="odd"),
        _jrow(160, "pair", new=4111, old=1136),
        _jrow(190, "pair", new=4119, old=1142),
    ]
    strip1 = [
        _jrow(100, "header", text="Street"),
        _jrow(130, "label", parity="even"),
        _jrow(160, "pair", new=4206, old=1197),
    ]
    strip2 = [
        _jrow(60, "label", parity="odd"),
        _jrow(90, "pair", new=4123, old=1150),
        _jrow(120, "pair", new=4145, old=1170),
    ]
    st = Stitcher()
    st.feed_page(1, _page([strip0, strip1, strip2]))
    named = [c for c in st.chains if c.street == "DOE STREET"]
    assert len(named) == 2
    odd = next(c for c in named if c.parity == "odd")
    assert [p.new for p in odd.agreed()] == [4111, 4119, 4123, 4145]


def test_stitcher_continued_links_across_pages() -> None:
    page1 = _page(
        [
            [
                _jrow(100, "header", text="Doe Street"),
                _jrow(160, "pair", new=4111, old=1136),
                _jrow(190, "pair", new=4119, old=1142),
            ]
        ]
    )
    page2 = _page(
        [
            [
                _jrow(100, "header", text="Doe Street CONTINUED"),
                _jrow(160, "pair", new=4123, old=1150),
            ]
        ]
    )
    st = Stitcher()
    st.feed_page(1, page1)
    st.feed_page(2, page2)
    doe = [c for c in st.chains if c.street == "DOE STREET"]
    assert len(doe) == 1
    assert [p.new for p in doe[0].agreed()] == [4111, 4119, 4123]


def test_stitcher_garbled_header_does_not_hijack() -> None:
    # continuation column whose label the engine mangled into header noise:
    # attribution must fall through to arithmetic continuity, not the noise
    strip0 = [
        _jrow(100, "header", text="Doe Street"),
        _jrow(160, "pair", new=4111, old=1136),
        _jrow(190, "pair", new=4119, old=1142),
    ]
    strip1 = [
        {
            "y": 60,
            "st": "context",
            "a": {"kind": "header", "text": '"WON PPO'},
            "b": None,
        },
        _jrow(90, "pair", new=4123, old=1150),
    ]
    st = Stitcher()
    st.feed_page(1, _page([strip0, strip1]))
    doe = [c for c in st.chains if c.street == "DOE STREET"]
    assert len(doe) == 1
    assert [p.new for p in doe[0].agreed()] == [4111, 4119, 4123]
