"""Cross-model numeral consensus, replayed from the REAL recorded outputs.

tests/data/p1_v2_{sonnet,opus,fable}.json are the actual three-model v2
readings of _024 p1 (the sheet whose Madison 2301-2359 frontage range is
verified against the modern centerline fields). The consensus set must
reproduce the measured reliability: 46/47 street-tagged numerals landing in
real centerline ranges, every Madison value inside the verified range.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.addresses import AddressNumeral, consensus_numerals, match_address
from autogeoref.names import load_aliases, normalize

DATA = Path(__file__).resolve().parent / "data"
MODELS = {
    "claude-sonnet-5": "p1_v2_sonnet.json",
    "claude-opus-4-8": "p1_v2_opus.json",
    "claude-fable-5": "p1_v2_fable.json",
}


def _numerals(raw: dict[str, Any]) -> list[AddressNumeral]:
    out: list[AddressNumeral] = []
    for n in raw.get("address_numerals") or []:
        try:
            street = n.get("street")
            left, top, right, bottom = (float(v) for v in n["bbox"])
            out.append(
                AddressNumeral(
                    value=int(n["value"]),
                    bbox=(left, top, right, bottom),
                    street_hint=street if isinstance(street, str) and street else None,
                )
            )
        except (KeyError, TypeError, ValueError):
            pass
    return out


@pytest.fixture(scope="module")
def per_model() -> dict[str, list[AddressNumeral]]:
    return {
        model: _numerals(json.loads((DATA / fname).read_text())) for model, fname in MODELS.items()
    }


def test_consensus_reproduces_measured_set(per_model: dict[str, list[AddressNumeral]]) -> None:
    consensus = consensus_numerals(per_model)
    assert len(consensus) == 47  # measured on the recorded outputs


def test_consensus_madison_values_in_verified_range(
    per_model: dict[str, list[AddressNumeral]],
) -> None:
    consensus = consensus_numerals(per_model)
    madison = [n.value for n in consensus if n.street_hint and "MADISON" in n.street_hint.upper()]
    assert len(madison) >= 20
    assert all(2301 <= v <= 2359 for v in madison), sorted(madison)


def test_consensus_external_validity_vs_centerlines(
    per_model: dict[str, list[AddressNumeral]], fixtures_dir: Path, aliases_dir: Path
) -> None:
    """>= 95% of street-tagged consensus numerals land in a real range."""
    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_024.json")
    features = json.loads((fixtures_dir / "reference" / "street_center_lines.geojson").read_text())[
        "features"
    ]
    consensus = consensus_numerals(per_model, aliases)
    tagged = [n for n in consensus if n.street_hint]
    assert tagged
    matched = sum(1 for n in tagged if match_address(n, n.street_hint or "", features, aliases))
    assert matched / len(tagged) >= 0.95, f"{matched}/{len(tagged)}"


def test_single_model_is_not_consensus(per_model: dict[str, list[AddressNumeral]]) -> None:
    one = {"claude-fable-5": per_model["claude-fable-5"]}
    assert consensus_numerals(one) == []


def test_min_agree_three_is_stricter(per_model: dict[str, list[AddressNumeral]]) -> None:
    two = consensus_numerals(per_model, min_agree=2)
    three = consensus_numerals(per_model, min_agree=3)
    assert len(three) < len(two)
    # 3-model agreement is a subset relationship on (value, street) keys
    keys3 = {(n.value, normalize(n.street_hint) if n.street_hint else None) for n in three}
    keys2 = {(n.value, normalize(n.street_hint) if n.street_hint else None) for n in two}
    assert keys3 <= keys2
