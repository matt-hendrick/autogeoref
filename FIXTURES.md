# Fixtures

`fixtures/` is gitignored test data pinned by tracked `FIXTURE-SHA256SUMS`,
which covers that one root and nothing else.
`tests/test_fixture_integrity.py` detects drift. Golden
tests skip without the tree, so a green test run on a fresh clone does not
validate fixture behaviour.

## Rules

- Treat fixtures as read-only. Do not regenerate, edit, symlink, or replace
  them to change a result.
- Resolve ground-truth findings by documentation only. The correction overlay
  is retired; ground truth is served exactly as the volunteer left it.
- Audit new ground truth before freezing it as a golden testbed, and triage
  what the audit flags. The checks that find a bent pin are self-referential —
  a page's implied SCALE, its ROTATION folded mod 90, and how it TILES against
  its neighbours, each judged against its own volume's median. Residual QA is
  structurally blind to these: a volunteer who mis-identified their three to
  five control points consistently leaves a fit that agrees with itself. Such
  a check REPORTS and never drops a pin — one that excluded a pin for
  disagreeing with the pipeline would grade the pipeline against its own
  output.
- After a legitimate fixture addition, run
  `uv run python scripts/make_fixture_manifest.py` and commit the manifest.

## Contents

| Path | Purpose |
| --- | --- |
| `ground-truth/` | Volunteer GCP exports and masks |
| `ref-volume/` | Reference-volume annotations, images, results, and seam data |
| `sanborn01790_*/` | Frozen inputs and recorded results for golden/census volumes |
| `reference/` | Centerlines, boundaries, and rail inputs |
| `keymaps/` | Key-map index-page crops for street-index and prior-window work |
| `prod/`, `loc-catalog-chicago.json` | Cached bounds sources and LOC catalog data |
| `viewer-manifest.json` | Harvested viewer manifest for counterpart bounds hints |

Use `fixtures/*/results/` to understand a frozen prior result. The fixture tree
is evidence, not an input to rewrite from code or a source of live status.

`reference/` holds Chicago's inputs and is not a template for a new city.
`docs/ADDING-A-CITY.md` §2 states the centerline schema contract these files
satisfy, §2a names the free nationwide substitute (Census TIGER/Line ADDRFEAT)
for a city with no comparable dataset, and §2b says which City of Chicago
datasets these two are and where to re-download them.

## Important Traps

### Reference Inputs Are Not Interchangeable

`fixtures/ref-volume/` and `work/sanborn01790_006.5/` describe the same volume
but contain separate annotation passes. Their funnels are not comparable. Quote
the source of any result and do not treat a difference as a matcher regression.

Golden replays exercise pipeline stages from frozen inputs; they are not a full
CLI production run. See `tests/test_golden_replay.py`,
`tests/test_golden_new_volumes.py`, and the relevant records for coverage.

### A Volume Can Have No Origin Baseline At All

`sanborn01790_017` has none, and nothing should be built that implies otherwise.
The origin ran that LOC item end to end, but every page it read was a
`sanborn01790_018` scan: both items are titled "Vol. 1, 1906" and the origin's
image store was keyed on the shared OHMG slug `chicago_ill_1906_vol_1`, which
physically held only `_018`'s pixels. Its `28/95` funnel therefore describes
`_018`'s ground and cannot be repaired into an `_017` baseline.

The misfiled `fixtures/sanborn01790_017/` tree was deleted for this reason.
`autogeoref status` prints `-` in `_017`'s frozen column wherever the
row appears at all, and on a checkout with neither its `work/` tree nor served
tiles it prints no row. Both are the correct output, not a gap to fill. Never
reconstruct a baseline from `work/` — that
manufactures a bar we set for ourselves. Verify identity with
`scripts/audit_fixture_volume_identity.py` before trusting any tree
as a baseline.

### Ground Truth Needs Matching Pixels

Ground truth is usable for scoring only when the corresponding sheet images are
available. Pair images and GCPs by Library of Congress item ID, never by an
OHMG slug, which is not unique across editions.

Scoring a volume needs three things at once — pixels, a result tree, and pins
filed under the same LOC item id — and most volumes are missing one of them.
On 2026-08-14: 39 exports carry at least one usable pinned page, 80 volumes
have results, and **36 have both** — thirty-five Chicago volumes plus
Cleveland's `sanborn06648_072`. Three pinned volumes remain unprocessed
(`_107`, `_132`, `_138`), and all three are suburban: they are blocked by the
centerline reference rather than waiting their turn, so none of them lands
scoreable merely by being run. A ground-truth file with zero pinned layers is
an absence, not a gap to fill: `_001`–`_006`, `_021`, `_024`, `_034` and `_038`
are in that state deliberately. Crystal Lake and Staunton are configured cities
with no pinned volume at all, so they are counted at zero rather than omitted.

**Enumerate the scoreable set from the city configs, never from a glob.**
Cleveland's pins sit with everyone else's in `fixtures/ground-truth/`, but its
tree is `work/city2-probe/sanborn06648_072`, so `work/sanborn*` still excludes
it silently — and a filesystem walk is not the fix either, because `work/`
holds hundreds of experiment arms and A/B controls shaped exactly like live
volume trees. The work root is the half of that trap the pin move did not
close. Two commands also disagree about where pins live: `autogeoref status`
reads only `fixtures/ground-truth`, so `_090` and `_093` print `-` for ground
truth they have in `fixtures/prod`, while `autogeoref score` takes
`--ground-truth` repeatably and sees whatever it is given.

Five frozen ~40-sheet golden subsets sit under `fixtures/` (`_040`, `_041`,
`_089`, `_110`, `_130`) and are samples of pinned pages, not whole volumes.
All five name volumes that are also among the thirty-six scoreable ones — the
fixture subset and the live `work/` tree are different objects for the same
volume, and their funnels are not comparable. `_130` was the exception until
2026-08-14; it is now placed in `work/` as well, and its frozen 33-page subset
still reports separately from its 83-sheet tree.

The README's headline figures are that measurement, dated and scoped there.

Volunteer pins can be internally consistent yet wrong. Do not edit or exclude a
pin to improve a result, and never derive a correction from the pipeline's own
placement. Document audit findings instead — two known-defective `_040` pins
score as they do deliberately. Read a
sheet's printed scale before classifying an off-scale pin as defective.

`_040` carries a larger version of the same thing in its `work/` tree. On
fourteen of that tree's pages the human layer is compressed along ONE pixel axis,
to 0.60–0.90 of the width the pipeline implies at a matching height,
because a pin sits out near the sheet's outer margin under the name of a street
drawn well inside it. On the three checked against the scan, the placement's own
transform lands those junctions on the ink and the human's does not. Nothing
keyed on pin count, pin spread or fit residual sees any of it. The frozen
fixture subset is a different object and holds only three of the fourteen, so a
count taken there is not the volume's;
`scripts/audit_score_extent.py` is what measures either, over one volume or
over the corpus.
Run it before quoting a per-sheet score, and read its suspect column as a pointer
rather than a verdict: it names the side departing from its own volume's typical
scale, so it inverts on a sheet printed at a scale the rest of its volume is not
— which is a Cleveland sheet, not one of these.

### A Frozen `OK` Can Be A Defect Today's Pipeline Already Refuses

The recorded funnels predate several gates, so a frozen accept is evidence of
what the RECORDED run did, never of what this port would do. Reading one as a
live specimen inverts the finding. Two named sets exist because agents have
walked into this twice: `test_golden_new_volumes.LOO_GATE_DIFFS` and
`test_golden_replay.MATCH_DEPARTURES` enumerate the pages the port rejects and
the record accepted.

The sharpest instance: **22 of the 726 committed fixture records refit to a
POSITIVE determinant** — a mirrored placement — pinned as
`test_rescue.MIRRORED_FIXTURE_RECORDS`. Twenty are pre-fix rescue records whose
GCP set is degenerate on the pixel or the world side, so the determinant's SIGN
is numerical noise rather than a reflection. The remaining two are match
accepts the port already refuses (`ref-volume` p75, `_089` p15). **None is
evidence that a live gate is missing**, and the baseline is permanent: a
future migration rewrites `work/`, and this tree is read-only. Audit
`work/` when you want the live answer
(`scripts/audit_reflected_placements.py`).

A corollary that cost a day: a fixture sweep must glob `*/results`, not
`sanborn*/results`. `ref-volume` is the golden volume and matches neither.

### Region Splits Are Not Whole Sheets

Some volunteer layers georeference a crop such as `p10_1`, whose pixels have no
known offset into the whole sheet. `bounds.load_ground_truth` intentionally
excludes them. Do not widen the page parser to map a crop to page 10: it creates
a plausible but wrong transform.

## Related Material

- Fixture tests: `tests/test_fixture_integrity.py` and `tests/test_golden_*.py`
- Scoring extent: `scripts/audit_score_extent.py`
- Reflected placements in `work/`: `scripts/audit_reflected_placements.py`
