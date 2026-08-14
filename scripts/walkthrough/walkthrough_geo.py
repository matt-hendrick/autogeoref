"""World-space figures: centerlines, warped scans, and served tiles.

Everything here works in EPSG:3857, the frame the pipeline's own placements are
recorded in. A sheet is warped by the affine its own recorded control points
determine, so what a plate shows is the placement, not a redrawing of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import walkthrough_theme as theme
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from autogeoref.affine import TO_3857

Box = tuple[float, float, float, float]
XY = tuple[float, float]
RGB = tuple[int, int, int]
#: A 2x3 placement, [X, Y] = coef @ [1, px, py].
Coef = NDArray[np.float64]

__all__ = [
    "TO_3857",
    "Box",
    "Coef",
    "MapCanvas",
    "MapView",
    "bounds_of",
    "canvas_for",
    "draw_streets",
    "polygon",
    "rings",
    "scale_bar",
    "tiles_image",
    "view_for",
    "warp_sheet",
]


@dataclass(frozen=True)
class MapView:
    """A web-mercator window drawn into a plate box."""

    left: float
    top: float
    #: plate pixels per 3857 metre
    ppm: float
    #: 3857 coordinate of the view's top-left corner
    origin: XY
    size: tuple[int, int]

    def xy(self, x: float, y: float) -> XY:
        """A 3857 point on the plate."""
        return (
            self.left + (x - self.origin[0]) * self.ppm,
            self.top + (self.origin[1] - y) * self.ppm,
        )

    def lnglat(self, lng: float, lat: float) -> XY:
        return self.xy(*TO_3857.transform(lng, lat))

    @property
    def rect(self) -> Box:
        return (self.left, self.top, self.left + self.size[0], self.top + self.size[1])


@dataclass
class MapCanvas:
    """A map drawn on its own surface, so nothing can spill onto the plate."""

    view: MapView
    image: Image.Image
    draw: ImageDraw.ImageDraw
    at: tuple[int, int]

    def commit(self, plate: theme.Plate) -> None:
        """Paste the finished map into the plate and rule its edge."""
        plate.image.paste(self.image.convert("RGB"), self.at)
        plate.draw.rectangle(
            (self.at[0], self.at[1], self.at[0] + self.image.width, self.at[1] + self.image.height),
            outline=theme.PLATE_EDGE,
            width=1,
        )


def canvas_for(
    box: Box, bounds3857: Box, *, pad: float = 0.06, fill: RGB = (250, 246, 236)
) -> MapCanvas:
    """A clipped map surface for ``bounds3857``, sized to ``box``."""
    view = view_for((0, 0, box[2] - box[0], box[3] - box[1]), bounds3857, pad=pad)
    image = Image.new("RGBA", view.size, (*fill, 255))
    return MapCanvas(
        view=view, image=image, draw=ImageDraw.Draw(image), at=(int(box[0]), int(box[1]))
    )


def view_for(box: Box, bounds3857: Box, *, pad: float = 0.06) -> MapView:
    """A view of ``bounds3857`` that fills ``box`` without distorting it."""
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    x0, y0, x1, y1 = bounds3857
    span_x, span_y = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    span_x *= 1 + 2 * pad
    span_y *= 1 + 2 * pad
    ppm = min(w / span_x, h / span_y)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return MapView(
        left=left,
        top=top,
        ppm=ppm,
        origin=(cx - w / (2 * ppm), cy + h / (2 * ppm)),
        size=(int(w), int(h)),
    )


def rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    kind = geometry.get("type")
    if kind == "LineString":
        return [geometry["coordinates"]]
    if kind in ("MultiLineString", "Polygon"):
        return list(geometry["coordinates"])
    if kind == "MultiPolygon":
        return [ring for part in geometry["coordinates"] for ring in part]
    return []


def draw_streets(
    draw: ImageDraw.ImageDraw,
    view: MapView,
    features: list[dict[str, Any]],
    *,
    fill: tuple[int, int, int] = theme.WORLD,
    width: int = 2,
    highlight: set[str] | None = None,
    name_property: str = "street_nam",
) -> None:
    """Modern centerlines inside the view; ``highlight`` names draw heavier."""
    for feature in features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        named = str(feature.get("properties", {}).get(name_property) or "").upper()
        hot = highlight is not None and any(key in named for key in highlight)
        if highlight is not None and not hot:
            continue
        for ring in rings(geometry):
            pts = [view.lnglat(c[0], c[1]) for c in ring]
            if len(pts) > 1:
                draw.line([p for pt in pts for p in pt], fill=fill, width=width * (2 if hot else 1))


def polygon(
    draw: ImageDraw.ImageDraw,
    view: MapView,
    geometry: dict[str, Any],
    *,
    outline: tuple[int, int, int],
    width: int = 3,
) -> None:
    """A 4326 polygon's outline, in view coordinates."""
    for ring in rings(geometry):
        pts = [view.lnglat(c[0], c[1]) for c in ring]
        if len(pts) > 2:
            draw.line([p for pt in pts for p in pt] + list(pts[0]), fill=outline, width=width)


def _plate_from_small(coef: Coef, scale: float, view: MapView) -> NDArray[np.float64]:
    """3x3 matrix taking a SMALL-frame pixel to a plate pixel."""
    full = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]], dtype=float)
    world = np.array(
        [[coef[0][1], coef[0][2], coef[0][0]], [coef[1][1], coef[1][2], coef[1][0]], [0, 0, 1]]
    )
    plate = np.array(
        [
            [view.ppm, 0, view.left - view.origin[0] * view.ppm],
            [0, -view.ppm, view.top + view.origin[1] * view.ppm],
            [0, 0, 1],
        ]
    )
    product: NDArray[np.float64] = plate @ world @ full
    return product


def warp_sheet(
    small: Image.Image,
    coef: Coef,
    scale: float,
    view: MapView,
    *,
    mask: dict[str, Any] | None = None,
) -> Image.Image:
    """The scan placed in the view by its own recorded affine, as RGBA.

    ``mask`` is the sheet's cutline, in 4326; without one the whole page paints.
    """
    matrix = _plate_from_small(coef, scale, view)
    inverse = np.linalg.inv(matrix)
    out = small.convert("RGB").transform(
        view.size,
        Image.Transform.AFFINE,
        tuple(inverse[:2].flatten()),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(255, 255, 255),
    )
    alpha = Image.new("L", view.size, 0)
    corners = [(0, 0), (small.width, 0), (small.width, small.height), (0, small.height)]
    hull = [
        (
            float(matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) - view.left,
            float(matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) - view.top,
        )
        for x, y in corners
    ]
    ImageDraw.Draw(alpha).polygon(hull, fill=255)
    if mask is not None:
        cut = Image.new("L", view.size, 0)
        cutter = ImageDraw.Draw(cut)
        for ring in rings(mask):
            pts = [view.lnglat(c[0], c[1]) for c in ring]
            if len(pts) > 2:
                cutter.polygon([(p[0] - view.left, p[1] - view.top) for p in pts], fill=255)
        alpha = Image.composite(alpha, Image.new("L", view.size, 0), cut)
    out.putalpha(alpha)
    return out


#: The web-mercator half-circumference, in metres.
HALF = 20037508.342789244


def _tile_span(zoom: int) -> float:
    return float(2 * HALF / (2**zoom))


def tiles_image(tiles_root: Path, view: MapView, *, ext: str = "webp") -> Image.Image:
    """The served XYZ tiles composited over the view, as RGBA."""
    zoom = _best_zoom(view)
    span = _tile_span(zoom)
    out = Image.new("RGBA", view.size, (0, 0, 0, 0))
    x0 = view.origin[0]
    y0 = view.origin[1]
    x1 = x0 + view.size[0] / view.ppm
    y1 = y0 - view.size[1] / view.ppm
    for tx in range(int((x0 + HALF) // span), int((x1 + HALF) // span) + 1):
        for ty in range(int((HALF - y0) // span), int((HALF - y1) // span) + 1):
            path = tiles_root / str(zoom) / str(tx) / f"{ty}.{ext}"
            if not path.is_file():
                continue
            tile = Image.open(path).convert("RGBA")
            side = max(round(span * view.ppm), 1)
            tile = tile.resize((side, side), Image.Resampling.LANCZOS)
            at = view.xy(tx * span - HALF, HALF - ty * span)
            out.alpha_composite(tile, (round(at[0] - view.left), round(at[1] - view.top)))
    return out


def _best_zoom(view: MapView) -> int:
    """The zoom whose native tile resolution is closest to the view's."""
    target = math.log2(2 * HALF * view.ppm / 256)
    return max(0, min(20, round(target)))


def bounds_of(points: list[XY], *, pad_m: float = 60.0) -> Box:
    """A padded 3857 bounding box over ``points``."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)


def scale_bar(draw: ImageDraw.ImageDraw, view: MapView, at: XY, metres: float = 200.0) -> None:
    """A metre bar, so a reader can size what they are looking at."""
    x, y = at
    length = metres * view.ppm
    draw.line((x, y, x + length, y), fill=theme.INK, width=3)
    for end in (x, x + length):
        draw.line((end, y - 6, end, y + 6), fill=theme.INK, width=3)
    theme.text(draw, (x, y - 10), f"{metres:.0f} m", size=theme.TINY, fill=theme.INK, anchor="ld")
