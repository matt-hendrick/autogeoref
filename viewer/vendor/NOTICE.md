# Vendored viewer dependencies

Pinned copies so the viewer (and its deploy bundle) is fully static and
self-contained — no CDN at serve time. Re-vendor deliberately, never by
letting a build step float versions.

| Asset | Version | Source | License |
|---|---|---|---|
| maplibre-gl.js / .css | 4.7.1 | https://unpkg.com/maplibre-gl@4.7.1/dist/ | BSD-3-Clause |
| maplibre-gl-compare.js / .css | 0.5.0 | https://unpkg.com/@maplibre/maplibre-gl-compare@0.5.0/dist/ | BSD-3-Clause (MapLibre fork of Mapbox GL Compare, ISC) |
| pmtiles.js | 3.2.1 | https://unpkg.com/pmtiles@3.2.1/dist/ | BSD-3-Clause |
| fonts/ (Fraunces, Newsreader) | Google Fonts css2 snapshot 2026-07-10 | fonts.googleapis.com / fonts.gstatic.com | OFL-1.1 |
| basemap/style-*.json | @protomaps/basemaps 5.7.2 | https://registry.npmjs.org/@protomaps/basemaps/-/basemaps-5.7.2.tgz | BSD-3-Clause |
| basemap/fonts/ (Noto Sans Regular / Medium / Italic glyph PBFs) | protomaps/basemaps-assets @ 028c18f | https://github.com/protomaps/basemaps-assets | OFL-1.1 (Noto) |
| basemap/sprites/ (light, grayscale, @2x) | protomaps/basemaps-assets @ 028c18f | https://github.com/protomaps/basemaps-assets | BSD-3-Clause |

`fonts/fonts.css` is the Google css2 response with each `fonts.gstatic.com`
URL rewritten to a local `font-NN.woff2` download (same axes as the original
viewer's `<link>`: Fraunces opsz 9..144 wght 400/600/900; Newsreader
ital/opsz 6..72 wght 400/500 + italic 400).

`basemap/` is the self-hosted vector basemap's static half; the PMTiles
archive it reads is named by `manifest.site.basemap.pmtiles` and uploaded
separately (`HANDOFF-VIEWER-DEPLOY-2026-07-24.md`). The style files are the
published Protomaps flavors, generated — not hand-edited — so a re-vendor is a
re-run, with two deployment facts patched in: `glyphs`/`sprite` point at the
vendored copies above, and `sources.protomaps.url` is left empty for the
viewer to fill from the manifest.

```sh
npm pack @protomaps/basemaps@5.7.2 && tar xzf protomaps-basemaps-5.7.2.tgz
node -e 'const fs=require("fs");import("./package/dist/esm/index.js").then(m=>{
  for (const f of ["grayscale","light"]) fs.writeFileSync("style-"+f+".json",
    JSON.stringify({version:8,
      glyphs:"vendor/basemap/fonts/{fontstack}/{range}.pbf",
      sprite:"vendor/basemap/sprites/"+(f==="grayscale"?"grayscale":"light"),
      sources:{protomaps:{type:"vector",url:""}},
      layers:m.layers("protomaps",m.namedFlavor(f),{lang:"en"})}, null, 1));})'
```

All 256 glyph ranges per stack are vendored (14 MB) rather than the Latin ones
a Chicago extract happens to need: a missing range is a silently unlabelled
feature, and which ranges a future city's labels reach is not knowable here.
