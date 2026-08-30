/* The viewer's pure decisions, kept apart from the DOM so they can be run
   directly by a test: era-label merging, district naming, layer stacking, the
   compare clamp, story-stop resolution, and the shared URL fragment.

   Nothing here touches `document`, `location` or a map — callers measure and
   pass the numbers in. Carries no city fact.

   A plain classic script, no bundler either side: a browser global here, a
   CommonJS export in node. */
const ViewerLib = (function () {
  /* The group a volume with no era joins. A layer whose year is unknown is
     still a layer, so it gets a chip like any other rather than being dropped:
     "no year" is a fact about the catalog, not a reason to hide the atlas. */
  const UNDATED = "undated";
  const eraOf = (volume) => volume.era || UNDATED;

  /* Era labels are display text with a leading year, so sort by that year
     rather than relying on numeric-string coercion. `undated` parses to NaN,
     which would sort unpredictably, so it is pinned LAST — a dated corpus
     keeps its exact order and the odd one out does not head the list. */
  const eraYear = (era) => Number.parseInt(era, 10) || 0;
  const compareErasNewestFirst = (a, b) => {
    if (a === UNDATED || b === UNDATED) return (a === UNDATED ? 1 : 0) - (b === UNDATED ? 1 : 0);
    return Number.parseInt(b, 10) - Number.parseInt(a, 10);
  };

  /* The selection as display text: adjacent decades merge into one range, and
     gaps keep a separator, so five chips read as a span, not a list. */
  function selectionLabel(chosen) {
    const isDecade = (e) => /^\d{4}s$/.test(e);
    // oldest first — but `undated` still trails, as it does in the chip row:
    // it has no year to place it among the others, at either end.
    const sorted = [...chosen].sort((a, b) => {
      if (a === UNDATED || b === UNDATED) return (a === UNDATED ? 1 : 0) - (b === UNDATED ? 1 : 0);
      return Number.parseInt(a, 10) - Number.parseInt(b, 10);
    });
    const runs = [];
    for (const era of sorted) {
      const last = runs[runs.length - 1];
      if (last && isDecade(last.end) && isDecade(era) &&
          Number.parseInt(era, 10) - Number.parseInt(last.end, 10) === 10) last.end = era;
      else runs.push({ start: era, end: era });
    }
    return runs.map(r => r.start === r.end ? r.start : r.start + "–" + r.end).join(" · ");
  }

  /* Fallback district label for volumes with no community-area names:
     configured coordinate bands (first match per axis, combined + collapsed),
     else a generic compass name relative to the home point. */
  function regionLabel(bounds, regionLabels, homePoint) {
    const [w, s, e, n] = bounds;
    const lat = (s + n) / 2, lng = (w + e) / 2;
    if (regionLabels) {
      const pick = (bands, val) => {
        for (const b of bands || []) {
          if (b.above != null && !(val > b.above)) continue;
          if (b.below != null && !(val < b.below)) continue;
          return b.label;
        }
        return "";
      };
      let label = (regionLabels.combine || "{lat} {lng}")
        .replace("{lat}", pick(regionLabels.lat, lat))
        .replace("{lng}", pick(regionLabels.lng, lng))
        .trim();
      for (const [from, to] of regionLabels.collapse || []) label = label.replace(from, to);
      return label || "District";
    }
    if (homePoint) {
      const ns = lat >= homePoint[1] ? "North" : "South";
      const ew = lng <= homePoint[0] ? "west" : "east";
      return ns + ew + " district";
    }
    return "District";
  }

  /* The box enclosing every volume in `list`, or null for an empty one — a
     union of nothing is not a box, and returning the seed would be an inverted
     one that a caller would hand straight to `fitBounds`. */
  function unionOf(list) {
    if (!list.length) return null;
    return list.reduce((u, v) => [
      Math.min(u[0], v.bounds[0]), Math.min(u[1], v.bounds[1]),
      Math.max(u[2], v.bounds[2]), Math.max(u[3], v.bounds[3]),
    ], [180, 90, -180, -90]);
  }

  const area = (v) => (v.bounds[2] - v.bounds[0]) * (v.bounds[3] - v.bounds[1]);

  /* Which district the atlas opens on: the SMALLEST one covering the city's
     configured zero point, else the largest on screen. Smallest-covering
     because a citywide sheet and a block sheet both contain the point and only
     the block sheet shows anything. Null when there is nothing to open on. */
  function startVolume(volumes, homePoint) {
    if (!volumes.length) return null;
    const covering = homePoint
      ? volumes.filter(v =>
          v.bounds[0] <= homePoint[0] && homePoint[0] <= v.bounds[2] &&
          v.bounds[1] <= homePoint[1] && homePoint[1] <= v.bounds[3])
      : [];
    return covering.length
      ? covering.reduce((a, b) => (area(b) < area(a) ? b : a), covering[0])
      : volumes.reduce((a, b) => (area(b) > area(a) ? b : a), volumes[0]);
  }

  /* Whether one layer draws: its era is switched on AND the reader has not
     hidden that district. Hiding is per district and independent of the era
     chips — a district hidden in one era stays hidden when its era comes back,
     because a control that silently forgets is worse than one that does not.
     Citywide era layers carry an id no district list can hide. */
  const layerVisibility = (id, era, selectedEras, hidden) =>
    selectedEras.has(era) && !hidden.has(id) ? "visible" : "none";

  /* The hidden districts a link carries, kept to ids the manifest still lists:
     a stale one hides nothing and would ride every forwarded link forever. */
  function hiddenFromLink(hash, key, knownIds) {
    const raw = linkText(hash, key);
    const known = new Set(knownIds);
    return new Set(raw ? raw.split(",").filter(id => known.has(id)) : []);
  }

  /* Layer ids bottom-first. With several eras visible at once, overlapping
     volumes need an order the eye can predict: strictly chronological, newest
     survey on top, so the map reads as the latest selected snapshot of each
     block and turning an era off reveals the older sheets beneath. A volume
     with no year of its own sorts at its era's year. */
  function stackOrder(added) {
    const key = (id) => {
      const m = added.get(id);
      return [eraYear(m.era), m.year || eraYear(m.era)];
    };
    return [...added.keys()].sort((a, b) => {
      const ka = key(a), kb = key(b);
      return (ka[0] - kb[0]) || (ka[1] - kb[1]);
    });
  }

  /* Hosts where borrowing a third party's raster tiles is nobody's problem. */
  const DEV_HOSTS = ["localhost", "127.0.0.1", "[::1]", ""];
  const DEV_ONLY_RASTER = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

  /* Which basemap a pane will try, decided apart from fetching it.

     A public deployment must own every tile it serves: third-party raster
     basemaps forbid this traffic shape, and the failure mode is a referer
     block and a blank map. So the raster default is offered on a dev host and
     nowhere else — a public host with nothing configured renders bare and says
     so, rather than borrowing somebody's tiles.

     Returns `{kind: "vector", styleHref}`, `{kind: "raster", tiles, dev}`, or
     `{kind: "bare", reason}`. Every one of them is renderable: the atlas is the
     product and no basemap failure may take the page with it. */
  function chooseBasemap(basemap, hostname, pane) {
    const config = basemap || {};
    if (config.type === "vector") {
      const styleHref = (config.styles || {})[pane];
      return styleHref
        ? { kind: "vector", styleHref }
        : { kind: "bare", reason: "no vector basemap style for the " + pane + " pane" };
    }
    if (config.tiles) return { kind: "raster", tiles: config.tiles, dev: false };
    if (DEV_HOSTS.includes(hostname)) return { kind: "raster", tiles: DEV_ONLY_RASTER, dev: true };
    return { kind: "bare", reason: "no site.basemap configured" };
  }

  /* Maps a requested compare-handle position to a legal one.

     `layout.width` is the map's width, `layout.edge` the margin that keeps the
     whole handle on screen at either end, and `layout.panelRight` the right
     edge of the panel when the panel floats over the map — null when it does
     not, because a docked panel takes no map width. A handle parked behind the
     floating panel can never be grabbed again. */
  function clampSlider(x, layout) {
    const edge = layout.edge;
    let min = edge;
    if (layout.panelRight != null) min = Math.max(min, layout.panelRight + edge);
    const max = Math.max(min, layout.width - edge);
    return Math.min(Math.max(Number.isFinite(x) ? x : min, min), max);
  }

  /* One arrow-key nudge and one coarse jump, as fractions of the map's width.
     Fine enough to line a street up, coarse enough to cross the map in ten. */
  const SWIPE_STEP = 0.02;
  const SWIPE_COARSE = 0.1;

  /* Where a key sends the compare divider, as a fraction of the map's width,
     or null for a key this does not answer.

     `fraction` is where the divider is now and `coarse` whether shift was
     held. Arrows step, Page keys and shift jump, Home and End go to the ends;
     up/right raise the fraction and down/left lower it, which is the slider
     convention a screen reader announces against.

     Returns a fraction in 0..1 and nothing narrower: which PIXELS are legal
     depends on the panel and the window, and that is `clampSlider`'s answer,
     applied after this one. Callers must not pre-clamp. */
  function swipeStep(key, fraction, coarse) {
    if (key === "Home") return 0;
    if (key === "End") return 1;
    // arrays rather than an object: a key named like an Object prototype
    // member ("constructor") reads as a truthy hit from a bare lookup
    const lower = ["ArrowLeft", "ArrowDown", "PageDown"];
    const raise = ["ArrowRight", "ArrowUp", "PageUp"];
    const direction = lower.includes(key) ? -1 : raise.includes(key) ? 1 : 0;
    if (!direction) return null;
    const jump = Boolean(coarse) || key === "PageUp" || key === "PageDown";
    const at = Number.isFinite(fraction) ? fraction : 0.5;
    const next = at + direction * (jump ? SWIPE_COARSE : SWIPE_STEP);
    return Math.min(Math.max(next, 0), 1);
  }

  /* How far down the map the compare handle sits, in pixels — the other axis
     of the same problem `clampSlider` solves on the horizontal.

     `layout.panelTop` is the top edge of a panel docked across the bottom of
     the map, `layout.mapHeight` the map's height, `layout.headroom` the depth
     of the chrome along the top edge, and `layout.radius` half the handle. The
     answer is the middle of the strip the panel leaves, held inside it where
     it fits. `null` for a panel that covers no part of the map, and for a
     layout that does not measure — both leave the stylesheet's own position
     alone, which is a position the reader can reach. */
  function handleTop(layout) {
    const { mapHeight, panelTop, headroom, radius } = layout;
    if (!Number.isFinite(panelTop) || !Number.isFinite(mapHeight)) return null;
    if (panelTop >= mapHeight) return null;
    const middle = (headroom + panelTop) / 2;
    const low = headroom + radius, high = panelTop - radius;
    // A strip thinner than the handle cannot hold it clear of both, and the
    // two overlaps are not equally bad: the panel is opaque across the whole
    // width and covers whatever it reaches, while the chrome along the top
    // edge is a couple of boxes at one side. So the handle is pushed up
    // against the panel rather than centred on a strip it would sink into.
    const wanted = high >= low ? Math.min(Math.max(middle, low), high) : high;
    return Math.round(Math.min(Math.max(wanted, radius), Math.max(radius, mapHeight - radius)));
  }

  /* What a page's query string says about the guided-story entry list: true to
     offer it, false to suppress it, null for a link with no opinion.

     Three answers rather than two, because the caller has a second reason to
     offer the list — the reader arrived on a story permalink — and an explicit
     `?<key>=0` has to beat that, or the off values mean nothing on the only
     visits that have another way in.

     Whether a `#story=` permalink OPENS its story is not decided here: the
     site produced those links, and they work under every answer above. */
  const STORIES_OFF = ["0", "false", "off", "no"];
  function storiesAsked(search, key) {
    const value = hashRead(search, key);          // null absent, "" for a bare key
    return value === null ? null : !STORIES_OFF.includes(value.toLowerCase());
  }

  /* Which stop a story opens at. An unknown or absent id is the first stop,
     never a -1 that would read as the last one. */
  const stopIndex = (stops, id) => Math.max(0, stops.findIndex(s => s.id === id));
  /* Held inside the story: at either end, `next`/`back` is a no-op rather than
     a re-fly to where the reader already is. */
  const clampStopIndex = (index, count) => Math.max(0, Math.min(index, count - 1));

  /* One hash holds every shareable view flag as `#key=value&key=value`, so a
     later feature adds a key instead of a second scheme. Keys the viewer does
     not own are carried through byte for byte — a `URLSearchParams` round trip
     would re-encode `/` and `,` and turn a bare `#anchor` into `#anchor=`, so
     one panel click would corrupt somebody else's key.

     A query string has the same grammar behind a different leading character,
     so `hashRead` serves both. Only the hash is ever WRITTEN. */
  const hashParts = (hash) => String(hash || "").replace(/^[#?]/, "").split("&").filter(Boolean);
  const partKey = (part) => part.split("=")[0];

  /* The value, "" for a key written with no value, null when the key is
     absent. Both of the first two must read as "this link says nothing" to a
     caller expecting a number: `Number(null)` is 0. */
  function hashRead(hash, key) {
    for (const part of hashParts(hash)) {
      if (partKey(part) !== key) continue;
      const eq = part.indexOf("=");
      return eq < 0 ? "" : part.slice(eq + 1);
    }
    return null;
  }

  /* The fragment with one key set, or removed when `value` is null — so a
     feature that stops applying leaves the URL as clean as it found it. Every
     other key keeps its place and its bytes. Returns "" for an empty result. */
  function hashWrite(hash, key, value) {
    const parts = hashParts(hash);
    const at = parts.findIndex((part) => partKey(part) === key);
    if (value === null) {
      if (at >= 0) parts.splice(at, 1);
    } else {
      const written = key + "=" + value;
      if (at < 0) parts.push(written); else parts[at] = written;   // keep its place
    }
    return parts.length ? "#" + parts.join("&") : "";
  }

  /* The query string with one key set — same grammar, same byte-for-byte
     treatment of every other key, a `?` on the front. Setting a key that is
     already there REPLACES it, so a caller that writes the same flag twice
     writes one key and not two. */
  const queryWrite = (search, key, value) =>
    hashWrite(search, key, value).replace(/^#/, "?");

  /* A link's numeric list for `key`, or null when the link says nothing about
     it. Both an absent key and a valueless one must read as nothing: a caller
     expecting a number gets `Number(null) === 0`, which silently means
     "divider hard left" on every plain visit. A list of the wrong length, or
     with anything unparseable in it, is nothing too. */
  function linkNumbers(hash, key, count) {
    const raw = hashRead(hash, key);
    if (!raw) return null;
    const parts = raw.split(",").map(Number);
    return parts.length === count && parts.every(Number.isFinite) ? parts : null;
  }
  const linkText = (hash, key) => hashRead(hash, key) || null;

  /* The camera as a link carries it. 5 d.p. of lng/lat is about a metre and
     2 d.p. of zoom is finer than one wheel notch, so a forwarded link
     describes the view without churning on sub-pixel drift. */
  const viewValue = (lng, lat, zoom) =>
    [lng.toFixed(5), lat.toFixed(5), zoom.toFixed(2)].join(",");

  /* The geocoder query with the configured ", City, ST" appended, unless it
     already names the city. The token matched is taken FROM the suffix — the
     city is configuration, not a fact this file may hold — and its grammar
     (a letter then word characters) is what keeps it safe to build a RegExp
     from. */
  function withCitySuffix(query, suffix) {
    if (!suffix) return query;
    const token = (suffix.match(/[A-Za-z][\w'-]*/) || [null])[0];
    if (token && new RegExp(token, "i").test(query)) return query;
    return query + suffix;
  }

  /* Where an address search goes, decided apart from sending it.

     Mapbox whenever a token is configured — `window.MAPBOX_TOKEN`, shipped by
     the deploy-time `config.js`. Nominatim is a dev-host convenience only:
     OSMF's public instance forbids this traffic shape, and the failure mode is
     a block and a search box that answers "No match" forever. So a public host
     with no token says search is unavailable rather than borrowing it, the
     same rule `chooseBasemap` applies to third-party tiles.

     `geocoder` is the manifest's block (`suffix`, `bbox` as
     west,south,east,north). Returns `{provider: "mapbox"|"nominatim", url}`,
     or `{provider: null, reason}` for a search that must not be sent.

     Mapbox is asked on the v6 endpoint. Both take `bbox` in the same
     west,south,east,north order, so the configured block is passed through
     unchanged; the ANSWER shape differs, and `geocodeHit` is the half that
     knows it. */
  function chooseGeocoder(query, geocoder, token, hostname) {
    const config = geocoder || {};
    const q = encodeURIComponent(withCitySuffix(query, config.suffix));
    const bbox = config.bbox;
    if (token) {
      return {
        provider: "mapbox",
        url: "https://api.mapbox.com/search/geocode/v6/forward?q=" + q + "&country=us&limit=1" +
             (bbox ? "&bbox=" + bbox.join(",") : "") +
             "&access_token=" + encodeURIComponent(token),
      };
    }
    if (!DEV_HOSTS.includes(hostname)) {
      return { provider: null, reason: "Address search is not configured on this site." };
    }
    return {
      provider: "nominatim",
      url: "https://nominatim.openstreetmap.org/search?format=json&limit=1" +
           (bbox ? "&bounded=1&viewbox=" + [bbox[0], bbox[3], bbox[2], bbox[1]].join(",") : "") +
           "&q=" + q,
    };
  }

  /* One geocoder's answer as `{lngLat, name}`, or null for no usable match.
     The two providers reply in different shapes and neither is trusted to be
     well formed — a truncated body must read as "no match", not throw.

     Mapbox v6 answers plain GeoJSON: the point is the feature's `geometry`,
     and the label its `properties`. Neither `center` nor `place_name` — the
     keys the v5 endpoint used — exists on a v6 feature, so reading those
     against a v6 body yields NaN and a silent "No match" on every valid
     address. That is why the URL and this function move together. */
  function geocodeHit(provider, payload) {
    // `Number(null)` and `Number("")` are both 0, which is a real place off
    // west Africa — so the RAW value is checked for being a number-ish thing
    // before it is coerced, or a body with null coordinates reads as a hit.
    const num = (value) =>
      (typeof value === "number" || (typeof value === "string" && value.trim() !== "")
        ? Number(value)
        : NaN);
    const point = (lng, lat, name) =>
      (Number.isFinite(lng) && Number.isFinite(lat) ? { lngLat: [lng, lat], name: name } : null);
    if (provider === "mapbox") {
      const f = (payload && payload.features || [])[0] || {};
      const coords = Array.isArray((f.geometry || {}).coordinates) ? f.geometry.coordinates : [];
      const props = f.properties || {};
      return point(num(coords[0]), num(coords[1]),
                   String(props.full_address || props.name || ""));
    }
    const f = (Array.isArray(payload) ? payload : [])[0] || {};
    const name = String(f.display_name || "").split(",").slice(0, 3).join(",");
    return point(num(f.lon), num(f.lat), name);
  }

  /* Which city's manifest this page load is for.

     One page, one city: every field in a manifest comes from one city's
     config, so the manifests are per city and the page is told which by
     `?city=<slug>` against the index published beside it. Returns
     `{manifest}` to load one, `{choose}` to offer the list, or `{error}` when
     the named city is not there — never nothing, because a page that renders
     neither a map nor a reason is the failure this whole thing is about.

     `cities` is the index as published (null when there is none). With no
     index at all the caller falls back to a manifest beside the page, which
     is the one-city layout a deploy bundle and a hand-made directory both
     have. */
  function chooseCity(cities, search) {
    const list = (cities && Array.isArray(cities.cities) ? cities.cities : [])
      .filter(entry => entry && entry.slug && entry.manifest);
    const wanted = hashRead(search, "city");
    if (wanted) {
      const hit = list.find(entry => entry.slug === wanted);
      return hit ? { manifest: hit.manifest, city: hit } : { error: wanted, choose: list };
    }
    if (list.length === 1) return { manifest: list[0].manifest, city: list[0] };
    if (list.length) return { choose: list };
    return { error: null, choose: [] };
  }

  /* Whether this browser will give the map a 3D context.

     `make` takes a context name and returns the context or null, so a test runs
     this without a canvas. Both names are tried because the map falls back to
     WebGL 1; a throw counts as a refusal, since a failing driver raises instead
     of returning null. The probe releases what it opens: browsers cap live
     contexts and the page builds two maps. */
  function webglAvailable(make) {
    for (const name of ["webgl2", "webgl"]) {
      let gl = null;
      try {
        gl = make(name);
      } catch (err) { /* a refusal, the same answer as null */ }
      if (!gl) continue;
      try {
        const lose = gl.getExtension("WEBGL_lose_context");
        if (lose) lose.loseContext();
      } catch (err) { /* best effort: releasing must not change the answer */ }
      return true;
    }
    return false;
  }

  return {
    UNDATED, eraOf, chooseCity, webglAvailable,
    compareErasNewestFirst, selectionLabel, regionLabel, stackOrder, chooseBasemap,
    unionOf, area, startVolume, layerVisibility, hiddenFromLink,
    clampSlider, swipeStep, handleTop, storiesAsked, stopIndex, clampStopIndex,
    hashRead, hashWrite, queryWrite,
    linkNumbers, linkText, viewValue, withCitySuffix, chooseGeocoder, geocodeHit,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = ViewerLib;
