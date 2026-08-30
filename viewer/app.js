/* The public viewer's behaviour. A plain classic script — no bundler, no
   module graph, no npm in the deploy path — loaded by index.html after the
   vendored maplibre, compare and pmtiles globals.

   City-fact-free by contract: every city-specific string, coordinate and
   credit comes from manifest.json's `site` block. */

/* The overlay's terminal state: reason shown, fade cancelled. File scope so the
   boot's `.catch()` can reach it after the boot failed. `done` is removed
   because the deadline may already have fired: a boot slow enough to outlast it
   and then throw would otherwise leave the reason behind a faded overlay. */
function overlayStop(message) {
  const overlay = document.getElementById("loading");
  if (!overlay) return;
  overlay.textContent = message;
  overlay.classList.remove("done");
  overlay.classList.add("choose");
}

(async function () {
  // lib.js: the decisions that need no DOM, so a test can run them directly
  const {
    eraOf, chooseCity,
    compareErasNewestFirst, selectionLabel, regionLabel, stackOrder, chooseBasemap,
    unionOf, startVolume, layerVisibility, hiddenFromLink,
    clampSlider, swipeStep, handleTop, storiesAsked, stopIndex, clampStopIndex,
    hashRead, hashWrite, queryWrite,
    linkNumbers, linkText, viewValue, chooseGeocoder, geocodeHit,
    webglAvailable, readWithRetry,
  } = ViewerLib;

  const loading = document.getElementById("loading");

  // ---------- which city's page this is ----------
  // Manifests are per city (`<slug>/manifest.json`); `cities.json` lists what
  // is published here, and a deploy bundle ships a one-entry copy so public and
  // local resolve identically.
  //
  // ABSENT and UNREADABLE must not collapse: no index means a one-city
  // directory with its manifest beside the page; a 500 or a truncated read is a
  // server fault, and calling that "manifest.json is missing" hides it.
  async function readJson(href) {
    let response;
    try {
      response = await fetch(href);
    } catch (err) {
      return { unreadable: String(err) };
    }
    if (response.status === 404) return { absent: true };
    if (!response.ok) return { unreadable: "HTTP " + response.status };
    try {
      return { data: await response.json() };
    } catch (err) {
      return { unreadable: String(err) };
    }
  }

  const cityIndex = await readJson("cities.json");
  const picked = cityIndex.data
    ? chooseCity(cityIndex.data, location.search)
    : { manifest: "manifest.json", choose: [] };

  /* A page with no atlas to draw still has to say so — the failure this whole
     resolution exists to stop is a blank page with nothing on it. */
  function explain(message, offer) {
    overlayStop(message);
    for (const entry of offer || []) {
      const link = document.createElement("a");
      link.href = "?city=" + encodeURIComponent(entry.slug);
      link.textContent = entry.name || entry.slug;
      loading.appendChild(link);
    }
  }

  if (!picked.manifest) {
    explain(
      picked.error
        ? `No atlas published here for “${picked.error}”.`
        : picked.choose.length
          ? "Choose an atlas:"
          : "No atlas is published here yet.",
      picked.choose);
    return;
  }

  // Paths INSIDE a manifest are relative to the manifest, not to the page:
  // that is what lets one page directory serve several cities' archives and
  // story images without either side knowing where the other lives.
  const manifestHref = picked.manifest;
  const manifestBase = new URL(manifestHref, location.href);
  const read = await readJson(manifestHref);
  const manifest = read.data;
  if (!manifest) {
    // Three different situations, and a stranger who has published nothing
    // meets the first of them. Telling them "could not load manifest.json"
    // reads as a fault in a viewer that is working exactly as it should.
    explain(
      cityIndex.unreadable
        ? `The list of atlases here could not be read (${cityIndex.unreadable}).`
        : cityIndex.absent && read.absent
          ? "No atlas is published here yet."
          : read.absent
            ? `The atlas listed as “${manifestHref}” is not here.`
            : `Could not load ${manifestHref} (${read.unreadable}).`,
      picked.choose);
    return;
  }
  // A manifest that parses is not yet a manifest. Without this the page threw
  // on the first loop over `volumes` and the loading overlay never resolved AT
  // ALL — no atlas and no reason, which is the one outcome every other branch
  // here exists to avoid.
  const vols = Array.isArray(manifest.volumes) ? manifest.volumes : null;
  if (!vols) {
    explain(`${manifestHref} is not a viewer manifest.`, picked.choose);
    return;
  }
  if (!vols.length) {
    explain("No layers are published for this atlas yet.", picked.choose);
    return;
  }
  const site = manifest.site || {};

  // Generic fallbacks: a manifest with no `site` block still renders a
  // working viewer with neutral wording (the generalization contract).
  const cityName = site.name || "Atlas";
  const S = {
    title: site.title || cityName + " {era} / Now",
    kicker: site.kicker || "Fire Insurance Atlas",
    heading: site.heading || cityName + " {era}",
    heading_note: site.heading_note ?? "against now",
    dek: site.dek || "Drag the divider to slide between the historical atlas and today.",
    loading_text: site.loading_text || "loading the atlas…",
    now_label: site.now_label || "NOW",
    default_era: site.default_era || null,
    default_eras: site.default_eras || null,
    home_point: site.home_point || null,          // [lng, lat]
    footer_source_html: site.footer_source_html || "",
    // `default_credits_html` is the pre-rename spelling. A manifest on disk
    // outlives the code that wrote it, so a site published before the rename
    // keeps its credit until it is republished. Drop the fallback once every
    // served manifest has been rebuilt.
    optional_credits_html: site.optional_credits_html || site.default_credits_html || "",
    era_credits: site.era_credits || {},
    // bias only (suffix, bbox); WHICH geocoder answers is `chooseGeocoder`'s
    geocoder: site.geocoder || {},
    basemap: Object.assign({
      type: "raster",
      attribution: "© OpenStreetMap contributors",
      maxzoom: 19,
    }, site.basemap || {}),
    region_labels: site.region_labels || null,
    // Optional guided stories. Absent or empty means NO story UI is created at
    // all — not a disabled button, nothing — which is what keeps this out of
    // the way of a city that configures none.
    stories: Array.isArray(site.stories) ? site.stories : [],
  };
  loading.textContent = S.loading_text;
  document.getElementById("kicker").textContent = S.kicker;
  document.getElementById("h1-note").textContent = S.heading_note;
  document.getElementById("dek").textContent = S.dek;
  document.getElementById("plate-now").textContent = S.now_label;

  // eras present = every era with volumes, newest first. Built from `eraOf`,
  // so a volume the catalog gave no year joins the single `undated` group
  // instead of vanishing; with no volumes at all the row is empty and
  // `renderEras` draws nothing.
  const eraSet = new Set();
  for (const v of vols) eraSet.add(eraOf(v));
  const eras = [...eraSet].sort(compareErasNewestFirst);
  // Several eras can be on at once: eras that barely overlap compose into one
  // continuous city. `default_eras` (list) wins over the legacy `default_era`.
  const configuredDefault = Array.isArray(S.default_eras) && S.default_eras.length
    ? S.default_eras
    : (S.default_era ? [S.default_era] : []);
  const selectedEras = new Set(configuredDefault.filter(e => eras.includes(e)));
  if (!selectedEras.size && eras.length) selectedEras.add(eras[0]);

  const eraVols = (era) => vols.filter(v => eraOf(v) === era);

  // PMTiles is the only atlas tile source (object storage + CDN, no server),
  // and — where a city configures a vector basemap — the basemap's too.
  // Volumes merged from an old-generator manifest may still carry a legacy
  // raster `tiles` URL template; nothing writes one, and it is honored here
  // only so a hand-kept manifest still draws.
  const protocol = new pmtiles.Protocol();
  maplibregl.addProtocol("pmtiles", protocol.tile);

  // The below is a hack because some of the PMTiles are greater than the cacheable
  // size limit of some edges. To avoid not rendering those on first read, we retry for that failure.
  // `readWithRetry` owns the retry and the abort; it is in lib.js so a test can
  // run it without a browser.
  class RetryingSource extends pmtiles.FetchSource {
    getBytes(offset, length, signal, etag) {
      return readWithRetry((owned) => super.getBytes(offset, length, owned, etag), signal);
    }
  }

  // Registering is what puts that source in the path: the protocol resolves an
  // unregistered `pmtiles://` URL by building its own source, so a key that does
  // not match what `tilev4` strips is a silent no-op. It strips exactly the
  // scheme, so the key is the href these two functions already build.
  function pmtilesUrl(href) {
    if (!protocol.get(href)) protocol.add(new pmtiles.PMTiles(new RetryingSource(href)));
    return "pmtiles://" + href;
  }

  // Two bases, and the difference is real. Volume archives are written by the
  // manifest builder RELATIVE TO THE MANIFEST, so they resolve against
  // it. The basemap block is verbatim config whose sibling `styles` name files
  // vendored beside the PAGE, so the whole block resolves against the page. In
  // a deploy bundle the two are the same directory and the distinction is moot.
  const archiveUrl = (path) => pmtilesUrl(new URL(path, manifestBase).href);
  const basemapArchiveUrl = (path) => pmtilesUrl(new URL(path, location.href).href);

  // Two flavors of one basemap: `atlas` sits under the historical sheets and
  // stays muted so their ink reads; `now` is the modern city at full contrast.
  // WHICH basemap is lib.js's `chooseBasemap`; this is the fetching half. The
  // atlas is the product, so no basemap failure may take the page with it:
  // every path here returns a renderable style, and `basemapDrawn` keeps the
  // footer from crediting a basemap nobody can see.
  const RASTER_BASEMAP = "basemap";
  let vectorBasemapSource = null;
  let basemapDrawn = false;
  const bareStyle = () => ({
    version: 8,
    sources: {},
    layers: [{ id: "bg", type: "background", paint: { "background-color": "#14110c" } }],
  });

  async function baseStyle(pane) {
    const choice = chooseBasemap(S.basemap, location.hostname, pane);
    if (choice.kind === "bare") {
      console.warn(choice.reason + " — the atlas is drawn over a bare background");
      return bareStyle();
    }
    if (choice.kind === "vector") {
      const style = await fetch(choice.styleHref).then(r => r.ok ? r.json() : null).catch(() => null);
      if (!style) {
        console.error(
          `basemap style ${choice.styleHref} did not load — drawing the atlas over a bare background`);
        return bareStyle();
      }
      // The vendored flavors ship with an empty source URL: where the archive
      // lives is a deployment fact from the manifest, not a style fact.
      for (const [id, source] of Object.entries(style.sources || {})) {
        if (source.type !== "vector") continue;
        source.url = basemapArchiveUrl(S.basemap.pmtiles);
        vectorBasemapSource = id;
      }
      // NOT credited here either: the style JSON is vendored in this bundle
      // and so is always present, while the ARCHIVE it points at is the thing
      // a deploy can get wrong. See `creditBasemapOnFirstTile`.
      return style;
    }
    const style = bareStyle();
    style.sources[RASTER_BASEMAP] = {
      type: "raster", tiles: [choice.tiles], tileSize: 256,
      maxzoom: S.basemap.maxzoom, attribution: S.basemap.attribution,
    };
    style.layers.push({
      id: "base", type: "raster", source: RASTER_BASEMAP,
      paint: pane === "atlas"
        ? { "raster-saturation": -1, "raster-brightness-max": 0.35, "raster-opacity": 0.9 }
        : {},
    });
    // NOT credited here: see `creditBasemapOnFirstTile`
    return style;
  }

  // Everything waiting on the basemap SOURCE, on the one event that reports it:
  // a basemap whose tiles or archive 404 fires neither `load` nor `error`.
  //
  //  * The credit, on a landed TILE. Crediting at style build or parse says
  //    nothing about whether a visitor can see a basemap — what a deployment
  //    gets wrong is the ARCHIVE — and a false credit misattributes a third
  //    party whose data is not on the page.
  //  * The max-bounds clamp, whose footprint arrives with archive metadata and
  //    is not readable at style load.
  //
  // A vector style naming no vector source credits nothing: the safe direction.
  function watchBasemapSource(map) {
    map.on("sourcedata", (event) => {
      if (event.sourceId !== (vectorBasemapSource || RASTER_BASEMAP)) return;
      clampToBasemap(map);
      if (basemapDrawn || !event.tile || event.tile.state !== "loaded") return;
      basemapDrawn = true;
      applySelection();   // recompose the credit line now that it is true
    });
  }

  // start view: the union of known districts, else the configured home point
  const view = vols.length
    ? { bounds: unionOf(vols), fitBoundsOptions: { padding: 40 } }
    : S.home_point
      ? { center: S.home_point, zoom: 12 }
      : { center: [0, 0], zoom: 2 };

  // `drawAtlas` (declared below, hoisted) reads this, so it must exist before
  // the first map event can reach it.
  let atlasDrawn = false;

  // ---------- a browser that cannot draw a map at all ----------
  // Every map here is WebGL, and a browser can refuse a 3D context: driver
  // blocked, acceleration off, remote session. Constructing the map anyway
  // throws inside the vendored library, which took the rest of the boot with it.
  // No city list: every one would fail identically.
  if (!webglAvailable(name => document.createElement("canvas").getContext(name))) {
    explain(
      "This atlas needs WebGL, and this browser is not providing it. Enabling " +
      "hardware acceleration, updating the graphics driver, or another browser " +
      "will show the map.");
    return;
  }

  // ---------- when the overlay comes down ----------
  // On the first tile that PAINTS, not on `style.load`: the style parses long
  // before anything is on screen, and dismissing there left both panes grey for
  // another ten to twenty seconds behind a drawn panel. Any tile in the left
  // map counts, atlas or basemap — the question is whether the pane is empty.
  //
  // The deadline is the guarantee, and why a tile is watched rather than `load`
  // or `idle`: an archive whose bytes never arrive fires none of those. A tile
  // may only clear the overlay EARLIER, never hold it past the deadline. Armed
  // here, before any map exists — inside `drawAtlas` it needed a loaded style
  // first, so it covered a missing archive but not a map never built.
  //
  // 20s rather than tighter: past it nothing has loaded and the message beats
  // the grey pane; cutting a slow-but-working load short is the whole defect.
  const LOADING_DEADLINE_MS = 20000;
  let loadingCleared = false;
  function clearLoading() {
    // A reason already on screen outranks the fade: fading it would leave a
    // blank page saying nothing.
    if (loadingCleared || loading.classList.contains("choose")) return;
    loadingCleared = true;
    loading.classList.add("done");
  }
  setTimeout(clearLoading, LOADING_DEADLINE_MS);

  // Both styles BEFORE either map: a map fires `style.load` a tick after it is
  // constructed, so an await between the two constructors would let the left
  // map's fire before anything is listening — and `drawAtlas` below waits on
  // exactly that event.
  const [atlasStyle, nowStyle] = await Promise.all([baseStyle("atlas"), baseStyle("now")]);

  // LEFT map: muted basemap + the atlas
  const before = new maplibregl.Map(Object.assign({
    container: "before", style: atlasStyle, attributionControl: false,
  }, view));
  // The atlas is drawn as soon as the STYLE is ready, which is all `addSource`
  // needs. Waiting for `load` hands the page to the basemap instead: `load`
  // also waits on the basemap's tiles, and a raster basemap whose tiles 404 is
  // reported by no event at all — not `load`, not `error`, not `idle` — so the
  // loading screen never cleared and the atlas never drew, for a layer that is
  // not even the product.
  before.on("style.load", drawAtlas);

  // RIGHT map: the modern city, full color
  const after = new maplibregl.Map(Object.assign({
    container: "after", style: nowStyle, attributionControl: false,
  }, view));
  after.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  window.beforeMap = before; window.afterMap = after;  // debugging convenience
  watchBasemapSource(before);
  watchBasemapSource(after);

  // Nothing here handles `webglcontextlost`: the vendored map calls
  // preventDefault() on it and rebuilds the painter on `webglcontextrestored`,
  // so a lost context recovers on its own. Reporting it would put an overlay
  // over a page that heals, and the software rasteriser drops contexts under
  // load for reasons no page can act on.

  // A self-hosted basemap is an EXTRACT: it stops at the coverage box, and a
  // user who zooms out otherwise ends up in an empty void with a city-shaped
  // patch floating in it. The archive states its own footprint, so hold both
  // panes inside it. (A raster basemap covers the world; nothing to clamp.)
  function clampToBasemap(map) {
    const source = vectorBasemapSource && map.getSource(vectorBasemapSource);
    const b = source && source.bounds;
    if (b) map.setMaxBounds([[b[0], b[1]], [b[2], b[3]]]);
  }

  // opacity control
  const slider = document.getElementById("opacity");
  const out = document.getElementById("op-out");
  slider.addEventListener("input", () => {
    const val = slider.value / 100;
    out.textContent = slider.value + "%";
    slider.style.setProperty("--fill", slider.value + "%");
    for (const id of added.keys()) before.setPaintProperty(id, "raster-opacity", val);
  });

  // layer id -> {era, year}, for opacity, era toggling, and stacking
  const added = new Map();

  // Districts the reader has switched off one at a time, by volume id. The
  // decision it feeds is lib.js's; this is the state and the wiring.
  const hidden = new Set();
  const visibility = (id, era) => layerVisibility(id, era, selectedEras, hidden);

  // bottom-first, so each moveLayer lifts its layer above the one before it
  function restack() {
    for (const id of stackOrder(added)) before.moveLayer(id);  // no beforeId: to the top
    raiseStoryOverlay();  // an outline never goes under the paint it marks up
  }

  function pmSource(path) {
    // tileSize MUST be 256: gdal2tiles bakes 256px tiles, and declaring 512
    // makes MapLibre stretch them 2x — fine-print text goes blurry at every
    // zoom (user-reported on the live viewer, 2026-07-10)
    return { type: "raster", url: archiveUrl(path), tileSize: 256 };
  }

  function addVolume(v) {
    // No supersession: a district always draws. The old rule dropped districts
    // UNLESS this pipeline had placed them; every district is this pipeline's
    // now, so the exception swallowed the rule and a re-baked district could be
    // baked, manifested and invisible.
    if (added.has(v.id)) return;
    let source;
    if (v.pmtiles) {
      source = pmSource(v.pmtiles);
    } else if (v.tiles) {
      source = { type: "raster", tiles: [v.tiles], tileSize: 512,
                 bounds: v.bounds, minzoom: 10, maxzoom: 21 };
    } else {
      return;  // list-only volume (its imagery is not published locally)
    }
    added.set(v.id, { era: eraOf(v), year: v.year });
    before.addSource(v.id, source);
    before.addLayer({ id: v.id, type: "raster", source: v.id,
      layout: { visibility: visibility(v.id, eraOf(v)) },
      paint: { "raster-opacity": slider.value / 100, "raster-fade-duration": 150 } });
  }

  function addAllLayers() {
    for (const v of vols) addVolume(v);
    restack();
  }

  function eraCredits(era) {
    return S.era_credits[era] || S.optional_credits_html;
  }

  function applySelection() {
    for (const [id, m] of added)
      before.setLayoutProperty(id, "visibility", visibility(id, m.era));
    restack();  // undo any district-click raise

    const label = selectionLabel(eras.filter(e => selectedEras.has(e)));
    document.title = S.title.replace("{era}", label || "—");
    const plate = document.getElementById("plate-era");
    plate.textContent = label;
    plate.style.display = label ? "" : "none";
    document.getElementById("h1-head").textContent = S.heading.replace("{era}", label || "—");
    // one credit per distinct source across the selection, oldest era first.
    // Dedupe WHOLE credit lines: a credit is HTML and may carry a link, and
    // splitting on the separator cuts an anchor in half — the opening fragment
    // then dedupes away against another era's, leaving one era's credit as bare
    // text with no link and no error.
    const credits = [];
    for (const era of [...eras].reverse()) {
      if (!selectedEras.has(era)) continue;
      const credit = eraCredits(era);
      if (credit && !credits.includes(credit)) credits.push(credit);
    }
    const creditLine = [
      S.footer_source_html,
      ...credits,
      // never credit a basemap that failed to load: the bare-background pane
      // owes nobody, and a false credit is the one part of the failure a
      // visitor would actually read
      basemapDrawn ? "basemap " + S.basemap.attribution : "",
    ].filter(Boolean).join(" · ");
    // the same line in both places: the footer reads well with the panel open,
    // and the sources control is what a collapsed panel leaves behind
    // eslint-disable-next-line no-unsanitized/property -- composed from configured credit fragments, which ARE HTML: they carry the source links a licence obliges
    document.getElementById("footer").innerHTML = creditLine;
    // eslint-disable-next-line no-unsanitized/property -- the same configured HTML, second home
    document.getElementById("attrib-text").innerHTML = creditLine;
    document.querySelectorAll("#eras .era").forEach(el =>
      el.classList.toggle("active", selectedEras.has(el.dataset.era)));
    renderList();
  }

  function toggleEra(era) {
    if (selectedEras.has(era)) {
      selectedEras.delete(era);
    } else {
      selectedEras.add(era);
      const evs = eraVols(era);
      if (evs.length) before.fitBounds(unionOf(evs), { padding: 60, duration: 1600 });
    }
    applySelection();
  }

  // era chips (toggles: any subset may be on)
  const erasDiv = document.getElementById("eras");
  function renderEras() {
    erasDiv.innerHTML = "";
    for (const era of eras) {
      const el = document.createElement("button");
      el.className = "era" + (selectedEras.has(era) ? " active" : "");
      // the chips are toggles and any subset may be on, so "which are on" is
      // carried in ARIA and not only in the colour the class paints
      el.setAttribute("aria-pressed", selectedEras.has(era) ? "true" : "false");
      el.dataset.era = era;
      el.textContent = era;
      el.onclick = () => toggleEra(era);
      erasDiv.appendChild(el);
    }
  }
  renderEras();

  function drawAtlas() {
    if (atlasDrawn) return;
    atlasDrawn = true;
    addAllLayers();
    // the list is first built before any layer exists, and it says which
    // districts can be switched off — so it is rebuilt once they do
    renderList();
    const firstTile = (event) => {
      if (!event.tile || event.tile.state !== "loaded") return;
      clearLoading();
      before.off("sourcedata", firstTile);   // one job, done once
    };
    before.on("sourcedata", firstTile);
    const start = startVolume(vols.filter(v => selectedEras.has(eraOf(v))), S.home_point);
    if (start) before.fitBounds(start.bounds, { padding: 60, duration: 0 });
    openFromLink();   // a forwarded link's view or story stop overrides all of it
  }

  const compare = new maplibregl.Compare(before, after, "#compare-wrap");
  window.compare = compare;   // debugging convenience, and the story hook

  // ---------- the divider, for a keyboard ----------
  // The vendored compare binds `mousedown` and `touchstart` and nothing else,
  // so the page's central control answered a pointer and nothing else. The
  // element is the vendored handle; what is added here is the slider role, the
  // value it reports, and the keys `swipeStep` decides. The opacity range
  // below is a second way to compare, not a substitute: it fades the atlas
  // everywhere rather than moving the line between the two panes.
  const swiper = compare._swiper;
  swiper.tabIndex = 0;
  swiper.setAttribute("role", "slider");
  swiper.setAttribute("aria-label", "Divider between the atlas and the modern map");
  swiper.setAttribute("aria-valuemin", "0");
  swiper.setAttribute("aria-valuemax", "100");

  // ---------- one clamp, both paths ----------
  // The compare control's travel is the whole viewport, so a handle parked
  // behind the floating panel can never be grabbed again. `_setPosition` is
  // the one place the vendored compare moves the handle from — the drag, its
  // own resize handler, and `setSlider` all land here — so wrapping it once
  // covers every path, including a position set from script.
  const HANDLE_EDGE = 24;   // keeps the whole 44px handle on screen at both ends
  const panel = document.getElementById("panel");
  const panelBody = document.getElementById("panel-body");
  const wrap = document.getElementById("compare-wrap");
  let panelCollapsed = false;   // read by the clamp, owned by setPanelCollapsed

  // 1 while the panel floats over the map; 0 once it docks. Declared in CSS so
  // the breakpoint lives in exactly one place.
  const panelOverlaysMap = () =>
    getComputedStyle(document.documentElement)
      .getPropertyValue("--panel-overlays-map").trim() === "1";

  // Measures the layout and hands the numbers to the clamp. The panel is
  // measured rather than assumed, because its width changes with the
  // breakpoint; a docked one blocks nothing, so it passes null.
  function clampSliderX(x) {
    try {
      const blocks = !panelCollapsed && panelOverlaysMap();
      return clampSlider(x, {
        width: wrap.getBoundingClientRect().width,
        edge: HANDLE_EDGE,
        panelRight: blocks ? panel.getBoundingClientRect().right : null,
      });
    } catch (e) {
      return x;   // the atlas outlives its chrome: never block a slide
    }
  }
  const setPosition = compare._setPosition.bind(compare);
  compare._setPosition = (x) => { setPosition(clampSliderX(x)); reportSwipe(); };

  // The announced value, refreshed wherever the handle lands — a drag, a
  // resize, a story stop, a forwarded link or a key — because the wrapper
  // above is the one place they all pass through. It reads `currentPosition`
  // AFTER the move rather than the x that was asked for, so what a screen
  // reader hears is where the handle actually is once the clamp has had it.
  function reportSwipe() {
    try {
      const width = wrap.getBoundingClientRect().width;
      if (!width) return;
      const pct = Math.round((compare.currentPosition / width) * 100);
      swiper.setAttribute("aria-valuenow", String(pct));
      swiper.setAttribute("aria-valuetext", pct + "% atlas");
    } catch (e) { /* the atlas outlives its chrome: never block a slide */ }
  }

  // The same problem on the other axis, and the vendored stylesheet has no
  // answer for it: the handle is pinned halfway down the map, and a panel
  // docked across the bottom of a phone covers that. Measured here, decided by
  // `handleTop`, and applied as the custom property the stylesheet falls back
  // from — so the divider is draggable whether or not this ever runs.
  const HANDLE_RADIUS = 22;         // half the 44px handle
  const HANDLE_HEADROOM = 88;       // the era plates and the sources control
  // `offsetParent` is null for any fixed element, so it cannot answer this
  const isShown = (el) => getComputedStyle(el).display !== "none";
  function placeHandle() {
    try {
      const map = wrap.getBoundingClientRect();
      const docked = isShown(panel) && !panelOverlaysMap();
      const top = docked
        ? handleTop({
          mapHeight: map.height,
          panelTop: panel.getBoundingClientRect().top - map.top,
          headroom: HANDLE_HEADROOM,
          radius: HANDLE_RADIUS,
        })
        : null;
      const root = document.documentElement.style;
      // A property that resolves to nonsense is NOT the stylesheet's fallback:
      // `var()` substitutes first and the declaration is then invalid, which
      // computes to `auto` and puts the handle off the top of the map. Clearing
      // it is what actually falls back.
      if (top === null || !Number.isFinite(top)) root.removeProperty("--handle-top");
      else root.setProperty("--handle-top", top + "px");
    } catch (e) { /* the atlas outlives its chrome: never block a slide */ }
  }

  // A handle that was legal at a wide window must be pulled back into reach
  // when the window narrows or the panel opens, not stranded.
  // Set once the shareable-link block below exists. The divider moves before
  // then — the plugin places it in its own constructor — and that is not news.
  let shareSoon = null;

  // The programmatic way to place the divider: a fraction of the map's width,
  // never a pixel count, because which pixels are legal depends on the layout.
  // A story stop's `swipe` comes through here, so it gets the clamp too.
  function swipeToFraction(fraction) {
    compare.setSlider(wrap.getBoundingClientRect().width * fraction);
    if (shareSoon) shareSoon();   // or the link would keep the old fraction
  }

  // Keys go through `swipeToFraction` for the same reason a story stop does:
  // it is the one path that clamps to the layout and refreshes the shareable
  // link. A modifier chord is left alone so browser and OS shortcuts still
  // work while the handle holds focus.
  swiper.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const width = wrap.getBoundingClientRect().width;
    if (!width) return;
    const next = swipeStep(event.key, compare.currentPosition / width, event.shiftKey);
    if (next === null) return;
    event.preventDefault();   // or Arrow/Page/Home would scroll the page too
    // ...and the story's own arrow keys listen on `window`, which this bubbles
    // to. Without this, one ArrowRight inside a story nudges the divider AND
    // advances the stop — and a stop that declares a `swipe` then overwrites
    // the nudge, so the reader is moved somewhere they did not ask to go.
    // Only the keys the divider actually took are stopped: `swipeStep` answers
    // null for Escape, so leaving a story still works while it holds focus.
    event.stopPropagation();
    swipeToFraction(next);
  });

  const reclampSlider = () => compare.setSlider(compare.currentPosition);
  const refitDivider = () => { reclampSlider(); placeHandle(); };
  after.on("resize", reclampSlider);   // after the compare refreshes its bounds
  addEventListener("resize", () => requestAnimationFrame(refitDivider));
  addEventListener("orientationchange", () => requestAnimationFrame(refitDivider));
  // The panel's own height moves without the window: an era switched on, a
  // district list rebuilt, a story opened, the sheet collapsed. Watching the
  // element covers all of them at once, and the first callback places the
  // handle for the load.
  if (window.ResizeObserver) new ResizeObserver(placeHandle).observe(panel);
  else placeHandle();
  // ...but not the panel's ARRIVAL. It settles in on a transform, which moves
  // the box this measures and not the box a resize observer reports, so the
  // only reading taken during it is of a panel still on its way.
  panel.addEventListener("animationend", placeHandle);

  // ---------- panel collapse, carried in the URL ----------
  // The fragment is a shared namespace; lib.js owns its grammar. Reads and
  // writes are guarded here: a malformed hash, or a browser that refuses
  // `replaceState`, must never keep the atlas from drawing.
  const HASH_PANEL = "panel";

  const hashState = (key) => hashRead(location.hash, key);
  function writeHashState(key, value) {
    try {
      const fragment = hashWrite(location.hash, key, value);
      history.replaceState(null, "", fragment || location.pathname + location.search);
    } catch (e) { /* file:// or a sandboxed frame: the state stays in-page */ }
  }

  const panelToggle = document.getElementById("panel-toggle");
  const panelReopen = document.getElementById("panel-reopen");
  // True once a link or a click has said what the panel should do. Until then
  // the state is inferred from the layout and must follow it across the
  // breakpoint: the same flag means "header strip" on a docked panel and
  // "gone entirely" on a floating one, so a rotation would otherwise take the
  // whole panel away from a visitor who never asked for that.
  let panelChoiceIsExplicit = false;

  function setPanelCollapsed(next, remember) {
    panelCollapsed = !!next;
    document.body.classList.toggle("panel-collapsed", panelCollapsed);
    // the header control is the only one a docked panel has, so it must say
    // what it will do next, not what it did once
    panelToggle.setAttribute("aria-expanded", String(!panelCollapsed));
    const label = (panelCollapsed ? "Show" : "Hide") + " the atlas panel";
    panelToggle.setAttribute("aria-label", label);
    panelToggle.title = label;
    panelToggle.textContent = panelCollapsed ? "☰" : "–";
    if (remember) {
      panelChoiceIsExplicit = true;
      writeHashState(HASH_PANEL, panelCollapsed ? "closed" : "open");
      try { localStorage.setItem("viewer.panel", panelCollapsed ? "closed" : "open"); }
      catch (e) { /* storage blocked; the hash still carries the choice */ }
    }
    reclampSlider();
  }

  panelToggle.addEventListener("click", () => {
    setPanelCollapsed(!panelCollapsed, true);
    // follow the control the keyboard just lost: the docked panel keeps its
    // own header strip, so only the floating one hands off to the reopener
    (panelCollapsed && isShown(panelReopen) ? panelReopen : panelToggle).focus();
  });
  panelReopen.addEventListener("click", () => {
    setPanelCollapsed(false, true);
    panelToggle.focus();
  });

  // the link wins over the visitor's last manual choice; with neither, a
  // docked panel starts as a header strip so the comparison is what loads
  const hashPanel = hashState(HASH_PANEL);
  let storedPanel = null;
  try { storedPanel = localStorage.getItem("viewer.panel"); } catch (e) { /* blocked */ }
  // { story, index } while a story is being read; declared here because the
  // resize handler below has to know not to close the panel over one
  let active = null;

  panelChoiceIsExplicit = hashPanel !== null || storedPanel !== null;
  setPanelCollapsed(
    hashPanel !== null ? hashPanel === "closed"      // present but empty: open
      : storedPanel !== null ? storedPanel === "closed"
      : !panelOverlaysMap(),
    false);

  // A hash someone edits or arrives back on wins — but only for the key it
  // actually carries: a fragment with no panel key expresses no opinion, and
  // silently reopening the panel on an unrelated `#anchor` jump would be worse
  // than leaving it. Only a reload re-runs the resolution above.
  addEventListener("hashchange", () => {
    const wanted = hashState(HASH_PANEL);
    if (wanted === null) return;
    panelChoiceIsExplicit = true;
    setPanelCollapsed(wanted === "closed", false);
  });

  // An inferred state has to be re-inferred when the layout changes under it.
  addEventListener("resize", () => {
    // ...but never out from under a story being read: the visitor did ask for
    // that panel, by following the link, even though they never touched a control
    if (!panelChoiceIsExplicit && !active) setPanelCollapsed(!panelOverlaysMap(), false);
  });

  const attribText = document.getElementById("attrib-text");
  const attribToggle = document.getElementById("attrib-toggle");
  attribToggle.addEventListener("click", () => {
    const opening = attribText.hasAttribute("hidden");
    attribText.toggleAttribute("hidden", !opening);
    attribToggle.setAttribute("aria-expanded", String(opening));
  });

  // The manifest is read once. It cannot change between page loads on a static
  // deploy, so there is nothing for a re-poll to find; an operator watching a
  // drain reads `autogeoref status` instead.

  // ---------- address search (modern addresses only) ----------
  // Provider, and whether to send at all, is `chooseGeocoder` in lib.js: Mapbox
  // on a token, Nominatim on a dev host, nothing on a public host without one.
  // Bias bbox and query suffix come from the manifest's geocoder block.
  //
  // Geocoded as typed. Historical street names are NOT rewritten here —
  // retired-name search was withdrawn. The alias tables under
  // configs/<city>/aliases/ still drive the pipeline's name matching; they are
  // no longer served to the viewer.
  const GEO = S.geocoder;
  let searchMarkers = [];

  const askGeocoder = (q) => chooseGeocoder(q, GEO, window.MAPBOX_TOKEN, location.hostname);

  // Said once on load, not only on submit: a reader should not have to type an
  // address to learn the box cannot answer. The empty query is a probe — the
  // decision reads the token and the host, never the query.
  const note = document.getElementById("search-note");
  const searchForm = document.getElementById("search-row");
  const unconfigured = askGeocoder("");
  if (!unconfigured.provider) {
    note.textContent = unconfigured.reason;
    document.getElementById("address").disabled = true;
    document.getElementById("search-go").disabled = true;
  }

  searchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const raw = document.getElementById("address").value.trim();
    if (!raw) return;
    const ask = askGeocoder(raw);
    if (!ask.provider) { note.textContent = ask.reason; return; }
    note.textContent = "searching…";
    // `the geocoder did not answer` and `it answered nothing` are different
    // things to tell a reader: a rejected token or a blocked instance would
    // otherwise say "No match" to every address forever, blaming the atlas
    const answered = await fetch(ask.url).then(r => r.ok ? r.json() : null).catch(() => null);
    if (answered === null) { note.textContent = "Address search is unavailable right now."; return; }
    const hit = geocodeHit(ask.provider, answered);
    if (!hit) { note.textContent = "No match — try a full address."; return; }
    searchMarkers.forEach(m => m.remove());
    searchMarkers = [before, after].map(m =>
      new maplibregl.Marker({ color: "#b0403a" }).setLngLat(hit.lngLat).addTo(m));
    before.flyTo({ center: hit.lngLat, zoom: Math.max(before.getZoom(), 17), duration: 1800 });
    note.textContent = `Showing ${hit.name}`;
  });

  // district list, grouped by era, chips' order (newest first)
  const nav = document.getElementById("volumes");
  function cell(className, text) {
    const el = document.createElement("span");
    if (className) el.className = className;
    el.textContent = text;
    return el;
  }
  // Switching one district off, without disturbing the era it belongs to: the
  // set outlives the list, which is rebuilt on every selection change.
  function repaint(ids) {
    for (const id of ids) {
      if (added.has(id)) {
        before.setLayoutProperty(id, "visibility", visibility(id, added.get(id).era));
      }
    }
    restack();     // a district raised by a row click must not stay on top of
                   // the era above it once the reader switches things back on
    renderList();
    shareView();   // the link has to describe what the reader is looking at
  }

  // Rebuilding the list takes the activated button out of the document with
  // it, and a keyboard reader would be returned to the top of the page after
  // every switch. Put them back on the control they just used.
  const refocus = (selector) => {
    const el = nav.querySelector(selector);
    if (el) el.focus();
  };

  function toggleVolume(id) {
    if (hidden.has(id)) hidden.delete(id); else hidden.add(id);
    repaint([id]);
    refocus(`.vol-eye[data-volume="${CSS.escape(id)}"]`);
  }

  function showAllVolumes() {
    const wasHidden = [...hidden];
    hidden.clear();
    repaint(wasHidden);
    refocus(".vol-eye");   // the control it replaced is gone; the list is not
  }

  function renderList() {
    nav.querySelectorAll(".vol, .vol-era, .vol-all").forEach(el => el.remove());
    let count = 0;
    for (const era of eras) {
      if (!selectedEras.has(era)) continue;
      const evs = eraVols(era);
      if (!evs.length) continue;
      count += evs.length;
      const head = document.createElement("div");
      head.className = "vol-era";
      head.textContent = era;
      nav.appendChild(head);
      const sorted = [...evs].sort((a, b) =>
        (parseFloat(a.volume_number) || 999) - (parseFloat(b.volume_number) || 999));
      for (const v of sorted) {
        // A row, not a button: it holds two controls, and a button inside a
        // button is not markup a browser will build.
        const row = document.createElement("div");
        row.className = "vol" + (hidden.has(v.id) ? " off" : "");
        const yr = v.year || "";
        const no = v.volume_number ? "Vol. " + v.volume_number : "—";
        // manifest label (declared or LOC subject) first: a special sheet's own
        // name beats the community areas its bbox happens to touch
        const label = v.label || ((v.areas && v.areas.length)
          ? v.areas.join(" · ")
          : regionLabel(v.bounds, S.region_labels, S.home_point));
        const go = document.createElement("button");
        go.type = "button";
        go.className = "vol-go";
        // every field as text, never markup: all three come from the same LOC
        // catalog free text, and what keeps the number and the year tame is a
        // parser two modules away rather than anything this line can see
        go.append(cell("no", no), cell("", label), cell("yr", yr));
        go.onclick = () => {
          before.fitBounds(v.bounds, { padding: 60, duration: 1600 });
          // raise the clicked district above any newer overlapping era until
          // the next selection change restores chronological stacking
          if (added.has(v.id)) before.moveLayer(v.id);
        };
        row.appendChild(go);
        // Only where there is a layer to switch: a volume listed without
        // published imagery has nothing this control could do.
        if (added.has(v.id)) {
          const eye = document.createElement("button");
          eye.type = "button";
          eye.className = "vol-eye";
          eye.dataset.volume = v.id;
          const drawn = !hidden.has(v.id);
          eye.textContent = drawn ? "●" : "○";
          // The name says what the control IS and `aria-pressed` says which way
          // it is set. Naming it for the action instead would have a reader
          // hear "Show X, pressed" — pressed, on a control named for what it
          // has not done yet.
          eye.setAttribute("aria-label", label + " on the map");
          eye.setAttribute("aria-pressed", String(drawn));
          eye.title = (drawn ? "Hide " : "Show ") + label;
          eye.onclick = () => toggleVolume(v.id);
          row.appendChild(eye);
        }
        nav.appendChild(row);
      }
    }
    document.getElementById("vol-ct").textContent = count;
    // the way back, where the way out was — and only while there is one
    if (hidden.size) {
      const all = document.createElement("button");
      all.type = "button";
      all.className = "vol-all";
      all.textContent = `Show all (${hidden.size} hidden)`;
      all.onclick = showAllVolumes;
      nav.insertBefore(all, nav.querySelector(".vol-era"));
    }
  }

  // ---------- shareable views, and guided stories on the same hash ----------
  // The most-forwarded thing this site will produce is "look at this corner",
  // so the address bar keeps describing what is on screen: the camera, the era
  // selection, the divider, and — inside a story — which stop. All of it on the
  // panel's hash, one key each, so no key here can disturb another.
  const HASH_VIEW = "at";        // lng,lat,zoom
  const HASH_ERAS = "eras";      // comma-joined chip labels
  const HASH_SWIPE = "swipe";    // 0..1 fraction of the map's width
  const HASH_OFF = "off";        // comma-joined ids of switched-off districts
  const HASH_STORY = "story";
  const HASH_STOP = "stop";

  // What the link that opened this page asks for, read once. `linkNumbers` and
  // `linkText` are lib.js's, and both answer null for a key the link does not
  // carry AND for one it carries empty.
  const hash = location.hash;
  const LINK = {
    view: linkNumbers(hash, HASH_VIEW, 3),
    eras: linkText(hash, HASH_ERAS) ? linkText(hash, HASH_ERAS).split(",") : null,
    swipe: (() => {
      const one = linkNumbers(hash, HASH_SWIPE, 1);
      return one && one[0] >= 0 && one[0] <= 1 ? one[0] : null;
    })(),
    story: linkText(hash, HASH_STORY),
    stop: linkText(hash, HASH_STOP),
  };

  // The story a permalink names, if this city still configures it.
  const linkedStory = LINK.story ? S.stories.find(s => s.id === LINK.story) : null;
  // Guided stories are OPT-IN per visit: the list is offered to a link that
  // asks (`?stories`) and to a reader arriving on a story permalink, who would
  // otherwise leave that story with no way back. `?stories=0` beats both. A
  // permalink still opens its story either way; only the LIST is gated.
  //
  // Gated on `LINK.story` being PRESENT, not resolving: a link naming a story
  // since renamed is the reader most in need of the list.
  const QUERY_STORIES = "stories";
  const storiesAsk = storiesAsked(location.search, QUERY_STORIES);   // true/false/null
  const storiesOn = Boolean(S.stories.length) && storiesAsk !== false &&
    (storiesAsk === true || Boolean(LINK.story));

  // The opt-in a reader expressed by FOLLOWING a story link, written into the
  // link itself. Without this the escape hatch above lasts exactly as long as
  // the story: leaving one clears the `story` key, so a reload — or the link
  // "Copy a link to this view" hands out — would land on a page with no way
  // back into the story the reader just came from.
  function rememberStoryOptIn() {
    // the LIVE search, not the page-load reading: this runs once per story
    // entered, and a stale reading would re-answer a question the previous
    // entry already wrote down
    if (storiesAsked(location.search, QUERY_STORIES) !== null) return;
    try {
      const search = queryWrite(location.search, QUERY_STORIES, "1");
      history.replaceState(null, "", location.pathname + search + location.hash);
    } catch (e) { /* file:// or a sandboxed frame: the in-page flag still stands */ }
  }

  const EMPTY_FC = { type: "FeatureCollection", features: [] };
  const OVERLAY = "story-overlay";
  const OVERLAY_LAYERS = [OVERLAY + "-fill", OVERLAY + "-line"];
  const storyEntry = document.getElementById("story-entry");
  const storyPanel = document.getElementById("story");
  const storyNav = document.getElementById("story-nav");
  const storyPrev = document.getElementById("story-prev");
  const storyNext = document.getElementById("story-next");

  function shareView() {
    if (active) return;
    const c = before.getCenter();
    writeHashState(HASH_VIEW, viewValue(c.lng, c.lat, before.getZoom()));
    writeHashState(HASH_ERAS, [...selectedEras].join(","));
    writeHashState(HASH_SWIPE,
      (compare.currentPosition / (wrap.getBoundingClientRect().width || 1)).toFixed(3));
    // absent rather than empty when nothing is hidden: the common link stays
    // short, and the key means what its absence means everywhere else here
    writeHashState(HASH_OFF, hidden.size ? [...hidden].join(",") : null);
    writeHashState(HASH_STORY, null);
    writeHashState(HASH_STOP, null);
  }
  // `jumpTo` fires moveend SYNCHRONOUSLY, so "jump, then set the divider" would
  // publish the divider from a line earlier. Every slider move re-publishes,
  // coalesced so a drag writes one entry and not sixty.
  let sharePending = null;
  shareSoon = () => {
    clearTimeout(sharePending);
    sharePending = setTimeout(shareView, 250);
  };
  before.on("moveend", shareView);
  compare.on("slideend", shareSoon);

  document.getElementById("copy-link").addEventListener("click", async (event) => {
    shareView();
    const button = event.currentTarget;
    try {
      await navigator.clipboard.writeText(location.href);
      button.textContent = "Link copied";
    } catch (err) {
      button.textContent = "Copy it from the address bar";
    }
    setTimeout(() => { button.textContent = "Copy a link to this view"; }, 2500);
  });

  // The outline is what turns a swipe into an argument — a right-of-way, a
  // clearance boundary — so it states the claim rather than leaving the visitor
  // to infer it, and it stays above every atlas raster the restack moves.
  function raiseStoryOverlay() {
    if (before.getSource(OVERLAY)) for (const id of OVERLAY_LAYERS) before.moveLayer(id);
  }
  function drawOverlay(overlay) {
    if (!before.getSource(OVERLAY)) {
      before.addSource(OVERLAY, { type: "geojson", data: EMPTY_FC });
      before.addLayer({ id: OVERLAY_LAYERS[0], type: "fill", source: OVERLAY,
        paint: { "fill-color": "#b0403a", "fill-opacity": 0 } });
      before.addLayer({ id: OVERLAY_LAYERS[1], type: "line", source: OVERLAY,
        paint: { "line-color": "#b0403a", "line-width": 2 } });
    }
    const style = (overlay && overlay.style) || {};
    const color = style.color || "#b0403a";
    before.getSource(OVERLAY).setData(overlay ? overlay.geojson : EMPTY_FC);
    before.setPaintProperty(OVERLAY_LAYERS[1], "line-color", color);
    before.setPaintProperty(OVERLAY_LAYERS[1], "line-width", style.width || 2);
    // unconditionally, like every other paint property here: a dash left over
    // from the previous stop would draw a claim this one did not make
    before.setPaintProperty(OVERLAY_LAYERS[1], "line-dasharray", style.dash || null);
    before.setPaintProperty(OVERLAY_LAYERS[0], "fill-color", color);
    before.setPaintProperty(OVERLAY_LAYERS[0], "fill-opacity", style.fill_opacity || 0);
    raiseStoryOverlay();
  }

  function renderStop() {
    const { story, index } = active;
    const stop = story.stops[index];
    storyPanel.innerHTML = "";
    const head = document.createElement("div");
    head.className = "stop-of";
    head.textContent = `${story.title} · ${index + 1} of ${story.stops.length}`;
    const title = document.createElement("h2");
    title.textContent = stop.title;
    storyPanel.append(head, title);
    if (stop.body_html) {
      const body = document.createElement("div");
      body.className = "body";
      // eslint-disable-next-line no-unsanitized/property -- story prose authored in the city TOML, trusted like the configured credits
      body.innerHTML = stop.body_html;
      storyPanel.appendChild(body);
    }
    for (const item of stop.media || []) {
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      // relative to the MANIFEST, like every other path it carries: the
      // images are staged beside it, not beside the page
      img.src = new URL(item.src, manifestBase).href;
      img.alt = item.alt || ""; img.loading = "lazy";
      if (item.href) {
        const link = document.createElement("a");
        link.href = item.href; link.target = "_blank"; link.rel = "noopener";
        link.appendChild(img);
        figure.appendChild(link);
      } else figure.appendChild(img);
      if (item.caption || item.credit) {
        const caption = document.createElement("figcaption");
        caption.textContent = [item.caption, item.credit].filter(Boolean).join(" — ");
        figure.appendChild(caption);
      }
      storyPanel.appendChild(figure);
    }
    if ((stop.sources || []).length) {
      const list = document.createElement("div");
      list.className = "sources";
      for (const source of stop.sources) {
        const row = document.createElement("div");
        if (source.href) {
          const link = document.createElement("a");
          link.href = source.href; link.target = "_blank"; link.rel = "noopener";
          link.textContent = source.label;
          row.appendChild(link);
        } else row.textContent = source.label;
        list.appendChild(row);
      }
      storyPanel.appendChild(list);
    }
    storyPrev.disabled = index === 0;
    storyNext.disabled = index === story.stops.length - 1;
    // to the top of the stop just opened. The panel body is the scroll region,
    // not the prose, so a reader who reached Next at the bottom of one stop
    // would otherwise open the next one already scrolled past its end.
    storyPanel.scrollIntoView({ block: "start" });

    if ((stop.eras || []).length) {
      selectedEras.clear();
      for (const era of stop.eras) if (eras.includes(era)) selectedEras.add(era);
    }
    // unconditionally: the chips, the plates, the heading and the credit line
    // all describe a selection, and a stop that names no era still moved
    applySelection();
    // The camera goes a frame after the layout. Resizing a pane makes the two
    // maps re-sync by jumping, and a jump landing mid-flight cancels it — which
    // is exactly what opening the bottom sheet does on a small screen.
    requestAnimationFrame(() => {
      before.flyTo({
        center: stop.camera.center,
        zoom: stop.camera.zoom,
        bearing: stop.camera.bearing || 0,
        pitch: stop.camera.pitch || 0,
        duration: 1400,
      });
      // through the clamp, always: a stop asking for 0.15 on a wide screen
      // would otherwise park the handle under the panel
      if (typeof stop.swipe === "number") swipeToFraction(stop.swipe);
    });
    drawOverlay(stop.overlay || null);
    writeHashState(HASH_STORY, story.id);
    writeHashState(HASH_STOP, stop.id);
    writeHashState(HASH_VIEW, null);
    writeHashState(HASH_ERAS, null);
    writeHashState(HASH_SWIPE, null);
  }

  function goToStop(index) {
    if (!active) return;
    const wanted = clampStopIndex(index, active.story.stops.length);
    if (wanted === active.index) return;   // at an end: re-flying is not "next"
    active.index = wanted;
    renderStop();
  }

  function enterStory(story, stopId) {
    active = { story, index: stopIndex(story.stops, stopId) };
    rememberStoryOptIn();
    storyEntry.classList.remove("on");
    nav.style.display = "none";
    storyPanel.classList.add("on");
    storyNav.classList.add("on");
    document.body.classList.add("story");
    setPanelCollapsed(false, false);   // a stop is unreadable behind a closed panel
    renderStop();
  }

  function exitStory() {
    active = null;
    storyPanel.classList.remove("on");
    storyPanel.innerHTML = "";
    storyNav.classList.remove("on");
    document.body.classList.remove("story");
    nav.style.display = "";
    if (storiesOn) storyEntry.classList.add("on");
    // back to the masthead: the story left the one scroll region parked
    // wherever its prose ended, which in the panel behind it is the middle of
    // the district list, with the chips and the search box scrolled away
    panelBody.scrollTop = 0;
    drawOverlay(null);
    shareView();   // clears the story keys and republishes the free view
  }

  // Wired for every city that configures a story, and deliberately NOT behind
  // the entry-list gate: a permalink opens its story whatever the query string
  // says, and controls wired on the other side of that gate would leave a
  // reader inside a rendered story with a live Next button that does nothing.
  function wireStoryControls() {
    if (!S.stories.length) return;
    storyPrev.onclick = () => goToStop(active.index - 1);
    storyNext.onclick = () => goToStop(active.index + 1);
    document.getElementById("story-exit").onclick = exitStory;
    addEventListener("keydown", (event) => {
      const focused = document.activeElement;
      if (!active || (focused && /^(INPUT|TEXTAREA)$/.test(focused.tagName))) return;
      if (event.key === "ArrowRight") goToStop(active.index + 1);
      else if (event.key === "ArrowLeft") goToStop(active.index - 1);
      else if (event.key === "Escape") exitStory();
      else return;
      event.preventDefault();   // or the map pans under the stop we just flew to
    });
  }

  function buildStoryEntry() {
    // configured but not offered this visit: the list is not built at all, so
    // there is nothing to reveal by clearing a class
    if (!storiesOn) return;
    for (const story of S.stories) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "story-open";
      button.textContent = story.title;
      if (story.dek) {
        const dek = document.createElement("div");
        dek.className = "story-dek";
        dek.textContent = story.dek;
        button.appendChild(dek);
      }
      button.onclick = () => enterStory(story, null);
      storyEntry.appendChild(button);
    }
    storyEntry.classList.add("on");
  }
  wireStoryControls();
  buildStoryEntry();

  // The link decides where this page opens. Guarded whole: the atlas outlives
  // its chrome, so a story that cannot be applied must not hold the map.
  function openFromLink() {
    try {
      // BEFORE the story branch, which returns: a story stop keeps whatever is
      // switched off behind it, and the link the page writes inside a story
      // carries `off` — so the link it writes has to be one it can read back.
      // Against the layers that EXIST, not every listed id: an id with no
      // layer would dim a row whose switch is not rendered, and a "show all"
      // would then be the only way out of a state nothing put the reader in.
      for (const id of hiddenFromLink(hash, HASH_OFF, [...added.keys()])) hidden.add(id);
      if (hidden.size) applySelection();
      if (linkedStory) { enterStory(linkedStory, LINK.stop); return; }
      if (LINK.eras) {
        const wanted = LINK.eras.filter(e => eras.includes(e));
        if (wanted.length) {
          selectedEras.clear();
          for (const era of wanted) selectedEras.add(era);
          applySelection();
        }
      }
      if (LINK.view) before.jumpTo({ center: [LINK.view[0], LINK.view[1]], zoom: LINK.view[2] });
      if (LINK.swipe !== null) swipeToFraction(LINK.swipe);
    } catch (err) {
      console.error("could not apply the link's view", err);
    }
  }

  applySelection();   // paints title, plates, footer, chips, list
})().catch((err) => {
  // One long async function: a throw skips every remaining line, including
  // whatever would have cleared the overlay, leaving a permanent load.
  console.error("the viewer could not start", err);
  overlayStop("This atlas could not start in this browser.");
});
