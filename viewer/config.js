/* Deploy-time configuration, loaded before app.js.

   `window.MAPBOX_TOKEN` is what turns address search on. A Mapbox PUBLIC token
   (`pk.…`) is public by design — the control is the URL restriction on the
   token, not secrecy — but it is still never committed, because a committed
   token is a token nobody rotates. `autogeoref viewer deploy-bundle` rewrites
   this file in the bundle from `--mapbox-token` / `AUTOGEOREF_MAPBOX_TOKEN`.

   Empty here so the local page loads it without a 404 and falls back to the
   dev-only geocoder; empty on a public host means search says so and stays
   off. */
window.MAPBOX_TOKEN = "";
