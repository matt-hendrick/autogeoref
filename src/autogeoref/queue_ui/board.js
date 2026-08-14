/* The queue console's decisions, kept apart from the markup so a test can run
   them: which table an entry belongs in, how old it is, what its context line
   says, and the read rate that makes a fan-out visible.

   Nothing here touches `document` or `Date.now()` — the caller measures and
   passes the numbers in. Carries no city fact.

   A plain classic script, no bundler either side: a browser global here, a
   CommonJS export in node. */
const Board = (function () {
  /* Age as the board prints it: seconds under a minute and a half, then whole
     minutes, then `h:mm`. `now` and `t` are both epoch SECONDS. A missing
     timestamp is an em dash, not "0s" — a job that never started has no age. */
  function ago(t, now) {
    if (!t) return "—";
    const s = Math.max(0, Math.floor(now - t));
    if (s < 90) return s + "s";
    if (s < 5400) return Math.floor(s / 60) + "m";
    return Math.floor(s / 3600) + "h" + String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  }

  /* The one-line "where and when" under a volume id: `city, year - areas`,
     with every empty field dropped rather than printed as a gap. "" when the
     payload carries no context for this volume at all. */
  function contextText(volume, context) {
    const c = context[volume];
    if (!c) return "";
    const place = [c.city, c.year].filter((v) => v !== null && v !== undefined && v !== "").join(", ");
    const areas = Array.isArray(c.neighborhoods) ? c.neighborhoods.filter(Boolean).join(" / ") : "";
    return [place, areas].filter(Boolean).join(" - ");
  }

  /* Which table each queue entry belongs in — four columns and one table per
     track, over one flat list. An entry in the wrong one is a volume the
     operator cannot find.

     `live` is what is running, then what is queued behind it in queue order.
     The two are kept in one list on purpose: the column is "Running — and
     queued behind it", and reading it in that order is the point. */
  function board(entries, tracks) {
    const byStatus = (status) => entries.filter((e) => e.status === status);
    const byTrack = {};
    for (const track of tracks) byTrack[track] = entries.filter((e) => e.track === track);
    return {
      live: byStatus("running").concat(byStatus("queued")),
      needsReview: byStatus("needs-review"),
      failed: byStatus("failed"),
      byTrack,
    };
  }

  /* Reads land only when a model call COMPLETES (~minutes each), so between
     landings a parallel annotate looks exactly like a stalled serial one — the
     tree cannot show an in-flight call. Successive polls can: a rate over a
     sliding window makes the fan-out visible, and a real stall visible as the
     rate decaying toward zero.

     Null rather than a number when there is too little history to mean
     anything, or when the count did not move. `t` is epoch MILLISECONDS. */
  const WINDOW_MS = 12 * 60000;
  const MIN_MINUTES = 1.5;

  const withinWindow = (samples, now) => samples.filter((s) => now - s.t <= WINDOW_MS);

  function readsPerMin(samples) {
    if (samples.length < 2) return null;
    const first = samples[0], last = samples[samples.length - 1];
    const minutes = (last.t - first.t) / 60000;
    if (minutes < MIN_MINUTES) return null;
    const rate = (last.reads - first.reads) / minutes;
    return rate > 0 ? rate : null;
  }

  return { ago, contextText, board, withinWindow, readsPerMin, WINDOW_MS, MIN_MINUTES };
})();

if (typeof module !== "undefined" && module.exports) module.exports = Board;
