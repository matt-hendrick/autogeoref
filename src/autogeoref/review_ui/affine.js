/* The review UI's mirror of the server's placement maths: the affine is 2x3
   with [X,Y] = M @ [1,px,py] in EPSG:3857, every edit is a world-space op
   (translate / uniform scale / rotate) composed onto it, and lng/lat comes in
   and out through spherical mercator.

   Display only — the server recomputes each saved verdict from the op log —
   but a divergence puts the ghost somewhere the server will not put the sheet,
   and the operator reads that as correct. Its own file so a test can run every
   piece of it against the Python that owns the same computation.

   A plain classic script, no bundler either side: a browser global here, a
   CommonJS export in node. */
const ReviewAffine = (function () {
  const applyAff = (m, px, py) => [m[0][0] + m[0][1] * px + m[0][2] * py,
                                   m[1][0] + m[1][1] * px + m[1][2] * py];

  /* one op's world map `world' = L @ world + t` */
  function opLin(op) {
    if (op.type === "translate") return { L: [[1, 0], [0, 1]], t: [op.dx_m, op.dy_m] };
    let L;
    if (op.type === "scale") L = [[op.factor, 0], [0, op.factor]];
    else { const r = op.deg * Math.PI / 180;
           L = [[Math.cos(r), -Math.sin(r)], [Math.sin(r), Math.cos(r)]]; }
    const [cx, cy] = op.center_3857;
    return { L, t: [cx - L[0][0] * cx - L[0][1] * cy, cy - L[1][0] * cx - L[1][1] * cy] };
  }

  const mul2 = (L, v) => [L[0][0] * v[0] + L[0][1] * v[1], L[1][0] * v[0] + L[1][1] * v[1]];

  function composeOps(base, ops) {
    let m = base.map(row => row.slice());
    for (const op of ops) {
      const { L, t } = opLin(op);
      const c = mul2(L, [m[0][0], m[1][0]]);
      const a = mul2(L, [m[0][1], m[1][1]]);
      const b = mul2(L, [m[0][2], m[1][2]]);
      m = [[c[0] + t[0], a[0], b[0]], [c[1] + t[1], a[1], b[1]]];
    }
    return m;
  }

  /* The affine over world coordinates that `ops` alone describes — `composeOps`
     from the identity, which is what makes a chain of ops applicable to a point
     already in 3857 without knowing the sheet it came from. */
  const IDENTITY = [[0, 1, 0], [0, 0, 1]];

  // --- EPSG:3857 <-> lng/lat (spherical mercator, the datum pyproj's 3857 uses)
  const R = 6378137;
  const toMerc = (lng, lat) => [R * lng * Math.PI / 180,
    R * Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360))];
  const toLngLat = (x, y) => [x / R * 180 / Math.PI,
    (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * 180 / Math.PI];

  /* World->pixel, as the coefficient bag `pxOf` reads. Divides by the linear
     part's determinant without checking it: a placement whose scale has
     collapsed yields Infinity/NaN pixels rather than throwing. */
  function invAff(m) {
    const [a, b, c, d] = [m[0][1], m[0][2], m[1][1], m[1][2]];
    const det = a * d - b * c;
    return { x0: m[0][0], y0: m[1][0], ia: d / det, ib: -b / det, ic: -c / det, id: a / det };
  }
  /* The pixel a world point falls on, rounded to 0.1 px — a mask vertex and a
     pin are both clicked, so more precision than that is noise. */
  const pxOf = (inv, w) => {
    const dx = w[0] - inv.x0, dy = w[1] - inv.y0;
    return [Math.round((inv.ia * dx + inv.ib * dy) * 10) / 10,
            Math.round((inv.ic * dx + inv.id * dy) * 10) / 10];
  };

  /* Least-squares similarity (rotation + uniform scale + translation, NO
     reflection) taking `source` onto `world`, both lists of [X,Y] in 3857 —
     expressed exactly as the op vocabulary: scale(centroid) . rotate(centroid)
     . translate. One pair moves without rescaling.

     Returns `{ops, residualsM, maxResidualM}`, or `{error}` for a pin set with
     nothing to fit against — coincident sources, or targets all on one spot.
     The residuals come back with the ops because they are the operator's only
     read on the fit, and computing them anywhere else would report a number
     from a different code path than the one that placed the sheet. */
  function similarityOps(source, world) {
    if (!source.length || source.length !== world.length) return { error: "no pin pairs to fit" };
    let ops;
    if (source.length === 1) {
      ops = [{ type: "translate",
               dx_m: world[0][0] - source[0][0], dy_m: world[0][1] - source[0][1] }];
    } else {
      const n = source.length;
      const sm = [source.reduce((a, p) => a + p[0], 0) / n,
                  source.reduce((a, p) => a + p[1], 0) / n];
      const wm = [world.reduce((a, p) => a + p[0], 0) / n,
                  world.reduce((a, p) => a + p[1], 0) / n];
      let c00 = 0, c01 = 0, c10 = 0, c11 = 0, varS = 0;
      for (let i = 0; i < n; i++) {
        const dsx = source[i][0] - sm[0], dsy = source[i][1] - sm[1];
        const dwx = world[i][0] - wm[0], dwy = world[i][1] - wm[1];
        c00 += dwx * dsx; c01 += dwx * dsy; c10 += dwy * dsx; c11 += dwy * dsy;
        varS += dsx * dsx + dsy * dsy;
      }
      // hypot/atan2 of the ROTATION-only pair: a reflected point set gets the
      // rotation that fits it least badly, never a mirrored transform
      const k = Math.hypot(c00 + c11, c10 - c01) / varS;
      if (!(varS > 1e-6) || !(k > 1e-9) || !isFinite(k)) {
        return { error: "degenerate pin set — spread the pins out" };
      }
      const theta = Math.atan2(c10 - c01, c00 + c11);
      ops = [];
      if (Math.abs(k - 1) > 1e-12) ops.push({ type: "scale", factor: k, center_3857: sm.slice() });
      if (Math.abs(theta) > 1e-12) {
        ops.push({ type: "rotate", deg: theta * 180 / Math.PI, center_3857: sm.slice() });
      }
      const dx = wm[0] - sm[0], dy = wm[1] - sm[1];
      if (dx || dy) ops.push({ type: "translate", dx_m: dx, dy_m: dy });
    }
    const fitted = composeOps(IDENTITY, ops);
    const residualsM = source.map((p, i) => {
      const q = applyAff(fitted, p[0], p[1]);
      return Math.hypot(q[0] - world[i][0], q[1] - world[i][1]);
    });
    return { ops, residualsM, maxResidualM: Math.max(...residualsM) };
  }

  /* The op log with `op` folded in. Consecutive translates add up, and so do
     rotates about the SAME centre — an arrow-key nudge should leave one entry,
     not sixty. Two rotates of the same angle about different centres are
     different edits and never merge.

     A NEW list of NEW objects: an undo snapshot taken before the call must not
     be rewritten by the merge, which is what keeps undo per-nudge while the
     log stays merged. */
  function coalesce(ops, op) {
    const last = ops[ops.length - 1];
    const replacingLast = (next) => [...ops.slice(0, -1), next];
    if (op.type === "translate" && last && last.type === "translate") {
      return replacingLast({ ...last, dx_m: last.dx_m + op.dx_m, dy_m: last.dy_m + op.dy_m });
    }
    if (op.type === "rotate" && last && last.type === "rotate" &&
        last.center_3857[0] === op.center_3857[0] &&
        last.center_3857[1] === op.center_3857[1]) {
      return replacingLast({ ...last, deg: last.deg + op.deg });
    }
    return [...ops, op];
  }

  /* A copy of the editable state an undo restores, sharing nothing with it.
     The classic defect here is a shallow copy: the snapshot and the live
     arrays keep the same objects, so a later edit rewrites the history it was
     supposed to be able to return to. That includes a rotate's `center_3857`,
     which is an array inside an op. */
  const cloneEdits = (state) => ({
    ops: state.ops.map(op =>
      op.center_3857 ? { ...op, center_3857: op.center_3857.slice() } : { ...op }),
    maskPx: state.maskPx ? state.maskPx.map(p => p.slice()) : null,
    maskDirty: state.maskDirty,
    pins: state.pins.map(p => ({ px: p.px, py: p.py, w: p.w ? p.w.slice() : null })),
  });

  return {
    applyAff, composeOps, toMerc, toLngLat, invAff, pxOf, similarityOps,
    coalesce, cloneEdits, IDENTITY,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = ReviewAffine;
