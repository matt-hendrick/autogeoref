/* The walkthrough stepper.
 *
 * Display only, and deliberately so. Every number, threshold, caption and
 * rendered state arrives in walkthrough/panels.json, written by
 * scripts/walkthrough/make_walkthrough_assets.py from a real run. Nothing here computes a
 * pipeline decision: a gate predicate re-implemented in this file could drift
 * from the code it claims to describe and teach something false with complete
 * confidence, where a rendered state cannot.
 *
 * Nothing writes markup either. Every string off the JSON reaches the page as
 * a text node, so a caption is incapable of carrying anything but text.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  // The shared fragment, read and written key by key. A URLSearchParams round
  // trip re-encodes characters other keys rely on, so it is not used here.
  const hashRead = (key) => {
    const found = location.hash.replace(/^#/, "").split("&")
      .map((pair) => pair.split("="))
      .find((pair) => pair[0] === key);
    return found ? decodeURIComponent(found[1] || "") : "";
  };
  const hashWrite = (values) => {
    const kept = location.hash.replace(/^#/, "").split("&")
      .filter((pair) => pair && !(pair.split("=")[0] in values));
    const added = Object.keys(values)
      .filter((key) => values[key] !== "")
      .map((key) => `${key}=${encodeURIComponent(values[key])}`);
    const next = kept.concat(added).join("&");
    history.replaceState(null, "", next ? `#${next}` : location.pathname);
  };

  let panels = [];
  let funnel = null;
  let index = 0;
  let stateKey = "";

  const panel = () => panels[index];
  const state = () => {
    const list = panel().states;
    return list.find((s) => s.key === stateKey) || list[0];
  };

  function buildSteps() {
    const list = $("steps");
    list.replaceChildren();
    let act = "";
    panels.forEach((p, i) => {
      if (act && p.act !== act) list.append(el("li", "act-break"));
      act = p.act;
      const item = el("li");
      const button = el("button", null, String(p.number));
      button.type = "button";
      button.title = `${p.act} - ${p.title}`;
      button.addEventListener("click", () => go(i, ""));
      item.append(button);
      list.append(item);
    });
  }

  function drawSteps() {
    const buttons = $("steps").querySelectorAll("button");
    buttons.forEach((button, i) => {
      if (i === index) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    const current = buttons[index];
    if (current) current.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function drawStates() {
    const box = $("states");
    box.replaceChildren();
    const list = panel().states;
    if (list.length < 2) return;
    list.forEach((s) => {
      const button = el("button", null, s.label);
      button.type = "button";
      button.setAttribute("aria-pressed", String(s.key === state().key));
      button.addEventListener("click", () => go(index, s.key));
      box.append(button);
    });
  }

  function drawFigures() {
    const box = $("figures");
    box.replaceChildren();
    panel().figures.forEach((figure) => {
      box.append(el("dt", null, figure.label));
      box.append(el("dd", null, figure.value));
    });
  }

  function drawTally() {
    const box = $("tally");
    const bar = $("tally-bar");
    // which stage's counter this panel shows is resolved by the generator, so a
    // panel that is not itself a stage carries the last one that resolved
    const stage = panel().tally;
    const at = funnel && funnel.stages[stage];
    if (!at) {
      box.classList.add("idle");
      $("tally-note").textContent =
        `${funnel ? funnel.total : "-"} sheets in this volume, none of them decided yet.`;
      bar.querySelectorAll(".seg").forEach((seg) => { seg.style.width = "0%"; });
      bar.querySelector(".seg.flagged").style.width = "100%";
      return;
    }
    box.classList.remove("idle");
    const total = funnel.total;
    const parts = [["placed", at.placed], ["prov", at.provisional], ["flagged", at.flagged]];
    parts.forEach(([name, count]) => {
      bar.querySelector(`.seg.${name}`).style.width = `${(count / total) * 100}%`;
    });
    const held = at.provisional ? `, ${at.provisional} held as proposals` : "";
    $("tally-note").textContent =
      `${funnel.labels[stage]}: ${at.placed} placed${held}, ${at.flagged} flagged, of ${total}.`;
  }

  function draw() {
    const p = panel();
    const s = state();
    $("act").textContent = p.act;
    $("title").textContent = p.title;
    $("lede").textContent = p.dek;
    $("caption").textContent = p.caption;
    // a panel told out of the order a run does it in says so here, and so does
    // one drawn from a different atlas; the CSS block exists for both
    $("order-note").textContent = p.note || "";
    $("order-note").hidden = !p.note;
    $("plate").src = `walkthrough/${s.file}`;
    $("plate").alt = s.alt;
    $("plate-link").href = `walkthrough/${s.file}`;
    $("plate-note").textContent =
      (p.states.length > 1 ? `${s.label}. ` : "") + "Open the figure to see it full size.";
    $("counter").textContent = `${p.number} of ${panels.length}`;
    $("prev").disabled = index === 0;
    $("next").disabled = index === panels.length - 1;
    document.title = `${p.number}. ${p.title} - how a scanned atlas becomes a map`;
    drawSteps();
    drawStates();
    drawFigures();
    drawTally();
  }

  function go(next, key) {
    index = Math.max(0, Math.min(panels.length - 1, next));
    const keys = panels[index].states.map((s) => s.key);
    stateKey = keys.indexOf(key) >= 0 ? key : keys[0];
    hashWrite({ step: String(panels[index].number), state: stateKey === "main" ? "" : stateKey });
    draw();
  }

  function fromHash() {
    const wanted = Number(hashRead("step"));
    const at = panels.findIndex((p) => p.number === wanted);
    go(at >= 0 ? at : 0, hashRead("state"));
  }

  fetch("walkthrough/panels.json")
    .then((response) => response.json())
    .then((data) => {
      panels = data.panels;
      funnel = data.funnel;
      $("dek").textContent =
        `Every figure here was rendered from a real run. The walkthrough follows one ` +
        `atlas - ${data.meta.title}, ${data.meta.sheets} scanned sheets, start to ` +
        `finish - and names the atlas on any panel that shows another.`;
      $("credits").textContent = `${data.meta.scan_credit}. ${data.meta.centerline_credit}.`;
      const glossary = $("glossary");
      data.glossary.forEach((entry) => {
        glossary.append(el("div", "term", entry.term));
        glossary.append(el("div", null, entry.gloss));
      });
      buildSteps();
      fromHash();
      document.body.dataset.ready = "1";
    })
    .catch((error) => {
      // A missing or broken panels.json is the one failure a reader can do
      // nothing about; say so on the page rather than leaving it blank.
      $("title").textContent = "The walkthrough could not load";
      $("caption").textContent = String(error);
      document.body.dataset.ready = "error";
    });

  $("prev").addEventListener("click", () => go(index - 1, ""));
  $("next").addEventListener("click", () => go(index + 1, ""));
  window.addEventListener("keydown", (event) => {
    if (event.target !== document.body && event.target !== $("panel")) return;
    if (event.key === "ArrowRight") go(index + 1, "");
    if (event.key === "ArrowLeft") go(index - 1, "");
  });
  window.addEventListener("hashchange", fromHash);
})();
