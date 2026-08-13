import { Matcher } from "./matcher.js";
import { initScan } from "./scan.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  wines: [],
  bySlug: new Map(),
  matcher: null,
  filters: { q: "", colors: new Set(), sweetness: "", region: "", grape: "" },
  shown: 0,
  filtered: [],
};

const store = {
  get(key, dflt) {
    try { return JSON.parse(localStorage.getItem(key)) ?? dflt; } catch { return dflt; }
  },
  set(key, val) { localStorage.setItem(key, JSON.stringify(val)); },
};

async function boot() {
  const res = await fetch("data/wines.json");
  const data = await res.json();
  state.wines = data.wines;
  for (const w of state.wines) state.bySlug.set(w.slug, w);
  state.matcher = new Matcher(state.wines);
  $("#dp-version").textContent = `база: ${data.meta.count} вин · v${data.meta.version}`;
  fillSelects();
  applyFilters();
  renderMine();
  initScan(state, { openDetail, addHistory: (slug, conf) => addHistory(slug, conf) });
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
}

function fillSelects() {
  const sweet = new Set(), regions = new Set(), grapes = new Set();
  for (const w of state.wines) {
    if (w.sweetness) sweet.add(w.sweetness);
    if (w.region) regions.add(w.region);
    for (const g of w.grapes) grapes.add(g);
  }
  const fill = (sel, values) => {
    for (const v of [...values].sort((a, b) => a.localeCompare(b, "ru"))) {
      const opt = document.createElement("option");
      opt.value = v; opt.textContent = v;
      sel.append(opt);
    }
  };
  fill($("#f-sweetness"), sweet);
  fill($("#f-region"), regions);
  fill($("#f-grape"), grapes);
}

function applyFilters() {
  const f = state.filters;
  const q = f.q.trim();
  let list = state.wines;
  if (f.colors.size) list = list.filter((w) => f.colors.has(w.color));
  if (f.sweetness) list = list.filter((w) => w.sweetness === f.sweetness);
  if (f.region) list = list.filter((w) => w.region === f.region);
  if (f.grape) list = list.filter((w) => w.grapes.includes(f.grape));
  if (q) {
    const hits = state.matcher.search(q, 400).map((r) => r.wine.slug);
    const rank = new Map(hits.map((s, i) => [s, i]));
    list = list.filter((w) => rank.has(w.slug)).sort((a, b) => rank.get(a.slug) - rank.get(b.slug));
  } else {
    list = [...list].sort((a, b) => (b.rating || 0) - (a.rating || 0));
  }
  state.filtered = list;
  state.shown = 0;
  $("#list").innerHTML = "";
  $("#count").textContent = `${list.length} вин`;
  renderMore();
}

function renderMore() {
  const chunk = state.filtered.slice(state.shown, state.shown + 40);
  const frag = document.createDocumentFragment();
  for (const w of chunk) frag.append(wineItem(w));
  $("#list").append(frag);
  state.shown += chunk.length;
}

function wineItem(w, extraSub) {
  const li = document.createElement("li");
  li.className = "wine-item";
  li.innerHTML = `
    ${w.photo ? `<img loading="lazy" src="${w.photo}" alt="">` : `<div class="noimg">🍷</div>`}
    <div class="wi-body">
      <div class="wi-title"></div>
      <div class="wi-sub"></div>
      <div class="wi-tags"></div>
    </div>`;
  li.querySelector(".wi-title").textContent = w.title;
  li.querySelector(".wi-sub").textContent = extraSub ?? [w.manufacturer, w.region].filter(Boolean).join(" · ");
  const tags = li.querySelector(".wi-tags");
  if (w.rating) {
    const r = document.createElement("span");
    r.className = "rating"; r.textContent = `★ ${w.rating}`;
    tags.append(r);
  }
  if (w.category) {
    const t = document.createElement("span");
    t.className = "tag"; t.textContent = w.category;
    tags.append(t);
  }
  li.addEventListener("click", () => openDetail(w.slug));
  return li;
}

function openDetail(slug) {
  const w = state.bySlug.get(slug);
  if (!w) return;
  const favs = store.get("favs", []);
  const my = store.get("notes", {})[slug] || {};
  const d = $("#detail");
  d.hidden = false;
  document.body.style.overflow = "hidden";
  d.innerHTML = `
    <div class="d-hero" style="${w.gradient ? `background:${w.gradient}` : ""}">
      <button class="d-close" aria-label="Закрыть">✕</button>
      <button class="d-fav" aria-label="В избранное">${favs.includes(slug) ? "♥" : "♡"}</button>
      ${w.photo ? `<img src="${w.photo}" alt="">` : ""}
    </div>
    <div class="d-body">
      <div class="d-title"></div>
      <div class="d-sub"></div>
      ${w.rating ? `<div class="d-rating">★ ${w.rating} · рейтинг платформы</div>` : ""}
      <div class="d-facts"></div>
      <div class="d-desc"></div>
      ${w.dishes.length ? `<div class="d-h">К блюдам</div><div class="d-facts dishes"></div>` : ""}
      <div class="d-h">Моя оценка</div>
      <div class="stars">${[1, 2, 3, 4, 5].map((n) => `<span data-n="${n}" class="${my.rating >= n ? "on" : ""}">★</span>`).join("")}</div>
      <textarea class="note" placeholder="Заметка: где пил, с чем, впечатления…"></textarea>
    </div>`;
  d.querySelector(".d-title").textContent = w.title;
  d.querySelector(".d-sub").textContent = [w.manufacturer, w.region].filter(Boolean).join(" · ");
  const facts = d.querySelector(".d-facts");
  const factList = [
    w.category, w.wineColor,
    w.alcohol ? `${w.alcohol}% алк.` : null,
    w.temperature ? `подача ${w.temperature}` : null,
    ...w.grapes,
  ].filter(Boolean);
  for (const t of factList) {
    const s = document.createElement("span");
    s.className = "tag"; s.textContent = t;
    facts.append(s);
  }
  d.querySelector(".d-desc").textContent = w.description;
  if (w.dishes.length) {
    const dl = d.querySelector(".dishes");
    for (const dish of w.dishes) {
      const s = document.createElement("span");
      s.className = "tag"; s.textContent = dish;
      dl.append(s);
    }
  }
  const note = d.querySelector(".note");
  note.value = my.note || "";
  note.addEventListener("change", () => saveNote(slug, { note: note.value }));
  d.querySelector(".stars").addEventListener("click", (e) => {
    const n = Number(e.target.dataset?.n);
    if (!n) return;
    saveNote(slug, { rating: n });
    d.querySelectorAll(".stars span").forEach((s) => s.classList.toggle("on", Number(s.dataset.n) <= n));
  });
  d.querySelector(".d-close").addEventListener("click", closeDetail);
  d.querySelector(".d-fav").addEventListener("click", (e) => {
    const cur = store.get("favs", []);
    const idx = cur.indexOf(slug);
    if (idx >= 0) cur.splice(idx, 1); else cur.unshift(slug);
    store.set("favs", cur);
    e.target.textContent = idx >= 0 ? "♡" : "♥";
    renderMine();
  });
  d.scrollTop = 0;
}

function closeDetail() {
  $("#detail").hidden = true;
  document.body.style.overflow = "";
}

function saveNote(slug, patch) {
  const notes = store.get("notes", {});
  notes[slug] = { ...notes[slug], ...patch };
  store.set("notes", notes);
}

function addHistory(slug, confidence) {
  const hist = store.get("history", []);
  hist.unshift({ slug, confidence, ts: Date.now() });
  store.set("history", hist.slice(0, 200));
  renderMine();
}

function renderMine() {
  const favs = store.get("favs", []);
  const hist = store.get("history", []);
  const favList = $("#fav-list"), histList = $("#hist-list");
  favList.innerHTML = ""; histList.innerHTML = "";
  for (const slug of favs) {
    const w = state.bySlug.get(slug);
    if (w) favList.append(wineItem(w));
  }
  for (const h of hist.slice(0, 50)) {
    const w = state.bySlug.get(h.slug);
    if (!w) continue;
    const when = new Date(h.ts).toLocaleDateString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    histList.append(wineItem(w, `скан · ${when}`));
  }
  $("#fav-empty").hidden = favs.length > 0;
  $("#hist-empty").hidden = hist.length > 0;
}

$$(".tabbtn").forEach((btn) => btn.addEventListener("click", () => {
  $$(".tabbtn").forEach((b) => b.classList.toggle("active", b === btn));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === btn.dataset.tab));
  if (btn.dataset.tab === "tab-mine") renderMine();
}));

$("#search").addEventListener("input", (e) => { state.filters.q = e.target.value; applyFilters(); });
$$("#color-chips .chip").forEach((chip) => chip.addEventListener("click", () => {
  const c = chip.dataset.color;
  if (state.filters.colors.has(c)) state.filters.colors.delete(c); else state.filters.colors.add(c);
  chip.classList.toggle("on");
  applyFilters();
}));
for (const [id, key] of [["#f-sweetness", "sweetness"], ["#f-region", "region"], ["#f-grape", "grape"]]) {
  $(id).addEventListener("change", (e) => { state.filters[key] = e.target.value; applyFilters(); });
}
new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting && state.shown < state.filtered.length) renderMore();
}).observe($("#list-sentinel"));

boot();
