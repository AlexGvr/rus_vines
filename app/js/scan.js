// Скан: фото → Tesseract.js (rus+eng, офлайн) → матчер → карточка / топ-5.
let worker = null;
let workerReady = null;

function ensureWorker(onProgress) {
  if (!workerReady) {
    const abs = (p) => new URL(p, location.href).href;
    workerReady = Tesseract.createWorker("rus+eng", 1, {
      workerPath: abs("vendor/tesseract/worker.min.js"),
      corePath: abs("vendor/tesseract/"),
      langPath: abs("vendor/tesseract/lang"),
      gzip: true,
      logger: (m) => {
        if (m.status === "recognizing text" && onProgress) onProgress(m.progress);
      },
    }).then((w) => { worker = w; return w; });
  }
  return workerReady;
}

function toCanvas(bmp, sx, sy, sw, sh, targetSide) {
  const scale = Math.min(targetSide / Math.max(sw, sh), 4);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(sw * scale);
  canvas.height = Math.round(sh * scale);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.filter = "grayscale(1) contrast(1.3)";
  ctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  return canvas;
}

// Два кадра: полный + кроп зоны этикетки (центр кадра с бутылкой).
// Зеркалится в core/eval.mjs.
async function prepImages(file, targetSide = 1400) {
  const bmp = await createImageBitmap(file);
  const full = toCanvas(bmp, 0, 0, bmp.width, bmp.height, targetSide);
  const crop = toCanvas(
    bmp,
    Math.round(bmp.width * 0.18), Math.round(bmp.height * 0.38),
    Math.round(bmp.width * 0.64), Math.round(bmp.height * 0.5),
    targetSide,
  );
  return [full, crop];
}

export function initScan(state, { openDetail, addHistory }) {
  const $ = (s) => document.querySelector(s);
  const input = $("#scan-input");
  const progress = $("#scan-progress");
  const hero = document.querySelector(".scan-hero");
  const result = $("#scan-result");
  const bar = $("#scan-bar");
  const status = $("#scan-status");

  $("#scan-btn").addEventListener("click", () => input.click());

  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    if (!file) return;
    input.value = "";
    hero.hidden = true;
    result.hidden = true;
    progress.hidden = false;
    bar.style.width = "4%";
    status.textContent = "Готовлю фото…";
    $("#scan-preview").src = URL.createObjectURL(file);
    try {
      const [full, crop] = await prepImages(file);
      status.textContent = "Загружаю распознавание…";
      let pass = 0;
      await ensureWorker((p) => {
        bar.style.width = `${8 + Math.round((pass + p) * 42)}%`;
        status.textContent = `Читаю этикетку… ${Math.round((pass + p) * 50)}%`;
      });
      const r1 = await worker.recognize(full);
      pass = 1;
      const r2 = await worker.recognize(crop);
      bar.style.width = "94%";
      status.textContent = "Ищу в базе…";
      const text = `${r1.data.text || ""}\n${r2.data.text || ""}`;
      const matches = state.matcher.search(text, 5);
      bar.style.width = "100%";
      showResult(text, matches);
    } catch (err) {
      console.error(err);
      status.textContent = "Не получилось распознать. Попробуйте ещё раз при хорошем свете.";
      setTimeout(reset, 2500);
    }
  });

  function reset() {
    progress.hidden = true;
    result.hidden = true;
    hero.hidden = false;
  }

  function confBadge(c) {
    const pct = Math.round(c * 100);
    const cls = c >= 0.45 ? "conf-hi" : c >= 0.22 ? "conf-mid" : "conf-lo";
    return `<span class="conf-badge ${cls}">${pct}%</span>`;
  }

  function showResult(text, matches) {
    progress.hidden = true;
    result.hidden = false;
    result.innerHTML = "";
    if (!matches.length) {
      result.innerHTML = `
        <div class="result-head">Ничего не нашлось. Текст с этикетки не прочитался —
        попробуйте ближе, ровнее, при хорошем свете.</div>`;
    } else {
      const [best, ...rest] = matches;
      const head = document.createElement("div");
      head.className = "result-head";
      head.innerHTML = best.confidence >= 0.22
        ? `Похоже, это: ${confBadge(best.confidence)}`
        : `Уверенности мало ${confBadge(best.confidence)} — проверьте варианты:`;
      result.append(head);
      const ul = document.createElement("ul");
      ul.className = "wine-list";
      for (const m of matches) {
        const li = document.createElement("li");
        li.className = "wine-item";
        const w = m.wine;
        li.innerHTML = `
          ${w.photo ? `<img loading="lazy" src="${w.photo}" alt="">` : `<div class="noimg">🍷</div>`}
          <div class="wi-body">
            <div class="wi-title"></div>
            <div class="wi-sub"></div>
            <div class="wi-tags">${confBadge(m.confidence)}${w.rating ? `<span class="rating">★ ${w.rating}</span>` : ""}</div>
          </div>`;
        li.querySelector(".wi-title").textContent = w.title;
        li.querySelector(".wi-sub").textContent = [w.manufacturer, w.region].filter(Boolean).join(" · ");
        li.addEventListener("click", () => {
          addHistory(w.slug, m.confidence);
          openDetail(w.slug);
        });
        ul.append(li);
      }
      result.append(ul);
    }
    const again = document.createElement("div");
    again.className = "rescan";
    again.innerHTML = `<button class="btn ghost">Сканировать ещё раз</button>`;
    again.querySelector("button").addEventListener("click", reset);
    result.append(again);
  }
}
