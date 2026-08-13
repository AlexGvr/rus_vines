// Eval ядра: OCR (tesseract.js, как в PWA) + матчер (app/js/matcher.js).
// Фото берутся в 1200px (как снимет пользователь), кэшируются в data/eval_photos/.
// Метрики: top-1 / top-5 accuracy. Запуск: node core/eval.mjs [N]
import Tesseract from "tesseract.js";
import sharp from "sharp";
import { readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { Matcher } from "../app/js/matcher.js";

const { createWorker } = Tesseract;
const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const N = Number(process.argv[2] || 150);
const IMG = "https://api.vino-svoe.ru/v1/img/str-api/1200/1200/resize";
const CACHE = path.join(ROOT, "data/eval_photos");
mkdirSync(CACHE, { recursive: true });

const data = JSON.parse(readFileSync(path.join(ROOT, "data/datapack/wines.json"), "utf8"));
const matcher = new Matcher(data.wines);
const withPhoto = data.wines.filter((w) => w.photo);

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(42);
const sample = [...withPhoto].sort(() => rand() - 0.5).slice(0, N);

async function photo1200(slug) {
  const cached = path.join(CACHE, `${slug}.webp`);
  if (existsSync(cached)) return cached;
  const card = JSON.parse(readFileSync(path.join(ROOT, "data/raw/cards", `${slug}.json`), "utf8"));
  const url = card.image?.url;
  if (!url) return null;
  const resp = await fetch(`${IMG}${url}`, { headers: { "User-Agent": "RusVinesMVP/0.1 eval" } });
  if (!resp.ok) return null;
  writeFileSync(cached, Buffer.from(await resp.arrayBuffer()));
  await new Promise((r) => setTimeout(r, 250));
  return cached;
}

const worker = await createWorker("rus+eng", 1, {
  langPath: path.join(ROOT, "app/vendor/tesseract/lang"),
  gzip: true,
});

let top1 = 0, top5 = 0, noText = 0, used = 0;
const misses = [];
const t0 = Date.now();
for (let i = 0; i < sample.length; i++) {
  const w = sample[i];
  const photoPath = await photo1200(w.slug).catch(() => null);
  if (!photoPath) continue;
  used++;
  let text = "";
  try {
    // два прохода, как в app/js/scan.js: полный кадр + кроп зоны этикетки
    const meta = await sharp(photoPath).metadata();
    const full = await sharp(photoPath).resize({ width: 1400, height: 1400, fit: "inside" })
      .grayscale().normalise().png().toBuffer();
    const cw = Math.round(meta.width * 0.64);
    const ch = Math.round(meta.height * 0.5);
    const crop = await sharp(photoPath)
      .extract({ left: Math.round(meta.width * 0.18), top: Math.round(meta.height * 0.38), width: cw, height: ch })
      .resize({ width: 1400 }).grayscale().normalise().png().toBuffer();
    const [r1, r2] = [await worker.recognize(full), await worker.recognize(crop)];
    text = `${r1.data.text || ""}\n${r2.data.text || ""}`;
  } catch { /* нечитаемое фото */ }
  if (!text.trim()) { noText++; misses.push([w.slug, "(нет текста)"]); continue; }
  const res = matcher.search(text, 5);
  const slugs = res.map((r) => r.wine.slug);
  if (slugs[0] === w.slug) top1++;
  if (slugs.includes(w.slug)) top5++;
  else misses.push([w.slug, slugs[0] || "-"]);
  if (used % 25 === 0) {
    console.log(`${used}/${sample.length}  top1=${(top1 / used * 100).toFixed(1)}%  top5=${(top5 / used * 100).toFixed(1)}%`);
  }
}
await worker.terminate();

const secs = ((Date.now() - t0) / 1000).toFixed(0);
console.log("\n=== EVAL (эталонные фото 1200px) ===");
console.log(`n=${used}, время ${secs}s`);
console.log(`top-1: ${(top1 / used * 100).toFixed(1)}%`);
console.log(`top-5: ${(top5 / used * 100).toFixed(1)}%`);
console.log(`фото без текста: ${noText}`);
console.log("\nпримеры промахов (истина -> top1):");
for (const [truth, got] of misses.slice(0, 15)) console.log(`  ${truth} -> ${got}`);
