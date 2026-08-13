// Зеркало core/matcher.py — менять синхронно.
import { tokenize, years } from "./normalize.js";

const FIELD_WEIGHTS = [["title", 3.0], ["manufacturer", 2.0], ["grapes", 1.0], ["extra", 0.5]];

function levenshteinLe(a, b, maxd) {
  if (Math.abs(a.length - b.length) > maxd) return false;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)));
      best = Math.min(best, cur[j]);
    }
    if (best > maxd) return false;
    prev = cur;
  }
  return prev[b.length] <= maxd;
}

function tokensMatch(q, d) {
  if (q === d) return 1.0;
  if (q.length >= 4 && (d.startsWith(q) || q.startsWith(d)) && Math.abs(q.length - d.length) <= 3) return 0.85;
  const n = Math.max(q.length, d.length);
  if (n >= 8 && levenshteinLe(q, d, 2)) return 0.7;
  if (n >= 5 && levenshteinLe(q, d, 1)) return 0.7;
  return 0.0;
}

export class Matcher {
  constructor(wines) {
    this.wines = wines;
    this.docs = [];
    const df = new Map();
    for (const w of wines) {
      const fields = {
        title: tokenize(w.title || "", true),
        manufacturer: tokenize(w.manufacturer || "", true),
        grapes: tokenize((w.grapes || []).join(" "), true),
        extra: tokenize(`${w.region || ""} ${w.category || ""}`),
      };
      const docTokens = new Map();
      for (const [field, weight] of FIELD_WEIGHTS) {
        for (const tok of fields[field]) {
          docTokens.set(tok, Math.max(docTokens.get(tok) || 0, weight));
        }
      }
      this.docs.push({
        tokens: docTokens,
        years: years(`${w.title || ""} ${w.slug || ""}`),
      });
      for (const tok of docTokens.keys()) df.set(tok, (df.get(tok) || 0) + 1);
    }
    const n = Math.max(wines.length, 1);
    this.idf = new Map();
    for (const [tok, c] of df) this.idf.set(tok, 0.5 + Math.log(n / c) / Math.log(n));
  }

  search(ocrText, top = 5) {
    const qTokens = [...new Set(tokenize(ocrText, true))].filter((t) => !/^\d+$/.test(t));
    const qYears = years(ocrText);
    if (!qTokens.length) return [];
    const results = [];
    for (let widx = 0; widx < this.docs.length; widx++) {
      const doc = this.docs[widx];
      let score = 0;
      let denom = 0;
      for (const qt of qTokens) {
        const qIdf = this.idf.get(qt) ?? 1.0;
        denom += qIdf;
        let best = 0;
        for (const [dt, w] of doc.tokens) {
          const m = tokensMatch(qt, dt);
          if (m) best = Math.max(best, m * w * qIdf);
        }
        score += best;
      }
      if (!denom) continue;
      let conf = score / (denom * 3.0);
      if (qYears.size && [...doc.years].some((y) => qYears.has(y))) conf = Math.min(1, conf + 0.1);
      if (conf > 0.02) results.push({ widx, confidence: conf });
    }
    results.sort((a, b) => (b.confidence - a.confidence)
      || ((this.wines[b.widx].rating || 0) - (this.wines[a.widx].rating || 0)));
    return results.slice(0, top).map((r) => ({ wine: this.wines[r.widx], confidence: r.confidence }));
  }
}
