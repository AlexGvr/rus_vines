// Зеркало core/normalize.py — менять синхронно.
export const LAT2CYR = [
  ["shch", "щ"], ["sch", "щ"],
  ["zh", "ж"], ["kh", "х"], ["ts", "ц"], ["ch", "ч"], ["sh", "ш"],
  ["yu", "ю"], ["ju", "ю"], ["ya", "я"], ["ja", "я"], ["yo", "е"], ["jo", "е"],
  ["ye", "е"], ["je", "е"], ["ck", "к"], ["qu", "кв"], ["ph", "ф"], ["th", "т"],
  ["a", "а"], ["b", "б"], ["c", "к"], ["d", "д"], ["e", "е"], ["f", "ф"],
  ["g", "г"], ["h", "х"], ["i", "и"], ["j", "й"], ["k", "к"], ["l", "л"],
  ["m", "м"], ["n", "н"], ["o", "о"], ["p", "п"], ["q", "к"], ["r", "р"],
  ["s", "с"], ["t", "т"], ["u", "у"], ["v", "в"], ["w", "в"], ["x", "кс"],
  ["y", "и"], ["z", "з"],
];

// Только шумовые слова. Категорийные (брют/белое/сухое...) НЕ стоп-слова:
// они матчатся полем категории с весом 0.5 и различают вина одной линейки.
export const STOPWORDS = new Set([
  "вино", "wine", "россия", "russia", "оф", "of", "и", "in",
  "the", "de", "ооо", "зао", "ао", "гост", "литр", "объем", "алк", "alc",
  "vol", "год", "урожай",
]);

export function translitToken(tok) {
  const low = tok.toLowerCase();
  let out = "";
  let i = 0;
  while (i < low.length) {
    let matched = false;
    for (const [src, dst] of LAT2CYR) {
      if (low.startsWith(src, i)) {
        out += dst;
        i += src.length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      out += low[i];
      i += 1;
    }
  }
  return out;
}

export function clean(text) {
  // 'İ' (U+0130): full vs simple case mapping расходятся между рантаймами —
  // унифицируем до lowercase во всех трёх зеркалах (js/py/dart).
  return text.replaceAll("İ", "I").toLowerCase().replaceAll("ё", "е")
    .replace(/[^a-zа-я0-9]+/g, " ")
    .replace(/\s+/g, " ").trim();
}

export function tokenize(text, dropStop = false) {
  const toks = [];
  for (let tok of clean(text).split(" ")) {
    if (tok.length < 2) continue;
    if (/^[a-z0-9]+$/.test(tok) && !/^\d+$/.test(tok)) tok = translitToken(tok);
    if (dropStop && STOPWORDS.has(tok)) continue;
    toks.push(tok);
  }
  return toks;
}

export function years(text) {
  return new Set(text.match(/\b(19[89]\d|20[0-3]\d)\b/g) || []);
}
