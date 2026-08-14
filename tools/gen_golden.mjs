// Генератор mobile/test/golden_parity.json — эталона паритета Dart-порта
// с JS-ядром (app/js/normalize.js + matcher.js).
// Запуск из корня репо: node tools/gen_golden.mjs
// Перегенерировать при любом изменении ядра или датапака, затем
// прогнать: cd mobile && flutter test test/parity_test.dart
import { readFileSync, writeFileSync } from "node:fs";
import { Matcher } from "../app/js/matcher.js";
import { tokenize, translitToken, years } from "../app/js/normalize.js";

const data = JSON.parse(readFileSync("data/datapack/wines.json", "utf8"));
const m = new Matcher(data.wines);
const queries = [
  "ARISTOV ARTE лесная просека игристое брют",
  "ALMA VALLEY riesling 2022 крым",
  "фанагория каберне", "ГРАФ ВОРОНЦОВ брют белое",
  "шато тамань саперави", "Fanagoria Cru Lermont Saperavi 2020",
  "абрау дюрсо brut rose", "INKERMAN пино нуар",
  "kuban vino шардоне сухое", "массандра портвейн",
  "ZB wine рислинг", "голубицкое estate blanc de blancs",
  "мысхако красностоп 2021", "esse muscat",
  "дербент вино дикий виноград", "sikory пти вердо",
  "новый свет брют розовое", "солнечная долина мускат белый",
  "gai kodzor viognier", "лефкадия рислинг",
  "опалиха пино гри", "усадьба дивноморское",
  "chateau tamagne reserve", "вилла звезда донская",
  "плечистик цимлянский", "кокур белый качинский",
  "high hopes шираз", "бельбек мерло розе",
  "twk вайн крафт", "просто вино красное полусухое",
];
const golden = {
  translit: ["aristov", "riesling", "shchedrin", "chateau", "yubileynaya", "quantum"]
    .map((t) => [t, translitToken(t)]),
  tokenize: ["ARISTOV ARTE «Лесная Просека» 2022 брют 12%", "Château Tamagne  сухое ГОСТ"]
    .map((t) => [t, tokenize(t, true)]),
  years: [["урожай 2021 и 1998", [...years("урожай 2021 и 1998")]]],
  queries: queries.map((q) => ({
    q,
    top: m.search(q, 5).map((r) => ({ slug: r.wine.slug, c: Number(r.confidence.toFixed(6)) })),
  })),
};
writeFileSync("mobile/test/golden_parity.json", JSON.stringify(golden, null, 1));
console.log("golden written:", queries.length, "queries");
