// Зеркало core/normalize.js — менять синхронно. Точный порт (бит-в-бит).

const List<List<String>> lat2cyr = [
  ['shch', 'щ'], ['sch', 'щ'],
  ['zh', 'ж'], ['kh', 'х'], ['ts', 'ц'], ['ch', 'ч'], ['sh', 'ш'],
  ['yu', 'ю'], ['ju', 'ю'], ['ya', 'я'], ['ja', 'я'], ['yo', 'е'], ['jo', 'е'],
  ['ye', 'е'], ['je', 'е'], ['ck', 'к'], ['qu', 'кв'], ['ph', 'ф'], ['th', 'т'],
  ['a', 'а'], ['b', 'б'], ['c', 'к'], ['d', 'д'], ['e', 'е'], ['f', 'ф'],
  ['g', 'г'], ['h', 'х'], ['i', 'и'], ['j', 'й'], ['k', 'к'], ['l', 'л'],
  ['m', 'м'], ['n', 'н'], ['o', 'о'], ['p', 'п'], ['q', 'к'], ['r', 'р'],
  ['s', 'с'], ['t', 'т'], ['u', 'у'], ['v', 'в'], ['w', 'в'], ['x', 'кс'],
  ['y', 'и'], ['z', 'з'],
];

// Только шумовые слова. Категорийные (брют/белое/сухое...) НЕ стоп-слова.
const Set<String> stopwords = {
  'вино', 'wine', 'россия', 'russia', 'оф', 'of', 'и', 'in',
  'the', 'de', 'ооо', 'зао', 'ао', 'гост', 'литр', 'объем', 'алк', 'alc',
  'vol', 'год', 'урожай',
};

String translitToken(String tok) {
  final low = tok.toLowerCase();
  final out = StringBuffer();
  var i = 0;
  while (i < low.length) {
    var matched = false;
    for (final pair in lat2cyr) {
      if (low.startsWith(pair[0], i)) {
        out.write(pair[1]);
        i += pair[0].length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      out.write(low[i]);
      i += 1;
    }
  }
  return out.toString();
}

final RegExp _nonWord = RegExp(r'[^a-zа-я0-9]+');
final RegExp _spaces = RegExp(r'\s+');
final RegExp _latin = RegExp(r'^[a-z0-9]+$');
final RegExp _digits = RegExp(r'^\d+$');
final RegExp _yearRe = RegExp(r'\b(19[89]\d|20[0-3]\d)\b');

String clean(String text) {
  // 'İ' (U+0130): JS full case mapping даёт i+U+0307, Dart simple mapping — i.
  // Унифицируем до lowercase во всех трёх зеркалах (js/py/dart).
  return text
      .replaceAll('İ', 'I')
      .toLowerCase()
      .replaceAll('ё', 'е')
      .replaceAll(_nonWord, ' ')
      .replaceAll(_spaces, ' ')
      .trim();
}

List<String> tokenize(String text, {bool dropStop = false}) {
  final toks = <String>[];
  for (var tok in clean(text).split(' ')) {
    if (tok.length < 2) continue;
    if (_latin.hasMatch(tok) && !_digits.hasMatch(tok)) tok = translitToken(tok);
    if (dropStop && stopwords.contains(tok)) continue;
    toks.add(tok);
  }
  return toks;
}

Set<String> years(String text) {
  return _yearRe.allMatches(text).map((m) => m[0]!).toSet();
}
