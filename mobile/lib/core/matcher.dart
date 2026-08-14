// Зеркало core/matcher.js — менять синхронно. Точный порт (бит-в-бит).
import 'dart:math' as math;

import '../models/wine.dart';
import 'normalize.dart';

const List<(String, double)> _fieldWeights = [
  ('title', 3.0),
  ('manufacturer', 2.0),
  ('grapes', 1.0),
  ('extra', 0.5),
];

final RegExp _digitsOnly = RegExp(r'^\d+$');

bool _levenshteinLe(String a, String b, int maxd) {
  if ((a.length - b.length).abs() > maxd) return false;
  var prev = List<int>.generate(b.length + 1, (j) => j);
  for (var i = 1; i <= a.length; i++) {
    final cur = <int>[i];
    var best = i;
    for (var j = 1; j <= b.length; j++) {
      cur.add(math.min(
        math.min(prev[j] + 1, cur[j - 1] + 1),
        prev[j - 1] + (a[i - 1] != b[j - 1] ? 1 : 0),
      ));
      best = math.min(best, cur[j]);
    }
    if (best > maxd) return false;
    prev = cur;
  }
  return prev[b.length] <= maxd;
}

double _tokensMatch(String q, String d) {
  if (q == d) return 1.0;
  if (q.length >= 4 &&
      (d.startsWith(q) || q.startsWith(d)) &&
      (q.length - d.length).abs() <= 3) {
    return 0.85;
  }
  final n = math.max(q.length, d.length);
  if (n >= 8 && _levenshteinLe(q, d, 2)) return 0.7;
  if (n >= 5 && _levenshteinLe(q, d, 1)) return 0.7;
  return 0.0;
}

class MatchResult {
  final Wine wine;
  final double confidence;
  MatchResult(this.wine, this.confidence);
}

class _Doc {
  final Map<String, double> tokens;
  final Set<String> years;
  _Doc(this.tokens, this.years);
}

class Matcher {
  final List<Wine> wines;
  final List<_Doc> _docs = [];
  final Map<String, double> _idf = {};

  Matcher(this.wines) {
    final df = <String, int>{};
    for (final w in wines) {
      final fields = <String, List<String>>{
        'title': tokenize(w.title, dropStop: true),
        'manufacturer': tokenize(w.manufacturer, dropStop: true),
        'grapes': tokenize(w.grapes.join(' '), dropStop: true),
        'extra': tokenize('${w.region} ${w.category}'),
      };
      final docTokens = <String, double>{};
      for (final (field, weight) in _fieldWeights) {
        for (final tok in fields[field]!) {
          final old = docTokens[tok] ?? 0;
          docTokens[tok] = math.max(old, weight);
        }
      }
      _docs.add(_Doc(docTokens, years('${w.title} ${w.slug}')));
      for (final tok in docTokens.keys) {
        df[tok] = (df[tok] ?? 0) + 1;
      }
    }
    final n = math.max(wines.length, 1);
    df.forEach((tok, c) {
      _idf[tok] = 0.5 + math.log(n / c) / math.log(n);
    });
  }

  List<MatchResult> search(String ocrText, {int top = 5}) {
    final qTokens = tokenize(ocrText, dropStop: true)
        .toSet()
        .where((t) => !_digitsOnly.hasMatch(t))
        .toList();
    final qYears = years(ocrText);
    if (qTokens.isEmpty) return [];
    final results = <({int widx, double confidence})>[];
    for (var widx = 0; widx < _docs.length; widx++) {
      final doc = _docs[widx];
      var score = 0.0;
      var denom = 0.0;
      for (final qt in qTokens) {
        final qIdf = _idf[qt] ?? 1.0;
        denom += qIdf;
        var best = 0.0;
        doc.tokens.forEach((dt, w) {
          final m = _tokensMatch(qt, dt);
          if (m != 0.0) best = math.max(best, m * w * qIdf);
        });
        score += best;
      }
      if (denom == 0.0) continue;
      var conf = score / (denom * 3.0);
      if (qYears.isNotEmpty && doc.years.any(qYears.contains)) {
        conf = math.min(1.0, conf + 0.1);
      }
      if (conf > 0.02) results.add((widx: widx, confidence: conf));
    }
    // JS Array.sort стабилен: эквивалентно доп. tie-break по widx asc.
    results.sort((a, b) {
      final byConf = b.confidence.compareTo(a.confidence);
      if (byConf != 0) return byConf;
      final ra = wines[a.widx].rating ?? 0;
      final rb = wines[b.widx].rating ?? 0;
      final byRating = rb.compareTo(ra);
      if (byRating != 0) return byRating;
      return a.widx.compareTo(b.widx);
    });
    return results
        .take(top)
        .map((r) => MatchResult(wines[r.widx], r.confidence))
        .toList();
  }
}
