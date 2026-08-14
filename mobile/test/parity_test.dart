import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:svoe_vino/core/matcher.dart' as core;
import 'package:svoe_vino/core/normalize.dart';
import 'package:svoe_vino/models/wine.dart';

void main() {
  final root = Directory.current.path;
  final golden = jsonDecode(
          File('$root/test/golden_parity.json').readAsStringSync())
      as Map<String, dynamic>;
  final winesJson =
      jsonDecode(File('$root/assets/wines.json').readAsStringSync())
          as Map<String, dynamic>;
  final wines = (winesJson['wines'] as List)
      .map((e) => Wine.fromJson(e as Map<String, dynamic>))
      .toList();

  test('translit parity', () {
    for (final pair in golden['translit'] as List) {
      expect(translitToken(pair[0] as String), pair[1],
          reason: 'translit("${pair[0]}")');
    }
  });

  test('tokenize parity', () {
    for (final pair in golden['tokenize'] as List) {
      expect(tokenize(pair[0] as String, dropStop: true),
          (pair[1] as List).cast<String>(),
          reason: 'tokenize("${pair[0]}")');
    }
  });

  test('years parity', () {
    for (final pair in golden['years'] as List) {
      expect(years(pair[0] as String),
          (pair[1] as List).cast<String>().toSet(),
          reason: 'years("${pair[0]}")');
    }
  });

  test('search parity: top-5 slugs and confidence', () {
    final matcher = core.Matcher(wines);
    for (final q in golden['queries'] as List) {
      final query = q['q'] as String;
      final expected = q['top'] as List;
      final got = matcher.search(query, top: 5);
      expect(got.length, expected.length, reason: 'len for "$query"');
      for (var i = 0; i < expected.length; i++) {
        expect(got[i].wine.slug, expected[i]['slug'],
            reason: 'slug[$i] for "$query"');
        final c = (expected[i]['c'] as num).toDouble();
        expect((got[i].confidence - c).abs() < 1e-4, isTrue,
            reason:
                'conf[$i] for "$query": got ${got[i].confidence}, want $c');
      }
    }
  });
}
