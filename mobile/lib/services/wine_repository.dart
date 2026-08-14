import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../core/matcher.dart';
import '../models/wine.dart';

class WineRepository {
  WineRepository._(this._wines, this._version)
      : _bySlug = {for (final w in _wines) w.slug: w},
        _matcher = Matcher(_wines);

  final List<Wine> _wines;
  final String _version;
  final Map<String, Wine> _bySlug;
  final Matcher _matcher;

  List<Wine> get wines => _wines;

  Wine? bySlug(String slug) => _bySlug[slug];

  Matcher get matcher => _matcher;

  String get version => _version;

  static Future<WineRepository> load() async {
    final raw = await rootBundle.loadString('assets/wines.json');
    final data = await compute(_parse, raw);
    final meta = (data['meta'] as Map<String, dynamic>?) ?? const {};
    final winesJson = (data['wines'] as List<dynamic>?) ?? const [];
    final wines = winesJson
        .map((e) => Wine.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
    return WineRepository._(wines, '${meta['version'] ?? ''}');
  }

  static Map<String, dynamic> _parse(String raw) =>
      jsonDecode(raw) as Map<String, dynamic>;
}
