import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

class HistoryEntry {
  HistoryEntry({required this.slug, required this.confidence, required this.ts});

  final String slug;
  final double confidence;
  final DateTime ts;

  factory HistoryEntry.fromJson(Map<String, dynamic> json) => HistoryEntry(
        slug: json['slug'] as String? ?? '',
        confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
        ts: DateTime.tryParse(json['ts'] as String? ?? '') ?? DateTime.now(),
      );

  Map<String, dynamic> toJson() => {
        'slug': slug,
        'confidence': confidence,
        'ts': ts.toIso8601String(),
      };
}

class MyWinesStore {
  MyWinesStore._(this._prefs)
      : _favs = List<String>.from(_prefs.getStringList(_kFavs) ?? const []),
        _history = _decodeHistory(_prefs.getString(_kHistory)),
        _notes = _decodeNotes(_prefs.getString(_kNotes));

  static const _kFavs = 'favs';
  static const _kHistory = 'history';
  static const _kNotes = 'notes';
  static const _maxHistory = 200;

  final SharedPreferences _prefs;
  final List<String> _favs;
  final List<HistoryEntry> _history;
  final Map<String, Map<String, dynamic>> _notes;

  static Future<MyWinesStore> load() async {
    final prefs = await SharedPreferences.getInstance();
    return MyWinesStore._(prefs);
  }

  // --- Favorites ---

  List<String> get favs => List.unmodifiable(_favs);

  bool isFav(String slug) => _favs.contains(slug);

  Future<void> toggleFav(String slug) async {
    if (!_favs.remove(slug)) {
      _favs.insert(0, slug);
    }
    await _prefs.setStringList(_kFavs, _favs);
  }

  // --- History ---

  List<HistoryEntry> get history => List.unmodifiable(_history);

  Future<void> addHistory(String slug, double confidence) async {
    _history.insert(
      0,
      HistoryEntry(slug: slug, confidence: confidence, ts: DateTime.now()),
    );
    if (_history.length > _maxHistory) {
      _history.removeRange(_maxHistory, _history.length);
    }
    await _prefs.setString(
      _kHistory,
      jsonEncode(_history.map((e) => e.toJson()).toList(growable: false)),
    );
  }

  // --- Notes ---

  ({int? rating, String? note}) noteFor(String slug) {
    final n = _notes[slug];
    if (n == null) return (rating: null, note: null);
    return (
      rating: (n['rating'] as num?)?.toInt(),
      note: n['note'] as String?,
    );
  }

  Future<void> saveNote(String slug, {int? rating, String? note}) async {
    if (rating == null && (note == null || note.isEmpty)) {
      _notes.remove(slug);
    } else {
      _notes[slug] = {'rating': rating, 'note': note};
    }
    await _prefs.setString(_kNotes, jsonEncode(_notes));
  }

  // --- Decoding helpers ---

  static List<HistoryEntry> _decodeHistory(String? raw) {
    if (raw == null || raw.isEmpty) return [];
    try {
      final list = jsonDecode(raw) as List<dynamic>;
      return list
          .map((e) => HistoryEntry.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  static Map<String, Map<String, dynamic>> _decodeNotes(String? raw) {
    if (raw == null || raw.isEmpty) return {};
    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return map.map(
        (k, v) => MapEntry(k, Map<String, dynamic>.from(v as Map)),
      );
    } catch (_) {
      return {};
    }
  }
}
