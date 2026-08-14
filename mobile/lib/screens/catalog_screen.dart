import 'package:flutter/material.dart';

import '../models/wine.dart';
import '../services/my_wines_store.dart';
import '../services/wine_repository.dart';
import '../widgets/wine_widgets.dart';
import 'wine_detail_screen.dart';

class CatalogScreen extends StatefulWidget {
  const CatalogScreen({super.key, required this.repository, required this.store});

  final WineRepository repository;
  final MyWinesStore store;

  @override
  State<CatalogScreen> createState() => _CatalogScreenState();
}

class _CatalogScreenState extends State<CatalogScreen> {
  final TextEditingController _searchCtrl = TextEditingController();

  static const List<String> _colorOptions = ['Белое', 'Красное', 'Розовое', 'Оранжевое'];

  final Set<String> _selectedColors = {};
  String? _sweetness;
  String? _region;
  String? _grape;

  late final List<String> _sweetnessOptions;
  late final List<String> _regionOptions;
  late final List<String> _grapeOptions;

  @override
  void initState() {
    super.initState();
    final wines = widget.repository.wines;
    _sweetnessOptions = _distinct(wines.map((w) => w.sweetness));
    _regionOptions = _distinct(wines.map((w) => w.region));
    _grapeOptions = _distinct(wines.expand((w) => w.grapes));
  }

  List<String> _distinct(Iterable<String> values) {
    final set = <String>{};
    for (final v in values) {
      final t = v.trim();
      if (t.isNotEmpty) set.add(t);
    }
    final list = set.toList()..sort((a, b) => a.toLowerCase().compareTo(b.toLowerCase()));
    return list;
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  bool _passesFilters(Wine w) {
    if (_selectedColors.isNotEmpty && !_selectedColors.contains(w.color)) return false;
    if (_sweetness != null && w.sweetness != _sweetness) return false;
    if (_region != null && w.region != _region) return false;
    if (_grape != null && !w.grapes.contains(_grape)) return false;
    return true;
  }

  // Семантика как в PWA (app/js/app.js applyFilters): сначала фильтры по
  // всему каталогу, затем пересечение с top-400 поиска, порядок — ранг поиска.
  List<Wine> _filteredWines() {
    final query = _searchCtrl.text.trim();
    final filtered = widget.repository.wines.where(_passesFilters).toList();
    if (query.isEmpty) {
      return filtered
        ..sort((a, b) => (b.rating ?? -1).compareTo(a.rating ?? -1));
    }
    final rank = <String, int>{};
    final hits = widget.repository.matcher.search(query, top: 400);
    for (var i = 0; i < hits.length; i++) {
      rank[hits[i].wine.slug] = i;
    }
    return filtered.where((w) => rank.containsKey(w.slug)).toList()
      ..sort((a, b) => rank[a.slug]!.compareTo(rank[b.slug]!));
  }

  void _openWine(Wine wine) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => WineDetailScreen(wine: wine, store: widget.store),
      ),
    );
  }

  Widget _dropdown({
    required String hint,
    required String? value,
    required List<String> options,
    required ValueChanged<String?> onChanged,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(10),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String?>(
          value: value,
          hint: Text(hint, style: const TextStyle(fontSize: 13)),
          isDense: true,
          style: TextStyle(
              fontSize: 13, color: Theme.of(context).colorScheme.onSurface),
          items: [
            DropdownMenuItem<String?>(value: null, child: Text('$hint: все')),
            for (final o in options)
              DropdownMenuItem<String?>(value: o, child: Text(o)),
          ],
          onChanged: onChanged,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final wines = _filteredWines();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
          child: TextField(
            controller: _searchCtrl,
            onChanged: (_) => setState(() {}),
            textInputAction: TextInputAction.search,
            decoration: InputDecoration(
              hintText: 'Поиск: название, винодельня, сорт…',
              prefixIcon: const Icon(Icons.search),
              suffixIcon: _searchCtrl.text.isEmpty
                  ? null
                  : IconButton(
                      icon: const Icon(Icons.clear),
                      onPressed: () {
                        _searchCtrl.clear();
                        setState(() {});
                      },
                    ),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
              isDense: true,
            ),
          ),
        ),
        // Цвета вина.
        SizedBox(
          height: 48,
          child: ListView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            children: [
              for (final c in _colorOptions)
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: FilterChip(
                    label: Text(c),
                    selected: _selectedColors.contains(c),
                    onSelected: (sel) {
                      setState(() {
                        if (sel) {
                          _selectedColors.add(c);
                        } else {
                          _selectedColors.remove(c);
                        }
                      });
                    },
                  ),
                ),
            ],
          ),
        ),
        // Выпадающие фильтры.
        SizedBox(
          height: 44,
          child: ListView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 12),
            children: [
              _dropdown(
                hint: 'Категория',
                value: _sweetness,
                options: _sweetnessOptions,
                onChanged: (v) => setState(() => _sweetness = v),
              ),
              const SizedBox(width: 8),
              _dropdown(
                hint: 'Регион',
                value: _region,
                options: _regionOptions,
                onChanged: (v) => setState(() => _region = v),
              ),
              const SizedBox(width: 8),
              _dropdown(
                hint: 'Сорт',
                value: _grape,
                options: _grapeOptions,
                onChanged: (v) => setState(() => _grape = v),
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
          child: Text(
            pluralWines(wines.length),
            style: theme.textTheme.bodySmall
                ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
          ),
        ),
        Expanded(
          child: wines.isEmpty
              ? Center(
                  child: Text(
                    'Ничего не найдено.\nПопробуйте изменить запрос или фильтры.',
                    textAlign: TextAlign.center,
                    style: theme.textTheme.bodyMedium
                        ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                  ),
                )
              : ListView.builder(
                  itemCount: wines.length,
                  itemBuilder: (context, i) {
                    final wine = wines[i];
                    return WineCard(wine: wine, onTap: () => _openWine(wine));
                  },
                ),
        ),
      ],
    );
  }
}
