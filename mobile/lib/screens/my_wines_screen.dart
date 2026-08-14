import 'package:flutter/material.dart';

import '../models/wine.dart';
import '../services/my_wines_store.dart';
import '../services/wine_repository.dart';
import '../widgets/wine_widgets.dart';
import 'wine_detail_screen.dart';

class MyWinesScreen extends StatefulWidget {
  const MyWinesScreen({super.key, required this.repository, required this.store});

  final WineRepository repository;
  final MyWinesStore store;

  @override
  State<MyWinesScreen> createState() => _MyWinesScreenState();
}

class _MyWinesScreenState extends State<MyWinesScreen> {
  Future<void> _openWine(Wine wine) async {
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => WineDetailScreen(wine: wine, store: widget.store),
      ),
    );
    // Возврат с деталей: избранное могло измениться.
    if (mounted) setState(() {});
  }

  String _formatTs(DateTime ts) {
    String two(int v) => v.toString().padLeft(2, '0');
    return '${two(ts.day)}.${two(ts.month)}.${ts.year} ${two(ts.hour)}:${two(ts.minute)}';
  }

  Widget _sectionHeader(BuildContext context, IconData icon, String title) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 8),
      child: Row(
        children: [
          Icon(icon, size: 20, color: theme.colorScheme.primary),
          const SizedBox(width: 8),
          Text(
            title,
            style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }

  Widget _emptyHint(BuildContext context, String text) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
      child: Text(
        text,
        style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final favWines = widget.store.favs
        .map(widget.repository.bySlug)
        .whereType<Wine>()
        .toList();

    final history = widget.store.history;

    final children = <Widget>[
      _sectionHeader(context, Icons.favorite, 'Избранное'),
      if (favWines.isEmpty)
        _emptyHint(context,
            'Пока пусто. Отмечайте вина сердечком на странице вина — они появятся здесь.')
      else
        for (final wine in favWines)
          WineCard(wine: wine, onTap: () => _openWine(wine)),
      _sectionHeader(context, Icons.history, 'История сканов'),
      if (history.isEmpty)
        _emptyHint(context,
            'Сканов ещё не было. Откройте вкладку «Скан» и сфотографируйте этикетку — распознанные вина сохранятся здесь.')
      else
        for (final entry in history)
          _historyCard(context, entry),
      const SizedBox(height: 24),
    ];

    return ListView(children: children);
  }

  Widget _historyCard(BuildContext context, HistoryEntry entry) {
    final wine = widget.repository.bySlug(entry.slug);
    if (wine == null) return const SizedBox.shrink();
    final theme = Theme.of(context);
    return WineCard(
      wine: wine,
      onTap: () => _openWine(wine),
      subtitleExtra: Wrap(
        spacing: 8,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          confidenceBadge(entry.confidence),
          Text(
            _formatTs(entry.ts),
            style: theme.textTheme.bodySmall
                ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
          ),
        ],
      ),
    );
  }
}
