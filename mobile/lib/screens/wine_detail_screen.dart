import 'dart:async';

import 'package:flutter/material.dart';

import '../models/wine.dart';
import '../services/my_wines_store.dart';
import '../widgets/wine_widgets.dart';

class WineDetailScreen extends StatefulWidget {
  const WineDetailScreen({super.key, required this.wine, required this.store});

  final Wine wine;
  final MyWinesStore store;

  @override
  State<WineDetailScreen> createState() => _WineDetailScreenState();
}

class _WineDetailScreenState extends State<WineDetailScreen> {
  late final TextEditingController _noteCtrl;
  Timer? _noteDebounce;
  int? _myRating;

  @override
  void initState() {
    super.initState();
    final saved = widget.store.noteFor(widget.wine.slug);
    _myRating = saved.rating;
    _noteCtrl = TextEditingController(text: saved.note ?? '');
  }

  @override
  void dispose() {
    // Незаписанный дебаунс сбрасываем на диск, иначе заметка теряется
    // при закрытии экрана раньше 700 мс после ввода.
    if (_noteDebounce?.isActive ?? false) {
      _noteDebounce!.cancel();
      _saveNote();
    }
    _noteCtrl.dispose();
    super.dispose();
  }

  Future<void> _toggleFav() async {
    await widget.store.toggleFav(widget.wine.slug);
    if (mounted) setState(() {});
  }

  Future<void> _saveNote() async {
    final text = _noteCtrl.text.trim();
    await widget.store.saveNote(
      widget.wine.slug,
      rating: _myRating,
      note: text.isEmpty ? null : text,
    );
  }

  void _onNoteChanged(String _) {
    _noteDebounce?.cancel();
    _noteDebounce = Timer(const Duration(milliseconds: 700), _saveNote);
  }

  Future<void> _setRating(int stars) async {
    setState(() => _myRating = stars);
    await _saveNote();
  }

  Widget _chip(BuildContext context, String label, {IconData? icon}) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: theme.colorScheme.secondaryContainer,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 14, color: theme.colorScheme.onSecondaryContainer),
            const SizedBox(width: 4),
          ],
          Text(
            label,
            style: TextStyle(
                fontSize: 12, color: theme.colorScheme.onSecondaryContainer),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(BuildContext context, String text) {
    return Padding(
      padding: const EdgeInsets.only(top: 20, bottom: 8),
      child: Text(
        text,
        style: Theme.of(context)
            .textTheme
            .titleMedium
            ?.copyWith(fontWeight: FontWeight.w700),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final wine = widget.wine;
    final isFav = widget.store.isFav(wine.slug);

    return Scaffold(
      appBar: AppBar(
        title: Text(wine.title, maxLines: 1, overflow: TextOverflow.ellipsis),
        actions: [
          IconButton(
            tooltip: isFav ? 'Убрать из избранного' : 'В избранное',
            icon: Icon(
              isFav ? Icons.favorite : Icons.favorite_outline,
              color: isFav ? kWineColor : null,
            ),
            onPressed: _toggleFav,
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
        children: [
          Center(
            child: SizedBox(
              height: 280,
              child: winePhoto(wine,
                  height: 280, fit: BoxFit.contain, placeholderFontSize: 72),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            wine.title,
            style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 4),
          Text(
            '${wine.manufacturer} · ${wine.region}',
            style: theme.textTheme.bodyMedium
                ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
          ),
          if (wine.rating != null) ...[
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.star, color: kGoldColor, size: 20),
                const SizedBox(width: 4),
                Text(
                  wine.rating!.toStringAsFixed(1),
                  style: theme.textTheme.titleMedium
                      ?.copyWith(fontWeight: FontWeight.w700),
                ),
                const SizedBox(width: 6),
                Text(
                  'рейтинг платформы',
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                ),
              ],
            ),
          ],
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _chip(context, wine.category, icon: Icons.wine_bar),
              if (wine.wineColor.isNotEmpty)
                _chip(context, wine.wineColor, icon: Icons.palette_outlined),
              if (wine.alcohol != null)
                _chip(context, 'Крепость ${_fmtAlcohol(wine.alcohol!)}%',
                    icon: Icons.percent),
              if (wine.temperature.isNotEmpty)
                _chip(context, 'Подача ${wine.temperature}°C',
                    icon: Icons.thermostat_outlined),
              for (final g in wine.grapes) _chip(context, g, icon: Icons.spa_outlined),
            ],
          ),
          if (wine.description.isNotEmpty) ...[
            _sectionTitle(context, 'Описание'),
            Text(wine.description, style: theme.textTheme.bodyMedium?.copyWith(height: 1.4)),
          ],
          if (wine.dishes.isNotEmpty) ...[
            _sectionTitle(context, 'К блюдам'),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final d in wine.dishes)
                  _chip(context, d, icon: Icons.restaurant_outlined),
              ],
            ),
          ],
          _sectionTitle(context, 'Моя оценка'),
          Row(
            children: [
              for (int i = 1; i <= 5; i++)
                IconButton(
                  visualDensity: VisualDensity.compact,
                  icon: Icon(
                    (_myRating ?? 0) >= i ? Icons.star : Icons.star_outline,
                    color: kGoldColor,
                    size: 32,
                  ),
                  onPressed: () => _setRating(i),
                ),
              if (_myRating != null)
                TextButton(
                  onPressed: () async {
                    setState(() => _myRating = null);
                    await _saveNote();
                  },
                  child: const Text('Сбросить'),
                ),
            ],
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _noteCtrl,
            onChanged: _onNoteChanged,
            onEditingComplete: () {
              _noteDebounce?.cancel();
              _saveNote();
              FocusScope.of(context).unfocus();
            },
            maxLines: 3,
            decoration: InputDecoration(
              hintText: 'Заметка: впечатления, с чем пили, где купили…',
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
            ),
          ),
        ],
      ),
    );
  }

  String _fmtAlcohol(double v) {
    return v == v.roundToDouble() ? v.round().toString() : v.toStringAsFixed(1);
  }
}
