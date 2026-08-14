import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../core/matcher.dart';
import '../services/my_wines_store.dart';
import '../services/ocr_service.dart';
import '../services/wine_repository.dart';
import '../widgets/wine_widgets.dart';
import 'wine_detail_screen.dart';

enum _ScanState { idle, processing, results }

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key, required this.repository, required this.store});

  final WineRepository repository;
  final MyWinesStore store;

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  final ImagePicker _picker = ImagePicker();
  late final OcrEngine _ocr = MlkitOcrEngine();

  _ScanState _state = _ScanState.idle;
  List<MatchResult> _results = const [];
  bool _cameraFailed = false;

  @override
  void dispose() {
    _ocr.dispose();
    super.dispose();
  }

  Future<void> _scan(ImageSource source) async {
    XFile? file;
    try {
      file = await _picker.pickImage(source: source, maxWidth: 1920, imageQuality: 90);
    } catch (_) {
      if (source == ImageSource.camera) {
        // Камера недоступна — предлагаем галерею.
        setState(() => _cameraFailed = true);
        _showSnack('Камера недоступна — выберите фото из галереи');
        return;
      }
      _showSnack('Не удалось открыть галерею');
      return;
    }
    if (file == null) return; // пользователь отменил

    setState(() => _state = _ScanState.processing);
    try {
      final text = await _ocr.recognizeFile(file.path);
      final results = widget.repository.matcher.search(text, top: 5);
      if (!mounted) return;
      setState(() {
        _results = results;
        _state = _ScanState.results;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _state = _ScanState.idle);
      _showSnack('Не удалось распознать текст на этикетке');
    }
  }

  void _showSnack(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _openResult(MatchResult result) async {
    await widget.store.addHistory(result.wine.slug, result.confidence);
    if (!mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => WineDetailScreen(wine: result.wine, store: widget.store),
      ),
    );
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    switch (_state) {
      case _ScanState.idle:
        return _buildIdle(context);
      case _ScanState.processing:
        return _buildProcessing(context);
      case _ScanState.results:
        return _buildResults(context);
    }
  }

  Widget _buildIdle(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Сканер этикеток',
                style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Text(
              'Наведите камеру на этикетку российского вина — найдём его в каталоге',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium
                  ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
            ),
            const SizedBox(height: 32),
            // Большая круглая кнопка сканирования.
            InkWell(
              customBorder: const CircleBorder(),
              onTap: () => _scan(ImageSource.camera),
              child: Ink(
                width: 180,
                height: 180,
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [Color(0xFF9C2B4C), Color(0xFF5A1428)],
                  ),
                  boxShadow: [
                    BoxShadow(
                        color: Color(0x557B1E3A), blurRadius: 24, offset: Offset(0, 8)),
                  ],
                ),
                child: const Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.photo_camera, size: 56, color: Colors.white),
                    SizedBox(height: 8),
                    Text(
                      'Сканировать\nэтикетку',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            TextButton.icon(
              onPressed: () => _scan(ImageSource.gallery),
              icon: const Icon(Icons.photo_library_outlined),
              label: const Text('Из галереи'),
            ),
            if (_cameraFailed)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  'Камера недоступна на этом устройстве',
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: theme.colorScheme.error),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildProcessing(BuildContext context) {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 48,
            height: 48,
            child: CircularProgressIndicator(strokeWidth: 4),
          ),
          SizedBox(height: 20),
          Text('Читаю этикетку…', style: TextStyle(fontSize: 16)),
        ],
      ),
    );
  }

  Widget _buildResults(BuildContext context) {
    final theme = Theme.of(context);
    final best = _results.isNotEmpty ? _results.first : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
          child: _results.isEmpty
              ? Text('Ничего не нашлось',
                  style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700))
              : Row(
                  children: [
                    Expanded(
                      child: Text(
                        best!.confidence >= 0.22
                            ? 'Похоже, это:'
                            : 'Уверенности мало — проверьте варианты:',
                        style: theme.textTheme.titleLarge
                            ?.copyWith(fontWeight: FontWeight.w700),
                      ),
                    ),
                    confidenceBadge(best.confidence),
                  ],
                ),
        ),
        Expanded(
          child: _results.isEmpty
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Text(
                      'Попробуйте снять этикетку крупнее и при хорошем освещении,\nлибо поищите вино в каталоге',
                      textAlign: TextAlign.center,
                      style: theme.textTheme.bodyMedium
                          ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                    ),
                  ),
                )
              : ListView.builder(
                  itemCount: _results.length,
                  itemBuilder: (context, i) {
                    final r = _results[i];
                    return WineCard(
                      wine: r.wine,
                      onTap: () => _openResult(r),
                      subtitleExtra: confidenceBadge(r.confidence),
                    );
                  },
                ),
        ),
        Padding(
          padding: const EdgeInsets.all(16),
          child: FilledButton.icon(
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 14),
            ),
            onPressed: () {
              setState(() {
                _state = _ScanState.idle;
                _results = const [];
              });
              _scan(ImageSource.camera);
            },
            icon: const Icon(Icons.photo_camera),
            label: const Text('Сканировать ещё раз'),
          ),
        ),
      ],
    );
  }
}
