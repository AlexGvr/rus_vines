import 'package:flutter/material.dart';

import '../models/wine.dart';

/// Винная палитра.
const Color kWineColor = Color(0xFF7B1E3A);
const Color kGoldColor = Color(0xFFC9A227);

/// Плейсхолдер вместо фото (🍷 на бордовом фоне).
Widget winePhotoPlaceholder({double? width, double? height, double fontSize = 28}) {
  return Container(
    width: width,
    height: height,
    alignment: Alignment.center,
    decoration: BoxDecoration(
      gradient: const LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [Color(0xFF8E2B47), Color(0xFF5A1428)],
      ),
      borderRadius: BorderRadius.circular(10),
    ),
    child: Text('🍷', style: TextStyle(fontSize: fontSize)),
  );
}

/// Фото вина из ассетов с плейсхолдером на ошибку.
Widget winePhoto(Wine wine,
    {double? width, double? height, BoxFit fit = BoxFit.cover, double placeholderFontSize = 28}) {
  final photo = wine.photo;
  if (photo == null || photo.isEmpty) {
    return winePhotoPlaceholder(width: width, height: height, fontSize: placeholderFontSize);
  }
  return ClipRRect(
    borderRadius: BorderRadius.circular(10),
    child: Image.asset(
      'assets/$photo',
      width: width,
      height: height,
      fit: fit,
      errorBuilder: (context, error, stack) =>
          winePhotoPlaceholder(width: width, height: height, fontSize: placeholderFontSize),
    ),
  );
}

/// Бейдж процента уверенности: >=45% зелёный, >=22% жёлтый, иначе красный.
Widget confidenceBadge(double confidence) {
  final Color color;
  if (confidence >= 0.45) {
    color = const Color(0xFF2E7D32);
  } else if (confidence >= 0.22) {
    color = const Color(0xFFB8860B);
  } else {
    color = const Color(0xFFB3261E);
  }
  return Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
    decoration: BoxDecoration(
      color: color,
      borderRadius: BorderRadius.circular(999),
    ),
    child: Text(
      '${(confidence * 100).round()}%',
      style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w700),
    ),
  );
}

/// «N вин» с русской плюрализацией.
String pluralWines(int n) {
  final m10 = n % 10;
  final m100 = n % 100;
  if (m100 >= 11 && m100 <= 14) return '$n вин';
  if (m10 == 1) return '$n вино';
  if (m10 >= 2 && m10 <= 4) return '$n вина';
  return '$n вин';
}

/// Карточка вина для списков (каталог, сканы, «мои вина»).
class WineCard extends StatelessWidget {
  const WineCard({
    super.key,
    required this.wine,
    required this.onTap,
    this.trailing,
    this.subtitleExtra,
  });

  final Wine wine;
  final VoidCallback onTap;
  final Widget? trailing;
  final Widget? subtitleExtra;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              winePhoto(wine, width: 64, height: 84, placeholderFontSize: 26),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      wine.title,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '${wine.manufacturer} · ${wine.region}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.bodySmall
                          ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                    ),
                    const SizedBox(height: 6),
                    Wrap(
                      spacing: 8,
                      runSpacing: 4,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        if (wine.rating != null)
                          Text(
                            '★ ${wine.rating!.toStringAsFixed(1)}',
                            style: const TextStyle(
                                color: kGoldColor, fontWeight: FontWeight.w700, fontSize: 13),
                          ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.secondaryContainer,
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Text(
                            wine.category,
                            style: TextStyle(
                                fontSize: 11,
                                color: theme.colorScheme.onSecondaryContainer),
                          ),
                        ),
                        ?subtitleExtra,
                      ],
                    ),
                  ],
                ),
              ),
              ?trailing,
            ],
          ),
        ),
      ),
    );
  }
}
