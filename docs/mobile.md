# Мобильное приложение (Flutter)

Каталог `mobile/` — Flutter-приложение «Своё вино»: полностью офлайновый сканер этикеток российских вин. Тот же продукт, что и PWA в `app/`, но с нативным OCR (ML Kit) вместо Tesseract.js. Проверено end-to-end на Android-эмуляторе; сборка — Flutter 3.44, release APK 97MB (universal).

## Структура lib/

~1700 строк Dart:

- `mobile/lib/core/` — зеркало поискового ядра, порт [app/js/](../app/js/) и [core/](../core/):
  - `normalize.dart` — канонизация в кириллицу (жадный диграфный транслит), `tokenize`, `translitToken`, `years`; `'İ'` заменяется на `'I'` до lowercase, как во всех зеркалах;
  - `matcher.dart` — `Matcher` и `MatchResult`: IDF-взвешенный скоринг по полям (title 3.0 / manufacturer 2.0 / grapes 1.0 / region+category 0.5), confidence, бонус года.
- `mobile/lib/models/wine.dart` — `Wine.fromJson`, все поля карточки датапака (slug, title, manufacturer, grapes, photo и т.д.), устойчив к null.
- `mobile/lib/services/`:
  - `wine_repository.dart` — загрузка `assets/wines.json` через `rootBundle`, парсинг JSON в изоляте (`compute`), индекс по slug, экземпляр `Matcher`, версия датапака из `meta`;
  - `ocr_service.dart` — OCR-слой (см. ниже);
  - `my_wines_store.dart` — избранное/история/заметки поверх `shared_preferences`.
- `mobile/lib/screens/` — `scan_screen.dart` (камера/галерея → OCR → топ-5 матчей), `catalog_screen.dart` (список + поиск тем же матчером), `wine_detail_screen.dart` (карточка, избранное, оценка/заметка), `my_wines_screen.dart` (избранное + история).
- `mobile/lib/widgets/wine_widgets.dart` — `WineCard`, `winePhoto` (ассет `assets/{photo}` с плейсхолдером на ошибку), `confidenceBadge` с теми же порогами, что в PWA: ≥0.45 зелёный, ≥0.22 жёлтый («Похоже, это:»), ниже — «Уверенности мало».
- `mobile/lib/main.dart` — `SvoeVinoApp` (Material 3, ru-локаль, светлая/тёмная тема), сплэш-загрузчик репозитория и store, `HomeShell` с тремя вкладками (Скан / Каталог / Мои вина) на `IndexedStack`.

## Зависимости

Из [mobile/pubspec.yaml](../mobile/pubspec.yaml):

| Пакет | Зачем |
|---|---|
| `google_mlkit_text_recognition ^0.16.0` | OCR on-device, latin script |
| `image_picker ^1.2.3` | камера и галерея (в `_scan`: maxWidth 1920, quality 90) |
| `shared_preferences ^2.5.5` | локальное хранение favs/history/notes |
| `flutter_localizations` | русская локализация Material-виджетов |
| `flutter_lints ^6.0.0` (dev) | стандартный линт, `flutter analyze` чистый |

## Ассеты

- `mobile/assets/wines.json` — датапак 2.1MB, 2017 вин, копия [data/datapack/wines.json](../data/datapack/wines.json) (та же схема `meta` + `wines`).
- `mobile/assets/photos/` — 2017 фото 400px (~10KB шт., 18MB всего), скачаны через resize-прокси `api.vino-svoe.ru`. Отсюда 97MB universal APK; `--split-per-abi` даёт примерно на треть меньше на устройство.

При обновлении датапака (`etl/update.sh`) ассеты нужно перекопировать и перегенерировать golden (см. паритет).

## OCR-слой

[mobile/lib/services/ocr_service.dart](../mobile/lib/services/ocr_service.dart): абстракция `OcrEngine` (`recognizeFile(path) → Future<String>`, `dispose()`) и единственная реализация `MlkitOcrEngine` — `TextRecognizer(script: TextRecognitionScript.latin)`.

**Ограничение:** у ML Kit Text Recognition v2 нет кириллической модели (только latin/chinese/devanagari/japanese/korean). Компенсация — транслит-нормализация ядра: бренды на этикетках почти всегда дублируются латиницей, `LAT2CYR` приводит их к кириллическим токенам каталога. Чисто кириллические названия этим движком не читаются.

**Прод-альтернативы** (за интерфейсом `OcrEngine`, без переписывания UI): Apple Vision на iOS (русский поддерживается), Tesseract через FFI, серверный fallback (Qwen2.5-VL — спроектирован, не реализован).

## Хранение

`MyWinesStore` — три ключа `shared_preferences`:

- `favs` — StringList slug'ов, новые в начало, toggle;
- `history` — JSON-строка, список `{slug, confidence, ts}` (ISO 8601), максимум 200 записей;
- `notes` — JSON-строка, map slug → `{rating, note}`; пустая заметка без оценки удаляет запись.

Битый JSON молча заменяется пустым состоянием (`_decodeHistory`/`_decodeNotes`).

## Сборка и запуск

```bash
cd mobile
flutter analyze
flutter test                       # включая паритет-тест
flutter build apk --release        # → build/app/outputs/flutter-apk/app-release.apk (~97MB)
flutter build apk --release --split-per-abi   # app-arm64-v8a-release.apk и др.
```

Release подписывается debug-ключом ([mobile/android/app/build.gradle.kts](../mobile/android/app/build.gradle.kts), TODO для прода). `applicationId`: `ru.rusvines.svoe_vino`.

## Proguard-грабли ML Kit

R8 в release-сборке ломает ML Kit двумя независимыми способами; лечение — [mobile/android/app/proguard-rules.pro](../mobile/android/app/proguard-rules.pro), подключённый в `buildTypes.release.proguardFiles`:

1. **Сборка падает** с missing classes: плагин `google_mlkit_text_recognition` ссылается на все скриптовые модели (chinese/devanagari/japanese/korean), а в APK подключена только latin. Симптом — R8 ошибки на `com.google.mlkit.vision.text.chinese.**` и т.п. Лечение:
   ```
   -dontwarn com.google.mlkit.vision.text.chinese.**
   -dontwarn com.google.mlkit.vision.text.devanagari.**
   -dontwarn com.google.mlkit.vision.text.japanese.**
   -dontwarn com.google.mlkit.vision.text.korean.**
   ```
2. **Рантайм NPE** при создании распознавателя (через MethodChannel, т.е. крэш только в release на устройстве, debug работает): R8 стрипает внутренности ML Kit. Лечение:
   ```
   -keep class com.google.mlkit.** { *; }
   -keep class com.google.android.odml.** { *; }
   -keep class com.google.android.gms.internal.mlkit_vision_text_common.** { *; }
   ```

Без обоих блоков release-APK либо не собирается, либо падает на первом скане.

## Паритет с JS-ядром

[mobile/test/parity_test.dart](../mobile/test/parity_test.dart) сверяет Dart-порт с эталоном [mobile/test/golden_parity.json](../mobile/test/golden_parity.json): транслит, токенизация, годы и топ-5 (slug + confidence, точность 1e-4) по 30 запросам.

Регенерация golden при любой правке ядра или датапака — генератор [tools/gen_golden.mjs](../tools/gen_golden.mjs) гоняет JS-ядро `app/js/` на `data/datapack/wines.json`:

```bash
node tools/gen_golden.mjs        # из корня репо
cd mobile && flutter test test/parity_test.dart
```

Править ядро нужно синхронно во всех трёх зеркалах (`core/`, `app/js/`, `mobile/lib/core/`).

## Эмулятор

```bash
flutter emulators --launch <avd>          # или emulator -avd <name>
flutter devices
flutter install                            # либо adb install -r build/app/outputs/flutter-apk/app-release.apk
```

Проверено end-to-end на эмуляторе: холодный старт со сплэшем, скан фото из галереи (камера эмулятора недоступна — экран показывает fallback «Камера недоступна» и предлагает галерею), топ-5 с бейджами уверенности, карточка вина, избранное/история/заметки переживают перезапуск.

## Отличия от PWA

- **OCR:** ML Kit latin вместо Tesseract.js rus+eng — нет кириллицы, но нет и 12MB vendor-бандла; распознавание нативное и быстрое.
- **Нет двухпроходного кропа** (полный кадр + кроп 18%/38%/64%×50% с grayscale/contrast в PWA): это компенсация слабой сегментации Tesseract; ML Kit сам детектит текстовые блоки на полном кадре, препроцессинг выигрыша не даёт.
- **Фото** в ассетах APK, а не cache-on-demand через Service Worker; датапак вкомпилирован, а не network-first.
- **Хранение** — `shared_preferences` вместо localStorage; ключи и семантика (favs/history/notes) совпадают.
- Пороги UI и ядро матчинга идентичны — гарантируется паритет-тестом.