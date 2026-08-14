# Своё Вино — офлайн-скан этикеток (MVP)

Сервис: фото этикетки → карточка вина из базы платформы «Своё Вино»
(vino-svoe.ru, 2017 вин). Две поставки: устанавливаемая офлайн-PWA
([app/](app/)) и Flutter-приложение ([mobile/](mobile/)); после первой
установки всё работает без сети, включая распознавание.

## Документация

| Документ | О чём |
|---|---|
| [docs/architecture.md](docs/architecture.md) | обзор системы, потоки данных, три зеркала ядра, роадмап |
| [docs/etl.md](docs/etl.md) | выгрузка базы, формат датапака, версионирование, юр. вопросы |
| [docs/recognition.md](docs/recognition.md) | нормализация, алгоритм матчера с формулами, OCR-движки, eval |
| [docs/pwa.md](docs/pwa.md) | PWA: скан-пайплайн, офлайн-архитектура SW, разработка |
| [docs/mobile.md](docs/mobile.md) | Flutter: структура, сборка, proguard-грабли ML Kit, паритет-тест |

## Структура

```
etl/        выгрузка и сборка датапака (серверный контур)
  scrape.py           sitemap → карточки JSON + фото (resume-safe)
  build_datapack.py   нормализация → wines.json + wines.sqlite + фото в app/
  update.sh           крон-обновление: докачка новых вин + пересборка
core/       ядро распознавания (Python-референс + eval)
  normalize.py        канонизация текста, транслитерация лат→кир
  matcher.py          fuzzy-матчер с IDF-весами (зеркало app/js/matcher.js)
  eval.mjs            метрики top-1/top-5: тот же OCR и матчер, что в PWA
app/        PWA (три таба: Скан / Каталог / Мои вина)
  js/normalize.js     зеркало core/normalize.py
  js/matcher.js       зеркало core/matcher.py
  js/scan.js          камера → Tesseract.js (rus+eng, офлайн) → матчер
  vendor/tesseract/   OCR-движок + языковые данные (~12 MB, в бандле)
  data/wines.json     датапак (генерируется build_datapack.py)
  photos/             фото бутылок 400px (~25 MB, генерируется)
  sw.js               service worker: полный офлайн после первой загрузки
mobile/     Flutter-приложение (Android/iOS), тот же продукт нативно
  lib/core/           Dart-зеркало ядра (normalize + matcher)
  lib/services/       датапак, «Мои вина» (shared_preferences), OCR (ML Kit)
  lib/screens/        Скан / Каталог / Детальная / Мои вина
  test/parity_test.dart  паритет Dart-ядра с JS по golden-файлу
tools/gen_golden.mjs   генератор golden-эталона (перегенерять при правке ядра)
data/       сырые карточки и собранный датапак (генерируется)
```

## Запуск

```bash
# 1. Выгрузка базы и сборка датапака (~20 мин, щадящий rate limit)
python3 etl/scrape.py
python3 etl/build_datapack.py

# 2. Метрики ядра (Node, тот же OCR-стек, что в приложении)
npm install
node core/eval.mjs 150

# 3. Приложение (PWA)
cd app && python3 -m http.server 8000
# открыть http://localhost:8000 (на телефоне — «Добавить на экран Домой»)

# 4. Flutter-приложение
cd mobile && flutter test && flutter build apk --release
# APK: mobile/build/app/outputs/flutter-apk/app-release.apk
```

Flutter-порт: ядро идентично JS (паритет-тест по 30 запросам с точностью
1e-4, эталон — tools/gen_golden.mjs). OCR — ML Kit (latin script): у ML Kit
нет кириллической модели, латинские надписи этикеток закрываются
транслитерацией в матчере; для кириллицы — Apple Vision (iOS), Tesseract FFI
или серверный fallback. Заметки/оценки/избранное — shared_preferences.

## Как работает распознавание (офлайн)

1. Фото → даунскейл до 1400px → Tesseract.js LSTM, rus+eng.
2. Токены канонизируются в кириллицу (ARISTOV → аристов) — этикетки
   часто на латинице, база на кириллице.
3. Матчер: вес поля (название 3.0 > винодельня 2.0 > сорта 1.0 >
   регион/категория 0.5) × IDF; совпадение точное / префиксное /
   Левенштейн ≤2. Год на этикетке даёт бонус.
4. confidence ≥ 0.22 — показываем лучший результат; ниже — топ-5
   «возможно, вы искали».

## Серверный контур

`etl/update.sh` по крону: докачивает новые вина, пересобирает датапак
(версия + hash в meta). Приложение при наличии сети сравнивает hash и
обновляет `data/wines.json`; фото докачиваются лениво через service worker.

## Ограничения и следующие шаги

- robots.txt платформы запрещает ботов в `/api/*` — для продакшна нужна
  договорённость об официальной выгрузке (для пилота — щадящий обход,
  честный User-Agent с контактом).
- Eval меряется на эталонных фото каталога — это верхняя граница; на
  «полевых» фото (блики, угол) точность ниже. Следующий шаг — полевой
  тестсет и визуальный путь (MobileCLIP kNN) как второй сигнал.
- Fallback-контур (спорные фото → серверная VLM Qwen2.5-VL при сети)
  спроектирован, в MVP не включён.
- Нативная оболочка (Flutter + ML Kit вместо Tesseract) — следующая
  итерация; ядро и датапак переносятся как есть.
