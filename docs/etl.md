# ETL и датапак

> **Частично архив.** Обход платформы (`etl/scrape.py` → `data/raw/cards`,
> `data/datapack/wines.json`) по-прежнему нужен сборке каталога
> (`pipeline/build_catalog.py`: доливка полей, `--refresh-map`) и текстовому
> каналу (сладость из карточки). Доставка датапака в PWA и мобильное
> приложение относится к первому MVP; текущая архитектура —
> [ARCHITECTURE.md](../ARCHITECTURE.md).

Конвейер: выгрузка каталога «Своё Вино» → сборка датапака (JSON + SQLite + фото) → доставка в PWA/мобильное приложение. Три файла: [etl/scrape.py](../etl/scrape.py), [etl/build_datapack.py](../etl/build_datapack.py), [etl/update.sh](../etl/update.sh).

## Источник данных

Платформа vino-svoe.ru, два эндпоинта:

- `https://vino-svoe.ru/wines-sitemap.xml` — полный список slug'ов (парсится regex'ом по `<loc>https://vino-svoe.ru/wines/{slug}</loc>`, дедуп + сортировка). На момент выгрузки — 2017 вин, 0 ошибок.
- `https://vino-svoe.ru/api/wines/{slug}` — карточка вина, JSON.

Поля карточки (пример — [data/raw/cards/](../data/raw/cards/)):

| Поле | Тип | Пример |
|---|---|---|
| `slug` | string | `abrau-dyurso-abrau-durso-brut-rose-reserve-pino-nuar-beloe-bryut-12` |
| `title` | string | `Abrau-Durso Brut Rose Reserve` (часто латиница) |
| `manufacturer` | object `{name, slug}` | `{"name": "Абрау-Дюрсо", ...}` |
| `region` | object `{name, image}` | `{"name": "Кубань", ...}` |
| `category` | object `{name, backgroundGradient}` | `{"name": "Белое брют", ...}` |
| `color` | string | `Насыщенный розовый` (цвет напитка, не категория) |
| `alcohol` | number | `12` |
| `temperature` | string | `8-10` |
| `publicRating` | number | `5` |
| `description` | string | текст дегустационной заметки |
| `grapes` | array of `{name, image, backgroundImage}` | `[{"name": "Пино Нуар", ...}]` |
| `dishes` | array of `{name, image}` | `[{"name": "Сыры", ...}]` |
| `image` | object `{url, altText}` | `{"url": "/uploads/abrau_..._.webp"}` |

Любое из вложенных полей может отсутствовать/быть `null` — сборщик это учитывает.

## Фото-прокси и размеры

Фото отдаются через ресайз-прокси: `https://api.vino-svoe.ru/v1/img/str-api/{W}/{H}/resize{image.url}`. В скрейпере зашито `400/400` — этого хватает для карточки в UI (~10 KB на файл, ~18 MB на всю базу, webp). Ответы короче 500 байт отбрасываются как заглушки. Сырые фото складываются в `data/raw/photos/{slug}.webp`.

## Скрейпер: etl/scrape.py

- **Resume-safe**: карточка качается только если нет `data/raw/cards/{slug}.json`, фото — только если нет `data/raw/photos/{slug}.webp`. Повторный запуск докачивает недостающее и стоит почти ноль запросов.
- **Rate limit**: `WORKERS = 2` потока, `DELAY = 0.25` c на воркер после каждого запроса — щадящий режим для чужого API.
- **UA**: честный `RusVinesMVP/0.1 (educational prototype; contact: ...)` — не маскируемся под браузер.
- **Ретраи**: до 4 попыток с нарастающей паузой `1.5×(attempt+1)` c; HTTP 404 не ретраится. Скачанный JSON валидируется `json.loads` до записи на диск.
- Ошибки собираются в `data/raw/errors.log`, прогресс печатается каждые 100 карточек.

Запуск: `python3 etl/scrape.py` из любого каталога (пути строятся от расположения скрипта).

## Сборка: etl/build_datapack.py

Читает все `data/raw/cards/*.json` и нормализует:

- `manufacturer` и `region` — объекты; берётся только `.name` (`name_of()` терпит и строку, и `null`).
- `grapes`/`dishes` — массивы объектов → списки имён.
- `parse_category()` режет `category.name` на цвет и сладость: `'Белое брют' → ('Белое', 'Брют')` по префиксному словарю `COLORS = (белое, красное, оранжевое, розовое)`. Использует `clean()` из [core/normalize.py](../core/normalize.py) — та же канонизация, что и в матчере.
- `publicRating → rating`, `color → wineColor` (чтобы не путать с цветом категории), `image → photo` (относительный путь `photos/{slug}.webp`, `null` если фото нет).

Артефакты:

- `data/datapack/wines.json` — датапак для PWA (2.2 MB), копия кладётся в `app/data/wines.json`;
- `data/datapack/wines.sqlite` — SQLite (2.2 MB): таблица `wines` + виртуальная `wines_fts` (FTS5, `tokenize='unicode61'`, contentless) по title/manufacturer/region/grapes — для серверного/нативного варианта;
- `app/photos/{slug}.webp` — фото копируются в приложение (только недостающие).

Запуск: `python3 etl/build_datapack.py`. В конце печатает счётчики, размеры и `version`/`hash`.

## Формат wines.json

```json
{
  "meta": { "version": "2026-08-14", "count": 2017, "source": "vino-svoe.ru", "hash": "a1b2c3d4e5f6" },
  "wines": [ { ...карточка... } ]
}
```

Поля карточки в датапаке:

| Поле | Тип | Назначение |
|---|---|---|
| `slug` | string | первичный ключ |
| `title` | string | матчинг, вес 3.0 |
| `manufacturer` | string | матчинг, вес 2.0 |
| `region` | string | матчинг, вес 0.5 |
| `category` | string | исходная категория («Белое брют»), матчинг 0.5 |
| `color` / `sweetness` | string | разобранная категория, фильтры UI |
| `wineColor` | string | цвет напитка (описание) |
| `alcohol` | number\|null | крепость, % |
| `temperature` | string | температура подачи |
| `rating` | number\|null | публичный рейтинг, tie-break в матчере |
| `description` | string | карточка в UI |
| `grapes` | string[] | матчинг, вес 1.0 |
| `dishes` | string[] | карточка в UI |
| `gradient` | string | CSS-градиент фона категории |
| `photo` | string\|null | `photos/{slug}.webp` |

## Версионирование и доставка обновлений

- `meta.version` = дата сборки (ISO), `meta.hash` = первые 12 hex-символов sha256 от сериализованного payload (считается по blob без hash, затем вписывается и blob пересериализуется). В payload входит `meta.version` (дата сборки), поэтому hash меняется при каждой пересборке в новый день — даже без изменения контента; сравнение hash отвечает на вопрос «датапак другой?», а не «данные другие?».
- PWA: service worker держит `wines.json` в стратегии **network-first** — при наличии сети клиент получает свежий датапак сразу после пересборки, офлайн падает на кэш. Shell precache'ится (16 записей), фото — cache-on-demand, т.е. новые фото докачиваются по мере просмотра карточек.
- В Flutter-приложении датапак вкомпилирован в ассеты; обновление — новой сборкой. Сетевое сравнение hash + докачка — задел на следующую итерацию.

## Крон: etl/update.sh

```
python3 etl/scrape.py          # докачка новых вин (resume-safe)
python3 etl/build_datapack.py  # пересборка всех артефактов
```

Целевой режим — крон раз в сутки (сейчас запускается вручную; авто-крон включать только после договорённости с платформой — см. юридический раздел). Благодаря resume-safe скрейпу ежедневный прогон — это один запрос за sitemap плюс запросы только по новым slug'ам; полный цикл дорог лишь при первом запуске. `set -euo pipefail`: упавший скрейп не даст собрать битый датапак.

## Юридические ограничения

`robots.txt` vino-svoe.ru **запрещает `/api/*`**. Текущая выгрузка — пилот/прототип: щадящий rate limit (2 воркера × 0.25 c), честный UA с контактом, одноразовая полная выгрузка + инкрементальные догрузки. Для продакшна обязательна договорённость с платформой (официальный доступ к API или лицензия на данные); без неё ежедневный крон и распространение датапака в приложениях не запускать.
