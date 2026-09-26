<script setup lang="ts">
const apiBase = useApiBase()
const route = useRoute()
const { preview, file, result, details, analogs } = useScan()
const history = useHistory()

const fileInput = ref<HTMLInputElement>()
const galleryInput = ref<HTMLInputElement>()
const busy = ref(false)
const failed = ref<string | null>(null)
const shotZoomed = ref(false)

// Технические числа (уверенность, инлаеры, время, F1) пользователю не
// нужны: ТЗ просит их в API, а не в интерфейсе. Для демо — ?debug.
const debug = computed(() => route.query.debug !== undefined)

const LIST_KEYS = ['alternatives', 'maker', 'analogs']

async function scan(event: Event) {
  const picked = (event.target as HTMLInputElement).files?.[0]
  if (!picked) return
  ;(event.target as HTMLInputElement).value = ''

  preview.value = URL.createObjectURL(picked)
  file.value = picked
  busy.value = true
  failed.value = null
  result.value = undefined
  details.value = undefined
  analogs.value = undefined
  clearNuxtState(LIST_KEYS.map((key) => `wine-list-${key}`))

  const body = new FormData()
  body.append('image', picked)
  try {
    result.value = await $fetch(`${apiBase}/v1/search`, { method: 'POST', body })
    const match = result.value?.match
    if (match) {
      history.add(match)
      details.value = await $fetch(`${apiBase}/v1/wines/${match.slug}`)
    } else {
      loadAnalogs(picked)
    }
  } catch (error: any) {
    failed.value = error?.statusCode === 400
      ? 'Не получилось открыть это фото. Попробуйте другое.'
      : 'Не получилось связаться с сервисом. Попробуйте ещё раз.'
  } finally {
    busy.value = false
  }
}

// Вино не найдено: отдельным запросом читаем стиль с этикетки и берём
// аналоги из каталога. Ответ поиска к этому моменту уже на экране.
async function loadAnalogs(picked: File) {
  analogs.value = 'loading'
  const body = new FormData()
  body.append('image', picked)
  try {
    const data = await $fetch<LabelAnalogs>(`${apiBase}/v1/analogs`, { method: 'POST', body })
    if (file.value === picked) analogs.value = data
  } catch {
    if (file.value === picked) analogs.value = 'failed'
  }
}

function reset() {
  preview.value = undefined
  file.value = undefined
  result.value = undefined
  details.value = undefined
  analogs.value = undefined
  failed.value = null
}

const alternatives = computed<WineItem[]>(() =>
  (result.value?.alternatives || []).map((wine) => ({ wine })))

const pairingText = computed(() =>
  splitDescription(result.value?.match?.description).pairing)

const label = computed(() =>
  typeof analogs.value === 'object' ? analogs.value : undefined)

// «По этикетке: «Пино Нуар», Табия · красное · полусухое»
const labelSummary = computed(() => {
  const read = label.value?.label
  if (!read) return null
  const name = [read.name && `«${read.name}»`, read.producer_ru || read.producer]
    .filter(Boolean).join(', ')
  const style = [read.color, read.sparkling && !['брют', 'экстра брют'].includes(read.sweetness || '')
    ? 'игристое' : null, read.sweetness, ...(read.grapes || [])].filter(Boolean).join(' · ')
  return { name, style }
})

const sommelierWine = computed(() => {
  const match = result.value?.match
  if (!match) return undefined
  return {
    slug: match.slug,
    title: match.title,
    dishes: details.value?.pairings,
    sweetness: details.value?.style?.sweetness,
  }
})

function toSommelier() {
  document.getElementById('sommelier')?.scrollIntoView({ behavior: 'smooth' })
}
</script>

<template>
  <main>
    <!-- Экран съёмки -->
    <section v-if="!result && !busy" class="scan">
      <h1>Что это за вино?</h1>
      <p class="lead muted">
        Наведите камеру на этикетку — найдём карточку в каталоге «Своё Вино».
      </p>
      <button class="shutter" @click="fileInput?.click()">
        <svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2" />
          <circle cx="12" cy="12" r="3.4" />
        </svg>
        <span>Сканировать<br>этикетку</span>
      </button>
      <button class="btn ghost gallery" @click="galleryInput?.click()">Выбрать из галереи</button>
      <p v-if="failed" class="error">{{ failed }}</p>

      <NuxtLink to="/sommelier" class="ask">
        Не знаете, что взять? <strong>Спросите сомелье ›</strong>
      </NuxtLink>

      <div v-if="history.items.value.length" class="recent">
        <h3>Вы недавно сканировали</h3>
        <WineList
          :items="history.items.value.map((wine) => ({ wine }))"
          :api-base="apiBase"
          :initial="3"
        />
      </div>

      <input ref="fileInput" type="file" accept="image/*" capture="environment" hidden @change="scan">
      <input ref="galleryInput" type="file" accept="image/*" hidden @change="scan">
    </section>

    <!-- Поиск -->
    <section v-if="busy" class="working">
      <img v-if="preview" :src="preview" alt="">
      <p>Ищем в каталоге…</p>
    </section>

    <!-- Результат -->
    <section v-if="result && !busy" class="result">
      <div class="head">
        <p :class="['verdict', result.status]">
          <template v-if="result.status === 'confident'">Это вино</template>
          <template v-else-if="result.status === 'uncertain'">Скорее всего, это</template>
          <template v-else>Не удалось уверенно определить вино</template>
        </p>
        <!-- Свой снимок рядом с карточкой: сверить этикетку глазами -->
        <button v-if="preview" type="button" class="shot" @click="shotZoomed = true">
          <img :src="preview" alt="Ваш снимок">
          <span>ваш снимок</span>
        </button>
        <PhotoZoom v-if="preview" v-model="shotZoomed" :src="preview" alt="Ваш снимок" />
      </div>

      <!-- Найдено: одна карточка -->
      <template v-if="result.match">
        <WineCard :wine="result.match" :style-info="details?.style" :api-base="apiBase" />

        <!-- Спорный ответ: варианты свёрнуты сразу под карточкой -->
        <div v-if="result.status === 'uncertain' && alternatives.length" class="block not-it">
          <h3>Не оно?</h3>
          <p class="hint">
            Сверьте этикетку со своим снимком. Если вино другое — поищите его среди
            похожих ниже или переснимите этикетку крупнее.
          </p>
          <WineList :items="alternatives" :api-base="apiBase" state-key="alternatives" />
        </div>

        <button type="button" class="btn ghost ask-inline" @click="toSommelier">
          Спросить сомелье
        </button>

        <Pairings :dishes="details?.pairings" :text="pairingText" />

        <div v-if="details?.similar?.length" class="block">
          <h3>Похожие вина других виноделен</h3>
          <WineList :items="details.similar" :api-base="apiBase" :initial="6" />
        </div>
      </template>

      <!-- Не найдено: честно, с тем, что есть в каталоге похожего -->
      <template v-else>
        <p class="hint lead-hint">
          Этого вина может не быть в каталоге, а может не хватить кадра —
          тогда переснимите этикетку крупнее и ровнее.
        </p>

        <p v-if="analogs === 'loading'" class="reading">
          <span class="dot" aria-hidden="true" /> Читаем этикетку и подбираем похожие…
        </p>
        <template v-else-if="label">
          <p v-if="labelSummary && (labelSummary.name || labelSummary.style)" class="label-read">
            По этикетке: <strong>{{ labelSummary.name }}</strong>
            <span v-if="labelSummary.style">{{ labelSummary.name ? ' · ' : '' }}{{ labelSummary.style }}</span>
          </p>
          <div v-if="label.maker_wines.length" class="block">
            <h3>Вина «{{ label.maker }}» в каталоге</h3>
            <WineList :items="label.maker_wines" :api-base="apiBase" state-key="maker" />
          </div>
          <div v-if="label.analogs.length" class="block">
            <h3>Похожие вина других виноделен</h3>
            <WineList :items="label.analogs" :api-base="apiBase" state-key="analogs" />
          </div>
        </template>

        <div v-if="alternatives.length" class="block">
          <h3>Похожее по виду</h3>
          <WineList :items="alternatives" :api-base="apiBase" state-key="alternatives" />
        </div>
      </template>

      <SommelierPanel id="sommelier" :api-base="apiBase" :current="sommelierWine" />

      <template v-if="debug">
        <p class="meta muted">
          уверенность в карточке {{ Math.round(result.confidence.probability * 100) }}% ·
          в пятёрке {{ Math.round(result.confidence.probability_top5 * 100) }}% ·
          совпавших точек {{ result.confidence.inliers }} ·
          {{ Math.round(result.latency_ms.total) }} мс
        </p>
        <p v-if="result.quality" class="meta muted">
          F1 конвейера на синтетическом наборе (запросы из фото каталога):
          топ-1 {{ (result.quality.f1_top1 * 100).toFixed(1) }}% ·
          топ-5 {{ (result.quality.f1_top5 * 100).toFixed(1) }}%
        </p>
      </template>
      <button class="btn ghost again" @click="reset">Сканировать ещё раз</button>
    </section>
  </main>
</template>

<style scoped>
.scan { text-align: center; padding-top: 6dvh; }
h1 { font-size: 27px; line-height: 1.15; }
.lead { font-size: 15px; margin: 8px auto 30px; max-width: 30ch; }
.shutter {
  width: 188px;
  height: 188px;
  border-radius: 50%;
  border: 0;
  background: var(--wine);
  color: #fff;
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  font-size: 16px;
  font-weight: 600;
  line-height: 1.25;
  cursor: pointer;
  box-shadow: 0 10px 26px rgba(143, 61, 66, .28);
  transition: transform .12s, background .15s;
}
.shutter:hover { background: var(--wine-deep); }
.shutter:active { transform: scale(.96); }
.gallery { display: block; margin: 22px auto 0; }
.error { color: var(--wine-darkest); font-size: 14px; margin-top: 18px; }
.ask {
  display: inline-block;
  margin-top: 26px;
  font-size: 14.5px;
  color: var(--muted);
  text-decoration: none;
}
.ask strong { color: var(--wine-deep); font-weight: 600; }
.ask:hover strong { text-decoration: underline; }
.recent { margin-top: 34px; text-align: left; }
.recent h3 { font-size: 18px; margin-bottom: 11px; }

.working { text-align: center; padding-top: 8dvh; }
.working img {
  max-width: 62%;
  max-height: 34dvh;
  border-radius: var(--radius);
  object-fit: cover;
}
.working p { color: var(--muted); margin-top: 18px; }

.head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.verdict {
  font-family: var(--display);
  font-size: 15px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin: 0;
}
.verdict.confident { color: var(--wine-deep); }
.verdict.uncertain { color: #9a7a2e; }
.verdict.unsure { color: var(--muted); }
.shot {
  flex: none;
  border: 0;
  padding: 0;
  background: none;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  cursor: zoom-in;
}
.shot img {
  width: 46px;
  height: 58px;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid var(--line);
}
.shot span { font-size: 11px; color: var(--muted); }
.hint { font-size: 14px; color: var(--muted); margin: 0 0 12px; }
.lead-hint { margin-top: 2px; }

.block { margin-top: 26px; }
.block h3 { font-size: 18px; margin-bottom: 11px; }
.not-it { margin-top: 20px; }
.not-it h3 { margin-bottom: 4px; }
.ask-inline { display: block; margin: 18px auto 0; }

.reading {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  color: var(--muted);
  margin: 18px 0 0;
}
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--wine);
  animation: pulse 1s ease-in-out infinite alternate;
}
@keyframes pulse { from { opacity: .25; } to { opacity: 1; } }
.label-read { font-size: 14.5px; margin: 16px 0 0; color: var(--ink); }
.label-read span { color: var(--muted); }

.meta { font-size: 12px; margin-top: 22px; text-align: center; }
.again { display: block; margin: 10px auto 0; }
</style>
