<script setup lang="ts">
const apiBase = useApiBase()
const { preview, result, details } = useScan()

const fileInput = ref<HTMLInputElement>()
const galleryInput = ref<HTMLInputElement>()
const busy = ref(false)
const failed = ref(false)

async function scan(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return
  ;(event.target as HTMLInputElement).value = ''

  preview.value = URL.createObjectURL(file)
  busy.value = true
  failed.value = false
  result.value = undefined
  details.value = undefined

  const body = new FormData()
  body.append('image', file)
  try {
    result.value = await $fetch(`${apiBase}/v1/search`, { method: 'POST', body })
    const slug = result.value?.match?.slug
    if (slug) {
      details.value = await $fetch(`${apiBase}/v1/wines/${slug}`)
    }
  } catch {
    failed.value = true
  } finally {
    busy.value = false
  }
}

function reset() {
  preview.value = undefined
  result.value = undefined
  details.value = undefined
  failed.value = false
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
      <p v-if="failed" class="error">Сервис не ответил. Проверьте, что бэкенд запущен, и повторите.</p>

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
      <!-- Уверенный ответ: одна карточка, без экрана вариантов -->
      <template v-if="result.status === 'confident'">
        <p class="verdict found">Это вино</p>
        <WineCard :wine="result.match!" :api-base="apiBase" />
      </template>

      <!-- Спорный: карточку показываем, но честно говорим о сомнении -->
      <template v-else-if="result.status === 'uncertain'">
        <p class="verdict maybe">Скорее всего, это</p>
        <WineCard :wine="result.match!" :api-base="apiBase" />
        <p class="hint">
          Уверенности не хватает. Если это не оно — посмотрите близкие варианты ниже
          или переснимите этикетку крупнее.
        </p>
      </template>

      <!-- Слабое совпадение: не выдумываем ответ и не врём про каталог -->
      <template v-else>
        <p class="verdict missing">Не удалось уверенно определить вино</p>
        <p class="hint">
          Совпадение слишком слабое: этого вина может не быть в каталоге,
          а может не хватить кадра. Переснимите этикетку крупнее и ровнее —
          или посмотрите, что похоже по виду:
        </p>
      </template>

      <!-- Гастропары найденного вина -->
      <div v-if="details?.pairings?.length" class="block">
        <h3>К чему подать</h3>
        <div class="chips">
          <span v-for="dish in details.pairings" :key="dish" class="chip">{{ dish }}</span>
        </div>
      </div>

      <!-- Похожие вина других виноделен -->
      <div v-if="details?.similar?.length" class="block">
        <h3>Похожие вина других виноделен</h3>
        <div class="list">
          <WineCard
            v-for="item in details.similar"
            :key="item.wine.slug"
            compact
            :wine="item.wine"
            :reasons="item.reasons"
            :api-base="apiBase"
            :to="`/wine/${item.wine.slug}`"
          />
        </div>
      </div>

      <!-- Кандидаты: только когда ответ неточный -->
      <div v-if="result.status !== 'confident' && result.alternatives.length" class="block">
        <h3>{{ result.status === 'unsure' ? 'Ближайшее по виду' : 'Другие варианты' }}</h3>
        <div class="list">
          <WineCard
            v-for="item in result.alternatives"
            :key="item.slug"
            compact
            :wine="item"
            :api-base="apiBase"
            :to="`/wine/${item.slug}`"
          />
        </div>
      </div>

      <SommelierPanel :api-base="apiBase" />

      <p class="meta muted">
        уверенность в карточке {{ Math.round(result.confidence.probability * 100) }}% ·
        в пятёрке {{ Math.round(result.confidence.probability_top5 * 100) }}% ·
        совпавших точек {{ result.confidence.inliers }} ·
        {{ Math.round(result.latency_ms.total) }} мс
      </p>
      <p v-if="result.quality" class="meta muted">
        F1 конвейера на валидационном наборе:
        топ-1 {{ (result.quality.f1_top1 * 100).toFixed(1) }}% ·
        топ-5 {{ (result.quality.f1_top5 * 100).toFixed(1) }}%
      </p>
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

.working { text-align: center; padding-top: 8dvh; }
.working img {
  max-width: 62%;
  max-height: 34dvh;
  border-radius: var(--radius);
  object-fit: cover;
}
.working p { color: var(--muted); margin-top: 18px; }

.verdict {
  font-family: var(--display);
  font-size: 15px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin: 0 0 10px;
}
.verdict.found { color: var(--wine-deep); }
.verdict.maybe { color: #9a7a2e; }
.verdict.missing { color: var(--muted); }
.hint { font-size: 14px; color: var(--muted); margin: 12px 0 0; }

.block { margin-top: 26px; }
.block h3 { font-size: 18px; margin-bottom: 11px; }
.list { display: flex; flex-direction: column; gap: 9px; }
.meta { font-size: 12px; margin-top: 22px; text-align: center; }
.again { display: block; margin: 10px auto 0; }
</style>
