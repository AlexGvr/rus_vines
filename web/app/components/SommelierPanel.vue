<script setup lang="ts">
// current — найденное вино. Сомелье тогда начинает с его вкуса и после
// подбора говорит, подходит ли оно к выбранному блюду: вопрос у полки
// обычно «к ужину это подойдёт?», а не «что вообще взять».
const props = defineProps<{
  apiBase: string
  current?: { slug: string; title: string; dishes?: string[]; sweetness?: string | null }
}>()

type Option = { value: string; label: string }
type Question = { id: string; title: string; options: Option[] }

const questions = ref<Question[]>([])
// Ответы и подборка переживают переход на страницу вина и возврат назад.
const answers = useState<Record<string, string>>('sommelier-answers', () => ({}))
const picks = useState<any[]>('sommelier-picks', () => [])
const relaxed = useState('sommelier-relaxed', () => false)
const asked = useState('sommelier-asked', () => false)
// Для какого вина заполнены ответы: новое сканирование начинает заново.
const answeredFor = useState<string | null>('sommelier-for', () => null)
const loading = ref(false)

onMounted(async () => {
  try {
    const data = await $fetch<{ questions: Question[] }>(
      `${props.apiBase}/v1/sommelier/questions`)
    questions.value = data.questions
  } catch {
    questions.value = []
  }
})

watch(() => props.current?.slug ?? null, (slug) => {
  if (slug === answeredFor.value) return
  answeredFor.value = slug
  answers.value = {}
  picks.value = []
  asked.value = false
  const taste = props.current?.sweetness
  if (taste && ['сухое', 'полусухое', 'полусладкое', 'брют'].includes(taste)) {
    answers.value = { taste }
  }
}, { immediate: true })

const ready = computed(() =>
  questions.value.length > 0 &&
  questions.value.every((q) => answers.value[q.id] !== undefined))

// Подходит ли найденное вино к выбранному блюду — по карточке платформы.
const verdict = computed(() => {
  const dish = answers.value.dish
  if (!props.current || !dish || !asked.value) return null
  const name = `«${cleanTitle(props.current.title)}»`
  return (props.current.dishes || []).includes(dish)
    ? `${name} подходит к блюду «${dish.toLowerCase()}» — можно брать. Ещё варианты:`
    : `${name} платформа к блюду «${dish.toLowerCase()}» не советует. Что подойдёт лучше:`
})
const shown = computed(() => picks.value.filter((p) => p.wine.slug !== props.current?.slug))

async function recommend() {
  loading.value = true
  asked.value = true
  try {
    const data = await $fetch<{ recommendations: any[]; relaxed: boolean }>(
      `${props.apiBase}/v1/sommelier`,
      { method: 'POST', body: { ...answers.value, like: props.current?.slug } })
    picks.value = data.recommendations
    relaxed.value = data.relaxed
  } catch {
    picks.value = []
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="sommelier">
    <h3>Цифровой сомелье</h3>
    <p class="lead muted">
      <template v-if="current">Три вопроса — подскажем, подойдёт ли это вино, и что взять ещё.</template>
      <template v-else>Три вопроса — и подберём вино из каталога под ваш случай.</template>
    </p>

    <div v-for="question in questions" :key="question.id" class="question">
      <p class="q-title">{{ question.title }}</p>
      <div class="chips">
        <button
          v-for="option in question.options"
          :key="option.label"
          :class="['chip', 'pick', { on: answers[question.id] === option.value }]"
          @click="answers[question.id] = option.value"
        >{{ option.label }}</button>
      </div>
    </div>

    <button class="btn" :disabled="!ready || loading" @click="recommend">
      {{ loading ? 'Подбираем…' : 'Подобрать вино' }}
    </button>

    <div v-if="asked && !loading" class="results">
      <p v-if="verdict" class="verdict">{{ verdict }}</p>
      <p v-if="relaxed" class="note">
        Под все условия сразу ничего не нашлось — показываем близкое по вкусу.
      </p>
      <p v-if="!shown.length" class="note">
        Ничего не подобралось. Попробуйте смягчить условия — например, «не знаю» во вкусе.
      </p>
      <WineCard
        v-for="pick in shown"
        :key="pick.wine.slug"
        compact
        :wine="pick.wine"
        :reasons="pick.reasons"
        :api-base="apiBase"
        :to="`/wine/${pick.wine.slug}`"
      />
    </div>
  </section>
</template>

<style scoped>
.sommelier {
  margin-top: 28px;
  padding-top: 22px;
  border-top: 1px solid var(--line);
  scroll-margin-top: 12px;
}
h3 { font-size: 20px; }
.lead { font-size: 14px; margin: 4px 0 16px; }
.question { margin-bottom: 14px; }
.q-title { font-size: 14px; font-weight: 600; margin: 0 0 7px; }
.pick { cursor: pointer; transition: all .12s; }
.pick.on {
  background: var(--wine);
  border-color: var(--wine);
  color: #fff;
}
.btn:disabled { opacity: .45; cursor: default; }
.results { display: flex; flex-direction: column; gap: 9px; margin-top: 18px; }
.verdict { font-size: 14.5px; font-weight: 600; color: var(--wine-deep); margin: 0 0 4px; }
.note { font-size: 13.5px; color: var(--muted); margin: 0 0 6px; }
</style>
