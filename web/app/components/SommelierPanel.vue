<script setup lang="ts">
const props = defineProps<{ apiBase: string }>()

type Option = { value: string; label: string }
type Question = { id: string; title: string; options: Option[] }

const questions = ref<Question[]>([])
const answers = reactive<Record<string, string>>({})
const picks = ref<any[]>([])
const relaxed = ref(false)
const loading = ref(false)
const asked = ref(false)

onMounted(async () => {
  try {
    const data = await $fetch<{ questions: Question[] }>(
      `${props.apiBase}/v1/sommelier/questions`)
    questions.value = data.questions
  } catch {
    questions.value = []
  }
})

const ready = computed(() =>
  questions.value.length > 0 &&
  questions.value.every((q) => answers[q.id] !== undefined))

async function recommend() {
  loading.value = true
  asked.value = true
  try {
    const data = await $fetch<{ recommendations: any[]; relaxed: boolean }>(
      `${props.apiBase}/v1/sommelier`, { method: 'POST', body: { ...answers } })
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
    <p class="lead muted">Три вопроса — и подберём вино из каталога под ваш случай.</p>

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
      <p v-if="relaxed" class="note">
        Под все условия сразу ничего не нашлось — показываем близкое по вкусу.
      </p>
      <p v-if="!picks.length" class="note">
        Ничего не подобралось. Попробуйте смягчить условия — например, «не знаю» во вкусе.
      </p>
      <WineCard
        v-for="pick in picks"
        :key="pick.wine.slug"
        compact
        :wine="pick.wine"
        :reasons="pick.reasons"
        :api-base="apiBase"
      />
    </div>
  </section>
</template>

<style scoped>
.sommelier {
  margin-top: 28px;
  padding-top: 22px;
  border-top: 1px solid var(--line);
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
.note { font-size: 13.5px; color: var(--muted); margin: 0 0 6px; }
</style>
