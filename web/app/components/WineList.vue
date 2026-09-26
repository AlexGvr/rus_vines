<script setup lang="ts">
// Список компактных карточек: сначала несколько, остальные — по кнопке.
// Длинный список вариантов под карточкой выглядит как экран выбора,
// которого ТЗ просит избегать; при этом ни один вариант не теряется.
// stateKey сохраняет раскрытие при уходе на страницу вина и возврате.
const props = withDefaults(defineProps<{
  items: WineItem[]
  apiBase: string
  initial?: number
  stateKey?: string
}>(), { initial: 3, stateKey: undefined })

const expanded = props.stateKey
  ? useState(`wine-list-${props.stateKey}`, () => false)
  : ref(false)
const shown = computed(() =>
  expanded.value ? props.items : props.items.slice(0, props.initial))
</script>

<template>
  <div class="list">
    <WineCard
      v-for="item in shown"
      :key="item.wine.slug"
      compact
      :wine="item.wine"
      :reasons="item.reasons"
      :api-base="apiBase"
      :to="`/wine/${item.wine.slug}`"
    />
    <button
      v-if="!expanded && items.length > initial"
      type="button"
      class="more"
      @click="expanded = true"
    >
      Показать ещё {{ items.length - initial }}
    </button>
  </div>
</template>

<style scoped>
.list { display: flex; flex-direction: column; gap: 9px; }
.more {
  align-self: flex-start;
  border: 0;
  background: none;
  padding: 4px 2px;
  color: var(--wine-deep);
  font-size: 14.5px;
  font-weight: 600;
  cursor: pointer;
}
.more:hover { text-decoration: underline; }
</style>
