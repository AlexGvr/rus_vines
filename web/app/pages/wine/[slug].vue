<script setup lang="ts">
// Страница вина: полная карточка, гастропары и похожие вина других
// виноделен. Сюда ведут варианты и похожие вина с экрана результата;
// с похожих можно идти дальше по цепочке.
const apiBase = useApiBase()
const route = useRoute()
const router = useRouter()

type WineDetails = {
  wine: Record<string, any>
  pairings: string[]
  similar: { wine: Record<string, any>; reasons: string[] }[]
}

const { data, pending, error } = await useFetch<WineDetails>(
  () => `${apiBase}/v1/wines/${encodeURIComponent(String(route.params.slug))}`,
  { key: () => `wine-${route.params.slug}`, server: false },
)

function back() {
  // Назад по истории, если пришли из приложения; по прямой ссылке — на сканер.
  if (window.history.state?.back) router.back()
  else navigateTo('/')
}
</script>

<template>
  <main>
    <button class="back" @click="back">
      <span aria-hidden="true">‹</span> Назад
    </button>

    <p v-if="pending" class="muted state">Загружаем карточку…</p>
    <p v-else-if="error || !data" class="muted state">
      Карточка не найдена. Возможно, позиция убрана из каталога.
    </p>

    <template v-else>
      <WineCard :wine="data.wine" :api-base="apiBase" />

      <div v-if="data.pairings?.length" class="block">
        <h3>К чему подать</h3>
        <div class="chips">
          <span v-for="dish in data.pairings" :key="dish" class="chip">{{ dish }}</span>
        </div>
      </div>

      <div v-if="data.similar?.length" class="block">
        <h3>Похожие вина других виноделен</h3>
        <div class="list">
          <WineCard
            v-for="item in data.similar"
            :key="item.wine.slug"
            compact
            :wine="item.wine"
            :reasons="item.reasons"
            :api-base="apiBase"
            :to="`/wine/${item.wine.slug}`"
          />
        </div>
      </div>
    </template>

    <NuxtLink to="/" class="btn ghost again">К сканеру</NuxtLink>
  </main>
</template>

<style scoped>
.back {
  border: 0;
  background: none;
  padding: 4px 0;
  margin: -6px 0 12px;
  color: var(--wine-deep);
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
}
.back span { font-size: 22px; line-height: 1; vertical-align: -2px; margin-right: 2px; }
.state { text-align: center; margin-top: 10dvh; }
.block { margin-top: 26px; }
.block h3 { font-size: 18px; margin-bottom: 11px; }
.list { display: flex; flex-direction: column; gap: 9px; }
.again {
  display: block;
  width: fit-content;
  margin: 28px auto 0;
  text-decoration: none;
}
</style>
