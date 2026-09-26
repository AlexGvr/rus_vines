<script setup lang="ts">
// Страница вина: полная карточка, гастропары и похожие вина других
// виноделен. Сюда ведут варианты и похожие вина с экрана результата;
// с похожих можно идти дальше по цепочке.
const apiBase = useApiBase()
const route = useRoute()
const router = useRouter()

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
      <WineCard :wine="data.wine" :style-info="data.style" :api-base="apiBase" />

      <Pairings :dishes="data.pairings" :text="splitDescription(data.wine.description).pairing" />

      <div v-if="data.similar?.length" class="block">
        <h3>Похожие вина других виноделен</h3>
        <WineList :items="data.similar" :api-base="apiBase" :initial="6" />
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
.again {
  display: block;
  width: fit-content;
  margin: 28px auto 0;
  text-decoration: none;
}
</style>
