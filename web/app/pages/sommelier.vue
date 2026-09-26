<script setup lang="ts">
// Сомелье без сканирования: у полки вопрос часто не «что это за вино»,
// а «что взять к ужину».
const apiBase = useApiBase()
const router = useRouter()

function back() {
  if (window.history.state?.back) router.back()
  else navigateTo('/')
}
</script>

<template>
  <main>
    <button class="back" @click="back">
      <span aria-hidden="true">‹</span> Назад
    </button>
    <SommelierPanel :api-base="apiBase" class="standalone" />
    <NuxtLink to="/" class="btn ghost again">К сканеру</NuxtLink>
  </main>
</template>

<style scoped>
.back {
  border: 0;
  background: none;
  padding: 4px 0;
  margin: -6px 0 4px;
  color: var(--wine-deep);
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
}
.back span { font-size: 22px; line-height: 1; vertical-align: -2px; margin-right: 2px; }
.standalone { border-top: 0; margin-top: 0; padding-top: 8px; }
.again {
  display: block;
  width: fit-content;
  margin: 28px auto 0;
  text-decoration: none;
}
</style>
