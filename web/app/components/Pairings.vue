<script setup lang="ts">
// «К чему подать»: блюда из карточки и абзац «Сочетания» из описания.
defineProps<{ dishes?: string[]; text?: string }>()
</script>

<template>
  <div v-if="dishes?.length || text" class="block">
    <h3>К чему подать</h3>
    <div v-if="dishes?.length" class="chips">
      <span v-for="dish in dishes" :key="dish" class="chip">{{ dish }}</span>
    </div>
    <p v-if="text" class="text">{{ text }}</p>
  </div>
</template>

<style scoped>
.block { margin-top: 26px; }
h3 { font-size: 18px; margin-bottom: 11px; }
.text { margin: 10px 0 0; font-size: 14px; line-height: 1.55; color: #4a4441; }
</style>
