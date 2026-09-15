<script setup lang="ts">
const props = defineProps<{
  wine: Record<string, any>
  apiBase: string
  compact?: boolean
  reasons?: string[]
}>()

const facts = computed(() => [
  props.wine.category,
  props.wine.alcohol ? `${props.wine.alcohol}% алк.` : null,
  props.wine.temperature ? `подача ${props.wine.temperature}` : null,
  ...(props.wine.grapes || []),
].filter(Boolean))
</script>

<template>
  <article :class="['card', { compact }]">
    <img
      :src="`${apiBase}/v1/photo/${wine.slug}`"
      :alt="wine.title"
      loading="lazy"
      @error="($event.target as HTMLImageElement).style.visibility = 'hidden'"
    >
    <div class="body">
      <h2 class="title">{{ wine.title }}</h2>
      <p class="origin">{{ [wine.manufacturer, wine.region].filter(Boolean).join(' · ') }}</p>

      <p v-if="wine.rating" class="rating">★ {{ wine.rating }} <span class="muted">рейтинг платформы</span></p>

      <p v-if="reasons?.length" class="reasons">{{ reasons.join(' · ') }}</p>

      <div v-if="!compact" class="chips facts">
        <span v-for="fact in facts" :key="fact" class="chip">{{ fact }}</span>
      </div>
      <p v-if="!compact && wine.description" class="description">{{ wine.description }}</p>
    </div>
  </article>
</template>

<style scoped>
.card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  overflow: hidden;
}
.card:not(.compact) { text-align: center; }
.card:not(.compact) img {
  width: 100%;
  max-height: 340px;
  object-fit: contain;
  padding: 22px 0 8px;
  background: linear-gradient(180deg, #fbf7f4 0%, #fff 100%);
}
.card:not(.compact) .body { padding: 6px 18px 20px; }
.card:not(.compact) .facts { justify-content: center; margin: 14px 0; }

.card.compact {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 10px;
}
.card.compact img {
  width: 58px;
  height: 78px;
  object-fit: contain;
  flex: none;
}
.card.compact .body { min-width: 0; text-align: left; }

.title { font-size: 21px; line-height: 1.2; }
.card.compact .title { font-size: 15.5px; font-family: var(--sans); font-weight: 600; }
.origin { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
.card.compact .origin { font-size: 12.5px; }
.rating { margin: 8px 0 0; color: var(--wine-deep); font-weight: 600; font-size: 14.5px; }
.card.compact .rating { font-size: 13px; margin-top: 3px; }
.rating .muted { font-weight: 400; font-size: 13px; }
.reasons { margin: 5px 0 0; font-size: 12.5px; color: var(--wine-deep); }
.description {
  margin: 12px 0 0;
  font-size: 14.5px;
  line-height: 1.6;
  text-align: left;
  color: #4a4441;
  white-space: pre-line;
}
</style>
