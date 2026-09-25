<script setup lang="ts">
const props = defineProps<{
  wine: Record<string, any>
  apiBase: string
  compact?: boolean
  reasons?: string[]
  // Куда ведёт карточка: страница вина. Без него карточка не ссылка.
  to?: string
}>()

const NuxtLink = resolveComponent('NuxtLink')
const zoomed = ref(false)
const missing = ref(false)
const photo = computed(() => `${props.apiBase}/v1/photo/${props.wine.slug}`)

const facts = computed(() => [
  props.wine.category,
  props.wine.alcohol ? `${props.wine.alcohol}% алк.` : null,
  props.wine.temperature ? `подача ${props.wine.temperature}` : null,
  ...(props.wine.grapes || []),
].filter(Boolean))
</script>

<template>
  <article :class="['card', { compact, link: !!to }]">
    <button
      type="button"
      class="photo"
      :disabled="missing"
      :aria-label="`Увеличить фото: ${wine.title}`"
      @click="zoomed = true"
    >
      <img
        :src="photo"
        :alt="wine.title"
        loading="lazy"
        @error="missing = true"
      >
      <span v-if="!compact && !missing" class="zoom-hint" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="6.5" /><path d="M16 16l4.5 4.5M11 8v6M8 11h6" />
        </svg>
      </span>
    </button>
    <component :is="to ? NuxtLink : 'div'" :to="to" class="go">
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
      <span v-if="to" class="chevron" aria-hidden="true">›</span>
    </component>
    <PhotoZoom v-model="zoomed" :src="photo" :alt="wine.title" />
  </article>
</template>

<style scoped>
.card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  overflow: hidden;
}
.card.link { transition: border-color .15s, box-shadow .15s; }
.card.link:hover { border-color: var(--wine); box-shadow: 0 4px 14px rgba(143, 61, 66, .1); }

.photo {
  position: relative;
  display: block;
  border: 0;
  padding: 0;
  background: none;
  cursor: zoom-in;
}
.photo:disabled { cursor: default; visibility: hidden; }
.photo img { display: block; }

.card:not(.compact) { text-align: center; }
.card:not(.compact) .photo { width: 100%; }
.card:not(.compact) .photo img {
  width: 100%;
  max-height: 340px;
  object-fit: contain;
  padding: 22px 0 8px;
  background: linear-gradient(180deg, #fbf7f4 0%, #fff 100%);
}
.zoom-hint {
  position: absolute;
  right: 12px;
  bottom: 10px;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, .92);
  border: 1px solid var(--line);
  color: var(--wine-deep);
}
.card:not(.compact) .body { padding: 6px 18px 20px; }
.card:not(.compact) .facts { justify-content: center; margin: 14px 0; }

.card.compact {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 10px;
}
.card.compact .photo { flex: none; }
.card.compact .photo img {
  width: 58px;
  height: 78px;
  object-fit: contain;
}
.go { color: inherit; text-decoration: none; }
.card.compact .go {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  align-self: stretch;
}
.card.compact .body { min-width: 0; flex: 1; text-align: left; }
.chevron {
  flex: none;
  font-size: 26px;
  line-height: 1;
  color: var(--wine);
  padding: 0 4px 0 2px;
}

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
