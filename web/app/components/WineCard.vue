<script setup lang="ts">
const props = defineProps<{
  wine: Record<string, any>
  apiBase: string
  compact?: boolean
  reasons?: string[]
  // Куда ведёт карточка: страница вина. Без него карточка не ссылка.
  to?: string
  // Сладость и игристость из /v1/wines: в выгрузке отдельных полей нет.
  styleInfo?: WineStyle
}>()

const NuxtLink = resolveComponent('NuxtLink')
const zoomed = ref(false)
const failed = ref(false)
watch(() => props.wine.slug, () => { failed.value = false })
// У дюжины позиций в дампе нет фото: вместо пустого места — силуэт бутылки.
const missing = computed(() => !props.wine.photo || failed.value)
const photo = computed(() => `${props.apiBase}/v1/photo/${props.wine.slug}`)
const title = computed(() => cleanTitle(props.wine.title))
// «Сочетания: …» показывает блок «К чему подать» рядом с карточкой.
const description = computed(() => splitDescription(props.wine.description).body)

const facts = computed(() => {
  const sweetness = props.styleInfo?.sweetness
  const sparkling = props.styleInfo?.sparkling
    && !['брют', 'экстра брют'].includes(sweetness || '')
  return [
    props.wine.category,
    sparkling ? 'игристое' : null,
    sweetness,
    props.wine.alcohol ? `${props.wine.alcohol}% алк.` : null,
    props.wine.temperature ? `подача ${props.wine.temperature}` : null,
    ...(props.wine.grapes || []),
  ].filter(Boolean)
})
</script>

<template>
  <article :class="['card', { compact, link: !!to }]">
    <div v-if="missing" class="photo placeholder" role="img" aria-label="Фото нет в каталоге">
      <svg viewBox="0 0 40 110" aria-hidden="true">
        <path d="M16 4h8v26c0 4 8 8 8 20v52a4 4 0 0 1-4 4H12a4 4 0 0 1-4-4V50c0-12 8-16 8-20z" />
      </svg>
      <span v-if="!compact">Фото нет в каталоге</span>
    </div>
    <button
      v-else
      type="button"
      class="photo"
      :aria-label="`Увеличить фото: ${title}`"
      @click="zoomed = true"
    >
      <img
        :src="photo"
        :alt="title"
        loading="lazy"
        @error="failed = true"
      >
      <span v-if="!compact" class="zoom-hint" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="6.5" /><path d="M16 16l4.5 4.5M11 8v6M8 11h6" />
        </svg>
      </span>
    </button>
    <component :is="to ? NuxtLink : 'div'" :to="to" class="go">
      <div class="body">
        <h2 class="title">{{ title }}</h2>
        <p class="origin">{{ [wine.manufacturer, wine.region].filter(Boolean).join(' · ') }}</p>

        <p v-if="wine.rating" class="rating">★ {{ wine.rating }} <span class="muted">рейтинг платформы</span></p>

        <p v-if="reasons?.length" class="reasons">{{ reasons.join(' · ') }}</p>

        <div v-if="!compact" class="chips facts">
          <span v-for="fact in facts" :key="fact" class="chip">{{ fact }}</span>
        </div>
        <p v-if="!compact && description" class="description">{{ description }}</p>
      </div>
      <span v-if="to" class="chevron" aria-hidden="true">›</span>
    </component>
    <PhotoZoom v-if="!missing" v-model="zoomed" :src="photo" :alt="title" />
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
.photo img { display: block; }
.placeholder {
  cursor: default;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--muted);
  font-size: 13px;
}
.placeholder svg { fill: #ece4df; }
.card:not(.compact) .placeholder {
  height: 220px;
  background: linear-gradient(180deg, #fbf7f4 0%, #fff 100%);
}
.card:not(.compact) .placeholder svg { height: 150px; }
.card.compact .placeholder { width: 58px; height: 78px; }
.card.compact .placeholder svg { height: 70px; }

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
