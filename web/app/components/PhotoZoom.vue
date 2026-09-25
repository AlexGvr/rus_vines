<script setup lang="ts">
// Фото на весь экран. Эталоны в дампе бывают мелкими скриншотами, и
// сверить свою бутылку с карточкой по миниатюре трудно. Касание фото
// переключает «вписать в экран» и «вдвое крупнее с прокруткой»,
// касание фона, крестик или Esc закрывают.
const props = defineProps<{ src: string; alt?: string }>()
const open = defineModel<boolean>({ default: false })
const natural = ref(false)
const frame = ref<HTMLElement>()

// Увеличение — в ту точку, которой коснулись: иначе прокрутка начинается
// с левого верхнего угла, а бутылка стоит посередине кадра.
async function toggle(event: MouseEvent) {
  const img = event.currentTarget as HTMLImageElement
  const box = img.getBoundingClientRect()
  const fx = (event.clientX - box.left) / box.width
  const fy = (event.clientY - box.top) / box.height
  natural.value = !natural.value
  if (!natural.value) return
  await nextTick()
  const el = frame.value
  if (!el) return
  el.scrollLeft = fx * el.scrollWidth - el.clientWidth / 2
  el.scrollTop = fy * el.scrollHeight - el.clientHeight / 2
}

function close() {
  open.value = false
  natural.value = false
}

function onKey(event: KeyboardEvent) {
  if (event.key === 'Escape') close()
}

watch(open, (value) => {
  document.body.style.overflow = value ? 'hidden' : ''
  if (value) window.addEventListener('keydown', onKey)
  else window.removeEventListener('keydown', onKey)
})

onBeforeUnmount(() => {
  document.body.style.overflow = ''
  window.removeEventListener('keydown', onKey)
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="zoom"
      role="dialog"
      aria-modal="true"
      :aria-label="props.alt || 'Фото'"
      @click.self="close"
    >
      <div ref="frame" :class="['frame', { natural }]" @click.self="close">
        <img :src="props.src" :alt="props.alt" @click="toggle">
      </div>
      <button class="close" aria-label="Закрыть" @click="close">✕</button>
      <p class="tip">{{ natural ? 'Коснитесь фото, чтобы вписать в экран' : 'Коснитесь фото, чтобы увеличить' }}</p>
    </div>
  </Teleport>
</template>

<style scoped>
.zoom {
  position: fixed;
  inset: 0;
  z-index: 100;
  background: rgba(24, 20, 19, .94);
  display: flex;
  align-items: center;
  justify-content: center;
}
.frame {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 56px 12px 44px;
}
/* Вписать в экран: мелкий эталон (скриншоты около 290×556) растягивается
   до размеров экрана, а не остаётся в натуральную величину. */
.frame img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  cursor: zoom-in;
}
/* Крупнее экрана: вдвое шире, остальное — прокруткой. */
.frame.natural {
  overflow: auto;
  display: block;
  -webkit-overflow-scrolling: touch;
}
.frame.natural img {
  width: 200%;
  height: auto;
  cursor: zoom-out;
}
.close {
  position: absolute;
  top: 12px;
  right: 12px;
  width: 40px;
  height: 40px;
  border: 0;
  border-radius: 50%;
  background: rgba(255, 255, 255, .14);
  color: #fff;
  font-size: 18px;
  cursor: pointer;
}
.tip {
  position: absolute;
  bottom: 10px;
  left: 0;
  right: 0;
  margin: 0;
  text-align: center;
  font-size: 12.5px;
  color: rgba(255, 255, 255, .6);
  pointer-events: none;
}
</style>
