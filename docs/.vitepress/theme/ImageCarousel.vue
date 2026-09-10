<script setup>
import { ref, computed } from 'vue'

// 사용 예:
// <ImageCarousel :images="[
//   { src: '/colab/step1.png', caption: '1. 새 노트북 만들기' },
//   { src: '/colab/step2.png', caption: '2. 런타임 → GPU 설정' }
// ]" />
// 스크린샷은 docs/public/ 아래에 두고 절대경로(/colab/step1.png)로 참조하세요.
// src를 비워두면 '이미지 준비 중' 플레이스홀더가 표시됩니다(구조 먼저 잡고 나중에 교체 가능).
const props = defineProps({
  images: { type: Array, default: () => [] }
})

const idx = ref(0)
const count = computed(() => props.images.length)

function go(n) {
  if (!count.value) return
  idx.value = (n + count.value) % count.value
}
function prev() { go(idx.value - 1) }
function next() { go(idx.value + 1) }
function onKey(e) {
  if (e.key === 'ArrowLeft') prev()
  else if (e.key === 'ArrowRight') next()
}
</script>

<template>
  <div v-if="count" class="ic" tabindex="0" @keydown="onKey">
    <div class="ic-stage">
      <button v-if="count > 1" class="ic-arrow ic-prev" @click="prev" aria-label="이전">‹</button>

      <div class="ic-frame">
        <img
          v-if="images[idx] && images[idx].src"
          :src="images[idx].src"
          :alt="images[idx].caption || ''"
          class="no-zoom"
        />
        <div v-else class="ic-placeholder">
          <span class="ic-ph-icon">🖼️</span>
          <span class="ic-ph-text">이미지 준비 중</span>
          <small v-if="images[idx]">{{ images[idx].caption }}</small>
        </div>
      </div>

      <button v-if="count > 1" class="ic-arrow ic-next" @click="next" aria-label="다음">›</button>
    </div>

    <div v-if="images[idx] && images[idx].caption" class="ic-caption">
      {{ images[idx].caption }}
    </div>

    <div v-if="count > 1" class="ic-dots">
      <button
        v-for="(im, i) in images"
        :key="i"
        class="ic-dot"
        :class="{ active: i === idx }"
        @click="go(i)"
        :aria-label="`${i + 1}번째 이미지`"
      />
      <span class="ic-counter">{{ idx + 1 }} / {{ count }}</span>
    </div>
  </div>
</template>

<style scoped>
.ic {
  margin: 20px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  overflow: hidden;
  background-color: var(--vp-c-bg-soft);
  outline: none;
}
.ic-stage {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}
.ic-frame {
  flex: 1;
  min-height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
}
.ic-frame img {
  max-width: 100%;
  max-height: 70vh;
  border-radius: 6px;
  display: block;
}
.ic-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 48px 16px;
  color: var(--vp-c-text-3);
  border: 2px dashed var(--vp-c-divider);
  border-radius: 8px;
  width: 100%;
  text-align: center;
}
.ic-ph-icon { font-size: 28px; }
.ic-ph-text { font-weight: 600; }
.ic-arrow {
  flex: 0 0 auto;
  width: 40px;
  height: 40px;
  margin: 0 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 50%;
  background-color: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 22px;
  line-height: 1;
  cursor: pointer;
  transition: background-color 0.2s, color 0.2s;
}
.ic-arrow:hover {
  background-color: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}
.ic-caption {
  padding: 10px 16px;
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  border-top: 1px solid var(--vp-c-divider);
  text-align: center;
}
.ic-dots {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px;
  flex-wrap: wrap;
}
.ic-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  border: none;
  padding: 0;
  background-color: var(--vp-c-divider);
  cursor: pointer;
  transition: background-color 0.2s, transform 0.2s;
}
.ic-dot.active {
  background-color: var(--vp-c-brand-1);
  transform: scale(1.25);
}
.ic-counter {
  margin-left: 6px;
  font-size: 12px;
  color: var(--vp-c-text-2);
}
</style>
