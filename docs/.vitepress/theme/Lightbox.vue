<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'

const open = ref(false)
const src = ref('')
const alt = ref('')

function onClick(e) {
  const t = e.target
  if (!(t instanceof HTMLImageElement)) return
  // 본문(.vp-doc) 이미지만 대상, 캐러셀(.ic)·no-zoom 은 제외
  if (!t.closest('.vp-doc')) return
  if (t.closest('.ic') || t.classList.contains('no-zoom')) return
  src.value = t.currentSrc || t.src
  alt.value = t.alt || ''
  open.value = true
}

function close() {
  open.value = false
}

function onKey(e) {
  if (e.key === 'Escape') close()
}

// 라이트박스 열려있는 동안 본문 스크롤 잠금
watch(open, (v) => {
  try {
    document.body.style.overflow = v ? 'hidden' : ''
  } catch {}
})

onMounted(() => {
  document.addEventListener('click', onClick)
  document.addEventListener('keydown', onKey)
})

onUnmounted(() => {
  document.removeEventListener('click', onClick)
  document.removeEventListener('keydown', onKey)
  try { document.body.style.overflow = '' } catch {}
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="lb"
      role="dialog"
      aria-modal="true"
      aria-label="이미지 확대 보기"
      @click="close"
    >
      <img :src="src" :alt="alt" class="lb-img" />
      <button class="lb-close" @click.stop="close" aria-label="닫기">×</button>
    </div>
  </Teleport>
</template>

<style scoped>
.lb {
  position: fixed;
  inset: 0;
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px;
  background-color: rgba(0, 0, 0, 0.85);
  cursor: zoom-out;
  animation: lb-fade 0.15s ease;
}
.lb-img {
  max-width: 92vw;
  max-height: 92vh;
  object-fit: contain;
  border-radius: 6px;
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.5);
  cursor: default;
}
.lb-close {
  position: fixed;
  top: 16px;
  right: 20px;
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background-color: rgba(255, 255, 255, 0.12);
  color: #fff;
  font-size: 26px;
  line-height: 1;
  cursor: pointer;
  transition: background-color 0.2s;
}
.lb-close:hover {
  background-color: rgba(255, 255, 255, 0.28);
}
@keyframes lb-fade {
  from { opacity: 0; }
  to { opacity: 1; }
}
</style>
