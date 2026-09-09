<script setup>
import { onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import { setLast, addVisited, getScroll, setScroll } from './progress.js'

const route = useRoute()
let ticking = false
let currentPath = ''

// 스크롤을 throttle(rAF)해서 현재 페이지 위치 저장
function onScroll() {
  if (ticking) return
  ticking = true
  requestAnimationFrame(() => {
    if (currentPath) setScroll(currentPath, window.scrollY)
    ticking = false
  })
}

// 페이지 진입 시: 이어보기/방문 기록 + (앵커 없을 때만) 저장된 스크롤 복원
function handleEnter(path) {
  currentPath = path
  setLast(path)
  addVisited(path)

  // #heading 앵커로 들어온 경우엔 복원하지 않음 (VitePress 기본 동작 존중)
  if (window.location.hash) return

  const y = getScroll(path)
  if (y > 0) {
    // 콘텐츠 렌더 후 복원 (약간의 지연)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => window.scrollTo({ top: y, behavior: 'auto' }))
    })
  }
}

onMounted(() => {
  window.addEventListener('scroll', onScroll, { passive: true })
  handleEnter(route.path)
})

onUnmounted(() => {
  window.removeEventListener('scroll', onScroll)
})

// SPA 라우트 변경 감지
watch(
  () => route.path,
  (path) => handleEnter(path)
)
</script>

<template>
  <!-- 화면에 아무것도 그리지 않는 로직 전용 컴포넌트 -->
  <span aria-hidden="true" style="display: none" />
</template>
