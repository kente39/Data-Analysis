<script setup>
import { onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import { setLast, addVisited, getScroll, setScroll, normalizePath } from './progress.js'

const route = useRoute()
const RESUME_KEY = 'plantguide:resumeTo'

let ticking = false
let navSuppress = false     // 이동 직후 자동 스크롤(맨 위로)이 저장값을 덮어쓰지 않도록 잠깐 저장 중단
let currentPath = ''

function onScroll() {
  if (navSuppress || ticking) return
  ticking = true
  requestAnimationFrame(() => {
    if (currentPath) setScroll(currentPath, window.scrollY)
    ticking = false
  })
}

function record(path) {
  currentPath = normalizePath(path)
  setLast(path)
  addVisited(path)
}

// '이어보기' 버튼으로 들어온 경우에만 저장된 스크롤 위치로 복원.
// 이전/다음·사이드바 등 일반 이동은 복원하지 않음 → VitePress 기본대로 맨 위.
function maybeResume(path) {
  let target = ''
  try { target = sessionStorage.getItem(RESUME_KEY) || '' } catch {}
  if (!target) return
  if (normalizePath(path) !== normalizePath(target)) return
  try { sessionStorage.removeItem(RESUME_KEY) } catch {}
  if (window.location.hash) return
  const y = getScroll(path)
  if (y > 0) {
    // VitePress 기본 스크롤(맨 위로)보다 나중에 실행되도록 약간 지연
    setTimeout(() => window.scrollTo({ top: y, behavior: 'auto' }), 120)
  }
}

function enter(path) {
  navSuppress = true
  record(path)
  maybeResume(path)
  setTimeout(() => { navSuppress = false }, 250)
}

onMounted(() => {
  window.addEventListener('scroll', onScroll, { passive: true })
  enter(route.path)
})

onUnmounted(() => {
  window.removeEventListener('scroll', onScroll)
})

watch(() => route.path, (path) => enter(path))
</script>

<template>
  <span aria-hidden="true" style="display: none" />
</template>
