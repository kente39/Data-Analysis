<script setup>
import { ref, onMounted } from 'vue'
import { withBase } from 'vitepress'
import { getLast, titleFor, getVisited } from './progress.js'

const last = ref('')
const lastTitle = ref('')
const visitedCount = ref(0)

onMounted(() => {
  last.value = getLast()
  lastTitle.value = last.value ? titleFor(last.value) : ''
  visitedCount.value = getVisited().length
})

// '이어보기'로 들어갈 때만 스크롤 복원되도록 플래그를 심는다
function onResume() {
  try {
    sessionStorage.setItem('plantguide:resumeTo', last.value)
  } catch {}
}
</script>

<template>
  <div v-if="last" class="resume-banner">
    <div class="resume-info">
      <span class="resume-label">이어보기</span>
      <span class="resume-title">{{ lastTitle }}</span>
      <span v-if="visitedCount" class="resume-count">· 지금까지 {{ visitedCount }}개 페이지 학습</span>
    </div>
    <a class="resume-btn" :href="withBase(last + '.html')" @click="onResume">이어서 학습하기 →</a>
  </div>
</template>

<style scoped>
.resume-banner {
  max-width: 1152px;
  margin: 0 auto;
  padding: 16px 24px;
  margin-top: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background-color: var(--vp-c-bg-soft);
}
.resume-info {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}
.resume-label {
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  background-color: var(--vp-c-brand-1);
  padding: 2px 8px;
  border-radius: 6px;
}
.resume-title {
  font-weight: 600;
  color: var(--vp-c-text-1);
}
.resume-count {
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.resume-btn {
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  white-space: nowrap;
}
.resume-btn:hover {
  text-decoration: underline;
}
@media (max-width: 640px) {
  .resume-banner {
    padding: 14px 16px;
  }
}
</style>
