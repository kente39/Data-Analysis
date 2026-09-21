<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// 사용 예:
// <Tooltip content="watering_per_week(주당 물주기 횟수), fertilizer_ml(비료량 ml)">수치형(6개)</Tooltip>
// content는 콤마로 구분하면 박스 안에서 항목별로 줄바꿈됩니다.
const props = defineProps({
  content: { type: String, default: '' }
})

const open = ref(false)
const root = ref(null)

function toggle() { open.value = !open.value }
function onDocClick(e) {
  if (root.value && !root.value.contains(e.target)) open.value = false
}

onMounted(() => document.addEventListener('click', onDocClick))
onUnmounted(() => document.removeEventListener('click', onDocClick))

const items = () => props.content.split(',').map(s => s.trim()).filter(Boolean)
</script>

<template>
  <span
    ref="root"
    class="tt"
    @mouseenter="open = true"
    @mouseleave="open = false"
  >
    <span
      class="tt-label"
      tabindex="0"
      @click="toggle"
      @focus="open = true"
      @blur="open = false"
    ><slot /></span>
    <span v-show="open" class="tt-box" role="tooltip">
      <span v-for="(it, i) in items()" :key="i" class="tt-item">{{ it }}</span>
    </span>
  </span>
</template>

<style scoped>
.tt {
  position: relative;
  display: inline-block;
}
.tt-label {
  border-bottom: 1px dashed var(--vp-c-brand-1);
  cursor: help;
  outline: none;
  white-space: nowrap;
}
.tt-label:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
  border-radius: 2px;
}
.tt-box {
  position: absolute;
  bottom: calc(100% + 8px);
  /* 라벨 왼쪽에 맞춰 펼치되, 화면 밖으로 나가면 안쪽으로 당겨짐 */
  left: 0;
  z-index: 50;
  width: max-content;
  max-width: min(300px, 80vw);
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background-color: var(--vp-c-bg-elv);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
  font-size: 14px;
  line-height: 1.55;
  text-align: left;
  white-space: normal;      /* 길면 줄바꿈 */
  overflow-wrap: anywhere;
}
/* 왼쪽 아래를 가리키는 작은 꼬리 */
.tt-box::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 18px;
  border: 6px solid transparent;
  border-top-color: var(--vp-c-bg-elv);
}
.tt-item {
  color: var(--vp-c-text-1);
}
/* 괄호 앞의 컬럼명(영문)만 모노폰트로 */
.tt-item {
  font-family: var(--vp-font-family-mono);
  font-size: 13.5px;
}
</style>
