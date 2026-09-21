<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// 사용 예:
// <Tooltip content="watering_per_week, days_since_repot, fertilizer_ml">수치형(6개)</Tooltip>
// content는 콤마로 구분하면 박스 안에서 줄바꿈되어 보기 좋습니다.
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
      aria-describedby="tt-box"
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
}
.tt-label:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
  border-radius: 2px;
}
.tt-box {
  position: absolute;
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  z-index: 50;
  min-width: 160px;
  max-width: 260px;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background-color: var(--vp-c-bg-elv);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
  font-size: 12.5px;
  line-height: 1.5;
  white-space: nowrap;
  text-align: left;
}
/* 아래를 가리키는 작은 꼬리 */
.tt-box::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: var(--vp-c-bg-elv);
}
.tt-item {
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
}
</style>
