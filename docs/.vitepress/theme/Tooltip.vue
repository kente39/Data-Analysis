<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'

// 사용 예:
// <Tooltip content="watering_per_week(주당 물주기 횟수), fertilizer_ml(비료량 ml)">수치형(6개)</Tooltip>
// content는 콤마로 구분하면 박스 안에서 항목별로 줄바꿈됩니다.
// 박스는 <body>로 Teleport되어 표/컨테이너의 overflow에 잘리지 않습니다.
const props = defineProps({
  content: { type: String, default: '' }
})

const open = ref(false)
const labelEl = ref(null)
const boxEl = ref(null)
const pos = ref({ top: 0, left: 0 })

const items = () => props.content.split(',').map(s => s.trim()).filter(Boolean)

function place() {
  const el = labelEl.value, box = boxEl.value
  if (!el || !box) return
  const r = el.getBoundingClientRect()
  const bw = box.offsetWidth, bh = box.offsetHeight
  const margin = 8
  const vw = window.innerWidth

  // 기본: 라벨 왼쪽에 맞춰 위쪽에 표시
  let left = r.left
  let top = r.top - bh - 10           // 라벨 위 (스크롤 좌표 아님: fixed 사용)
  // 좌우 화면 밖으로 나가지 않게 보정
  if (left + bw > vw - margin) left = vw - margin - bw
  if (left < margin) left = margin
  // 위 공간이 부족하면 아래로 뒤집기
  if (top < margin) top = r.bottom + 10

  pos.value = { top, left }
}

async function show() {
  open.value = true
  await nextTick()
  place()
}
function hide() { open.value = false }
function toggle() { open.value ? hide() : show() }

function onDocClick(e) {
  if (labelEl.value && labelEl.value.contains(e.target)) return
  if (boxEl.value && boxEl.value.contains(e.target)) return
  hide()
}
function onScrollResize() { if (open.value) place() }

onMounted(() => {
  document.addEventListener('click', onDocClick)
  window.addEventListener('scroll', onScrollResize, true)
  window.addEventListener('resize', onScrollResize)
})
onUnmounted(() => {
  document.removeEventListener('click', onDocClick)
  window.removeEventListener('scroll', onScrollResize, true)
  window.removeEventListener('resize', onScrollResize)
})
</script>

<template>
  <span class="tt">
    <span
      ref="labelEl"
      class="tt-label"
      tabindex="0"
      @mouseenter="show"
      @mouseleave="hide"
      @click="toggle"
      @focus="show"
      @blur="hide"
    ><slot /></span>

    <Teleport to="body">
      <span
        v-show="open"
        ref="boxEl"
        class="tt-box"
        role="tooltip"
        :style="{ top: pos.top + 'px', left: pos.left + 'px' }"
      >
        <span v-for="(it, i) in items()" :key="i" class="tt-item">{{ it }}</span>
      </span>
    </Teleport>
  </span>
</template>

<style scoped>
.tt { position: relative; display: inline-block; }
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
</style>

<style>
/* Teleport로 body에 붙으므로 전역 스타일(scoped 아님) */
.tt-box {
  position: fixed;
  z-index: 2000;
  width: max-content;
  max-width: min(320px, 86vw);
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background-color: var(--vp-c-bg-elv);
  box-shadow: 0 6px 22px rgba(0, 0, 0, 0.22);
  text-align: left;
  white-space: normal;
  overflow-wrap: anywhere;
  pointer-events: none;   /* 박스가 hover를 가로채 깜빡이는 것 방지 */
}
.tt-box .tt-item {
  font-family: var(--vp-font-family-mono);
  font-size: 13.5px;
  line-height: 1.55;
  color: var(--vp-c-text-1);
}
</style>