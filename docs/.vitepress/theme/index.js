import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import BackToTop from './BackToTop.vue'
import ProgressTracker from './ProgressTracker.vue'
import ResumeBanner from './ResumeBanner.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // 모든 페이지 전역: 플로팅 '맨 위로' 버튼 + 진도 추적기(비가시)
      'layout-bottom': () => [h(BackToTop), h(ProgressTracker)],
      // 홈 전용: 히어로 아래 '이어보기' 배너
      'home-hero-after': () => h(ResumeBanner)
    })
  }
}
