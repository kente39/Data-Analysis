import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import BackToTop from './BackToTop.vue'
import ProgressTracker from './ProgressTracker.vue'
import ResumeBanner from './ResumeBanner.vue'
import Lightbox from './Lightbox.vue'
import ImageCarousel from './ImageCarousel.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // 전역: 맨 위로 버튼 + 진도 추적 + 이미지 라이트박스
      'layout-bottom': () => [h(BackToTop), h(ProgressTracker), h(Lightbox)],
      // 홈 전용: 이어보기 배너
      'home-hero-after': () => h(ResumeBanner)
    })
  },
  enhanceApp({ app }) {
    // 마크다운 어디서나 <ImageCarousel> 사용 가능하도록 전역 등록
    app.component('ImageCarousel', ImageCarousel)
  }
}
