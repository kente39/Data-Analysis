// 학습 진도 저장용 localStorage 헬퍼
// - 마지막으로 본 페이지(이어보기)
// - 페이지별 마지막 스크롤 위치
// - 방문한 페이지 목록(진도 표시용)
// 모든 접근은 try/catch로 감싸 시크릿 모드/차단 환경에서도 사이트가 죽지 않게 함.

const PREFIX = 'plantguide:'

export const KEYS = {
  last: PREFIX + 'lastVisited',
  visited: PREFIX + 'visited',
  scroll: (path) => PREFIX + 'scroll:' + path
}

// 홈은 이어보기 대상에서 제외
export function isTrackable(path) {
  return !!path && path !== '/' && !path.endsWith('/index.html')
}

// 경로 → 사람이 읽을 수 있는 제목 (사이드바와 동일)
export const PAGE_TITLES = {
  '/module00': '0. 개발환경 셋업',
  '/stats-intro': '통계학 기초 입문',
  '/module01': '1. NumPy & Pandas 기초',
  '/module02': '2. 기술통계를 코드로',
  '/module03': '3. 시각화 (Matplotlib·Seaborn)',
  '/module04': '4. 통계 검정 (SciPy·statsmodels)',
  '/module05': '5. 전처리 (scikit-learn)',
  '/module06': '6. 머신러닝 (scikit-learn)',
  '/module07': '7. 딥러닝 (TensorFlow·Keras)',
  '/data': '데이터 다운로드'
}

// 배포 시 경로가 '/module03.html' 형태일 수 있어 확장자를 떼고 매핑
export function normalizePath(path) {
  if (!path) return path
  return path.replace(/\.html$/, '').replace(/\/index$/, '/')
}

export function titleFor(path) {
  const p = normalizePath(path)
  return PAGE_TITLES[p] || p
}

export function getLast() {
  try {
    return localStorage.getItem(KEYS.last) || ''
  } catch {
    return ''
  }
}

export function setLast(path) {
  try {
    if (isTrackable(path)) localStorage.setItem(KEYS.last, normalizePath(path))
  } catch {}
}

export function getVisited() {
  try {
    return JSON.parse(localStorage.getItem(KEYS.visited) || '[]')
  } catch {
    return []
  }
}

export function addVisited(path) {
  try {
    if (!isTrackable(path)) return
    const p = normalizePath(path)
    const list = getVisited()
    if (!list.includes(p)) {
      list.push(p)
      localStorage.setItem(KEYS.visited, JSON.stringify(list))
    }
  } catch {}
}

export function getScroll(path) {
  try {
    const v = localStorage.getItem(KEYS.scroll(normalizePath(path)))
    return v ? parseInt(v, 10) : 0
  } catch {
    return 0
  }
}

export function setScroll(path, y) {
  try {
    localStorage.setItem(KEYS.scroll(normalizePath(path)), String(Math.max(0, y | 0)))
  } catch {}
}
