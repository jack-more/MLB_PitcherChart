/**
 * Stat color + thickness helpers shared across components.
 */

const STAT_RANGES = {
  wOBA:          { min: 0.220, mid: 0.320, max: 0.420, invert: false },
  BA:            { min: 0.180, mid: 0.260, max: 0.340, invert: false },
  SLG:           { min: 0.280, mid: 0.420, max: 0.560, invert: false },
  K_pct:         { min: 0.10,  mid: 0.22,  max: 0.35,  invert: true },
  BB_pct:        { min: 0.04,  mid: 0.09,  max: 0.16,  invert: false },
  whiff_rate_vs: { min: 0.15,  mid: 0.25,  max: 0.38,  invert: true },
}

/** Returns a normalized 0-1 value for how "good" the stat is */
export function getStatT(stat, value) {
  const range = STAT_RANGES[stat] || STAT_RANGES.wOBA
  const { min, max, invert } = range
  let t = (value - min) / (max - min)
  t = Math.max(0, Math.min(1, t))
  if (invert) t = 1 - t
  return t
}

/** Returns a hex color from red (bad) through amber to green (good) */
export function getStatColor(stat, value) {
  const t = getStatT(stat, value)
  // Red -> Orange -> Yellow -> Green
  if (t < 0.25) return lerpColor('#c1292e', '#e07840', t / 0.25)
  if (t < 0.50) return lerpColor('#e07840', '#d4a053', (t - 0.25) / 0.25)
  if (t < 0.75) return lerpColor('#d4a053', '#7bc47f', (t - 0.50) / 0.25)
  return lerpColor('#7bc47f', '#2a9d5f', (t - 0.75) / 0.25)
}

/** Returns line thickness (for 3D tube radius) based on PA count */
export function getPAThickness(pa) {
  // 10 PA -> thin, 200+ PA -> thick
  const t = Math.min(1, Math.max(0, (pa - 10) / 190))
  return 0.02 + t * 0.08
}

export function formatStat(stat, value) {
  if (stat === 'K_pct' || stat === 'BB_pct' || stat === 'whiff_rate_vs') {
    return (value * 100).toFixed(1) + '%'
  }
  return value.toFixed(3)
}

function lerpColor(a, b, t) {
  const ar = parseInt(a.slice(1, 3), 16)
  const ag = parseInt(a.slice(3, 5), 16)
  const ab = parseInt(a.slice(5, 7), 16)
  const br = parseInt(b.slice(1, 3), 16)
  const bg = parseInt(b.slice(3, 5), 16)
  const bb = parseInt(b.slice(5, 7), 16)
  const r = Math.round(ar + (br - ar) * t)
  const g = Math.round(ag + (bg - ag) * t)
  const bl = Math.round(ab + (bb - ab) * t)
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${bl.toString(16).padStart(2, '0')}`
}
