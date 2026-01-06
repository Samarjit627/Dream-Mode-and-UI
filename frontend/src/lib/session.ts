import type { StylePack } from './packs'

const KEY = 'chosenPack'

export function getChosenPack(): StylePack | null {
  try {
    const v = localStorage.getItem(KEY)
    return v ? JSON.parse(v) : null
  } catch { return null }
}

export function setChosenPack(p: StylePack | null) {
  try {
    if (p) localStorage.setItem(KEY, JSON.stringify(p))
    else localStorage.removeItem(KEY)
    window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: p ? '1' : '' }))
  } catch {}
}
