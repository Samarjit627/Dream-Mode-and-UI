import { useEffect, useState, useCallback } from 'react'

type Theme = 'light' | 'dark' | 'system'
const KEY = 'theme'

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window === 'undefined') return 'system'
    const ls = localStorage.getItem(KEY) as Theme | null
    return ls || 'system'
  })

  const apply = useCallback((t: Theme) => {
    const root = document.documentElement
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    const effective = t === 'system' ? (prefersDark ? 'dark' : 'light') : t
    root.classList.toggle('dark', effective === 'dark')
  }, [])

  useEffect(() => {
    apply(theme)
    try { localStorage.setItem(KEY, theme) } catch {}
  }, [theme, apply])

  const setTheme = (t: Theme) => setThemeState(t)

  return { theme, setTheme }
}
