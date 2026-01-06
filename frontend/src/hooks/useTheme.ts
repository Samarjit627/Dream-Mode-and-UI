import { useEffect, useState } from 'react'

export type Theme = 'dark'|'light'
export const useTheme = ()=>{
  const [theme,setTheme]=useState<Theme>('dark')
  useEffect(()=>{
    const saved = localStorage.getItem('axis5.theme') as Theme | null
    const sysDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches
    setTheme(saved ?? (sysDark?'dark':'light'))
  },[])
  useEffect(()=>{
    document.documentElement.classList.toggle('light', theme==='light')
    localStorage.setItem('axis5.theme', theme)
  },[theme])
  return { theme, setTheme }
}
export default useTheme
