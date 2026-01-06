import React from 'react'
import { useTheme } from '../hooks/useTheme'

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm">Theme</label>
      <select
        className="bg-transparent border rounded px-2 py-1"
        value={theme}
        onChange={(e) => setTheme(e.target.value as any)}
      >
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </div>
  )
}
