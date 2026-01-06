import React from 'react'

export function Chip({ active, children, onClick }: { active?: boolean; children: React.ReactNode; onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-2 py-1 rounded text-xs border"
      style={{
        background: active ? 'var(--panel-2)' : 'transparent',
        borderColor: 'var(--border)',
        color: 'var(--text)'
      }}
    >
      {children}
    </button>
  )
}

export function Card({ children }: { children: React.ReactNode }) {
  return (
    <div className="border rounded-lg p-3 shadow-sm" style={{ background: 'var(--panel-2)', borderColor: 'var(--border)', color: 'var(--text)' }}>{children}</div>
  )
}

export function Slider({ value, onChange, min = 0, max = 1, step = 0.01 }: { value: number; onChange: (v:number)=>void; min?: number; max?: number; step?: number }) {
  return (
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))} />
  )
}
