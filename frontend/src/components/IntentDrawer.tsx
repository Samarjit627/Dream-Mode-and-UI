import React from 'react'
import type { Intent } from '../pages/Shell'

type Props = {
  open: boolean
  onClose: () => void
  intent: Intent
  setIntent: React.Dispatch<React.SetStateAction<Intent>>
}

export default function IntentDrawer({ open, onClose, intent, setIntent }: Props) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="absolute right-0 top-0 h-full w-full max-w-md bg-white dark:bg-slate-900 shadow-xl p-4 overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Intent</h2>
          <button onClick={onClose} className="border rounded px-2 py-1 text-sm">Esc</button>
        </div>
        <div className="space-y-3 text-sm">
          {(
            [
              ['goal','Goal'],
              ['use','Use'],
              ['character','Character'],
              ['priority','Priority'],
              ['grammar','Grammar'],
              ['surface','Surface'],
              ['constraints','Constraints'],
            ] as Array<[keyof Intent, string]>
          ).map(([k,label]) => (
            <label key={k} className="block">
              <div className="text-xs mb-1 opacity-70">{label}</div>
              <input
                className="w-full border rounded px-2 py-1 bg-transparent"
                value={(intent[k] ?? '') as string}
                onChange={e => setIntent(s => ({ ...s, [k]: e.target.value }))}
              />
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
