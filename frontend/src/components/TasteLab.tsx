import React from 'react'
import type { StylePack } from '../lib/packs'

type Props = {
  packs: StylePack[]
  onClose: () => void
  onSelect?: (p: StylePack) => void
  selectedPack?: StylePack | null
}

export default function TasteLab({ packs, onClose, onSelect, selectedPack }: Props) {
  return (
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="absolute right-0 top-0 h-full w-full max-w-md bg-white dark:bg-slate-900 shadow-xl p-4 overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Taste Lab</h2>
          <button onClick={onClose} className="border rounded px-2 py-1 text-sm">Close</button>
        </div>
        <div className="space-y-3">
          {packs.map((p) => (
            <div key={p.id} className={`border rounded p-3 text-sm ${selectedPack?.id===p.id ? 'ring-1 ring-emerald-500' : ''}`}>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{p.id}</div>
                  <div className="text-xs opacity-70">{p.detail_cadence || ''}</div>
                </div>
                {onSelect && <button className="px-2 py-1 border rounded text-xs" onClick={() => onSelect(p)}>{selectedPack?.id===p.id ? 'Selected' : 'Select'}</button>}
              </div>
              <div className="text-xs mt-1">Finishes: {(p.finishes||[]).join(', ')}</div>
              <div className="text-xs mt-1">Bias Rules: {(p.bias_rules||[]).join(', ')}</div>
            </div>
          ))}
          {packs.length === 0 && <div className="text-sm opacity-70">No packs found</div>}
        </div>
      </div>
    </div>
  )
}
