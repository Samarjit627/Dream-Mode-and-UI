import React, { useEffect, useState } from 'react'
import { Card, Chip } from '../lib/ui'
import type { Intent } from './Shell'
import { postIdeateGenerate4 } from '../lib/api'
import type { StylePack } from '../lib/packs'
import { getChosenPack } from '../lib/session'

export type Concept = {
  name: string
  tagline: string
  form_tags: string[]
  params: Record<string, string | number>
}

async function fetch4(intent: Intent, biasOn: boolean): Promise<Concept[]> {
  const payload = { intent, biasOn }
  const data = await postIdeateGenerate4(payload)
  const seed: Array<{ title: string; desc: string }> = data.ideas || []
  return seed.slice(0, 4).map((c, idx) => ({
    name: c.title || `Concept ${idx+1}`,
    tagline: intent.goal ? `${c.desc} — ${intent.goal}` : c.desc,
    form_tags: [intent.grammar || 'neutral', ...(biasOn ? ['finish:matte','detail:low'] : [])],
    params: { index: idx, bias: biasOn ? 1 : 0 }
  }))
}

export default function DreamIdeate() {
  const [intent] = useState<Intent>({})
  const [biasOn, setBiasOn] = useState(false)
  const [cards, setCards] = useState<Concept[]>([])
  const [chosenPack, setChosenPackState] = useState<StylePack | null>(getChosenPack())

  useEffect(() => { fetch4(intent, biasOn).then(setCards) }, [])
  useEffect(() => {
    const onStorage = (e: StorageEvent) => { if (e.key === 'chosenPack') setChosenPackState(getChosenPack()) }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  const onGenerate = async () => setCards(await fetch4(intent, biasOn))

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button className="px-3 py-1 border rounded text-sm" onClick={onGenerate}>Generate 4</button>
        <label className="text-sm flex items-center gap-2" title={!chosenPack ? 'Select a Style Pack in Taste Lab first' : ''}>
          <input type="checkbox" checked={biasOn} onChange={e => setBiasOn(e.target.checked)} disabled={!chosenPack} />
          Style Bias (off by default){chosenPack && <span className="opacity-60 ml-2">Pack: {chosenPack.id}</span>}
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {cards.map((c, i) => (
          <Card key={i}>
            <div className="text-sm font-medium">{c.name}</div>
            <div className="text-xs opacity-70 mb-2">{c.tagline}</div>
            <div className="flex flex-wrap gap-1 mb-2">
              {c.form_tags.map((t, j) => <Chip key={j}>{t}</Chip>)}
            </div>
            <div className="h-32 grid place-items-center text-slate-500 dark:text-slate-400">Preview</div>
          </Card>
        ))}
      </div>
    </div>
  )
}
