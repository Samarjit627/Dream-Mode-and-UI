import React, { useEffect, useMemo, useState } from 'react'
import { Card, Chip } from '../lib/ui'
import { getKnowledgeCards, getTastePacks, postMentorCritique } from '../lib/api'
import TasteLab from '../components/TasteLab'
import type { StylePack, KnowledgeCard } from '../lib/packs'
import { getChosenPack, setChosenPack } from '../lib/session'

export default function DreamMentor() {
  const [tasteOpen, setTasteOpen] = useState(false)
  const [packs, setPacks] = useState<StylePack[]>([])
  const [cards, setCards] = useState<KnowledgeCard[]>([])
  const [selectedCard, setSelectedCard] = useState<KnowledgeCard | null>(null)
  const [pushes, setPushes] = useState<Array<{ overlay: string; note: string }>>([])
  const [chosenPack, setChosen] = useState<StylePack | null>(getChosenPack())
  const [belt, setBelt] = useState(0.6)

  useEffect(() => {
    getTastePacks().then(d => setPacks(d.packs || d))
    getKnowledgeCards('form_readability').then(d => setCards(d.cards || d))
    const onStorage = (e: StorageEvent) => {
      if (e.key === 'chosenPack') setChosen(getChosenPack())
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  const canPush = pushes.length < 3

  const onCritique = async () => {
    if (!canPush) return
    const out = await postMentorCritique({})
    const overlay = selectedCard?.overlay?.type || 'band_ratio'
    const note = out?.critique?.[0]?.note || 'Consider refinement'
    setPushes(prev => [...prev, { overlay, note }])
  }

  const onTrial = (overlay: string) => {
    if (/band/i.test(overlay)) {
      const next = Math.min(0.95, Math.max(0.05, belt + 0.04))
      setBelt(next)
      window.dispatchEvent(new CustomEvent('axis5:beltline:set', { detail: { y: next } }))
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `Applied trial: beltline → ${(next*100)|0}% height` }))
    } else {
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `Trial not implemented for ${overlay}` }))
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <button className="px-3 py-1 border rounded text-sm" onClick={onCritique} disabled={!canPush}>
          {canPush ? 'Push Critique' : 'Max 3 pushes reached'}
        </button>
        <button className="px-3 py-1 border rounded text-sm" onClick={() => setTasteOpen(v => !v)}>Taste Lab</button>
        {chosenPack && <span className="text-xs opacity-70">Pack: {chosenPack.id}</span>}
      </div>

      <Card>
        <div className="text-sm font-medium mb-2">Rubric</div>
        <div className="flex flex-wrap gap-2 text-xs">
          {['Intent','Package Truth','Readability','Affordance','Detail'].map((r,i) => <Chip key={i}>{r}</Chip>)}
        </div>
      </Card>

      <Card>
        <div className="text-sm font-medium mb-2">Knowledge Cards (track: form_readability)</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {cards.slice(0,6).map((c,i) => (
            <div key={i} className={`border rounded p-2 text-sm ${selectedCard?.id===c.id ? 'ring-1 ring-emerald-500' : ''}`}>
              <div className="font-medium">{c.title}</div>
              <div className="text-xs opacity-70">{c.track}</div>
              <div className="mt-1 text-xs">Overlay: {c.overlay?.type}</div>
              <div className="mt-2 flex gap-2">
                <button className="px-2 py-1 border rounded text-xs" onClick={() => setSelectedCard(c)}>Select</button>
                {c.actions?.[0] && <button className="px-2 py-1 border rounded text-xs">{c.actions[0].label}</button>}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <div className="text-sm font-medium mb-2">Pushes (≤3)</div>
        <ul className="list-disc pl-5 text-sm">
          {pushes.map((p,i) => (
            <li key={i} className="mb-1">Overlay: <span className="font-medium">{p.overlay}</span> — {p.note} <button className="ml-2 px-2 py-0.5 border rounded text-xs" onClick={() => onTrial(p.overlay)}>Trial Preview</button></li>
          ))}
          {pushes.length === 0 && <li className="opacity-70">No pushes yet</li>}
        </ul>
      </Card>

      {tasteOpen && (
        <TasteLab
          packs={packs}
          onClose={() => setTasteOpen(false)}
          onSelect={(p: StylePack) => { setChosenPack(p); setChosen(p) }}
          selectedPack={chosenPack}
        />
      )}
    </div>
  )
}
