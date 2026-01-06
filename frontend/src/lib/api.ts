export const API_BASE = import.meta.env.VITE_API_URL || '/api'

export async function getAnalyze(mode: 'sketch'|'cad', payload?: any) {
  const res = await fetch(`${API_BASE}/analyze/${mode}`, { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify(payload || {}) })
  if (!res.ok) throw new Error('analyze failed')
  return res.json()
}

export async function postIdeateGenerate4(payload: any) {
  const res = await fetch(`${API_BASE}/ideate/generate4`, { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify(payload) })
  if (!res.ok) throw new Error('ideate failed')
  return res.json()
}

export async function postMentorCritique(payload: any) {
  const res = await fetch(`${API_BASE}/mentor/critique`, { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify(payload) })
  if (!res.ok) throw new Error('mentor failed')
  return res.json()
}

export async function getTastePacks() {
  const res = await fetch(`${API_BASE}/taste/packs`)
  if (!res.ok) throw new Error('packs failed')
  return res.json()
}

export async function getKnowledgeCards(track?: string) {
  const url = new URL(`${API_BASE}/knowledge/cards`, window.location.origin)
  if (track) url.searchParams.set('track', track)
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error('cards failed')
  return res.json()
}

export async function uploadStep(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('upload failed')
  return res.json()
}
