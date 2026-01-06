const USE_WORKER = (process.env.USE_WORKER || 'false').toLowerCase() === 'true'
const WORKER_BASE_URL = process.env.WORKER_BASE_URL || 'http://localhost:7072'

export async function maybeProxyPost(path: string, body: any) {
  if (!USE_WORKER) return null
  const res = await fetch(WORKER_BASE_URL + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {})
  })
  if (!res.ok) throw new Error(`Worker ${path} HTTP ${res.status}`)
  return await res.json()
}

export async function maybeProxyGet(path: string) {
  if (!USE_WORKER) return null
  const res = await fetch(WORKER_BASE_URL + path)
  if (!res.ok) throw new Error(`Worker ${path} HTTP ${res.status}`)
  return await res.json()
}
