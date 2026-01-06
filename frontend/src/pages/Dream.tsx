import React, { useState } from 'react'

export default function Dream() {
  const [prompt, setPrompt] = useState('Generate 4 ideas for a modern water bottle with 300ml capacity')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch('/api/ideate/generate4', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Dream</h1>
      <form onSubmit={onSubmit} className="flex items-center gap-2">
        <input
          className="flex-1 border rounded px-3 py-2 bg-transparent"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button className="px-3 py-2 rounded bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900" disabled={loading}>
          {loading ? 'Generating…' : 'Generate4'}
        </button>
      </form>
      {error && <div className="text-red-600">{error}</div>}
      {result && (
        <pre className="text-xs bg-slate-100 dark:bg-slate-800 p-3 rounded overflow-auto">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  )
}
