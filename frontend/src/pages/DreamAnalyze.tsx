import React, { useState } from 'react'
import { Card } from '../lib/ui'

export default function DreamAnalyze() {
  const [error, setError] = useState<string | null>(null)
  const [reqId, setReqId] = useState<string | null>(null)
  const [result, setResult] = useState<any | null>(null)
  const [uploading, setUploading] = useState(false)

  // Dream uploader: POST file to /api/v1/dream/analyze then poll result
  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    setUploading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', f)
      const res = await fetch('/api/v1/dream/analyze', { method: 'POST', body: fd })
      if (!res.ok) throw new Error(`analyze ${res.status}`)
      const j = await res.json()
      const id = j.request_id || j.requestId || j.id
      setReqId(id)
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Analyze started' }))
      // poll
      const poll = async () => {
        if (!id) return
        const r = await fetch(`/api/v1/dream/result/${id}`)
        const dj = await r.json()
        if (dj?.status === 'processing') {
          setTimeout(poll, 1200)
        } else {
          setResult(dj)
          setUploading(false)
          window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Analyze complete' }))
        }
      }
      poll()
    } catch (err: any) {
      setUploading(false)
      const msg = `Upload failed: ${err?.message || 'network'}`
      setError(msg)
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: msg }))
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex items-center justify-between">
          <div className="text-sm font-medium">Dream Uploader</div>
          <input type="file" accept="image/*,application/pdf" onChange={onUpload} />
        </div>
        {reqId && <div className="mt-2 text-xs opacity-70">request_id: {reqId} {uploading && '(processing...)'}</div>}
      </Card>

      <Card>
        {error && <div className="mb-2 text-xs" style={{color:'var(--muted)'}}>{error} — check Gateway on 4000</div>}
        <div className="text-sm font-medium mb-2">Dream Results</div>
        {!result && <div className="text-xs opacity-70">{uploading ? 'Processing…' : 'Upload an image/PDF above to analyze'}</div>}
        {!!result && (
          <div className="text-xs space-y-2">
            <div>Objects: {(result.objects?.length ?? 0)}</div>
            {result.objects?.[0]?.class && <div>Top object: {result.objects[0].class}</div>}
            {result.text?.summary && <div>OCR summary: {result.text.summary}</div>}
            {!!result.perception_v1 && (
              <div className="space-y-1">
                <div className="text-xs font-medium">Perception v1</div>
                <div>Text blocks: {(result.perception_v1.text_blocks?.length ?? 0)}</div>
                <div>Lines: {(result.perception_v1.lines?.length ?? 0)}</div>
                <div>Closed shapes: {(result.perception_v1.closed_shapes?.length ?? 0)}</div>
                <div>Arrow candidates: {(result.perception_v1.arrow_candidates?.length ?? 0)}</div>
                {result.perception_v1.validation?.ocr_confidence_histogram && (
                  <pre className="mt-1 overflow-auto" style={{maxHeight:120}}>{JSON.stringify(result.perception_v1.validation.ocr_confidence_histogram, null, 2)}</pre>
                )}
              </div>
            )}
            <pre className="mt-2 overflow-auto" style={{maxHeight:200}}>{JSON.stringify(result.logs ?? [], null, 2)}</pre>
          </div>
        )}
      </Card>
    </div>
  )
}
