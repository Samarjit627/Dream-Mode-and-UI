import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

type Props = {
  onAttach: (kind: 'sketch' | 'cad') => void
  context?: {
    artifact?: 'none' | 'sketch' | 'cad'
    understanding?: { kind: string; payload?: any } | null
    detectedLabel?: string | null
    detectedProb?: number | null
    sceneCandidates?: Array<{ label: string; score: number }> | null
    sceneUncertainty?: { margin?: number; entropy_norm?: number } | null
    ocrSummary?: string | null
    ocrFullText?: string | null
    ocrPages?: Array<{ page: number; text: string }> | null
    perception_v1?: any | null
    judgement?: { kind?: string; confidence?: number; evidence?: any; summary?: string } | null
    engineering?: any | null
    engineeringDom?: any | null
    engineeringDrawing?: any | null
    engineeringUnderstanding?: any | null
    contacts?: {
      emails?: string[]
      phones?: string[]
      urls?: string[]
      address_like?: string[]
      icon_hints?: any
    } | null
    dimensions?: Array<{ kind?: string; value?: number; unit?: string | null; prefix?: string | null; raw?: string }> | null
    objects?: Array<{ class?: string; confidence?: number; class_clip_guess?: string; clip_score?: number }> | null
  }
}

type ChatMsg = {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export default function RightChat({ onAttach, context }: Props) {
  const [text, setText] = useState('')
  const [menu, setMenu] = useState(false)
  const [sending, setSending] = useState(false)
  const [messages, setMessages] = useState<ChatMsg[]>([
    { id: 'a0', role: 'assistant', content: 'Chat is ready. Upload a file, then tell me what you want to do.' }
  ])
  const listRef = useRef<HTMLDivElement>(null)
  const lastContextSigRef = useRef<string>('')
  const messagesRef = useRef<ChatMsg[]>(messages)

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  const canSend = useMemo(() => !!text.trim() && !sending, [text, sending])

  const appendGuessDebug = useCallback(() => {
    const cands = Array.isArray(context?.sceneCandidates) ? context!.sceneCandidates!.slice(0, 7) : []
    const unc = context?.sceneUncertainty
    if (!cands.length) {
      setMessages((prev) => [...prev, { id: `a_dbg_${Date.now()}`, role: 'assistant', content: 'No debug guesses available for the current upload.' }])
      return
    }
    const lines = cands.map((c) => `- ${c.label} (${Math.round((c.score || 0) * 100)}%)`).join('\n')
    const uncLine = (unc && (typeof unc.margin === 'number' || typeof unc.entropy_norm === 'number'))
      ? `\n\nUncertainty: margin=${typeof unc.margin === 'number' ? unc.margin.toFixed(4) : 'n/a'}, entropy=${typeof unc.entropy_norm === 'number' ? unc.entropy_norm.toFixed(2) : 'n/a'}`
      : ''
    setMessages((prev) => [
      ...prev,
      { id: `a_dbg_${Date.now()}`, role: 'assistant', content: `Debug classification guesses:\n${lines}${uncLine}` },
    ])
  }, [context?.sceneCandidates, context?.sceneUncertainty])

  const send = useCallback(async () => {
    const content = text.trim()
    if (!content || sending) return

    const wantsGuesses = /\b(show guesses|debug classification)\b/i.test(content)
    setSending(true)
    setMenu(false)
    setText('')

    const userMsg: ChatMsg = { id: `u_${Date.now()}`, role: 'user', content }
    setMessages((prev) => [...prev, userMsg])

    if (wantsGuesses) {
      appendGuessDebug()
    }

    try {
      const compactPerceptionV1 = (() => {
        try {
          const pv1 = (context as any)?.perception_v1
          if (!pv1 || typeof pv1 !== 'object') return null

          const existing = (pv1 as any).page_text
          const pageText: Record<string, string> =
            (existing && typeof existing === 'object' && !Array.isArray(existing))
              ? { ...existing }
              : {}

          if (!Object.keys(pageText).length) {
            const blocks: any[] = Array.isArray((pv1 as any).text_blocks) ? (pv1 as any).text_blocks : []
            for (const b of blocks) {
              const pn = Number((b as any)?.page)
              const t = String((b as any)?.text || '').trim()
              if (!pn || !t) continue
              const k = String(pn)
              if (!pageText[k]) pageText[k] = ''
              pageText[k] = (pageText[k] + ' ' + t).replace(/\s+/g, ' ').trim()
              if (pageText[k].length > 12000) pageText[k] = pageText[k].slice(0, 12000)
            }
          }

          // clamp any existing page_text values
          for (const k of Object.keys(pageText)) {
            const v = String((pageText as any)[k] || '').replace(/\s+/g, ' ').trim()
            ;(pageText as any)[k] = v.length > 12000 ? v.slice(0, 12000) : v
          }
          return {
            image_metadata: (pv1 as any).image_metadata || null,
            validation: (pv1 as any).validation || null,
            page_text: pageText,
          }
        } catch {
          return null
        }
      })()

      const ctxForChat = {
        ...(context || {}),
        // prevent huge request bodies
        perception_v1: compactPerceptionV1,
      }

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          messages: [...messagesRef.current, userMsg].map((m) => ({ role: m.role, content: m.content })),
          context: ctxForChat,
        })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const reply = String(data?.reply || data?.message || '').trim() || 'No reply.'
      const a: ChatMsg = { id: String(data?.id || `a_${Date.now()}`), role: 'assistant', content: reply }
      setMessages((prev) => [...prev, a])
    } catch (e: any) {
      const a: ChatMsg = { id: `a_err_${Date.now()}`, role: 'assistant', content: `Chat failed: ${e?.message || 'network error'}` }
      setMessages((prev) => [...prev, a])
    } finally {
      setSending(false)
      setTimeout(() => {
        const el = listRef.current
        if (el) el.scrollTop = el.scrollHeight
      }, 0)
    }
  }, [text, sending, context])

  useEffect(() => {
    const sig = JSON.stringify({
      a: context?.artifact || null,
      l: context?.detectedLabel || null,
      p: context?.detectedProb ?? null,
      c: context?.sceneCandidates || null,
      u: context?.sceneUncertainty || null,
      j: context?.judgement || null,
      e: context?.engineering || null,
      ed: context?.engineeringDom || null,
      eg: (context as any)?.engineeringDrawing || null,
      k: context?.contacts || null,
      d: context?.dimensions || null,
      o: context?.objects || null,
      os: context?.ocrSummary || null,
    })
    if (!sig || sig === lastContextSigRef.current) return
    lastContextSigRef.current = sig
    const kind = String(context?.judgement?.kind || '').trim().toLowerCase()
    const pct = typeof context?.judgement?.confidence === 'number'
      ? ` (${Math.round(context.judgement.confidence * 100)}%)`
      : (typeof context?.detectedProb === 'number' ? ` (${Math.round(context.detectedProb * 100)}%)` : '')

    const c = context?.contacts || null
    const emails = Array.isArray(c?.emails) ? c!.emails!.filter(Boolean).slice(0, 4) : []
    const phones = Array.isArray(c?.phones) ? c!.phones!.filter(Boolean).slice(0, 4) : []
    const urls = Array.isArray(c?.urls) ? c!.urls!.filter(Boolean).slice(0, 4) : []
    const addr = Array.isArray(c?.address_like) ? c!.address_like!.filter(Boolean).slice(0, 1) : []
    const hasContactFields = !!(emails.length || phones.length || urls.length || addr.length)

    const showTop = typeof context?.detectedProb === 'number' && context.detectedProb < 0.2
    const cands = showTop && Array.isArray(context?.sceneCandidates) ? context!.sceneCandidates!.slice(0, 5) : []
    const topK = cands.length
      ? `\n\nTop guesses:\n${cands.map((cc) => `- ${cc.label} (${Math.round((cc.score || 0) * 100)}%)`).join('\n')}`
      : ''
    const unc = showTop ? context?.sceneUncertainty : null
    const uncLine = (unc && (typeof unc.margin === 'number' || typeof unc.entropy_norm === 'number'))
      ? `\n\nUncertainty: margin=${typeof unc.margin === 'number' ? unc.margin.toFixed(4) : 'n/a'}, entropy=${typeof unc.entropy_norm === 'number' ? unc.entropy_norm.toFixed(2) : 'n/a'}`
      : ''

    let summary = ''
    const backendSummary = String(context?.judgement?.summary || '').trim()
    if (backendSummary) {
      summary = backendSummary
    } else if (kind === 'business_card' || (hasContactFields && !kind)) {
      const lines: string[] = ['I can see you uploaded a visiting card. Here is the information I found:']
      if (emails.length) lines.push(`\nEmail:\n${emails.map((e) => `- ${e}`).join('\n')}`)
      if (phones.length) lines.push(`\nPhone:\n${phones.map((p) => `- ${p}`).join('\n')}`)
      if (urls.length) lines.push(`\nWebsite:\n${urls.map((u) => `- ${u}`).join('\n')}`)
      if (addr.length) lines.push(`\nAddress (best guess):\n- ${addr[0]}`)
      lines.push(`\nYou can ask: "what's the email?", "what's the phone number?", or "what's written on page 2".`)
      summary = lines.join('\n')
    } else if (kind === 'engineering_drawing') {
      // Never let generic visual labels (e.g., YOLO/CLIP "clock") override engineering-drawing UX.
      const ed = (context as any)?.engineeringDrawing || null
      const dom = (context as any)?.engineeringDom || null

      const lines: string[] = ['I can see you uploaded an engineering drawing.']

      const borderConf = ed?.confidence?.layers?.sheet
      const globalConf = ed?.confidence?.global
      if (typeof globalConf === 'number') lines.push(`\nGlobal confidence: ${Math.round(globalConf * 100)}%`)
      if (typeof borderConf === 'number') lines.push(`Border confidence: ${Math.round(borderConf * 100)}%`)

      const regions = Array.isArray(ed?.regions) ? ed.regions : (Array.isArray(dom?.regions) ? dom.regions : [])
      const views = Array.isArray(ed?.views) ? ed.views : (Array.isArray(dom?.views) ? dom.views : [])
      if (regions.length) {
        const byType: Record<string, number> = {}
        regions.forEach((r: any) => {
          const t = String(r?.type || r?.region_type || 'UNKNOWN')
          byType[t] = (byType[t] || 0) + 1
        })
        const breakdown = Object.keys(byType).sort().map((k) => `${k}:${byType[k]}`).join(', ')
        lines.push(`\nRegions: ${regions.length}${breakdown ? ` (${breakdown})` : ''}`)
      }
      if (views.length) lines.push(`Views (leaf): ${views.length}`)

      const vdbg = (ed?.diagnostics?.viewport_child_v2) || (dom?.diagnostics?.viewport_child_v2) || null
      if (Array.isArray(vdbg) && vdbg.length) {
        const parts = vdbg.slice(0, 4).map((x: any) => {
          const p = String(x?.parent || '')
          const n = typeof x?.children === 'number' ? x.children : null
          const st = String(x?.status || '')
          return `${p}:${n ?? 'n/a'}${st ? ` (${st})` : ''}`
        })
        if (parts.length) lines.push(`Viewport child split: ${parts.join('; ')}`)
      }

      const meta = ed?.metadata || null
      const scale = String(meta?.scale || meta?.fields?.scale || '').trim()
      const material = String(meta?.material || meta?.fields?.material || '').trim()
      const rev = String(meta?.revision || meta?.fields?.revision || meta?.fields?.rev || '').trim()
      if (scale || material || rev) {
        const m: string[] = []
        if (scale) m.push(`Scale=${scale}`)
        if (material) m.push(`Material=${material}`)
        if (rev) m.push(`Rev=${rev}`)
        lines.push(`\nTitle block: ${m.join(', ')}`)
      }

      lines.push(`\nYou can ask: "how many views?", "show regions", or "list dimension texts".`)
      summary = lines.join('\n')
    } else if (context?.detectedLabel) {
      summary = `Hi, I can see that you have uploaded: ${context.detectedLabel}${pct}.${topK}${uncLine}\n\nNext, tell me what you want to know about it.`
    } else if (context?.ocrSummary) {
      const s = String(context.ocrSummary || '').trim().slice(0, 220)
      summary = `I found readable text in the upload${pct}.\n\nPreview:\n${s}${String(context.ocrSummary || '').length > 220 ? '…' : ''}\n\nAsk me: "what's written" or "extract email/phone".`
    }

    if (!summary) return

    const nextActions: string[] = []
    const dimsCount = Array.isArray(context?.dimensions) ? context!.dimensions!.length : 0
    const objs = Array.isArray(context?.objects) ? context!.objects! : []
    const hasPages = Array.isArray(context?.ocrPages) ? context!.ocrPages!.length > 0 : false
    const hasText = !!String(context?.ocrFullText || context?.ocrSummary || '').trim()
    const hasContacts = !!(context?.contacts && (
      (Array.isArray(context.contacts.emails) && context.contacts.emails.length) ||
      (Array.isArray(context.contacts.phones) && context.contacts.phones.length) ||
      (Array.isArray(context.contacts.urls) && context.contacts.urls.length)
    ))

    if (kind === 'engineering_drawing') {
      if (dimsCount) nextActions.push('List detected dimensions/tolerances')
      nextActions.push('Ask what the part resembles (based on text + shapes)')
      if (hasText) nextActions.push('Extract key notes/annotations from the drawing')
      if (hasPages) nextActions.push('Ask: what\'s written on page 2')
    } else if (kind === 'business_card') {
      if (hasContacts) nextActions.push('Extract email/phone/website/address')
      nextActions.push('Ask: summary of the visiting card')
    } else if (kind === 'mixed_pdf' || kind === 'text_document' || kind === 'scanned_document' || kind === 'pdf' || kind === 'document' || kind === 'document_with_illustration') {
      if (hasPages) nextActions.push('Ask: what\'s written on page 2')
      if (hasText) nextActions.push('Summarize the document')
      if (objs.length) nextActions.push('List detected objects/illustrations in the document')
    } else {
      if (objs.length) nextActions.push('Identify the main object(s) in the image')
      if (hasText) nextActions.push('Read the text in the image')
      nextActions.push('Ask for design feedback / improvement suggestions')
    }

    const nextBlock = nextActions.length
      ? `\n\nRecommended next actions:\n${nextActions.slice(0, 5).map((a) => `- ${a}`).join('\n')}`
      : ''

    setMessages((prev) => [
      ...prev,
      {
        id: `a_ctx_${Date.now()}`,
        role: 'assistant',
        content: `${summary}${nextBlock}`,
      }
    ])
    setTimeout(() => {
      const el = listRef.current
      if (el) el.scrollTop = el.scrollHeight
    }, 0)
  }, [
    context?.artifact,
    context?.detectedLabel,
    context?.detectedProb,
    context?.sceneCandidates,
    context?.sceneUncertainty,
    context?.ocrSummary,
    context?.ocrFullText,
    context?.ocrPages,
    context?.judgement,
    context?.engineeringDom,
    context?.engineeringDrawing,
    context?.contacts,
    context?.dimensions,
    context?.objects,
  ])

  return (
    <div className="border rounded-lg p-3 h-full flex flex-col" style={{background:'var(--panel)', borderColor:'var(--border)', color:'var(--text)'}}>
      <div ref={listRef} className="flex-1 overflow-auto text-sm space-y-2">
        {messages.map((m) => (
          <div key={m.id} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
            <div
              className="max-w-[92%] rounded-xl px-3 py-2 border"
              style={{
                background: m.role === 'user' ? 'var(--panel-2)' : 'transparent',
                borderColor: 'var(--border)',
                color: 'var(--text)'
              }}
            >
              <div className="whitespace-pre-wrap">{m.content}</div>
            </div>
          </div>
        ))}
        {sending && (
          <div className="text-xs" style={{color:'var(--muted)'}}>Thinking…</div>
        )}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <button className="px-2 py-1 border rounded" onClick={() => setMenu(v => !v)} aria-label="Attach">📎</button>
        {menu && (
          <div className="relative">
            <div className="absolute z-10 border rounded shadow p-2 flex gap-2" style={{background:'var(--panel-2)', borderColor:'var(--border)'}}>
              <button className="px-2 py-1 border rounded text-xs" onClick={() => { onAttach('sketch'); setMenu(false) }}>Mock Sketch</button>
              <button className="px-2 py-1 border rounded text-xs" onClick={() => { onAttach('cad'); setMenu(false) }}>Mock CAD</button>
            </div>
          </div>
        )}
        <input
          className="flex-1 border rounded px-2 py-1 text-sm"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          placeholder="Message"
          style={{background:'var(--panel-2)', borderColor:'var(--border)', color:'var(--text)'}}
          disabled={sending}
        />
        <button className="px-3 py-1 border rounded text-sm" onClick={send} disabled={!canSend}>
          {sending ? '...' : 'Send'}
        </button>
      </div>
    </div>
  )
}
