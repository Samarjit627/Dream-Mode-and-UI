import React, { useEffect, useRef, useState } from 'react'

type Box = { label: string; score: number; bbox: [number, number, number, number] }
type Word = { text: string; bbox: [number, number, number, number] }

export default function SketchCanvas({ url, boxes = [], words = [] }: { url: string; boxes?: Box[]; words?: Word[] }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [beltY, setBeltY] = useState(0.6)
  const [showCenter, setShowCenter] = useState(true)
  const [showBand, setShowBand] = useState(true)

  useEffect(() => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const wrap = wrapRef.current!
      const canvas = canvasRef.current!
      const maxW = wrap.clientWidth || 600
      const maxH = wrap.clientHeight || 400
      // fit image preserving aspect
      let w = img.width
      let h = img.height
      const scale = Math.min(maxW / w, maxH / h)
      w = Math.max(1, Math.floor(w * scale))
      h = Math.max(1, Math.floor(h * scale))
      canvas.width = w
      canvas.height = h
      canvas.style.width = w + 'px'
      canvas.style.height = h + 'px'
      draw(img)
    }
    img.src = url

    const ro = new ResizeObserver(() => {
      // re-trigger image load sizing on resize
      if (img.complete) {
        const wrap = wrapRef.current!
        const canvas = canvasRef.current!
        const maxW = wrap.clientWidth || 600
        const maxH = wrap.clientHeight || 400
        let w = img.width
        let h = img.height
        const scale = Math.min(maxW / w, maxH / h)
        w = Math.max(1, Math.floor(w * scale))
        h = Math.max(1, Math.floor(h * scale))
        canvas.width = w
        canvas.height = h
        canvas.style.width = w + 'px'
        canvas.style.height = h + 'px'
        draw(img)
      }
    })
    ro.observe(wrapRef.current!)

    const onOverlay = (e: any) => {
      setShowCenter(!!e?.detail?.centerline || false)
      setShowBand(!!e?.detail?.band || false)
      if (img.complete) draw(img)
    }
    const onBelt = (e: any) => {
      const yn = Math.min(1, Math.max(0, e?.detail?.y ?? 0.6))
      setBeltY(yn)
      if (img.complete) draw(img)
    }
    window.addEventListener('axis5:overlay:set', onOverlay as any)
    window.addEventListener('axis5:beltline:set', onBelt as any)

    return () => {
      window.removeEventListener('axis5:overlay:set', onOverlay as any)
      window.removeEventListener('axis5:beltline:set', onBelt as any)
      ro.disconnect()
    }
  }, [url])

  const draw = (img: HTMLImageElement) => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    // clear
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    // draw image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    // overlays
    // overlays
    // (Disabled per user request: No green/red lines)
    if (false) {
      // logic removed
    }
    // detection boxes
    if (boxes && boxes.length) {
      ctx.save()
      ctx.lineWidth = 2
      ctx.strokeStyle = '#22c55e'
      ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto'
      ctx.fillStyle = 'rgba(34,197,94,0.8)'
      boxes.slice(0, 12).forEach(b => {
        const [nx, ny, nw, nh] = b.bbox
        const x = Math.round(nx * canvas.width)
        const y = Math.round(ny * canvas.height)
        const w = Math.round(nw * canvas.width)
        const h = Math.round(nh * canvas.height)
        ctx.strokeRect(x, y, w, h)
        const label = `${b.label} ${(b.score * 100 | 0)}%`
        const tw = Math.ceil(ctx.measureText(label).width + 6)
        const th = 16
        ctx.fillRect(x, Math.max(0, y - th), tw, th)
        ctx.fillStyle = 'white'
        ctx.fillText(label, x + 3, Math.max(12, y - 4))
        ctx.fillStyle = 'rgba(34,197,94,0.8)'
      })
      ctx.restore()
    }
    // OCR words
    if (words && words.length) {
      ctx.save()
      ctx.strokeStyle = 'rgba(59,130,246,0.9)'
      ctx.lineWidth = 1
      ctx.font = '11px system-ui, -apple-system, Segoe UI, Roboto'
      ctx.fillStyle = 'rgba(59,130,246,0.85)'
      words.slice(0, 40).forEach(wd => {
        const [nx, ny, nw, nh] = wd.bbox
        const x = Math.round(nx * canvas.width)
        const y = Math.round(ny * canvas.height)
        const w = Math.round(nw * canvas.width)
        const h = Math.round(nh * canvas.height)
        ctx.strokeRect(x, y, w, h)
        const label = wd.text
        const tw = Math.ceil(ctx.measureText(label).width + 6)
        const th = 14
        ctx.fillRect(x, Math.max(0, y - th), tw, th)
        ctx.fillStyle = 'white'
        ctx.fillText(label, x + 3, Math.max(10, y - 4))
        ctx.fillStyle = 'rgba(59,130,246,0.85)'
      })
      ctx.restore()
    }
  }

  return (
    <div ref={wrapRef} style={{ width: '100%', height: '100%', display: 'grid', placeItems: 'center' }}>
      <canvas ref={canvasRef} style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: 12 }} />
    </div>
  )
}
