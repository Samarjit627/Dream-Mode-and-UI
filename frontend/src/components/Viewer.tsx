import React, { useCallback, useRef, useState } from 'react'
import ObjCanvas from './ObjCanvas'
import GltfCanvas from './GltfCanvas'
import SketchCanvas from './SketchCanvas'
import type { Detection, OcrWord } from '../lib/vision'

type Props = {
  glbUrl?: string | null
  sketchUrl?: string | null
  pdfUrl?: string | null
  objUrl?: string | null
  objFile?: File | null
  mtlFile?: File | null
  onDropFile?: (file: File) => void
  message?: string | null
  preferSketch?: boolean
  boxes?: Detection[]
  words?: OcrWord[]
  busy?: boolean
  busyLabel?: string
}

export default function Viewer({ glbUrl, sketchUrl, pdfUrl, objUrl, objFile, mtlFile, onDropFile, message, preferSketch, boxes, words, busy, busyLabel }: Props) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])
  const onDragLeave = useCallback(() => setDragOver(false), [])
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const files = e.dataTransfer.files
    if (files && files.length && onDropFile) {
      for (const f of Array.from(files)) onDropFile(f)
    }
  }, [onDropFile])

  const hasContent = !!(glbUrl || objUrl || sketchUrl || pdfUrl)

  return (
    <div onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop} onClick={!hasContent ? () => inputRef.current?.click() : undefined}
      className={`w-full h-full min-h-[300px] border rounded-3xl grid place-items-center ${dragOver ? 'ring-2 ring-emerald-500' : ''}`}
      style={{ background: 'var(--panel)', borderColor: 'var(--border)', color: 'var(--text)', cursor: 'pointer', position: 'relative' }}>
      {pdfUrl ? (
        <iframe title="pdf" src={pdfUrl} className="w-full h-full" style={{ border: 0 }} />
      ) : preferSketch && sketchUrl ? (
        <SketchCanvas url={sketchUrl} boxes={boxes} words={words} />
      ) : glbUrl ? (
        <GltfCanvas url={glbUrl} />
      ) : objUrl ? (
        <ObjCanvas url={objUrl} file={objFile || undefined} mtlFile={mtlFile || undefined} />
      ) : sketchUrl ? (
        <SketchCanvas url={sketchUrl} boxes={boxes} words={words} />
      ) : (
        <div className="text-center">
          <div>Viewer Placeholder</div>
          <div className="text-xs opacity-70">Drop an image (PNG/JPG) or CAD (STEP/GLB/OBJ) to preview</div>
        </div>
      )}
      {message && (
        <div className="text-xs" style={{position:'absolute', bottom: '8px', right: '12px', color: 'var(--muted)'}}>{message}</div>
      )}
      {hasContent && (
        <button onClick={() => inputRef.current?.click()} className="text-xs border px-2 py-1 rounded" style={{position:'absolute', top: 10, right: 12}}>Upload</button>
      )}
      {busy && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'grid',
            placeItems: 'center',
            background: 'rgba(0,0,0,0.35)',
            backdropFilter: 'blur(2px)',
          }}
        >
          <div className="flex flex-col items-center gap-2 text-sm" style={{ color: 'white' }}>
            <div className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
            <div>{busyLabel || 'Analyzing…'}</div>
          </div>
        </div>
      )}
      <input ref={inputRef} type="file" hidden multiple accept=".png,.jpg,.jpeg,.webp,.pdf,.step,.stp,.glb,.obj,.mtl" onChange={(e) => { const fl = e.target.files; if (fl && onDropFile) { for (const f of Array.from(fl)) onDropFile(f) } }} />
    </div>
  )
}
