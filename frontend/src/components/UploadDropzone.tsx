import React, { useCallback, useRef } from 'react'

export default function UploadDropzone(){
  const inputRef = useRef<HTMLInputElement>(null)
  const onFiles = useCallback((files: FileList|null)=>{
    if(!files?.length) return
    const f = files[0]
    const ext = (f.name.split('.').pop() || '').toLowerCase()
    const isSketch = ['png','jpg','jpeg','webp'].includes(ext)
    const isCAD = ['step','stp','glb','obj'].includes(ext)
    window.dispatchEvent(new CustomEvent('axis5:artifact', { detail:{ kind: isSketch?'sketch': isCAD?'cad':'unknown', file:f } }))
  },[])

  return (
    <div
      onDragOver={(e)=>e.preventDefault()}
      onDrop={(e)=>{ e.preventDefault(); onFiles(e.dataTransfer?.files || null); }}
      onClick={()=>inputRef.current?.click()}
      className="rounded-xl border border-dashed px-6 py-10 text-center cursor-pointer"
      style={{borderColor:'var(--border)', background:'var(--panel-2)', color:'var(--text)'}}
    >
      <div>Drop file here</div>
      <div className="text-xs" style={{color:'var(--muted)'}}>or click to choose</div>
      <input ref={inputRef} type="file" hidden accept=".png,.jpg,.jpeg,.webp,.step,.stp,.glb,.obj" onChange={(e)=>onFiles(e.target.files)} />
    </div>
  )
}
