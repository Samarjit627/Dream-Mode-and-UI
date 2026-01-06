import React from 'react'

type Props = {
  artifact: 'none'|'sketch'|'cad'
  onAction: () => void
}

export default function BestNextStep({ artifact, onAction }: Props) {
  const text = artifact === 'none'
    ? 'Start with Intent'
    : artifact === 'sketch' ? 'Analyze the Sketch' : 'Mentor on the CAD'
  return (
    <div>
      <div className="mx-auto max-w-6xl px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-xl px-3 py-1.5 text-xs" style={{background:'rgba(16,185,129,.1)', color:'var(--accent)', border:'1px solid rgba(16,185,129,.3)'}}>
            <b>Best Next Step:</b> {text}
          </div>
        </div>
        <button onClick={onAction} className="px-2.5 py-1 rounded-lg text-xs" style={{background:'var(--accent)', color:'#fff'}}>Do it</button>
      </div>
    </div>
  )
}
