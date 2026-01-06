import React, { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import UploadDropzone from '../components/UploadDropzone'

export default function Home() {
  const navigate = useNavigate()
  useEffect(() => {
    const onArtifact = (e: any) => {
      const detail = e?.detail
      navigate('/dream')
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('axis5:artifact', { detail }))
      }, 150)
    }
    window.addEventListener('axis5:artifact', onArtifact as any)
    return () => window.removeEventListener('axis5:artifact', onArtifact as any)
  }, [navigate])
  return (
    <div className="max-w-5xl mx-auto px-6 py-16">
      <h1 className="text-2xl font-semibold" style={{color:'var(--text)'}}>
        Hi — I’m your Axis5 mentor. Upload a design to start.
      </h1>
      <p className="text-sm mt-2" style={{color:'var(--muted)'}}>
        I’ll analyze it, ideate 4 concept directions, and mentor like a senior designer.
      </p>

      <div className="mt-8 grid md:grid-cols-2 gap-6">
        <section className="rounded-2xl border" style={{background:'var(--panel)', borderColor:'var(--border)'}}>
          <header className="p-4 border-b" style={{borderColor:'var(--border)'}}>Upload a design</header>
          <div className="p-6">
            <UploadDropzone />
            <div className="text-xs mt-3" style={{color:'var(--muted)'}}>
              PNG/JPG (sketch) or STEP/GLB (CAD). Drag & drop or click.
            </div>
          </div>
        </section>

        <section className="rounded-2xl border flex flex-col" style={{background:'var(--panel)', borderColor:'var(--border)'}}>
          <header className="p-4 border-b" style={{borderColor:'var(--border)'}}>Or start with ideas</header>
          <div className="p-4 text-sm" style={{color:'var(--muted)'}}>
            Open the Intent panel to set goals, character, and constraints.
          </div>
          <div className="p-4 border-t" style={{borderColor:'var(--border)'}}>
            <button className="px-3 py-2 rounded-lg text-sm" style={{background:'var(--accent)'}} onClick={()=>window.dispatchEvent(new CustomEvent('axis5:intent:open'))}>
              Open Intent
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}
