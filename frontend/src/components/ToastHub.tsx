import React, { useEffect, useState } from 'react'

export default function ToastHub() {
  const [msg, setMsg] = useState<string | null>(null)
  useEffect(() => {
    const onToast = (e: any) => {
      const text = typeof e?.detail === 'string' ? e.detail : (e?.detail?.message || 'Done')
      setMsg(text)
      const t = setTimeout(() => setMsg(null), 2400)
      return () => clearTimeout(t)
    }
    window.addEventListener('axis5:toast', onToast as any)
    return () => window.removeEventListener('axis5:toast', onToast as any)
  }, [])
  if (!msg) return null
  return (
    <div style={{ position: 'fixed', top: 14, right: 16, zIndex: 50 }}>
      <div className="border rounded-lg px-3 py-2 text-sm shadow" style={{ background: 'var(--panel-2)', borderColor: 'var(--border)', color: 'var(--text)' }}>
        {msg}
      </div>
    </div>
  )
}
