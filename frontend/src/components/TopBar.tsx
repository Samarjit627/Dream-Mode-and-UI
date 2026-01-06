import React from 'react'

type Props = { children?: React.ReactNode }
export default function TopBar({ children }: Props) {
  return (
    <header className="border-b border-slate-200 dark:border-slate-700">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
        {children}
      </div>
    </header>
  )
}
