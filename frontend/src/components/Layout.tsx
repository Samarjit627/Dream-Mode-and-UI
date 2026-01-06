import React from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import ThemeToggle from './ThemeToggle'

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-200 dark:border-slate-700">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <nav className="flex gap-4 text-sm">
            <NavLink to="/" className={({isActive}) => isActive ? 'font-semibold' : ''}>Home</NavLink>
            <NavLink to="/dream" className={({isActive}) => isActive ? 'font-semibold' : ''}>Dream</NavLink>
            <NavLink to="/build" className={({isActive}) => isActive ? 'font-semibold' : ''}>Build</NavLink>
            <NavLink to="/scale" className={({isActive}) => isActive ? 'font-semibold' : ''}>Scale</NavLink>
          </nav>
          <ThemeToggle />
        </div>
      </header>
      <main className="flex-1 mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
