import React from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './styles/index.css'
import './styles/theme.css'
import Shell from './pages/Shell'
import DreamAnalyze from './pages/DreamAnalyze'
import DreamIdeate from './pages/DreamIdeate'
import DreamMentor from './pages/DreamMentor'
import Home from './pages/Home'

const router = createBrowserRouter([
  { path: '/', element: <Home /> },
  {
    path: '/dream',
    element: <Shell />,
    children: [
      { path: 'analyze', element: <DreamAnalyze /> },
      { path: 'ideate', element: <DreamIdeate /> },
      { path: 'mentor', element: <DreamMentor /> },
    ]
  }
])

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)
