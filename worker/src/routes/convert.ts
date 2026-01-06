import { Router } from 'express'

const r = Router()

r.post('/convert/step-to-glb', async (req, res) => {
  // Mock conversion response
  res.json({ glbUrl: 'https://example.com/placeholder.glb', note: 'mocked by worker stub' })
})

r.post('/trial/preview', async (req, res) => {
  // Mock preview response
  res.json({ glbUrl: 'https://example.com/placeholder.glb', note: 'mocked by worker stub' })
})

export default r
