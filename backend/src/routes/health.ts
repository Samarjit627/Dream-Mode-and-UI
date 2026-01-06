import { Router } from 'express'

const r = Router()

r.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'axis5-gateway', version: '0.1.0' })
})

export default r
