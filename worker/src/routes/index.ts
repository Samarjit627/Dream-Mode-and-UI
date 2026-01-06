import { Router } from 'express'
import convert from './convert.js'

const r = Router()

r.use(convert)

r.get('/health', (req, res) => res.json({ status: 'ok', service: 'axis5-worker', version: '0.1.0' }))

export default r
