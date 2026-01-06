import { Router, Request, Response, NextFunction } from 'express'
import { maybeProxyPost } from '../services/proxy.js'
import sketchFixture from '../fixtures/analyze_sketch.js'
import cadFixture from '../fixtures/analyze_cad.js'

const r = Router()

r.post('/analyze/sketch', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const out = await maybeProxyPost('/analyze/sketch', req.body)
    if (out) return res.json(out)
    return res.json(sketchFixture)
  } catch (e) { next(e) }
})

r.post('/analyze/cad', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const out = await maybeProxyPost('/analyze/cad', req.body)
    if (out) return res.json(out)
    return res.json(cadFixture)
  } catch (e) { next(e) }
})

export default r
