import { Router, Request, Response, NextFunction } from 'express'
import { maybeProxyPost } from '../services/proxy.js'
import fixture from '../fixtures/ideate_generate4.js'

const r = Router()

r.post('/ideate/generate4', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const out = await maybeProxyPost('/ideate/generate4', req.body)
    if (out) return res.json(out)
    return res.json(fixture)
  } catch (e) { next(e) }
})

export default r
