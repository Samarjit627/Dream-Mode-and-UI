import { Router, Request, Response, NextFunction } from 'express'
import { maybeProxyPost } from '../services/proxy.js'
import fixture from '../fixtures/mentor_critique.js'

const r = Router()

r.post('/mentor/critique', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const out = await maybeProxyPost('/mentor/critique', req.body)
    if (out) return res.json(out)
    return res.json(fixture)
  } catch (e) { next(e) }
})

export default r
