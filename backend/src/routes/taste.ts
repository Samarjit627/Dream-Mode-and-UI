import { Router, Request, Response, NextFunction } from 'express'
import { maybeProxyGet } from '../services/proxy.js'
import fixture from '../fixtures/taste_packs.js'

const r = Router()

r.get('/taste/packs', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const out = await maybeProxyGet('/taste/packs')
    if (out) return res.json(out)
    return res.json(fixture)
  } catch (e) { next(e) }
})

export default r
