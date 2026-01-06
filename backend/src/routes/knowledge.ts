import { Router, Request, Response, NextFunction } from 'express'
import { maybeProxyGet } from '../services/proxy.js'
import fixture from '../fixtures/knowledge_cards.js'

const r = Router()

r.get('/knowledge/cards', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const out = await maybeProxyGet('/knowledge/cards')
    if (out) return res.json(out)
    return res.json(fixture)
  } catch (e) { next(e) }
})

export default r
