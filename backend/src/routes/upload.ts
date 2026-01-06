import { Router, Request, Response, NextFunction } from 'express'

const r = Router()

r.post('/upload', async (req: Request, res: Response, next: NextFunction) => {
  try {
    // Stub: return a fixed GLB preview URL
    return res.json({ id: 'stub-1', previewUrl: 'https://modelviewer.dev/shared-assets/models/Astronaut.glb' })
  } catch (e) { next(e) }
})

export default r
