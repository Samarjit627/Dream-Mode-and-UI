import { Router, Request, Response, NextFunction } from 'express'
import { execManager, ExecRequest } from '../services/exec.js'

const r = Router()

r.post('/exec/run', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const body = req.body as ExecRequest
    if (!body || (body.mode !== 'argv' && body.mode !== 'shell')) {
      return res.status(400).json({ error: 'Invalid body: mode must be argv|shell' })
    }
    const job = execManager.createJob(body)
    // fire and forget
    execManager.run(job, body).catch((e) => {
      job.status = 'failed'
      job.logs.push({ ts: new Date().toISOString(), level: 'error', jobId: job.jobId, stream: 'stderr', line: String(e) })
    })
    return res.json({ jobId: job.jobId })
  } catch (e) { next(e) }
})

r.get('/exec/jobs/:id', (req: Request, res: Response) => {
  const job = execManager.get(req.params.id)
  if (!job) return res.status(404).json({ error: 'job not found' })
  return res.json({ jobId: job.jobId, status: job.status, manifest: job.manifest })
})

r.get('/exec/jobs/:id/logs', (req: Request, res: Response) => {
  const job = execManager.get(req.params.id)
  if (!job) return res.status(404).json({ error: 'job not found' })
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders?.()
  res.write(': connected\n\n')
  execManager.subscribe(job.jobId, res)
})

export default r
