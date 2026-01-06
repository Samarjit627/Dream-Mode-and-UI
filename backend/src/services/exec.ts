import { spawn, SpawnOptions } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import type { Response } from 'express'

export type ExecMode = 'argv' | 'shell'

export interface ExecRequestArgv {
  mode: 'argv'
  command: string
  args?: string[]
  cwd?: string
  env?: Record<string, string>
  timeoutMs?: number
}

export interface ExecRequestShell {
  mode: 'shell'
  script: string
  cwd?: string
  env?: Record<string, string>
  timeoutMs?: number
}

export type ExecRequest = ExecRequestArgv | ExecRequestShell

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'timeout'

export interface LogLine {
  ts: string
  level: 'info' | 'error'
  jobId: string
  pid?: number
  stream: 'stdout' | 'stderr'
  line: string
}

export interface ExecManifest {
  jobId: string
  mode: ExecMode
  argv?: { command: string; args: string[] }
  script?: string
  resolved?: { cwd?: string }
  envKeys?: string[]
  startTs: string
  endTs?: string
  exitCode?: number | null
  signal?: NodeJS.Signals | null
}

export interface Job {
  jobId: string
  status: JobStatus
  manifest: ExecManifest
  logs: LogLine[]
  subscribers: Set<Response>
  timeout?: NodeJS.Timeout
}

class ExecManager {
  private jobs = new Map<string, Job>()

  createJob(req: ExecRequest): Job {
    const jobId = randomUUID()
    const manifest: ExecManifest = {
      jobId,
      mode: req.mode,
      argv: req.mode === 'argv' ? { command: req.command, args: req.args ?? [] } : undefined,
      script: req.mode === 'shell' ? req.script : undefined,
      resolved: { cwd: req.cwd },
      envKeys: req.env ? Object.keys(req.env) : [],
      startTs: new Date().toISOString(),
    }
    const job: Job = { jobId, status: 'queued', manifest, logs: [], subscribers: new Set() }
    this.jobs.set(jobId, job)
    return job
  }

  get(jobId: string): Job | undefined {
    return this.jobs.get(jobId)
  }

  subscribe(jobId: string, res: Response) {
    const job = this.get(jobId)
    if (!job) return
    job.subscribers.add(res)
    // Send backlog
    for (const l of job.logs) this.sendSSE(res, l)
    res.on('close', () => {
      job.subscribers.delete(res)
    })
  }

  private broadcast(job: Job, line: LogLine) {
    for (const res of job.subscribers) this.sendSSE(res, line)
  }

  private sendSSE(res: Response, data: any) {
    res.write(`data: ${JSON.stringify(data)}\n\n`)
  }

  private pushLog(job: Job, line: LogLine) {
    job.logs.push(line)
    this.broadcast(job, line)
  }

  async run(job: Job, req: ExecRequest) {
    job.status = 'running'
    const opts: SpawnOptions = {
      shell: false,
      cwd: req.cwd || process.cwd(),
      env: req.env ? { ...process.env, ...req.env } : process.env,
    }

    let child: ReturnType<typeof spawn>
    if (req.mode === 'argv') {
      if (!req.command) throw new Error('command required')
      child = spawn(req.command, req.args ?? [], opts)
    } else {
      const script = req.script || ''
      // Enforce no-skip guarantees in shell mode
      const bashCmd = `set -euo pipefail; set -x; ${script}`
      child = spawn('/bin/bash', ['-c', bashCmd], opts)
    }

    const pid = child.pid

    const onStdout = (chunk: Buffer) => {
      const lines = chunk.toString().split(/\r?\n/)
      for (const line of lines) if (line.length) this.pushLog(job, { ts: new Date().toISOString(), level: 'info', jobId: job.jobId, pid, stream: 'stdout', line })
    }
    const onStderr = (chunk: Buffer) => {
      const lines = chunk.toString().split(/\r?\n/)
      for (const line of lines) if (line.length) this.pushLog(job, { ts: new Date().toISOString(), level: 'error', jobId: job.jobId, pid, stream: 'stderr', line })
    }

    child.stdout?.on('data', onStdout)
    child.stderr?.on('data', onStderr)

    const timeoutMs = req.timeoutMs && req.timeoutMs > 0 ? req.timeoutMs : undefined
    if (timeoutMs) {
      job.timeout = setTimeout(() => {
        this.pushLog(job, { ts: new Date().toISOString(), level: 'error', jobId: job.jobId, pid, stream: 'stderr', line: `timeout after ${timeoutMs}ms` })
        job.status = 'timeout'
        try { child.kill('SIGKILL') } catch {}
      }, timeoutMs)
    }

    child.on('close', (code, signal) => {
      if (job.timeout) clearTimeout(job.timeout)
      job.manifest.endTs = new Date().toISOString()
      job.manifest.exitCode = code
      job.manifest.signal = signal
      if (job.status !== 'timeout') job.status = code === 0 ? 'succeeded' : 'failed'
      // final event
      this.pushLog(job, { ts: new Date().toISOString(), level: 'info', jobId: job.jobId, pid, stream: 'stdout', line: `process exited code=${code} signal=${signal ?? 'null'}` })
    })

    child.on('error', (err) => {
      this.pushLog(job, { ts: new Date().toISOString(), level: 'error', jobId: job.jobId, pid, stream: 'stderr', line: `spawn error: ${String(err)}` })
      job.status = 'failed'
    })
  }
}

export const execManager = new ExecManager()
