#!/usr/bin/env tsx
import { promises as fs } from 'node:fs'
import * as path from 'node:path'
import { planPath } from '../planPath.js'

const caps = ['analyze','ideate','mentor','taste','knowledge','upload'] as const

type Cap = typeof caps[number]

type Flags = { cap: Cap; name: string; parts?: string }

function parseFlags(argv: string[]): Flags {
  const f: any = {}
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) {
      const k = a.slice(2)
      const v = argv[i+1] && !argv[i+1].startsWith('--') ? argv[++i] : 'true'
      f[k] = v
    }
  }
  if (!caps.includes(f.cap)) throw new Error(`--cap must be one of ${caps.join(', ')}`)
  if (!f.name) throw new Error('--name is required')
  return f as Flags
}

async function ensureDir(p: string) {
  await fs.mkdir(p, { recursive: true })
}

async function writeIfMissing(file: string, content: string) {
  try {
    await fs.access(file)
    console.log(`[skip] exists: ${file}`)
  } catch {
    await ensureDir(path.dirname(file))
    await fs.writeFile(file, content, 'utf8')
    console.log(`[create] ${file}`)
  }
}

function controllerBoilerplate(cap: string, name: string) {
  return `import { Router, Request, Response, NextFunction } from 'express'
const r = Router()
r.get('/${cap}/${name}', (req: Request, res: Response, next: NextFunction) => { res.json({ ok: true }) })
export default r
`
}

function serviceBoilerplate(name: string) {
  const fn = name.charAt(0).toLowerCase() + name.slice(1)
  return `export async function ${fn}() { return null }
`
}

function validatorBoilerplate(name: string) {
  return `export const ${name}Schema = {}
`
}

async function main() {
  const { cap, name, parts } = parseFlags(process.argv.slice(2))
  const wants = (parts ? parts.split(',') : ['controller','service','validator']) as Array<'controller'|'service'|'validator'>

  const ctrlPath = planPath('gateway', 'shared', cap as any, 'controller', 'controller')
  const svcPath = planPath('gateway', 'shared', cap as any, 'service', 'service')
  const valPath = planPath('gateway', 'shared', cap as any, 'validator', 'validator')

  if (wants.includes('controller')) await writeIfMissing(ctrlPath, controllerBoilerplate(cap, name))
  if (wants.includes('service')) await writeIfMissing(svcPath, serviceBoilerplate(name))
  if (wants.includes('validator')) await writeIfMissing(valPath, validatorBoilerplate(name))
}

main().catch((e) => { console.error(e); process.exit(1) })
