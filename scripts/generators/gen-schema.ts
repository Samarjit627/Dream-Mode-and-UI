#!/usr/bin/env tsx
import { promises as fs } from 'node:fs'
import * as path from 'node:path'
import { spawn } from 'node:child_process'

type Flags = { name: string; type?: 'Overlay'|'StylePack'|'KnowledgeCard'|'TrialPreview'|'Session'; noTypes?: boolean; noPyd?: boolean }

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

function schemaBoilerplate(name: string, type?: Flags['type']) {
  const title = type ?? name
  return JSON.stringify({
    $schema: 'https://json-schema.org/draft/2020-12/schema',
    $id: `https://axis5.local/schemas/${title}.json`,
    title,
    type: 'object',
    properties: {},
    additionalProperties: true
  }, null, 2) + '\n'
}

async function main() {
  const { name, type, noTypes, noPyd } = parseFlags(process.argv.slice(2))
  const file = path.join('docs','SCHEMAS', `${name}.json`)
  await writeIfMissing(file, schemaBoilerplate(name, type))
  const run = (cmd: string, args: string[]) => new Promise<void>((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: 'inherit' })
    p.on('close', (code) => code === 0 ? resolve() : reject(new Error(`${cmd} ${args.join(' ')} exited ${code}`)))
  })
  if (!noTypes) {
    try { await run('npx', ['tsx', 'scripts/generators/gen-types.ts']) } catch (e) { console.warn('[gen-schema] types generation skipped:', e instanceof Error ? e.message : e) }
  }
  if (!noPyd) {
    try { await run('npx', ['tsx', 'scripts/generators/gen-pydantic.ts']) } catch (e) { console.warn('[gen-schema] pydantic generation skipped:', e instanceof Error ? e.message : e) }
  }
}

main().catch((e) => { console.error(e); process.exit(1) })
