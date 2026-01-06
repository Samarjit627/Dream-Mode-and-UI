#!/usr/bin/env tsx
import { promises as fs } from 'node:fs'
import * as path from 'node:path'

async function ensureDir(p: string) { await fs.mkdir(p, { recursive: true }) }

async function main() {
  const schemasDir = path.join('docs','SCHEMAS')
  const outDir = path.join('docs','SCHEMAS','pydantic')
  await ensureDir(outDir)
  const files = await fs.readdir(schemasDir)
  const jsons = files.filter(f => f.endsWith('.json'))
  for (const f of jsons) {
    const full = path.join(schemasDir, f)
    const name = path.basename(f, '.json')
    const raw = await fs.readFile(full, 'utf8')
    const schema = JSON.parse(raw)
    const className = (schema.title as string) || name
    const props = (schema.properties && typeof schema.properties === 'object') ? schema.properties : {}
    const fields = Object.keys(props).map(k => `    ${k}: Any | None = None`).join('\n')
    const body = `from typing import Any, None
from pydantic import BaseModel

class ${className}(BaseModel):
${fields || '    pass'}
`
    const outFile = path.join(outDir, `${name}.py`)
    await fs.writeFile(outFile, body, 'utf8')
    console.log(`[pydantic] ${outFile}`)
  }
}

main().catch((e) => { console.error('[gen-pydantic] generation failed', e); process.exit(1) })
