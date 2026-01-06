#!/usr/bin/env tsx
import { compileFromFile } from 'json-schema-to-typescript'
import { promises as fs } from 'node:fs'
import * as path from 'node:path'

async function ensureDir(p: string) { await fs.mkdir(p, { recursive: true }) }

async function main() {
  const schemasDir = path.join('docs','SCHEMAS')
  const outDir = path.join('docs','SCHEMAS','types')
  await ensureDir(outDir)
  const files = await fs.readdir(schemasDir)
  const jsons = files.filter(f => f.endsWith('.json'))
  const outputs: string[] = []
  for (const f of jsons) {
    const full = path.join(schemasDir, f)
    const name = path.basename(f, '.json')
    const ts = await compileFromFile(full, { bannerComment: '' })
    const outFile = path.join(outDir, `${name}.d.ts`)
    await fs.writeFile(outFile, ts, 'utf8')
    outputs.push(outFile)
    console.log(`[types] ${outFile}`)
  }
  const index = outputs.map(p => `/// <reference path="./${path.basename(p)}" />`).join('\n') + '\n'
  await fs.writeFile(path.join(outDir, 'index.d.ts'), index, 'utf8')
}

main().catch((e) => { 
  console.error('[gen-types] Failed. Did you install json-schema-to-typescript? npm i -D json-schema-to-typescript')
  console.error(e); process.exit(1) 
})
