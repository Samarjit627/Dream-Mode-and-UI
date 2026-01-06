#!/usr/bin/env tsx
import * as path from 'node:path'

// Simple path linter: accepts --files comma-separated and checks that dream/build/scale UI live under frontend/src/features
// For full enforcement, integrate with planPath and CI later.

function parseFlags(argv: string[]) {
  const f: any = {}
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) {
      const k = a.slice(2)
      const v = argv[i+1] && !argv[i+1].startsWith('--') ? argv[++i] : 'true'
      f[k] = v
    }
  }
  return f as { files?: string }
}

function lintFile(rel: string): string | null {
  const p = rel.replaceAll('\\', '/')
  // Allow simple layout
  if (p.startsWith('frontend/src/features/')) return null
  if (p.startsWith('frontend/src/shared/')) return null
  if (p.startsWith('backend/src/')) return null
  if (p.startsWith('worker/')) return null
  // Allow packages layout
  if (p.startsWith('packages/frontend/src/features/')) return null
  if (p.startsWith('packages/frontend/src/shared/')) return null
  if (p.startsWith('packages/gateway/src/')) return null
  if (p.startsWith('packages/worker/')) return null
  if (p.startsWith('docs/')) return null
  if (p.startsWith('scripts/')) return null
  // Allow root files like package.json, tsconfig.json
  const rootAllowed = ['package.json','pnpm-workspace.yaml','tsconfig.json','README.md']
  if (rootAllowed.includes(p)) return null
  return `Path policy violation: ${p} \nAllowed roots include: frontend/src/features, frontend/src/shared, backend/src, worker/, docs/, scripts/, packages/*`
}

async function main() {
  const { files } = parseFlags(process.argv.slice(2))
  if (!files) {
    console.log('No files specified; pass --files <comma-separated-relative-paths>')
    return
  }
  const rels = files.split(',').map(s => s.trim()).filter(Boolean)
  const errors: string[] = []
  for (const rel of rels) {
    const err = lintFile(rel)
    if (err) errors.push(err)
  }
  if (errors.length) {
    console.error(errors.join('\n'))
    process.exit(2)
  } else {
    console.log('Path lint passed')
  }
}

main().catch(e => { console.error(e); process.exit(1) })
