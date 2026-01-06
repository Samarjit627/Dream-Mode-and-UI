#!/usr/bin/env tsx
import { promises as fs } from 'node:fs'
import * as path from 'node:path'
import { planPath } from '../planPath.js'

const domains = ['dream','build','scale','shared'] as const
const caps = ['analyze','ideate','mentor','taste','knowledge','upload','ui','api','contracts','schemas','viewer','exports'] as const

type Domain = typeof domains[number]
type Cap = typeof caps[number]

type Flags = { domain: Domain; cap: Cap; name: string; kind?: 'ui'|'hook'|'test' }

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
  if (!domains.includes(f.domain)) throw new Error(`--domain must be one of ${domains.join(', ')}`)
  if (!caps.includes(f.cap)) throw new Error(`--cap must be one of ${caps.join(', ')}`)
  if (!f.name) throw new Error('--name is required')
  if (!['ui','hook','test', undefined].includes(f.kind)) f.kind = 'ui'
  return f as Flags
}

function planFrontendPath(domain: Domain, cap: Cap, name: string, kind: 'ui'|'hook'|'test') {
  const rel = planPath('frontend', domain, cap, name, kind)
  return { file: rel, dir: path.dirname(rel) }
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

function componentBoilerplate(name: string) {
  return `import React from 'react'

type Props = { className?: string }
export default function ${name}(props: Props) {
  return <div className={props.className}>${name}</div>
}
`
}

function hookBoilerplate(name: string) {
  return `import { useState } from 'react'
export function ${name}() {
  const [state, setState] = useState(null)
  return { state, setState }
}
`
}

function testBoilerplate(name: string) {
  return `import { describe, it, expect } from 'vitest'
import React from 'react'
import { render } from '@testing-library/react'
import ${name} from './${name}'

describe('${name}', () => {
  it('renders', () => {
    const r = render(<${name} />)
    expect(r.getByText('${name}')).toBeTruthy()
  })
})
`
}

async function main() {
  const { domain, cap, name, kind = 'ui' } = parseFlags(process.argv.slice(2))
  const { file } = planFrontendPath(domain, cap, name, kind)
  const content = kind === 'ui' ? componentBoilerplate(name)
    : kind === 'hook' ? hookBoilerplate(name)
    : testBoilerplate(name)
  await writeIfMissing(file, content)
}

main().catch((e) => { console.error(e); process.exit(1) })
