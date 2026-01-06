export type Domain = 'dream'|'build'|'scale'|'shared'
export type Cap = 'analyze'|'ideate'|'mentor'|'taste'|'knowledge'|'upload'|'ui'|'api'|'contracts'|'schemas'|'viewer'|'exports'

// Current layout support (axis5/frontend, axis5/backend, axis5/worker)
// Packages layout (packages/*) can be toggled via env PACKAGES_LAYOUT=true.
const usePackagesLayout = (process.env.PACKAGES_LAYOUT === 'true')

export function planPath(
  layer: 'frontend'|'gateway'|'worker'|'schema'|'fixture',
  domain: Domain,
  cap: Cap,
  name: string,
  kind: 'ui'|'hook'|'service'|'controller'|'validator'|'schema'|'fixture'|'test'
): string {
  if (usePackagesLayout) {
    switch (layer) {
      case 'frontend': {
        if (domain === 'shared') {
          const base = `packages/frontend/src/shared/${cap}`
          if (kind === 'ui' || kind === 'hook' || kind === 'test') return `${base}/${name}.${kind === 'test' ? 'test.' : ''}tsx`
          return `${base}/${name}.ts`
        }
        const base = `packages/frontend/src/features/${domain}/${cap}`
        if (kind === 'ui' || kind === 'hook' || kind === 'test') return `${base}/${name}.${kind === 'test' ? 'test.' : ''}tsx`
        if (kind === 'fixture') return `${base}/shared/fixtures/${name}.json`
        return `${base}/${name}.ts`
      }
      case 'gateway': {
        const base = `packages/gateway/src/modules/${cap}`
        if (kind === 'controller') return `${base}/controller.ts`
        if (kind === 'service') return `${base}/service.ts`
        if (kind === 'validator') return `${base}/validator.ts`
        if (kind === 'fixture') return `packages/gateway/src/fixtures/${name}.json`
        if (kind === 'test') return `${base}/${name}.spec.ts`
        return `${base}/${name}.ts`
      }
      case 'worker': {
        const base = `packages/worker/app/services/${cap}`
        if (kind === 'test') return `packages/worker/tests/${name}.py`
        return `${base}/${name}.py`
      }
      case 'schema': {
        return `docs/SCHEMAS/${name}.json`
      }
      case 'fixture': {
        if (domain === 'shared') return `packages/gateway/src/fixtures/${name}.json`
        return `packages/frontend/src/features/${domain}/shared/fixtures/${name}.json`
      }
    }
  }

  // Validate inputs
  const assert = (cond: boolean, msg: string) => { if (!cond) throw new Error(msg) }
  const domOk = ['dream','build','scale','shared'].includes(domain)
  const capsOk = ['analyze','ideate','mentor','taste','knowledge','upload','ui','api','contracts','schemas','viewer','exports'].includes(cap)
  assert(domOk && capsOk, 'invalid domain/capability')

  switch (layer) {
    case 'frontend': {
      if (domain === 'shared') {
        // shared area
        const base = `frontend/src/shared/${cap}`
        if (kind === 'ui' || kind === 'hook' || kind === 'test') return `${base}/${name}.${kind === 'test' ? 'test.' : ''}tsx`
        return `${base}/${name}.ts`
      }
      // feature slice
      const base = `frontend/src/features/${domain}/${cap}`
      if (kind === 'ui' || kind === 'hook' || kind === 'test') return `${base}/${name}.${kind === 'test' ? 'test.' : ''}tsx`
      if (kind === 'fixture') return `${base}/shared/fixtures/${name}.json`
      return `${base}/${name}.ts`
    }
    case 'gateway': {
      const base = `backend/src/modules/${cap}`
      if (kind === 'controller') return `${base}/controller.ts`
      if (kind === 'service') return `${base}/service.ts`
      if (kind === 'validator') return `${base}/validator.ts`
      if (kind === 'fixture') return `backend/src/fixtures/${name}.json`
      if (kind === 'test') return `${base}/${name}.spec.ts`
      return `${base}/${name}.ts`
    }
    case 'worker': {
      const base = `worker/app/services/${cap}`
      if (kind === 'test') return `worker/tests/${name}.py`
      return `${base}/${name}.py`
    }
    case 'schema': {
      return `docs/SCHEMAS/${name}.json`
    }
    case 'fixture': {
      if (domain === 'shared') return `backend/src/fixtures/${name}.json`
      return `frontend/src/features/${domain}/shared/fixtures/${name}.json`
    }
  }
}

export default planPath
