# Axis5 File Allocator Playbook

Single source of truth for where files live as the app grows.

## Allocation Rules (summary)
- Rule 0: Tag everything with { domain: dream|build|scale|shared, capability }.
- Rule 1 (frontend feature slice):
  - domain in {dream,build,scale} → packages/frontend/src/features/<domain>/<capability>/...
  - shared capability (ui, api, viewer, exports, theme, utils) → packages/frontend/src/shared/<capability>/...
- Rule 2 (gateway modules): packages/gateway/src/modules/<capability>/(controller|service|model|validator).ts
- Rule 3 (worker): packages/worker/app/services/<capability>/...
- Rule 4 (schemas canonical): docs/SCHEMAS/*.json are the source of truth.
- Rule 5 (fixtures):
  - Frontend: packages/frontend/src/features/<domain>/shared/fixtures/*.json
  - Gateway: packages/gateway/src/fixtures/*.json
  - Worker: packages/worker/app/fixtures/*.json
- Rule 6–9: exports under frontend shared/exports; viewer under frontend shared/viewer; knowledge content under frontend/features/dream/shared/knowledge; tests colocated.

## Path Planner API
```
export type Domain = 'dream'|'build'|'scale'|'shared';
export type Cap = 'analyze'|'ideate'|'mentor'|'taste'|'knowledge'|'upload'|'ui'|'api'|'contracts'|'schemas'|'viewer'|'exports';
export function planPath(
  layer: 'frontend'|'gateway'|'worker'|'schema'|'fixture',
  domain: Domain,
  cap: Cap,
  name: string,
  kind: 'ui'|'hook'|'service'|'controller'|'validator'|'schema'|'fixture'|'test'
): string
```

## Implementation Notes
- Current repo uses axis5/frontend, axis5/backend, axis5/worker.
- The planner supports current layout and future packages/ layout. A migration ADR will lock in packages/ later.
- Generators (gen:ui, gen:api, gen:schema) call planPath, create folders/files, and update index barrels.
- Path Linter enforces policy in CI.
