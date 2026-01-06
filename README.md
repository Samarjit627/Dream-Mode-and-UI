# Axis5 Monorepo

Gateway: Node.js + Express + TypeScript. Frontend: Vite + React + TS + Tailwind. Optional worker stub for CAD/ML (to be swapped later with FastAPI).

## Structure
- frontend/ — Vite app (Home, Dream, Build, Scale) with Night Mode persistence
- backend/ — Express TS gateway with REST routes returning fixtures (proxies to worker when USE_WORKER=true)
- worker/ — Optional Express TS worker stub exposing CAD endpoints (mocked)

## Quickstart

1) Install deps
- backend: `npm install` (from axis5/backend)
- worker: `npm install` (from axis5/worker)
- frontend: `npm install` (from axis5/frontend)

2) Dev servers
- backend: `npm run dev` (port 7071)
- worker: `npm run dev` (port 7072)
- frontend: `npm run dev` (port 5173, proxies /api to backend)

3) Env flags (backend)
- `USE_WORKER=false` (default): gateway serves fixtures
- `USE_WORKER=true`: gateway proxies to worker service at `WORKER_BASE_URL` (default http://localhost:7072)

## API (Gateway)
- POST /api/analyze/sketch
- POST /api/analyze/cad
- POST /api/ideate/generate4
- POST /api/mentor/critique
- GET  /api/taste/packs
- GET  /api/knowledge/cards
- GET  /api/health

## API (Worker stub)
- POST /convert/step-to-glb  → { glbUrl }
- POST /trial/preview       → { glbUrl }

## Notes
- Keep contracts REST/SSE so we can swap in a Python FastAPI worker for FreeCAD/ML later.
- Night Mode persists via localStorage and prefers-color-scheme.
