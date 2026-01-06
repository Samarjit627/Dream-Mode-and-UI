# Dream Mode and UI (Axis5)

This repository contains the Dream Mode UI and services used for document/image ingestion, OCR, and perception-only extraction.

## Components

- **frontend/**
  - Vite + React + TypeScript + Tailwind.
  - Main UI route: `/dream/analyze`.
- **backend/**
  - Node.js + Express + TypeScript gateway.
  - Proxies Dream PoC endpoints and hosts `/api/chat`.
- **services/dream-poc/**
  - FastAPI service providing `/api/v1/dream/analyze`.
  - Runs the perception pipeline (PDF render, OCR, geometry signals).
- **worker/**
  - Optional stub service (not required for Dream Mode OCR/perception).

## Run locally (dev)

### 1) Start Dream PoC (FastAPI)

From repo root:

```bash
docker compose -f services/dream-poc/docker/docker-compose.yml up -d --build
```

### 2) Start backend gateway

From `backend/`:

```bash
pnpm install
pnpm dev
```

Backend listens on:

- `http://localhost:4000`

### 3) Start frontend

From `frontend/`:

```bash
pnpm install
pnpm dev
```

Frontend listens on:

- `http://localhost:5173`

## Key endpoints

### Backend (gateway)

- **POST** `/api/chat`
  - Deterministic/local responses for page OCR queries and grounded intents.
- **GET** `/api/health`

### Dream PoC (FastAPI)

- **POST** `/api/v1/dream/analyze`
  - Upload a PDF/image and get an analysis JSON response.
- **GET** `/api/v1/dream/result/{job_id}`

## Output highlights

Dream PoC returns a `perception_v1` object intended to be **signals-only** (no semantic reasoning):

- `text_blocks[]` (OCR blocks with bboxes)
- `page_text` (per-page merged text)
- `lines[]`, `closed_shapes[]`, `arrow_candidates[]`
- `regions` (heuristic proposals: `title_block`, `viewports`, `notes`)
- `dimension_candidates[]` (numeric OCR blocks paired with nearby arrow candidates)
- `validation` (confidence histogram, coverage metrics)

The frontend and chat route use `perception_v1.page_text` for page reading queries (e.g. “what’s written on page 5”).

## Environment notes

- Backend expects Dream PoC reachable from Docker/network configuration.
- The system is tuned for development iteration; multiple dev servers on the same port will fail with `EADDRINUSE`.

## Known limits (important)

- PDF processing is capped to the first ~10 pages by default (performance safeguard).
- OCR quality depends on scan quality, font size, and effective DPI.
- Engineering dimension extraction is currently **signals-only** (candidates), not full dimension semantics.
