# Axis5 Dream PoC

Minimal Dream-mode pipeline service (FastAPI) that returns rich `dream_metadata.json` for images/sketches/PDFs.

Stack: FastAPI, YOLOv8 (Ultralytics), optional SAM (Segment Anything), CLIP (OpenAI), Tesseract OCR. CPU-ready; GPU recommended for production.

## Quickstart (Docker Compose)

1) Build & run

```
cd docker
docker-compose up --build
```

2) Health

```
curl -s http://localhost:8000/health | jq
```

3) Analyze an image (background async)

```
curl -s -X POST "http://localhost:8000/api/v1/dream/analyze" \
  -F "file=@app/sample_data/example.jpg" | jq
# => { request_id, status: "processing", result_url }
```

4) Poll result

```
REQUEST_ID=<id-from-previous>
curl -s http://localhost:8000/api/v1/dream/result/$REQUEST_ID | jq
```

Notes:
- SAM is optional. Set `SAM_WEIGHT_PATH` env var to a local SAM checkpoint to enable segmentation.
- Models may auto-download small weights on first run (YOLOv8n, CLIP). This may take time.
- Artifacts and results are stored under `/app/data` (mounted from `../data`).

## Local dev (without Docker)

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoints
- POST `/api/v1/dream/analyze`  → `{ request_id, status, result_url }`
- GET  `/api/v1/dream/result/{request_id}` → `dream_metadata.json` when ready or `{status: processing}`

## Directory
```
services/dream-poc/
  app/
    main.py
    processing.py
    models_loader.py
    utils.py
    sample_data/
  docker/
    Dockerfile
    docker-compose.yml
  requirements.txt
  README.md
```

## Next integration (optional)
- Proxy from Axis5 gateway: `/api/v1/dream/* -> http://localhost:8000/*`
- Wire Dream UI upload to POST analyze and poll result.
