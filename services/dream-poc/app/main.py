import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from .processing import process_upload_sync, process_upload_async
from .models_loader import ModelRegistry
from .perception_router import router as perception_router
from .dream.api import router as dream_router
# BUILD MODE disabled for now (focus on Dream Mode)
# from .build.api import router as build_router

app = FastAPI(title="Axis5 Dream PoC")
app.include_router(perception_router)
app.include_router(dream_router)
# app.include_router(build_router)  # Disabled: BUILD MODE
models = ModelRegistry()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/dream/analyze")
async def analyze(file: UploadFile = File(...), background: bool = True):
    contents = await file.read()
    filename = file.filename or "upload"
    if background:
        job_id = process_upload_async(contents, filename, models)
        return JSONResponse({
            "request_id": job_id,
            "status": "processing",
            "result_url": f"/api/v1/dream/result/{job_id}"
        })
    else:
        result = process_upload_sync(contents, filename, models)
        return JSONResponse(result)

@app.get("/api/v1/dream/result/{job_id}")
def get_result(job_id: str):
    out_path = f"/app/data/results/{job_id}.json"
    if not os.path.exists(out_path):
        return JSONResponse({"request_id": job_id, "status": "processing"})
    import json
    with open(out_path, "r") as f:
        data = json.load(f)
    return JSONResponse(data)
