from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio, uuid, json, os, threading
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, 
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"])

jobs = {}  # job_id -> {status, progress, message, filename, result_path}
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    content = await file.read()
    pdf_path.write_bytes(content)
    jobs[job_id] = {
        "status": "queued", "progress": 0,
        "message": "Queued", "filename": file.filename,
        "job_id": job_id, "result_path": None
    }
    thread = threading.Thread(
        target=run_pipeline_sync, 
        args=(job_id, str(pdf_path)), 
        daemon=True
    )
    thread.start()
    return {"job_id": job_id, "filename": file.filename}

def run_pipeline_sync(job_id: str, pdf_path: str):
    try:
        import sys; sys.path.insert(0, ".")
        from tender_extraction.main import TenderExtractionPipeline
        
        job = jobs[job_id]
        
        def update(progress: int, message: str):
            job["progress"] = progress
            job["message"] = message
            job["status"] = "running"
        
        update(5,  "Ingesting document pages...")
        update(20, "Extracting tables...")
        update(35, "Building ChromaDB + BM25 hybrid index...")
        update(50, "Running hybrid retrieval...")
        update(65, "Phi-3 extracting specifications (this takes 1-3 min)...")
        
        output_path = str(OUTPUT_DIR / f"{job_id}.json")
        pipeline = TenderExtractionPipeline()
        result = pipeline.run(pdf_path, output_path=output_path)
        
        specs = len(result.get("technical_specifications", []))
        tasks = len(result.get("scope_of_work", {}).get("tasks", []))
        
        job["progress"] = 100
        job["status"] = "done"
        job["result_path"] = output_path
        job["message"] = f"Complete — {specs} specs, {tasks} tasks extracted"
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)

@app.get("/jobs/{job_id}/status")
def get_status(job_id: str):
    if job_id not in jobs:
        return {"error": "not found"}
    return jobs[job_id]

@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "not ready"}
    return json.loads(Path(job["result_path"]).read_text())

@app.get("/jobs")
def list_jobs():
    return list(jobs.values())

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id in jobs:
        del jobs[job_id]
    return {"deleted": job_id}
