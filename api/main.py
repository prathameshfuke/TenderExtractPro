from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio, uuid, json, os, threading
import time
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, 
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"])

jobs = {}  # job_id -> {status, progress, message, filename, result_path}
chat_sessions = {}
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)


class AskRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    now = time.time()
    job_id = str(uuid.uuid4())[:8]
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    content = await file.read()
    pdf_path.write_bytes(content)
    jobs[job_id] = {
        "status": "queued", "progress": 0,
        "message": "Queued", "filename": file.filename,
        "job_id": job_id, "result_path": None,
        "created_at": now, "started_at": None, "updated_at": now,
        "pdf_path": str(pdf_path),
    }
    thread = threading.Thread(
        target=run_pipeline_sync, 
        args=(job_id, str(pdf_path)), 
        daemon=True
    )
    thread.start()
    return {"job_id": job_id, "filename": file.filename}

def run_pipeline_sync(job_id: str, pdf_path: str):
    heartbeat_stop = threading.Event()

    try:
        import sys; sys.path.insert(0, ".")
        from tender_extraction.main import TenderExtractionPipeline
        
        job = jobs[job_id]
        started_at = time.time()
        job["started_at"] = started_at
        job["updated_at"] = started_at
        stage_state = {"message": "Starting pipeline...", "progress": 5}

        def heartbeat_loop():
            while not heartbeat_stop.wait(5):
                if job.get("status") != "running":
                    continue
                elapsed = int(time.time() - started_at)
                base = stage_state["message"]
                job["message"] = f"{base} ({elapsed}s)"
                job["updated_at"] = time.time()

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        job["status"] = "running"
        job["progress"] = 5
        job["message"] = "Starting pipeline..."
        job["updated_at"] = time.time()
        
        def progress_callback(progress: int, message: str):
            stage_state["message"] = message
            stage_state["progress"] = progress
            job["progress"] = progress
            job["message"] = message
            job["status"] = "running"
            job["updated_at"] = time.time()
        
        output_path = str(OUTPUT_DIR / f"{job_id}.json")
        pipeline = TenderExtractionPipeline()
        result = pipeline.run(pdf_path, output_path=output_path,
                              progress_callback=progress_callback)
        
        specs = len(result.get("technical_specifications", []))
        deliverables = len(result.get("scope_of_work", {}).get("deliverables", []))
        
        job["progress"] = 100
        job["status"] = "done"
        job["result_path"] = output_path
        job["message"] = f"Complete - {specs} specs, {deliverables} deliverables extracted"
        job["updated_at"] = time.time()
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)
        jobs[job_id]["updated_at"] = time.time()
    finally:
        heartbeat_stop.set()

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
    return json.loads(Path(job["result_path"]).read_text(encoding="utf-8"))


@app.post("/jobs/{job_id}/ask")
def ask_document(job_id: str, payload: AskRequest):
    job = jobs.get(job_id)
    if not job:
        return {"error": "not found"}

    question = (payload.question or "").strip()
    if not question:
        return {"error": "question is required"}

    pdf_path = job.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return {"error": "source document is unavailable"}

    session = chat_sessions.get(job_id)
    if session is None:
        import sys; sys.path.insert(0, ".")
        from tender_extraction.qa import DocumentChatSession

        session = DocumentChatSession(
            pdf_path,
            persist_dir=str(OUTPUT_DIR / "_qa_qdrant_storage"),
            force_reindex=False,
        )
        chat_sessions[job_id] = session

    try:
        return session.ask(question)
    except Exception as exc:
        return {"error": str(exc)}

@app.get("/jobs")
def list_jobs():
    return list(jobs.values())

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    session = chat_sessions.pop(job_id, None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
    if job_id in jobs:
        del jobs[job_id]
    return {"deleted": job_id}

@app.get("/profile")
def get_profile():
    profile_path = Path("company_profile.json")
    if profile_path.exists():
        return json.loads(profile_path.read_text(encoding="utf-8"))
    return {}

@app.post("/profile")
def update_profile(profile: dict):
    profile_path = Path("company_profile.json")
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return {"status": "success"}

@app.get("/jobs/{job_id}/score")
def get_job_score(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "not ready"}
    
    result = json.loads(Path(job["result_path"]).read_text(encoding="utf-8"))
    profile_path = Path("company_profile.json")
    if not profile_path.exists():
        return {"error": "company profile not set"}
    
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    
    import sys; sys.path.insert(0, ".")
    from tender_extraction.scoring import score_tender_match
    
    score_result = score_tender_match(profile, result)
    return score_result
