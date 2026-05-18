from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid, json, os, threading, queue
import time
from pathlib import Path
import logging

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"])

jobs = {}          # job_id -> job dict (includes chat_history list)
chat_sessions = {} # job_id -> DocumentChatSession
chat_session_locks: dict = {}  # job_id -> threading.Lock  (prevents concurrent init)
job_queue = queue.Queue()
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}

logger = logging.getLogger("uvicorn")


# ── Background worker ──────────────────────────────────────────────────────

def _worker():
    """Process extraction jobs sequentially in a daemon thread."""
    while True:
        try:
            job_data = job_queue.get()
            if job_data is None:
                break
            job_id, file_path = job_data
            _run_pipeline(job_id, file_path)
        except Exception as exc:
            logger.error(f"Worker error: {exc}")
        finally:
            job_queue.task_done()

threading.Thread(target=_worker, daemon=True).start()


# ── Models ─────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_upload_path(job_id: str, filename: str | None) -> Path:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError(f"Unsupported upload type: {suffix or 'unknown'}")
    return UPLOAD_DIR / f"{job_id}{suffix}"


# ── Upload ─────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    now = time.time()
    job_id = str(uuid.uuid4())[:8]
    try:
        file_path = _resolve_upload_path(job_id, file.filename)
    except ValueError as exc:
        return {"error": str(exc)}
    content = await file.read()
    file_path.write_bytes(content)
    jobs[job_id] = {
        "status": "queued", "progress": 0,
        "message": "Waiting in queue...", "filename": file.filename,
        "job_id": job_id, "result_path": None,
        "created_at": now, "started_at": None, "updated_at": now,
        "file_path": str(file_path),
        "chat_history": [],   # server-side persistent chat history
    }
    job_queue.put((job_id, str(file_path)))
    return {"job_id": job_id, "filename": file.filename}


# ── Pipeline runner ────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, file_path: str):
    heartbeat_stop = threading.Event()
    try:
        import sys; sys.path.insert(0, ".")
        from tender_extraction.main import TenderExtractionPipeline

        job = jobs[job_id]
        started_at = time.time()
        job["started_at"] = started_at
        job["updated_at"] = started_at
        stage_state = {"message": "Starting pipeline...", "progress": 5}

        def _heartbeat():
            while not heartbeat_stop.wait(5):
                if job.get("status") != "running":
                    continue
                elapsed = int(time.time() - started_at)
                job["message"] = f"{stage_state['message']} ({elapsed}s)"
                job["updated_at"] = time.time()

        threading.Thread(target=_heartbeat, daemon=True).start()

        job.update({"status": "running", "progress": 5,
                    "message": "Starting pipeline...", "updated_at": time.time()})

        def _progress(pct: int, msg: str):
            stage_state.update({"message": msg, "progress": pct})
            job.update({"progress": pct, "message": msg,
                        "status": "running", "updated_at": time.time()})

        output_path = str(OUTPUT_DIR / f"{job_id}.json")
        pipeline = TenderExtractionPipeline()
        result = pipeline.run(file_path, output_path=output_path,
                              progress_callback=_progress)

        # Auto-score against company profile if present
        profile_path = Path("company_profile.json")
        if profile_path.exists():
            try:
                job["message"] = "Ranking against company profile..."
                from tender_extraction.scoring import score_tender_match
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                score_res = score_tender_match(profile, result)
                job["match_score"] = score_res.get("match_score")
                job["match_data"] = score_res
            except Exception as exc:
                logger.warning(f"Auto-scoring failed for {job_id}: {exc}")

        specs = len(result.get("technical_specifications", []))
        deliverables = len(result.get("scope_of_work", {}).get("deliverables", []))
        match_score = job.get("match_score")
        msg = f"Complete — {specs} specs, {deliverables} deliverables"
        if match_score is not None:
            msg += f" (Match: {match_score}%)"

        job.update({"progress": 100, "status": "done",
                    "result_path": output_path, "message": msg,
                    "updated_at": time.time()})

    except Exception as exc:
        jobs[job_id].update({"status": "error", "message": str(exc),
                             "updated_at": time.time()})
    finally:
        heartbeat_stop.set()


# ── Job endpoints ──────────────────────────────────────────────────────────

@app.get("/jobs")
def list_jobs():
    # Omit chat_history from the list to keep the polling payload lean
    return [
        {k: v for k, v in job.items() if k != "chat_history"}
        for job in jobs.values()
    ]

@app.get("/jobs/{job_id}/status")
def get_status(job_id: str):
    if job_id not in jobs:
        return {"error": "not found"}
    return {k: v for k, v in jobs[job_id].items() if k != "chat_history"}

@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "not ready"}
    return json.loads(Path(job["result_path"]).read_text(encoding="utf-8"))

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    session = chat_sessions.pop(job_id, None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
    jobs.pop(job_id, None)
    return {"deleted": job_id}


# ── Chat / Q&A endpoints ───────────────────────────────────────────────────

@app.post("/jobs/{job_id}/ask")
def ask_document(job_id: str, payload: AskRequest):
    job = jobs.get(job_id)
    if not job:
        return {"error": "not found"}

    question = (payload.question or "").strip()
    if not question:
        return {"error": "question is required"}

    file_path = job.get("file_path")
    if not file_path or not Path(file_path).exists():
        return {"error": "source document is unavailable"}

    # Ensure we have a per-job lock before touching chat_sessions
    if job_id not in chat_session_locks:
        chat_session_locks[job_id] = threading.Lock()
    lock = chat_session_locks[job_id]

    with lock:
        session = chat_sessions.get(job_id)
        if session is None:
            import sys; sys.path.insert(0, ".")
            from tender_extraction.qa import DocumentChatSession

            # ":memory:" avoids ALL file-locking issues — the QA index only
            # needs to live as long as the server process, so disk persistence
            # gives us nothing but problems (concurrent access errors on reload).
            session = DocumentChatSession(
                file_path,
                persist_dir=":memory:",
                force_reindex=True,   # always fresh — no stale cache to worry about
            )
            chat_sessions[job_id] = session

    try:
        answer_data = session.ask(question)

        # Persist the exchange server-side so the frontend can restore it
        # after a tab switch without needing to re-ask.
        job["chat_history"].append({
            "id": str(uuid.uuid4())[:8],
            "ts": time.time(),
            "question": question,
            "answer": answer_data,
        })
        return answer_data

    except Exception as exc:
        logger.error(f"ask_document error for {job_id}: {exc}")
        return {"error": str(exc)}


@app.get("/jobs/{job_id}/history")
def get_chat_history(job_id: str):
    """Return the full server-side chat history so the client can restore it."""
    job = jobs.get(job_id)
    if not job:
        return {"error": "not found"}
    return {"history": job.get("chat_history", [])}


@app.delete("/jobs/{job_id}/history")
def clear_chat_history(job_id: str):
    """Wipe the chat history for a job."""
    job = jobs.get(job_id)
    if not job:
        return {"error": "not found"}
    job["chat_history"] = []
    return {"cleared": True}


# ── Profile & Score endpoints ──────────────────────────────────────────────

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
    return score_tender_match(profile, result)
