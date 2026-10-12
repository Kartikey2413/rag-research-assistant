"""FastAPI service: upload documents, then ask questions about them. Docs at /docs."""
import logging
import os
import secrets
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.llm_status import ollama_status
from src.qa_engine import RAGPipeline

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag.api")

EXTENSIONS = {".txt", ".md", ".pdf"}
ASK_LIMIT = os.getenv("RATE_LIMIT", "20/minute")
UPLOAD_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "5/minute")
limiter = Limiter(key_func=get_remote_address)


@dataclass
class Config:
    docs_dir: str = "data/sample_docs"
    backend: str = "tfidf"
    top_k: int = 4
    max_file_bytes: int = 5 * 1024 * 1024
    session_ttl_s: float = 3600
    max_sessions: int = 20
    max_files: int = 5

    @classmethod
    def from_env(cls) -> "Config":
        e = os.getenv
        return cls(
            docs_dir=e("DOCS_DIR", cls.docs_dir),
            backend=e("EMBEDDING_BACKEND", cls.backend),
            top_k=int(e("TOP_K", cls.top_k)),
            max_file_bytes=int(float(e("MAX_UPLOAD_MB", 5)) * 1024 * 1024),
            session_ttl_s=float(e("SESSION_TTL_MIN", 60)) * 60,
            max_sessions=int(e("MAX_SESSIONS", cls.max_sessions)),
            max_files=int(e("MAX_SESSION_FILES", cls.max_files)),
        )


@dataclass
class Session:
    pipeline: Any
    path: Path
    last_used: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    session_id: Optional[str] = Field(default=None, description="Search only this upload session. Omit for the sample corpus.")


class Source(BaseModel):
    source: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    mode: str = Field(description="'llm' if Ollama is reachable with the model pulled, else 'extractive'")
    scope: str = Field(description="'session' or 'shared'")
    latency_ms: float


class UploadResponse(BaseModel):
    session_id: str
    files: List[str]
    chunks: int
    expires_in_minutes: int


def build(cfg: Config, docs_dir: str):
    pipeline = RAGPipeline(docs_dir=docs_dir, backend=cfg.backend, top_k=cfg.top_k)
    return pipeline, pipeline.build()


def purge_expired(state) -> None:
    now = time.time()
    with state.lock:
        for sid in [k for k, v in state.sessions.items() if now - v.last_used > state.cfg.session_ttl_s]:
            shutil.rmtree(state.sessions.pop(sid).path, ignore_errors=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = cfg = Config.from_env()
    app.state.lock = threading.Lock()
    app.state.sessions = {}
    app.state.shared, app.state.chunks = None, 0
    Path(cfg.docs_dir).mkdir(parents=True, exist_ok=True)
    try:
        app.state.shared, app.state.chunks = build(cfg, cfg.docs_dir)
    except Exception:
        log.exception("Could not index the shared sample corpus")
    yield
    for s in app.state.sessions.values():
        shutil.rmtree(s.path, ignore_errors=True)


app = FastAPI(title="RAG Research Assistant", version="2.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health(request: Request):
    st = request.app.state
    return {
        "status": "ok" if st.shared is not None else "degraded",
        "chunks": st.chunks,
        "backend": st.cfg.backend,
        "mode": ollama_status(),
    }


@app.post("/upload", response_model=UploadResponse)
@limiter.limit(UPLOAD_LIMIT)
def upload(request: Request, files: List[UploadFile] = File(...)):
    """Index the uploaded files into a private, temporary session and return its id."""
    st = request.app.state
    cfg = st.cfg
    purge_expired(st)
    if len(files) > cfg.max_files:
        raise HTTPException(413, f"At most {cfg.max_files} files per upload.")
    if len(st.sessions) >= cfg.max_sessions:
        raise HTTPException(503, "Server is busy. Try again in a few minutes.")

    docs = {}
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in EXTENSIONS:
            raise HTTPException(415, f"'{name}': only {sorted(EXTENSIONS)} files are supported.")
        data = f.file.read(cfg.max_file_bytes + 1)
        if len(data) > cfg.max_file_bytes:
            raise HTTPException(413, f"'{name}' is larger than {cfg.max_file_bytes // 2**20} MB.")
        if not data.strip():
            raise HTTPException(422, f"'{name}' is empty.")
        docs[name] = data

    path = Path(tempfile.mkdtemp(prefix="rag_"))
    try:
        for name, data in docs.items():
            (path / name).write_bytes(data)
        pipeline, chunks = build(cfg, str(path))
    except Exception as exc:
        shutil.rmtree(path, ignore_errors=True)
        raise HTTPException(422, f"Could not read your files: {exc}")

    sid = secrets.token_urlsafe(16)
    with st.lock:
        st.sessions[sid] = Session(pipeline, path, time.time())
    return UploadResponse(session_id=sid, files=sorted(docs), chunks=chunks, expires_in_minutes=int(cfg.session_ttl_s // 60))


@app.delete("/session/{session_id}")
def delete_session(request: Request, session_id: str):
    with request.app.state.lock:
        session = request.app.state.sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(404, "Session not found or expired.")
    shutil.rmtree(session.path, ignore_errors=True)
    return {"deleted": True}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(ASK_LIMIT)
def ask(request: Request, body: AskRequest):
    st = request.app.state
    if body.session_id:
        purge_expired(st)
        session = st.sessions.get(body.session_id)
        if session is None:
            raise HTTPException(404, "Session expired. Please upload your files again.")
        session.last_used = time.time()
        pipeline, scope = session.pipeline, "session"
    elif st.shared is None:
        raise HTTPException(503, "Sample corpus is not indexed. Check the server logs.")
    else:
        pipeline, scope = st.shared, "shared"

    start = time.perf_counter()
    result = pipeline.ask(body.question)
    latency = (time.perf_counter() - start) * 1000
    log.info("ask scope=%s latency=%.0fms", scope, latency)
    return AskResponse(
        answer=result["answer"],
        sources=[Source(source=str(s["source"]), score=float(s["score"]), snippet=str(s["snippet"])) for s in result["sources"]],
        mode=ollama_status(),
        scope=scope,
        latency_ms=round(latency, 1),
    )
