import platform
# Monkey patch win32_ver to avoid subprocess hangs during torch imports on Windows
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.19045', '', 'Multiprocessor Free')

import os
import re
import time
import uuid
import shutil
import logging
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from rag import RAGService

# Setup logger
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ClarifAI-API")

app = FastAPI(
    title="ClarifAI RAG Customer Support API",
    description="Production-hardened, secure RAG customer support system API.",
    version="1.0.0"
)

# ----------------------------------------------------
# 1. Environment & Configurable Limits
# ----------------------------------------------------
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Verify API key is present
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.warning("[SECURITY] GOOGLE_API_KEY environment variable is not set. Gemini API calls will fail.")

# Configurable Limits
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
RATE_LIMIT_CHAT = int(os.getenv("RATE_LIMIT_CHAT", "30"))
RATE_LIMIT_UPLOAD = int(os.getenv("RATE_LIMIT_UPLOAD", "10"))

# ----------------------------------------------------
# 2. HTTP Security Headers Middleware
# ----------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# ----------------------------------------------------
# 3. Rate Limiting Middleware (Sliding Window Per IP)
# ----------------------------------------------------
class SlidingWindowRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        
    def is_allowed(self, ip: str, max_requests: int, window_seconds: int = 60) -> bool:
        now = time.time()
        cutoff = now - window_seconds
        self.requests[ip] = [ts for ts in self.requests[ip] if ts > cutoff]
        if len(self.requests[ip]) >= max_requests:
            return False
        self.requests[ip].append(now)
        return True

rate_limiter = SlidingWindowRateLimiter()

RATE_LIMITS = {
    "/chat": RATE_LIMIT_CHAT,
    "/upload": RATE_LIMIT_UPLOAD,
    "/documents": 30
}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "127.0.0.1"
    path = request.url.path
    
    for route_prefix, limit in RATE_LIMITS.items():
        if path.startswith(route_prefix):
            if not rate_limiter.is_allowed(client_ip, limit, window_seconds=60):
                logger.warning(f"[RATE LIMIT EXCEEDED] IP: {client_ip} exceeded limit on {path}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Please wait a minute before making more requests."},
                    headers={"Retry-After": "60"}
                )
            break
            
    return await call_next(request)

# ----------------------------------------------------
# 4. Global Safe Exception Handler (Mask Internal Traces)
# ----------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"[UNHANDLED EXCEPTION] Path: {request.url.path} | Error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing your request."}
    )

# ----------------------------------------------------
# 5. RAG Service Initialization & Upload Cleanup
# ----------------------------------------------------
UPLOADS_DIR = os.path.abspath("uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

try:
    rag_service = RAGService()
    clear_uploads_flag = os.getenv("CLEAR_UPLOADS_ON_START", "false").lower() in ("true", "1", "yes")
    if clear_uploads_flag and rag_service:
        logger.info("[STARTUP] CLEAR_UPLOADS_ON_START is enabled. Clearing document session...")
        rag_service.delete_all_uploaded_documents()
        for filename in os.listdir(UPLOADS_DIR):
            file_path = os.path.join(UPLOADS_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing {filename} on startup: {e}")
except Exception as e:
    logger.error(f"Error initializing RAG Service: {e}")
    rag_service = None

# ----------------------------------------------------
# 6. Helper Functions for Upload Security & Magic Bytes
# ----------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}

def sanitize_filename(filename: str) -> str:
    """Sanitizes user filename to prevent path traversal attacks."""
    base = os.path.basename(filename)
    base = re.sub(r'[\/\\:\*\?"<>\|]', '_', base)
    base = "".join(ch for ch in base if ord(ch) >= 32)
    return base.strip()

def get_display_filename(filename: str) -> str:
    """Strips internal 8-character hex UUID prefix (e.g. '8ce11ef3_') from filename for UI display."""
    if not filename:
        return ""
    base = os.path.basename(str(filename))
    return re.sub(r'^[0-9a-fA-F]{8}_', '', base)

def is_safe_upload_path(target_path: str) -> bool:
    """Ensures file path resides strictly inside the UPLOADS_DIR boundary."""
    abs_target = os.path.abspath(target_path)
    return abs_target.startswith(UPLOADS_DIR + os.sep) or abs_target == UPLOADS_DIR

def validate_file_signature(header_bytes: bytes, ext: str) -> bool:
    """Validates magic bytes / file signatures to reject disguised binary files."""
    if ext == ".pdf":
        return header_bytes.startswith(b"%PDF-")
    elif ext == ".docx":
        # DOCX files are Office Open XML ZIP archives
        return header_bytes.startswith(b"PK\x03\x04")
    elif ext in (".txt", ".csv"):
        # Ensure file contains plain text without null bytes or binary control sequences
        if b"\x00" in header_bytes[:4096]:
            return False
        try:
            header_bytes[:4096].decode("utf-8")
            return True
        except UnicodeDecodeError:
            try:
                header_bytes[:4096].decode("latin-1")
                return True
            except UnicodeDecodeError:
                return False
    return False

# ----------------------------------------------------
# 7. Request Models & Input Validations
# ----------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        if len(v) > 2000:
            raise ValueError("Message content exceeds 2000 character limit.")
        return v

class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace only.")
        if len(stripped) > 2000:
            raise ValueError("Question exceeds maximum length of 2000 characters.")
        return stripped

    @field_validator("history")
    @classmethod
    def validate_history_length(cls, v: list[Message]) -> list[Message]:
        if len(v) > 50:
            raise ValueError("History exceeds maximum limit of 50 messages.")
        return v

# ----------------------------------------------------
# 8. Secure API Endpoints
# ----------------------------------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG Service is not initialized. Check server logs.")

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided in request.")

    # 1. Sanitize filename against path traversal
    safe_name = sanitize_filename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # 2. Check for double extensions (e.g. file.pdf.exe)
    parts = safe_name.split(".")
    if len(parts) > 2:
        forbidden_exts = {"exe", "sh", "bat", "py", "js", "php", "vbs", "pl", "rb", "cmd"}
        for part in parts[1:-1]:
            if part.lower() in forbidden_exts:
                raise HTTPException(status_code=400, detail="File contains forbidden extension type.")

    # 3. Extension Whitelist Validation
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 4. Read Header Bytes & Validate Magic Byte Signature (File Signature Protection)
    header = await file.read(4096)
    await file.seek(0)
    if not validate_file_signature(header, ext):
        logger.warning(f"[SECURITY REJECTION] File '{safe_name}' failed magic byte signature validation for extension '{ext}'.")
        raise HTTPException(
            status_code=415,
            detail=f"Invalid file signature for extension '{ext}'. File content does not match the declared format."
        )

    # 5. Collision Protection: Generate Unique Stored Filename
    file_uuid = uuid.uuid4().hex[:8]
    stored_filename = f"{file_uuid}_{safe_name}"
    file_path = os.path.join(UPLOADS_DIR, stored_filename)

    # 6. Path Traversal Safety Verification
    if not is_safe_upload_path(file_path):
        raise HTTPException(status_code=400, detail="Path traversal attempt detected.")

    # 7. File Size Limitation Verification (10MB limit)
    logger.info(f"Saving uploaded file '{safe_name}' as '{stored_filename}' to disk.")
    size_read = 0
    try:
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 64):
                size_read += len(chunk)
                if size_read > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB."
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving uploaded file '{safe_name}': {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Error saving file to disk.")

    # 8. Safe Ingestion Execution
    try:
        logger.info(f"Ingesting file '{safe_name}' into RAG vector index...")
        num_chunks = rag_service.ingest_document(file_path)
        logger.info(f"File '{safe_name}' ingested successfully: {num_chunks} chunks.")
        return {"message": "Document uploaded and processed successfully", "chunks": num_chunks, "filename": safe_name}
    except Exception as e:
        logger.error(f"Error ingesting document '{safe_name}': {e}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Error processing and indexing document content.")

@app.get("/documents")
async def list_documents():
    if not os.path.exists(UPLOADS_DIR):
        return []
    
    docs = []
    for stored_filename in os.listdir(UPLOADS_DIR):
        file_path = os.path.join(UPLOADS_DIR, stored_filename)
        if os.path.isfile(file_path) and is_safe_upload_path(file_path):
            stats = os.stat(file_path)
            # Extract clean display name by stripping internal UUID prefix if present
            display_name = get_display_filename(stored_filename)
            document_type = os.path.splitext(display_name)[1].replace(".", "").upper()
            
            chunks_count = 0
            if rag_service and rag_service.vector_store:
                for doc_id, doc in rag_service.vector_store.docstore._dict.items():
                    source = doc.metadata.get("source", "")
                    normalized_source = os.path.normpath(source).replace("\\", "/")
                    normalized_target = os.path.normpath(file_path).replace("\\", "/")
                    if normalized_source == normalized_target:
                        chunks_count += 1
                        
            docs.append({
                "name": display_name,
                "size": stats.st_size,
                "chunks": chunks_count,
                "type": document_type
            })
    return docs

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    safe_name = sanitize_filename(filename)
    matching_file_path = None
    
    # Locate stored file by matching display name or stored name
    if os.path.exists(UPLOADS_DIR):
        for stored_name in os.listdir(UPLOADS_DIR):
            display_name = get_display_filename(stored_name)
            if display_name == safe_name or stored_name == safe_name or get_display_filename(safe_name) == display_name:
                matching_file_path = os.path.join(UPLOADS_DIR, stored_name)
                break

    if not matching_file_path or not is_safe_upload_path(matching_file_path) or not os.path.exists(matching_file_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    if rag_service:
        try:
            rag_service.delete_document(matching_file_path)
        except Exception as e:
            logger.error(f"Error removing chunks from FAISS vector store: {e}")
            
    clean_name = get_display_filename(matching_file_path)
    try:
        os.remove(matching_file_path)
        logger.info(f"File '{clean_name}' deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting file from disk: {e}")
        raise HTTPException(status_code=500, detail="Error deleting file from disk.")
        
    return {"message": f"Document '{clean_name}' deleted successfully."}

@app.delete("/documents")
async def delete_all_documents():
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG Service not initialized.")
        
    try:
        rag_service.delete_all_uploaded_documents()
    except Exception as e:
        logger.error(f"Error clearing FAISS index: {e}")
        
    try:
        if os.path.exists(UPLOADS_DIR):
            for filename in os.listdir(UPLOADS_DIR):
                file_path = os.path.join(UPLOADS_DIR, filename)
                if os.path.isfile(file_path) and is_safe_upload_path(file_path):
                    os.remove(file_path)
        logger.info("Deleted all uploaded documents from uploads directory.")
    except Exception as e:
        logger.error(f"Error clearing upload directory: {e}")
        raise HTTPException(status_code=500, detail="Error deleting documents from disk.")
        
    return {"message": "All uploaded documents deleted successfully."}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG Service is not initialized. Check server logs.")
    
    try:
        history_list = [msg.dict() for msg in request.history]
        response = rag_service.ask_question(
            request.question, 
            history=history_list
        )
        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while answering your question.")

@app.get("/")
def read_root():
    return {"status": "ClarifAI RAG Customer Support API is running", "security": "Production Hardened & Magic-Byte Verified"}

if __name__ == "__main__":
    import uvicorn
    logger.info("[RUNNER] Starting backend server via uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
