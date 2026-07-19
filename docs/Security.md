# ClarifAI Production Security Architecture

Security is a foundational pillar of **ClarifAI v1.0**. The system implements multi-layered protections across file uploads, API endpoints, LLM context handling, and network headers.

---

## 🛡️ Security Layers Overview

```
[ Incoming Request ]
        │
        ▼
1. Rate Limiting Middleware (Sliding Window IP Bucket: 30 req/min chat, 10 req/min upload)
        │
        ▼
2. HTTP Security Response Headers (CORS, HSTS, NoSniff, XSS, Frame Options)
        │
        ▼
3. Input Length & Schema Validation (Pydantic V2 Max Length Limits)
        │
        ▼
4. File Upload Security (Magic Byte Signature Validation, Path Traversal Sanitization, Size Limits)
        │
        ▼
5. Prompt Injection Shield (XML Grounding & System Isolation)
        │
        ▼
6. Storage Boundary & UUID Masking (Strict Upload Subdirectory & Hidden Internal UUIDs)
```

---

## Key Security Features

### 1. Magic-Byte Binary File Signature Inspection
To prevent malicious users from bypassing file extension checks by renaming executable files or malware to `.pdf` or `.docx`, ClarifAI reads the binary header bytes of every upload:
- **PDF Signature**: Must begin with `%PDF-` (`0x25 0x50 0x44 0x46 0x2D`).
- **DOCX Signature**: Must begin with Office Open XML ZIP header `PK\x03\x04` (`0x50 0x4B 0x03 0x04`).
- **TXT / CSV Validation**: Must contain valid UTF-8/Latin-1 text and contain zero null bytes (`\x00`).
- Any mismatch immediately raises an `HTTP 415 Unsupported Media Type` rejection.

### 2. Path Traversal & File Boundary Shields
- User-supplied filenames are sanitized (`sanitize_filename`) to strip invalid characters, path separators (`/`, `\`), and control characters.
- Upload targets are strictly checked against `UPLOADS_DIR` using `is_safe_upload_path` to prevent path traversal directory breakouts (`../../etc/passwd`).

### 3. Sliding-Window Rate Limiting
- **Chat API (`/chat`)**: 30 requests per minute per IP address.
- **Upload API (`/upload`)**: 10 requests per minute per IP address.
- Bounded IP sliding window prevents Denial-of-Service (DoS) and API key quota exhaustion.

### 4. Prompt Injection Defense
- User queries are sanitized and isolated inside strict XML context boundaries (`<context>...</context>`).
- System prompts instruct the LLM never to follow system override commands embedded inside uploaded documents.

### 5. HTTP Security Response Headers
Every FastAPI response automatically includes industry-standard security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`

### 6. Secrets & Privacy Protection
- `.env` files and API keys are strictly excluded from version control via `.env.example` templates.
- Exception masking ensures internal stack traces are never exposed to clients in API error responses.
