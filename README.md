# ClarifAI — RAG-Based Customer Support System

> **ClarifAI** is a high-performance, enterprise-grade AI Document Assistant powered by Retrieval-Augmented Generation (RAG). It enables customer support teams and users to instantly upload multi-format documents (PDF, TXT, Word) and receive accurate, grounded, multi-lingual answers with precise source citations.

---

## 🌟 Key Features

- **⚡ Hybrid Dense-Sparse Retrieval**: Combines FAISS 768-dimensional dense vector embeddings with BM25 lexical keyword matching for high-precision document search.
- **🌐 Multilingual & Cross-Lingual Support**: Seamlessly processes and answers queries across English, German, Spanish, French, Hindi, and Hinglish.
- **📄 Layout-Aware Document Parsing**: Intelligently chunks documents by sections, headers, and lists to preserve context boundaries instead of arbitrary token cuts.
- **🛡️ Enterprise Security Hardening**: Built-in sliding window rate limiting, magic-byte binary signature inspection, path traversal shielding, and prompt injection defense.
- **🎯 Precise Source Citations & Highlighting**: Returns exact document source citations and auto-highlights verified answer passages in the UI.
- **🎨 Glassmorphic Modern UI**: Responsive React interface featuring dark/light mode toggle, instant copy-to-clipboard, and interactive file upload queues.
- **🤖 Fallback Architecture**: Automatically falls back from Google Gemini to local HuggingFace embeddings (`all-MiniLM-L6-v2`) if offline or API quota is reached.

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|                        React.js + Vite + Modern CSS UI                            |
+-----------------------------------------------------------------------------------+
                                          |  HTTP REST Requests
                                          v
+-----------------------------------------------------------------------------------+
|                                FASTAPI BACKEND API                                |
|  - Rate Limiting Middleware (Sliding Window IP Bucket)                             |
|  - HTTP Security Response Headers (CORS, HSTS, NoSniff, XSS, Frame Options)       |
|  - Magic-Byte Binary File Signature Inspector                                      |
|  - Upload Path Traversal & File Boundary Guards                                   |
+-----------------------------------------------------------------------------------+
                                          |
                +-------------------------+-------------------------+
                | Ingest & Index                                    | Query & Answer
                v                                                   v
+-------------------------------+               +-----------------------------------+
|      DOCUMENT INGESTION       |               |     HYBRID RETRIEVAL ENGINE       |
| - Layout-Aware Sectioning     |               | - Sub-Query Decomposition         |
| - Text Extraction (PDF/Docx)  |               | - Semantic Vector Search (FAISS)  |
| - Language Detection (LangId) |               | - BM25 Keyword Search & BM25F     |
| - Metadata & Entity Indexing  |               | - Dynamic Hybrid Reciprocal Rank  |
+-------------------------------+               | - Multi-Layer Re-ranker           |
                |                               +-----------------------------------+
                v                                                   |
+-------------------------------+                                   | Grounded Context
|      FAISS VECTOR STORE       |                                   v
| - Gemini 768-dim Embeddings   |               +-----------------------------------+
| - MiniLM Local Fallback       |               |        GENERATION & PROMPT        |
| - UUID Storage Masking        |               | - Strict XML Context Boundaries   |
| - Section Metadata DB         |               | - Grounded Gemini-2.5-Flash LLM   |
+-------------------------------+               | - Auto-Highlighting & Citations   |
                                                +-----------------------------------+
```

---

## 💻 Tech Stack

### Backend
- **Framework**: Python 3.10+, FastAPI, Uvicorn
- **LLM**: Google Gemini 2.5 Flash (`google-generativeai`)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Gemini `gemini-embedding-001` (Fallback: HuggingFace `all-MiniLM-L6-v2`)
- **Document Parsers**: `pypdf`, `python-docx`, `langdetect`

### Frontend
- **Framework**: React 18, Vite
- **Styling**: Modern Vanilla CSS Design Tokens (Dark / Light Theme Toggle)
- **Icons**: Custom SVG Component Library

---

## 📂 Project Structure

```
ClarifAI/
│
├── docs/                               # System & Security Architecture Documentation
│   ├── Architecture.md
│   ├── SystemDesign.md
│   └── Security.md
│
├── data/                               # Sample Benchmark Support Documents
│   ├── Spanish_RAG_Customer_Support.docx
│   ├── german_support_doc.txt
│   ├── hospital_data.txt
│   └── sample_support_doc.txt
│
├── backend/                            # FastAPI REST API Server & RAG Engine
│   ├── main.py                         # REST Endpoints & Security Middleware
│   ├── rag.py                          # Hybrid Retrieval & RAG Pipeline
│   ├── run_backend.py                  # Startup Runner Script
│   ├── .env.example                    # Environment Template
│   └── requirements.txt                # Python Dependencies
│
├── frontend/                           # React + Vite User Interface
│   ├── public/                         # Static Assets
│   ├── src/
│   │   ├── components/                 # React UI Components
│   │   ├── App.jsx                     # Root Component
│   │   ├── App.css                     # Glassmorphic Stylesheet
│   │   ├── index.css                   # Core Design Tokens
│   │   └── main.jsx                    # React Entrypoint
│   ├── index.html                      # HTML Entrypoint
│   ├── package.json                    # Npm Dependencies & Scripts
│   ├── vite.config.js                  # Vite Config
│   ├── eslint.config.js                # Linter Config
│   └── README.md                       # Frontend README
│
├── .env.example                        # Root Secrets Template
└── README.md                           # Master Documentation & Setup Guide
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Python**: `3.10+`
- **Node.js**: `18.0+` & `npm`
- **Google Gemini API Key**: [Get an API Key](https://aistudio.google.com/)

---

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file from example
cp .env.example .env

# Add your Gemini API key inside .env
# GOOGLE_API_KEY=your_gemini_api_key_here

# Start backend server
python run_backend.py
```
The API server will run at: `http://localhost:8000`

---

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install

# Start local development server
npm run dev
```
The application interface will open at: `http://localhost:5173`

---

## 📖 Usage Guide

1. **Upload Support Documents**: Drag & drop or select PDF, TXT, or Word files using the upload panel.
2. **Ask Questions**: Type queries in plain English, German, Spanish, or Hindi in the chat input.
3. **Inspect Citations**: Review grounded answers with clickable source tags and highlighted text passages.
4. **Manage Documents**: View uploaded files and active vector chunks in the sidebar.

---

## 🔒 Security Features

- **Binary Signature Inspection**: Validates magic bytes (`%PDF-`, `PK\x03\x04`, text UTF-8) to reject disguised executable files (`HTTP 415`).
- **Path Traversal Protection**: Enforces strict upload directory boundary checks (`is_safe_upload_path`).
- **Sliding Window Rate Limiting**: Restricts `/chat` (30 req/min) and `/upload` (10 req/min) per IP.
- **HTTP Response Security Headers**: Includes `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, and `Referrer-Policy`.
- **UUID Filename Masking**: Strips internal 8-hex UUID prefixes from user-facing API responses to preserve privacy and display clean filenames.

---

## 📸 Screenshots

*(Add screenshots of the ClarifAI Dark & Light Mode Interface here)*

---

## 🔮 Future Enhancements

- [ ] Support for OCR image document extraction (Tesseract / EasyOCR).
- [ ] Role-based access control (RBAC) for enterprise document permissions.
- [ ] Streaming WebSocket responses for instant token rendering.
- [ ] Export chat history to PDF/Markdown support reports.

---

## 👤 Author

Developed by **ClarifAI Team**. Built with FastAPI, LangChain, FAISS, Google Gemini, and React.
