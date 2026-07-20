# ClarifAI – RAG-Based Customer Support System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF.svg)](https://vitejs.dev/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini--2.5--Flash-4285F4.svg)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-FF6F00.svg)](https://github.com/facebookresearch/faiss)

---

## 1. Project Overview

**ClarifAI** is a high-performance, production-ready Customer Support Assistant powered by **Retrieval-Augmented Generation (RAG)**. It allows enterprises, support teams, and users to upload multi-format documentation (PDFs, Word files, text manuals) and ask complex support questions in plain language.

### The Problem It Solves
Traditional customer support operations face high ticket volumes, escalating costs, and static FAQ portals that fail when users ask nuanced or multi-part questions. Generic LLMs often hallucinate policies, generate inaccurate specs, or leak unverified assumptions.

### Why RAG Is Used
RAG bridges the gap between static documents and generative AI. By fetching relevant chunks from an indexed vector database prior to answer generation, RAG grounds the LLM’s response strictly in proprietary organizational knowledge bases.

### Why Grounded AI Responses Matter
In customer support, accuracy is non-negotiable. Grounded responses prevent hallucinations, strictly enforce policy boundaries, and return exact document source attributions so support agents and customers can audit answer veracity instantly.

---

## 2. Features

- **Retrieval-Augmented Generation (RAG)**: Combines dense vector similarity search with grounded LLM synthesis for precise, hallucination-free responses.
- **Multi-Document Support**: Simultaneously indexes and retrieves answers across multiple uploaded knowledge base files.
- **Multilingual Document Support**: Processes and indexes documents in English, German, Spanish, French, Hindi, and Hinglish.
- **Multilingual Question Answering**: Understands user queries in various foreign languages and returns accurate, translated responses.
- **Multi-Question Understanding**: Automatically decomposes compound user questions into distinct sub-queries and aggregates grounded answers for each.
- **Hybrid Semantic Retrieval**: Blends FAISS 768-dimensional dense vector embeddings with BM25 lexical keyword matching for high-precision search.
- **Source Attribution**: Transparently cites source document filenames alongside each generated answer block.
- **Layout-Aware Document Processing**: Intelligently chunks documents by logical sections, headings, and lists to preserve semantic context.
- **Drag & Drop Upload**: Interactive frontend drag-and-drop interface supporting batch upload of `.pdf`, `.docx`, `.txt`, and `.csv` files.
- **UUID Filename Storage**: Internal UUID-prefixed file naming (`{uuid}_{filename}`) prevents storage collisions while exposing clean original filenames in UI and API outputs.
- **Responsive UI**: Glassmorphic dark/light theme interface featuring real-time loading feedback, code snippet formatting, and quick copy actions.
- **Production-Ready Backend**: Hardened FastAPI architecture featuring rate limiting, binary file signature inspection, CORS policies, and security headers.

---

## 3. Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend** | Python 3.10+, FastAPI, Uvicorn | Asynchronous REST API server with rate limiting and security middleware. |
| **Frontend** | React 18, Vite | Modular UI component framework with instant HMR bundler. |
| **LLM** | Google Gemini 2.5 Flash | Grounded answer generation and reasoning model. |
| **Embeddings** | Gemini `gemini-embedding-001` / HuggingFace `all-MiniLM-L6-v2` | Primary 768-dim dense embeddings with offline local fallback. |
| **Vector Database** | FAISS (Facebook AI Similarity Search) | High-speed dense vector similarity index. |
| **Document Processing**| `pypdf`, `python-docx`, `langdetect` | PDF/Word text extractors and automatic language identifier. |
| **Styling** | Vanilla CSS (Variables & Glassmorphism) | Modern tokenized design system with dark/light mode toggle. |
| **Deployment Readiness**| Rate Limiting, Magic Bytes, Security Headers | Hardened HTTP response headers, magic-byte inspection, and sliding-window limits. |

---

## 4. Installation

### Prerequisites
- **Python**: `3.10` or higher
- **Node.js**: `18.0` or higher & `npm`
- **Google Gemini API Key**: [Get an API Key](https://aistudio.google.com/)

---

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create environment configuration file from `.env.example`:
   ```bash
   # Create .env from template
   cp .env.example .env
   ```

5. Add your Google Gemini API key into `.env`:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

6. Launch the backend server:
   ```bash
   python run_backend.py
   ```
   The backend API will start at: `http://localhost:8000`

---

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install frontend npm packages:
   ```bash
   npm install
   ```

3. Start the frontend development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5173
   ```

---

## 5. Usage

1. **Upload Documents**: Drag & drop or select PDF, TXT, or Word files in the document upload card.
2. **Document Ingestion**: ClarifAI validates file signatures, parses text into section boundaries, extracts language metadata, and generates vector embeddings automatically.
3. **Embedding Generation**: Vector embeddings are saved directly into the local FAISS vector index.
4. **Ask Support Questions**: Type single or multi-part questions into the chat box.
5. **Receive Grounded Answers**: ClarifAI retrieves exact matching document context and synthesizes a grounded response complete with verified source citations.

---

## 6. Sample Use Cases

- **Customer Support**: Instantly query product return policies, shipping terms, and warranty coverage.
- **Knowledge Base Assistance**: Search across technical documentation, software specs, and troubleshooting guides.
- **Policy Documents**: Extract exact HR policies, compliance guidelines, and corporate procedures.
- **Product Manuals**: Retrieve step-by-step device setup, specifications, and maintenance steps.
- **Internal Documentation**: Index internal team onboarding material and operational SOPs.
- **FAQ Assistant**: Automate repetitive support inquiries with grounded citations.

---

## 7. Future Enhancements

- **OCR Support**: Integration with Tesseract / EasyOCR for extracting text from scanned PDF documents and images.
- **Image Understanding**: Multimodal RAG support for querying diagrammatic user manuals and architecture diagrams.
- **Authentication**: JWT & OAuth2 role-based access control (RBAC) for document-level permissions.
- **Streaming Responses**: Server-Sent Events (SSE) / WebSocket streaming for real-time token rendering.
- **Cloud Deployment**: One-click Docker containerization and Kubernetes deployment manifests.
- **Additional Vector Databases**: Modular adapters for Qdrant, Milvus, and Pinecone vector stores.

---

## 8. Author

Developed by **Shubham Ghodke**. Built with FastAPI, LangChain, FAISS, Google Gemini, and React.
