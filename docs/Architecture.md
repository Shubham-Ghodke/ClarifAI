# ClarifAI System Architecture

ClarifAI is an enterprise-grade, retrieval-augmented generation (RAG) Customer Support Document Assistant built to ingest, index, and query multi-format, multi-lingual organizational knowledge bases with high speed and precision.

---

## High-Level Architectural Diagram

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

## Core Component Breakdown

### 1. Document Processing & Ingestion Engine
- **Multi-Format Extraction**: Parses plain text, Markdown, PDF (`pypdf`), and Microsoft Word (`python-docx`) documents.
- **Layout-Aware Section Chunking**: Intelligently parses section titles, headings, and lists to preserve semantic context within boundaries instead of arbitrary token cuts.
- **Language Detection & Metadata Auto-Extraction**: Automatically identifies document language (English, German, Spanish, Hindi, etc.) and extracts key attributes, entities, and subjects for metadata filtering.

### 2. FAISS Vector Database & Fallback Embeddings
- **Primary Embedding Model**: Google Gemini (`models/gemini-embedding-001`) with 768-dimensional dense vector embeddings.
- **Offline Fallback Engine**: Local `sentence-transformers/all-MiniLM-L6-v2` HuggingFace embeddings run automatically if Gemini quota limit is reached or offline.
- **UUID-Collision Shielding**: Documents are assigned UUID prefixes internally for collision safety while API outputs strip UUID prefixes (`get_display_filename`) to keep UI outputs clean.

### 3. Hybrid Retrieval & Re-ranking Pipeline
- **Sub-Query Decomposition**: Deconstructs multi-part user questions into granular search units.
- **Dense-Sparse Hybrid Scoring**: Combines FAISS vector similarity with BM25 lexical term matching.
- **Multi-Layer Re-Ranker**: Boosts exact subject matches, section header alignments, and entity hits while penalizing mismatching domain entities.

### 4. Grounded LLM Generation & Citation System
- **Grounded Prompting**: Context is injected into strict XML boundaries (`<context>...</context>`).
- **Google Gemini-2.5-Flash Integration**: Generates concise, accurate responses strictly from provided text.
- **Source Citation & Highlighting**: Returns verified source filenames and highlights exact answer snippets.
