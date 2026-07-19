# ClarifAI System Design & Implementation Guide

This document details the software design principles, data models, state management, and API design of **ClarifAI**.

---

## Technical Stack Overview

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python 3.10+) | High-performance asynchronous REST API server |
| **LLM Provider** | Google Gemini 2.5 Flash | Grounded answer generation & reasoning |
| **Embeddings** | Gemini Embeddings / HuggingFace MiniLM | Vector embedding generation |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | High-speed dense vector similarity index |
| **Frontend Framework**| React 18 + Vite | Modular UI component architecture |
| **Styling** | Vanilla CSS (Variables & Tokens) | Modern glassmorphism dark/light design system |
| **API Client** | Native Fetch API | Asynchronous backend communication |

---

## REST API Specification

### 1. Document Upload
- **Endpoint**: `POST /upload`
- **Content-Type**: `multipart/form-data`
- **Request Parameters**: `file` (File binary object)
- **Response**:
  ```json
  {
    "message": "Document uploaded and processed successfully",
    "chunks": 14,
    "filename": "sample_support_doc.txt"
  }
  ```
- **Error Codes**:
  - `400 Bad Request`: Unsafe file extension or path traversal attempt.
  - `413 Payload Too Large`: Exceeds maximum 10MB file limit.
  - `415 Unsupported Media Type`: Binary file signature validation failure (magic byte mismatch).

### 2. Chat Query
- **Endpoint**: `POST /chat`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "question": "What is the return policy for smart watches?",
    "history": [
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello! How can I help you?"}
    ]
  }
  ```
- **Response Body**:
  ```json
  {
    "answer": "Customers can return smart watches within 30 days of purchase for a full refund.",
    "logic": "RAG HyDE + FAISS Vector Search + Grounded LLM",
    "sources": ["sample_support_doc.txt"],
    "answers": [
      {
        "sub_query": "return policy for smart watches",
        "answer": "Customers can return smart watches within 30 days of purchase...",
        "sources": ["sample_support_doc.txt"]
      }
    ]
  }
  ```

### 3. List Documents
- **Endpoint**: `GET /documents`
- **Response Body**:
  ```json
  [
    {
      "name": "sample_support_doc.txt",
      "size": 9234,
      "chunks": 14,
      "type": "TXT"
    }
  ]
  ```

### 4. Delete Document
- **Endpoint**: `DELETE /documents/{filename}`
- **Response Body**:
  ```json
  {
    "message": "Document 'sample_support_doc.txt' deleted successfully."
  }
  ```

---

## State & Session Management

- **Isolated File Sessions**: The backend supports automated session management via `CLEAR_UPLOADS_ON_START` configuration.
- **Display Filename Masking**: Storage layer maintains 8-hex character UUID prefixes (e.g., `8ce11ef3_sample_doc.txt`) to eliminate filename collision risks while all UI/API outputs strip UUID prefixes cleanly.
- **Vector Index Synchronization**: Deleting a document dynamically rebuilds the FAISS vector index in-memory and updates disk state atomically.
