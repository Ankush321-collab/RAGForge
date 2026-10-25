# RAGForge — Production Multimodal Agentic RAG Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.11-005FD3.svg)](https://opensearch.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.15-008CC1.svg)](https://neo4j.com/)
[![MinIO S3](https://img.shields.io/badge/MinIO-S3_Compatible-C72C48.svg)](https://min.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)

A production-grade **Multimodal Retrieval-Augmented Generation (RAG)** platform designed from first principles to overcome the fundamental bottlenecks of naive RAG: lost context in fixed chunking, visual diagram omission, lexical keyword blindspots, and unverified hallucinations.

---

## Architecture Overview

### High-Level Architecture (HLD)
![RAGForge High-Level Architecture](system.png)

### Low-Level End-to-End Pipeline (LLD)
![RAGForge Low-Level Flow](system_flow.png)

---

## Key Architectural Highlights

### 1. Ingestion & Multimodal Extraction
* **Any PDF Drag & Drop**: Ingests raw documents directly into local/cloud S3 storage (MinIO) via presigned URLs.
* **PyMuPDF Extraction**: Extracts page text with structural metadata retention.
* **Embedded Figure & Table Extraction**: Identifies visual architecture diagrams, flowcharts, and markdown tables for specialized multimodal vector search and tabular reasoning.

### 2. Hierarchical Parent-Child Chunking
* Naive chunking creates a trade-off: small chunks maximize search precision but starve the LLM of synthesis context, while large chunks dilute semantic cosine similarity.
* **Solution**: Chunks documents into **small child chunks (300 tokens, 50 overlap)** mapped to **large parent contexts (1000 tokens)**. The retriever matches the exact child passage, but serves the full parent context to the generator.

### 3. Dual-Stream Hybrid Search & Reciprocal Rank Fusion (RRF)
* **Dense Semantic Search**: 384-dimensional dense vector embeddings generated via `sentence-transformers/all-MiniLM-L6-v2` in ChromaDB / OpenSearch k-NN.
* **Sparse Lexical Search**: BM25 keyword matching for exact technical terms, error codes, and variable names.
* **RRF Scoring**: Fuses ranked candidate lists with $k=60$ to resolve scoring discrepancies:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

### 4. Cross-Encoder Reranking
* Top candidates are reranked using a deep cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) which evaluates query-passage pairs via full cross-attention, elevating true relevant passages to Rank #1.

### 5. Knowledge Graph RAG (Neo4j)
* Complex multi-hop queries spanning multiple chapters cannot be resolved via cosine distance alone.
* Extracts semantic entities (`Concept`, `Mechanism`, `OS_Concept`, `Technique`) and typed relationships (`ENABLES`, `IMPLEMENTS`, `BUILDS`, `POINTS_TO`) into Neo4j for Cypher graph traversals.

### 6. Agentic Query Router & Answer Grounding Guardrail
* **LangGraph Router**: Classifies query intent into `TEXT`, `IMAGE`, `TABLE`, `GRAPH`, or `HYBRID`.
* **AnswerValidator**: Evaluates sentence-by-sentence attribution against retrieved chunk passages to block ungrounded hallucinations before serving responses.

---

## Empirical Benchmark & Ablation Study

Evaluated across technical corpora (`os_notes.pdf` and `oops_notes.pdf`) through 3 consecutive independent evaluation passes:

| Retrieval Strategy | Recall@5 | MRR | NDCG@5 | Faithfulness | Gain vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BM25 Sparse Alone** | **0.7420** | **0.5340** | **0.6380** | **0.910** | *Baseline* |
| **2. Dense Vector Alone** | **0.7950** | **0.5980** | **0.7040** | **0.935** | **+7.1%** |
| **3. Hybrid (Dense + BM25 RRF)** | **0.8920** | **0.7120** | **0.7960** | **0.960** | **+20.2%** |
| **4. Hybrid + Cross-Encoder Reranker** | **0.9540** | **0.8250** | **0.8840** | **0.985** | **+28.6%** |
| **5. Full Multimodal Pipeline** | **0.9820** | **0.8960** | **0.9380** | **0.992** | **+32.3% GAIN** |

### Hard-Level Multi-Run Stress Test (14 Queries $\times$ 3 Consecutive Runs)
* **Cross-Corpus Isolation Precision**: **100%** (Zero token bleeding between OS and OOP document spaces).
* **Hallucination Detection Rate**: **100%** (Fabricated adversarial queries caught with $0.00$ Grounding Score).
* **Warm Steady-State Latency**: **198.7 ms mean**, **231.0 ms P95**.

---

## React 18 Dashboard ([`frontend/`](frontend/))

![RAGForge React Dashboard](react_dashboard.png)

A modular, clean white enterprise dashboard built with React 18, Vite, and modern CSS tokens:
* **Target Corpus Switcher**: Dynamically query a specific uploaded document or search across the entire knowledge base.
* **Agentic Chat View**: Interactive conversation stream with dynamic routing badges, source citation pills (`P.14 doc_oops_notes.pdf`), and a live **Grounding Guardrail Confidence Meter**.
* **Hybrid Retrieval Explorer**: Inspect raw candidate passages, Dense vs BM25 score breakdowns, and Cross-Encoder rerank comparisons.
* **Upload Any PDF**: Drag-and-drop dropzone with live progress and automatic indexing.
* **Neo4j Knowledge Graph**: Visualizes entity subgraphs and directed relationships with Cypher query inspection.

---

## Quickstart & Installation

### Option 1: Docker Compose (All Services)
Launch MinIO, OpenSearch, Neo4j, FastAPI Backend, and the React Frontend with a single command:
```bash
docker compose up --build
```
* **React Web App**: `http://localhost:8000/app/` (or `http://localhost:3000`)
* **FastAPI Swagger Docs**: `http://localhost:8000/docs`
* **MinIO Console**: `http://localhost:9001` (User: `minioadmin`, Pass: `minioadminpassword`)
* **OpenSearch Vector DB**: `http://localhost:9200`
* **Neo4j Browser**: `http://localhost:7474` (User: `neo4j`, Pass: `password123`)

---

### Option 2: Local Python & Node Setup

#### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/your-username/RAGForge.git
cd RAGForge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend service
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Run Vite development server
npm run dev
```

---

## API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Microservices infrastructure status and storage verification. |
| `/documents` | `GET` | Lists all ingested documents with page counts and chunk statistics. |
| `/documents` | `POST` | Uploads any PDF and executes the ingestion and vectorization pipeline. |
| `/documents/{id}/graph` | `GET` | Returns Knowledge Graph entity nodes and relationship edges for a document. |
| `/chat` | `POST` | Agentic RAG chat with query classification, hybrid retrieval, and grounding validation. |
| `/search` | `POST` | Executes hybrid search with optional Cross-Encoder reranking and metadata filters. |

---

## LinkedIn Presentation Slide Deck

A ready-to-upload 6-slide presentation carousel has been compiled for sharing on LinkedIn:
* **PDF Carousel**: [`RAGForge_Architecture_and_Benchmarks.pdf`](RAGForge_Architecture_and_Benchmarks.pdf)
* **Interactive Web Slides**: Open [`frontend/slides.html`](frontend/slides.html) in your browser.

---

## License

This project is licensed under the Apache 2.0 License.
