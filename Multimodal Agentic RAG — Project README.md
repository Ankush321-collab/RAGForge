# Multimodal Agentic RAG

A production-oriented **Multimodal Retrieval-Augmented Generation (RAG)** system that can understand and retrieve information from text, PDFs, tables, images, diagrams, and other document content.

The project will be developed incrementally. Every component will be implemented, tested, evaluated, and understood before moving to the next stage.

The goal is **not** to build a simple "Chat with PDF" application.

The goal is to understand and implement the retrieval pipeline from first principles:

```text
Document
   ↓
Object Storage (S3)
   ↓
Document Processing
   ↓
Text / Image / Table Extraction
   ↓
Chunking
   ↓
Embedding + Sparse Indexing
   ↓
Hybrid Retrieval
   ↓
Reranking
   ↓
Query Rewriting
   ↓
HyDE
   ↓
Parent-Child Retrieval
   ↓
Contextual Retrieval
   ↓
Multimodal Retrieval
   ↓
Graph RAG
   ↓
Agentic Routing
   ↓
LLM
   ↓
Answer + Citations
   ↓
Evaluation
```

---

# 1. Project Objective

Build a system that allows a user to upload documents and ask questions about them.

The system should eventually support:

- PDF documents
- Text
- Images
- Tables
- Charts
- Diagrams
- Research papers
- Technical documentation
- Multiple documents
- Metadata filtering
- Hybrid search
- Multimodal retrieval
- Graph-based retrieval
- Agentic query routing
- Source citations
- Retrieval evaluation
- Answer evaluation

Example:

```text
User:

"According to Figure 4, why does Model B
perform better than Model A?"

System:

1. Understands the query
2. Detects that visual information is required
3. Rewrites the query
4. Searches text
5. Searches images
6. Retrieves Figure 4
7. Retrieves surrounding text
8. Reranks results
9. Gives relevant context to vision-capable LLM
10. Generates answer
11. Shows source/page/figure citation
```

---

# 2. Important Development Rule

DO NOT build the entire system at once.

For every phase:

```text
Learn
 ↓
Implement
 ↓
Create small test
 ↓
Run test
 ↓
Inspect output
 ↓
Evaluate
 ↓
Document what happened
 ↓
Move to next phase
```

Never move to the next phase just because the code runs.

The objective is:

> Understand why the system works.

---

# 3. Final Architecture

Eventually the project should look approximately like:

````text
                         USER
                           |
                           v
                    React Frontend
                           |
                           v
                       FastAPI
                           |
                           v
                   Agentic RAG API
                           |
                           v
                    Query Analyzer
                           |
                           v
                    Query Rewriter
                           |
                           v
                    Query Router
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
     Text Retrieval   Image Retrieval   Graph Retrieval
          |                |                |
          v                v                v
       BM25             Image Vector      Neo4j
          |                |                |
          +----------------+----------------+
                           |
                           v
                    Hybrid Fusion
                           |

## 5.1 Implementation File Map

Use this map when starting a phase. Files marked **exists** are already present in
this repository. Files marked **create** are the next files to implement for that
phase. Keep the dependency direction clear:

```text
loader -> chunker -> embedding -> vector search -> RAG
                              |
                              +-> multimodal / graph / agent routing
````

| Phase   | Work                                                     | Files in this project                                                                                                                                                    |
| ------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0       | Environment and shared settings                          | `requirements.txt` (exists), `backend/app/core/config.py` (exists), `backend/app/core/logging.py` (exists), `docker-compose.yml` (exists)                                |
| 1A      | PDF loading, page text, metadata                         | `backend/app/ingestion/loader.py` (exists), `backend/tests/ingestion/test_loader.py` (create)                                                                            |
| 1B      | Page chunking and chunk metadata                         | `backend/app/ingestion/chunker.py` (exists), `backend/tests/ingestion/test_chunker.py` (create)                                                                          |
| 1C      | Text embeddings                                          | `backend/app/ingestion/embedding.py` (exists), `backend/tests/ingestion/test_embedding.py` (create)                                                                      |
| 1D      | Vector similarity search                                 | `backend/app/retrieval/vector.py` (exists), `backend/tests/retrieval/test_vector.py` (create)                                                                            |
| 1E      | Basic RAG answer generation and citations                | `backend/app/api/chat.py` (exists), `backend/app/api/search.py` (exists), `backend/tests/api/test_chat.py` (create)                                                      |
| 2       | Object storage foundation                                | `backend/app/core/config.py`, `backend/app/api/documents.py` (exists), `backend/app/storage/s3.py` (create), `backend/tests/storage/test_s3.py` (create)                 |
| 2A      | Upload documents to S3                                   | `backend/app/api/documents.py`, `backend/app/storage/s3.py`, `backend/tests/api/test_documents.py` (create)                                                              |
| 2B      | Generate presigned URLs                                  | `backend/app/storage/s3.py`, `backend/app/api/documents.py`, `backend/tests/storage/test_presigned_urls.py` (create)                                                     |
| 2C      | CDN and private asset delivery                           | `backend/app/storage/s3.py`, `backend/app/core/security.py` (exists), deployment configuration (create)                                                                  |
| 3       | Advanced and parent-child chunking                       | `backend/app/ingestion/chunker.py`, `backend/app/retrieval/parent_child.py` (exists), `backend/tests/retrieval/test_parent_child.py` (create)                            |
| 4       | BM25 sparse retrieval                                    | `backend/app/retrieval/bm25.py` (exists), `backend/tests/retrieval/test_bm25.py` (create)                                                                                |
| 5       | Hybrid retrieval and RRF fusion                          | `backend/app/retrieval/hybrid.py` (exists), `backend/app/retrieval/bm25.py`, `backend/app/retrieval/vector.py`, `backend/tests/retrieval/test_hybrid.py` (create)        |
| 6       | Cross-encoder reranking                                  | `backend/app/retrieval/reranker.py` (exists), `backend/tests/retrieval/test_reranker.py` (create)                                                                        |
| 7       | Query rewriting                                          | `backend/app/retrieval/query_transform.py` (exists), `backend/tests/retrieval/test_query_transform.py` (create)                                                          |
| 8       | Query expansion and multi-query fusion                   | `backend/app/retrieval/query_transform.py`, `backend/app/retrieval/hybrid.py`, `backend/tests/retrieval/test_query_expansion.py` (create)                                |
| 9       | HyDE retrieval                                           | `backend/app/retrieval/hyde.py` (exists), `backend/app/retrieval/vector.py`, `backend/tests/retrieval/test_hyde.py` (create)                                             |
| 10      | Parent-child context retrieval                           | `backend/app/retrieval/parent_child.py`, `backend/app/retrieval/hybrid.py`, `backend/tests/retrieval/test_parent_child.py` (create)                                      |
| 11      | Metadata filtering                                       | `backend/app/retrieval/vector.py`, `backend/app/retrieval/hybrid.py`, `backend/app/api/search.py`, `backend/tests/retrieval/test_metadata_filtering.py` (create)         |
| 12      | Contextualized chunks                                    | `backend/app/retrieval/contextual.py` (exists), `backend/app/ingestion/chunker.py`, `backend/tests/retrieval/test_contextual.py` (create)                                |
| 13A     | Extract and store document images                        | `backend/app/multimodal/image_extractor.py` (exists), `backend/app/storage/s3.py`, `backend/tests/multimodal/test_image_extractor.py` (create)                           |
| 13B     | Describe images with a vision model                      | `backend/app/multimodal/vision.py` (exists), `backend/tests/multimodal/test_vision.py` (create)                                                                          |
| 13C-13D | Image and text-to-image retrieval                        | `backend/app/multimodal/vision.py`, `backend/app/multimodal/image_embedder.py` (create), `backend/app/retrieval/hybrid.py`                                               |
| 13E     | Structured table extraction and questions                | `backend/app/multimodal/table_processor.py` (exists), `backend/tests/multimodal/test_table_processor.py` (create)                                                        |
| 14      | Fuse text, image, table, and metadata results            | `backend/app/retrieval/hybrid.py`, `backend/app/multimodal/`, `backend/tests/multimodal/test_multimodal_retrieval.py` (create)                                           |
| 15      | Extract entities and relationships                       | `backend/app/graph/entities.py` (exists), `backend/app/graph/relationships.py` (exists), `backend/app/graph/neo4j.py` (exists), `backend/tests/graph/`                   |
| 15A     | Graph retrieval and vector/graph fusion                  | `backend/app/graph/graph_retrieval.py` (create), `backend/app/graph/neo4j.py`, `backend/app/retrieval/hybrid.py`, `backend/tests/graph/test_graph_retrieval.py` (create) |
| 16      | Agentic query routing                                    | `backend/app/agents/state.py` (exists), `backend/app/agents/router.py` (exists), `backend/app/agents/graph.py` (create), `backend/tests/agents/`                         |
| 17      | Answer grounding and citation validation                 | `backend/app/agents/validator.py` (exists), `backend/app/api/chat.py`, `backend/tests/agents/test_validator.py` (create)                                                 |
| 18      | Retrieval and generation evaluation                      | `backend/app/evaluation/dataset.py` (exists), `backend/app/evaluation/metrics.py` (exists), `data/evaluation/`, `backend/tests/evaluation/`                              |
| 19      | Ablation experiments                                     | `backend/app/evaluation/ablation.py` (exists), `data/evaluation/processed/`, `backend/tests/evaluation/`                                                                 |
| 20      | Query, retrieval, and model observability                | `backend/app/core/logging.py`, `backend/app/api/`, `backend/tests/api/`                                                                                                  |
| 21      | FastAPI application endpoints                            | `backend/app/api/documents.py`, `backend/app/api/search.py`, `backend/app/api/chat.py`, `backend/app/api/health.py`, `backend/app/main.py` (create)                      |
| 22      | Upload, chat, viewer, and source UI                      | `frontend/src/api/`, `frontend/src/components/`, `frontend/src/hooks/`, `frontend/src/pages/`, `frontend/package.json`                                                   |
| 23      | Authentication, authorization, validation, and isolation | `backend/app/core/security.py`, `backend/app/api/`, `backend/tests/`                                                                                                     |
| 24      | Containerized local services                             | `docker-compose.yml`, `docker/`, `backend/`, `frontend/`                                                                                                                 |
| 25      | Production deployment and operations                     | `backend/app/`, `frontend/`, `docker/`, `docker-compose.yml`, deployment files (create)                                                                                  |

### Phase 1 Connection

Phase 1 must be implemented and tested in this order:

```python
from backend.app.ingestion.loader import load_document
from backend.app.ingestion.chunker import chunk_document
from backend.app.ingestion.embedding import embed_chunks

document = load_document("data/raw/oops_notes.pdf")
chunks = chunk_document(document, chunk_size=500, overlap=50)
embedded_chunks = embed_chunks(chunks)
```

The resulting records retain `document_id`, `page_number`, `text`, and
`embedding`. Phase 1D consumes `embedded_chunks`; later phases should extend
these records instead of rebuilding PDF extraction or chunking logic.
v
Reranker
|
v
Parent/Child Retrieval
|
v
Contextual Retrieval
|
v
Multimodal Context
|
v
LLM / VLM
|
v
Answer + Citations
|
v
Evaluation Layer

````

---

# 4. Recommended Technology Stack

## Backend

Python

FastAPI

Pydantic

AsyncIO where appropriate

## Agent orchestration

LangGraph

Use LangGraph primarily for orchestration.

Do not hide the retrieval logic inside an agent.

You should understand every retrieval operation independently.

## LLM

Start with one model.

Possible choices:

- Qwen
- Llama
- GPT
- another strong instruction model

For multimodal functionality you need a vision-capable model.

## Embeddings

Start with one text embedding model.

Examples:

```text
BGE
E5
Nomic
````

Later add image/multimodal embeddings.

## Sparse retrieval

BM25

Implement the concept yourself first.

Then use a production implementation.

## Vector database

Recommended:

```text
ChromaDB
```

Alternative:

```text
Weaviate
Milvus
pgvector
```

ChromaDB is a good learning choice for local development because persistence and vector search are straightforward.

## Graph database

```text
Neo4j
```

## Object storage

Use:

```text
Amazon S3
```

or an S3-compatible service during development.

Possible local/development alternative:

```text
MinIO
```

## CDN

Use a CDN in front of object storage for assets that need browser/client delivery.

Example architecture:

```text
User
 ↓
CDN
 ↓
S3
```

Important:

Do NOT expose your S3 bucket publicly just to make the project work.

Use:

```text
Private S3 bucket
       ↓
Presigned URL
       ↓
Client/CDN access
```

where appropriate.

## Document processing

Start with:

```text
PyMuPDF
```

Then explore:

```text
Unstructured
OCR
table extraction
```

## Evaluation

Use:

```text
Ragas
```

plus your own evaluation scripts.

---

# 5. Repository Structure

Start with:

```text
multimodal-agentic-rag/
│
├── backend/
│   │
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   ├── search.py
│   │   │   └── chat.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   │
│   │   ├── ingestion/
│   │   │   ├── loader.py
│   │   │   ├── parser.py
│   │   │   ├── chunker.py
│   │   │   └── metadata.py
│   │   │
│   │   ├── storage/
│   │   │   └── s3.py
│   │   │
│   │   ├── retrieval/
│   │   │   ├── bm25.py
│   │   │   ├── vector.py
│   │   │   ├── hybrid.py
│   │   │   ├── reranker.py
│   │   │   ├── hyde.py
│   │   │   └── parent_child.py
│   │   │
│   │   ├── multimodal/
│   │   │   ├── image_extractor.py
│   │   │   ├── image_embedder.py
│   │   │   ├── table_processor.py
│   │   │   └── vision.py
│   │   │
│   │   ├── graph/
│   │   │   ├── entities.py
│   │   │   ├── relationships.py
│   │   │   └── graph_retrieval.py
│   │   │
│   │   ├── agents/
│   │   │   ├── state.py
│   │   │   ├── router.py
│   │   │   └── graph.py
│   │   │
│   │   └── evaluation/
│   │       ├── dataset.py
│   │       ├── retrieval.py
│   │       └── generation.py
│   │
│   └── tests/
│       ├── test_parser.py
│       ├── test_chunking.py
│       ├── test_bm25.py
│       ├── test_vector.py
│       ├── test_hybrid.py
│       ├── test_reranker.py
│       └── test_multimodal.py
│
├── frontend/
│
├── evaluation/
│
├── docs/
│
├── scripts/
│
├── docker/
│
├── .env.example
├── .gitignore
└── README.md
```

Do not create all these files immediately.

Create them as the project grows.

---

# 6. PHASE 0 — Environment Setup

First learn:

```text
Python virtual environments
pip
environment variables
Git
Docker basics
FastAPI basics
Pydantic basics
```

Create:

```text
backend/
frontend/
```

Create Python environment.

Install only the libraries required for Phase 1.

Do NOT install 50 libraries on day one.

---

# 7. PHASE 1 — Understand Basic RAG

Before advanced RAG, build the smallest possible RAG.

Pipeline:

```text
PDF
 ↓
Extract text
 ↓
Chunk
 ↓
Embed
 ↓
Vector database
 ↓
Similarity search
 ↓
LLM
```

Your first objective:

Upload one PDF.

Ask:

```text
"What is this document about?"
```

The system retrieves relevant chunks.

---

# 8. PHASE 1A — PDF Processing

Learn:

- PDF structure
- pages
- text blocks
- coordinates
- metadata

Implement:

```python
extract_text(pdf_path)
```

Output:

```text
Page 1
Text...

Page 2
Text...

Page 3
Text...
```

Do not use a complicated framework yet.

Understand what the PDF parser actually returns.

---

# 9. PHASE 1B — Chunking

Start with simple chunking.

Example:

```text
chunk_size = 500
overlap = 50
```

Then inspect:

```text
Chunk 1
Chunk 2
Chunk 3
...
```

Test:

- very small chunks
- medium chunks
- large chunks
- different overlaps

Record retrieval quality.

---

# 10. PHASE 1C — Embeddings

Learn:

```text
What is an embedding?
Why does semantic similarity work?
Cosine similarity
Vector dimensions
Distance metrics
```

Implement:

```python
embed(text)
```

Store:

```text
chunk_id
document_id
page_number
text
embedding
metadata
```

---

# 11. PHASE 1D — Vector Search

Implement:

```text
query
 ↓
embedding
 ↓
vector similarity
 ↓
top-k
```

Test manually.

Print:

```text
Query

Retrieved chunk 1
score = ...

Retrieved chunk 2
score = ...

Retrieved chunk 3
score = ...
```

You should understand why each chunk was retrieved.

---

# 12. PHASE 1E — Basic RAG

Now connect:

```text
Retriever
    ↓
Context
    ↓
LLM
```

Prompt:

```text
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
```

Add citations:

```text
Source: research_paper.pdf
Page: 12
```

---

# 13. PHASE 2 — Object Storage

Now introduce S3.

Architecture:

```text
Frontend
    |
    v
FastAPI
    |
    v
S3
```

Store:

```text
documents/
    document_id/
        original.pdf

images/
    document_id/
        page_1/
        page_2/
```

Database should store references:

```text
document_id
s3_key
filename
content_type
created_at
```

Do NOT store large PDFs directly inside your database.

---

# 14. PHASE 2A — S3 Upload

Implement:

```text
POST /documents/upload
```

Flow:

```text
Frontend
 ↓
FastAPI
 ↓
S3
```

Return:

```json
{
  "document_id": "...",
  "filename": "...",
  "status": "uploaded"
}
```

---

# 15. PHASE 2B — Presigned URLs

Learn why presigned URLs exist.

Implement:

```text
GET /documents/{id}/url
```

The backend generates a temporary URL.

The client can access the object without exposing permanent credentials.

---

# 16. PHASE 2C — CDN

Only after S3 works.

Architecture:

```text
                         ┌── S3
                         │
Client → CDN → Object Storage
```

Use CDN primarily for assets that need efficient delivery.

For sensitive/private documents:

```text
Private bucket
      ↓
Authorized backend
      ↓
Presigned access
```

Do not make all uploaded documents publicly accessible.

---

# 17. PHASE 3 — Advanced Chunking

Now compare:

```text
Fixed Chunking
Recursive Chunking
Semantic Chunking
Structure-Aware Chunking
Parent-Child Chunking
```

Create an experiment.

Same document.

Same queries.

Different chunking strategies.

Measure:

```text
Recall@5
Recall@10
MRR
```

Choose the best strategy based on data.

Do not simply assume semantic chunking is always better.

---

# 18. PHASE 4 — BM25

Learn:

```text
TF
IDF
Term frequency
Document frequency
Inverse document frequency
BM25
```

Understand why sparse retrieval works.

Implement a small BM25 search independently.

Then integrate a production implementation.

Test queries containing:

```text
exact technical terms
model names
acronyms
numbers
rare keywords
```

Compare with vector search.

---

# 19. PHASE 5 — Hybrid Search

Now implement:

```text
                Query
                  |
        +---------+---------+
        |                   |
        v                   v
      BM25              Vector Search
        |                   |
        v                   v
      Top K                Top K
        |                   |
        +---------+---------+
                  |
                  v
             RRF Fusion
                  |
                  v
               Top K
```

Implement Reciprocal Rank Fusion.

Test:

```text
Vector only
BM25 only
Hybrid
```

Compare retrieval metrics.

---

# 20. PHASE 6 — Reranking

Pipeline:

```text
Hybrid Retrieval
      ↓
Top 20
      ↓
Cross Encoder
      ↓
Top 5
```

Understand:

```text
Bi-encoder
vs
Cross-encoder
```

Measure:

```text
Before reranking
After reranking
```

Your goal is to prove whether reranking actually improves your dataset.

---

# 21. PHASE 7 — Query Rewriting

Implement a separate query-rewriting component.

Input:

```text
"what about its limitations?"
```

Output:

```text
"What are the limitations of
Retrieval Augmented Generation?"
```

Keep:

```text
original_query
rewritten_query
```

Do not overwrite the original query.

---

# 22. PHASE 8 — Query Expansion

Generate multiple retrieval queries.

Example:

```text
Original:

How does RAG reduce hallucinations?

Query 1:
How does retrieval augmented generation reduce hallucination?

Query 2:
How does external knowledge grounding improve LLM factuality?

Query 3:
How does retrieved context improve factual accuracy?
```

Retrieve for each.

Fuse results.

Compare against single-query retrieval.

---

# 23. PHASE 9 — HyDE

Implement:

```text
User Query
    ↓
Hypothetical Answer
    ↓
Embedding
    ↓
Vector Search
    ↓
Real Documents
```

Important:

HyDE's hypothetical answer is NOT the final answer.

It is only a retrieval aid.

Compare:

```text
Vector Search
vs
Query Rewriting
vs
HyDE
vs
Hybrid + HyDE
```

---

# 24. PHASE 10 — Parent-Child Retrieval

Create:

```text
Parent document
       |
       +--- Child chunk
       +--- Child chunk
       +--- Child chunk
```

Search children.

Return parent context.

Example metadata:

```json
{
  "parent_id": "section_12",
  "child_id": "section_12_chunk_3"
}
```

Evaluate whether larger context improves answer quality.

---

# 25. PHASE 11 — Metadata Filtering

Create metadata:

```json
{
  "document_id": "123",
  "page": 42,
  "section": "Experiments",
  "year": 2025,
  "document_type": "research_paper",
  "author": "..."
}
```

Support filters:

```text
year > 2024
document_type = research_paper
page = 42
section = experiments
```

Pipeline:

```text
Query
 ↓
Metadata Filter
 ↓
Hybrid Retrieval
 ↓
Reranking
```

---

# 26. PHASE 12 — Contextual Retrieval

Normal chunk:

```text
"The model achieved 92% accuracy."
```

Contextual chunk:

```text
"In the medical image classification experiment,
the proposed EfficientNet model achieved 92%
accuracy on the test dataset."
```

Store both:

```text
original_text
contextualized_text
```

Embed the contextualized version.

But preserve original text for citations.

---

# 27. PHASE 13 — Multimodal Document Processing

Now the actual multimodal part begins.

A PDF may contain:

```text
Text
Images
Tables
Charts
Figures
Equations
Diagrams
```

Do not treat everything as text.

Create separate representations.

```text
PDF
 |
 +--- Text
 |
 +--- Images
 |
 +--- Tables
 |
 +--- Metadata
```

---

# 28. PHASE 13A — Extract Images

For each page:

```text
PDF page
 ↓
Detect embedded images
 ↓
Extract image
 ↓
Store image in S3
```

Example:

```text
s3://bucket/documents/123/images/page_10_img_2.png
```

Store metadata:

```json
{
  "document_id": "123",
  "page": 10,
  "image_id": "img_2",
  "s3_key": "...",
  "type": "figure"
}
```

---

# 29. PHASE 13B — Image Understanding

Use a vision-capable model.

For every image:

```text
Image
 ↓
Vision Model
 ↓
Description
```

Example:

```text
Image:
Architecture diagram

Description:
"The diagram shows a RAG pipeline consisting
of document ingestion, embedding generation,
vector retrieval, reranking and generation."
```

Store:

```text
image
description
metadata
```

---

# 30. PHASE 13C — Image Retrieval

Create embeddings for image representations.

Possible architecture:

```text
Image
 ↓
Vision / Multimodal Encoder
 ↓
Image Embedding
 ↓
Vector DB
```

Now support:

```text
Text → Text Search

Image → Image Search
```

Eventually:

```text
Text Query → Multimodal Search
```

---

# 31. PHASE 13D — Text-to-Image Retrieval

Example:

```text
User:

"Find the architecture diagram showing
the retrieval pipeline."
```

System:

```text
Query
 ↓
Text Embedding
 ↓
Image/Multimodal Index
 ↓
Relevant Figure
```

Return:

```text
Figure 4
Page 8
S3/CDN asset
```

---

# 32. PHASE 13E — Tables

Do not immediately convert tables into plain text.

Represent:

```text
Table ID
Page
Rows
Columns
Caption
Source
```

Example:

```text
Model | Accuracy | F1
----------------------
A     | 91.2     | 90.1
B     | 94.5     | 93.7
```

Then support:

> Which model has the highest F1 score?

For numerical/table questions, structured retrieval may be better than normal semantic retrieval.

---

# 33. PHASE 14 — Multimodal Retrieval

Now combine:

```text
Text
Images
Tables
Metadata
```

Architecture:

```text
                     Query
                       |
             ┌─────────┼─────────┐
             ↓         ↓         ↓
           Text      Image      Table
          Search     Search     Search
             |         |         |
             └─────────┼─────────┘
                       ↓
                   Fusion
                       ↓
                   Reranker
                       ↓
                Multimodal Context
                       ↓
                    Vision LLM
```

---

# 34. PHASE 15 — Graph RAG

Extract entities:

```text
OpenAI
GPT-4
Transformer
RAG
Multimodal
```

Extract relationships:

```text
GPT-4 → developed_by → OpenAI

RAG → uses → Retrieval

Multimodal RAG → includes → Image Retrieval
```

Store:

```text
Neo4j
```

Graph:

```text
Entity
  |
Relationship
  |
Entity
```

---

# 35. PHASE 15A — Graph Retrieval

Query:

```text
How are GPT-4 and multimodal RAG related?
```

Graph retrieval:

```text
GPT-4
 ↓
supports
 ↓
vision
 ↓
multimodal applications
```

Combine graph context with vector retrieval.

---

# 36. PHASE 16 — Agentic Query Router

Now introduce LangGraph.

Do NOT use LangGraph before understanding the individual components.

The agent should decide:

```text
What kind of query is this?
```

Possible routes:

```text
TEXT
IMAGE
TABLE
GRAPH
HYBRID
```

Example:

```text
"What is RAG?"

→ Text RAG
```

```text
"Explain Figure 5."

→ Image RAG
```

```text
"Which model has the highest accuracy?"

→ Table retrieval
```

```text
"How are these two technologies related?"

→ Graph RAG
```

---

# 37. PHASE 16A — LangGraph State

Create state containing:

```text
original_query
rewritten_query
query_type
retrieved_documents
retrieved_images
retrieved_tables
graph_context
final_context
answer
citations
```

Conceptually:

```text
START
  ↓
Analyze Query
  ↓
Rewrite Query
  ↓
Route
  ↓
Retrieve
  ↓
Rerank
  ↓
Build Context
  ↓
Generate
  ↓
Validate
  ↓
END
```

---

# 38. PHASE 17 — Answer Validation

Don't blindly trust the LLM.

Add:

```text
Answer
 ↓
Citation Check
 ↓
Grounding Check
 ↓
Final Answer
```

Detect:

```text
Unsupported claims
Missing citations
Low-quality context
```

If context is insufficient:

```text
Retrieve again
```

This creates a useful agentic loop.

---

# 39. PHASE 18 — RAG Evaluation

This is mandatory.

Create a dataset:

```text
question
ground_truth
relevant_document
relevant_page
```

Example:

```json
{
  "question": "What is BM25?",
  "answer": "...",
  "document": "rag.pdf",
  "page": 12
}
```

Measure retrieval:

```text
Recall@1
Recall@5
Recall@10
MRR
NDCG
Precision@K
```

Measure generation:

```text
Faithfulness
Answer Relevance
Context Relevance
Citation Accuracy
```

---

# 40. PHASE 19 — Ablation Study

This will make the project significantly stronger.

Compare:

```text
A. Vector Search

B. BM25

C. Hybrid Search

D. Hybrid + Reranker

E. Hybrid + Query Rewrite

F. Hybrid + HyDE

G. Hybrid + Reranker + HyDE

H. Full Multimodal Pipeline
```

Create a table:

```text
Method                  Recall@5    MRR    Faithfulness

Vector
BM25
Hybrid
Hybrid + Reranker
Hybrid + Rewrite
Hybrid + HyDE
Full Pipeline
```

Use actual measured results.

This becomes evidence for your architecture decisions.

---

# 41. PHASE 20 — Observability

Log every query:

```text
request_id

original_query

rewritten_query

route

retrieval_strategy

retrieved_document_ids

reranker_scores

latency

token_usage

final_answer
```

Example:

```text
Query
 ↓
Rewrite: 120ms
 ↓
BM25: 30ms
 ↓
Vector: 70ms
 ↓
Fusion: 5ms
 ↓
Reranker: 180ms
 ↓
LLM: 1.2s
```

Now you can identify bottlenecks.

---

# 42. PHASE 21 — FastAPI

Expose APIs.

Example:

```text
POST /documents/upload

GET /documents

GET /documents/{id}

POST /search

POST /chat

GET /documents/{id}/assets

GET /health
```

The frontend should never directly contain retrieval logic.

Architecture:

```text
React
 ↓
FastAPI
 ↓
Services
 ↓
Retrieval
 ↓
Databases / S3
```

---

# 43. PHASE 22 — Frontend

Build:

```text
Upload page
Chat page
Document viewer
Source panel
Retrieved chunks
Retrieved images
Citation links
```

Chat response:

```text
Answer

Sources:
──────────────
📄 paper.pdf — Page 12
📄 paper.pdf — Page 15
🖼 Figure 4 — Page 8
```

Clicking the source should open the relevant asset/page.

This is where S3 + CDN becomes useful.

---

# 44. PHASE 23 — Security

Before calling it production-oriented, implement:

```text
Authentication
Authorization
Private S3
Presigned URLs
File validation
File size limits
Rate limiting
Input validation
Prompt injection protection
Document isolation
```

Most importantly:

A user must never be able to retrieve another user's documents.

Every retrieval operation should be scoped by:

```text
user_id
workspace_id
document_id
```

---

# 45. PHASE 24 — Docker

Containerize:

```text
Frontend
Backend
ChromaDB
Neo4j
```

S3 remains external.

Development:

```text
docker-compose
```

Eventually:

```text
Frontend
Backend
ChromaDB
Neo4j
       ↓
AWS / Cloud
       ↓
S3 + CDN
```

---

# 46. PHASE 25 — Production Architecture

Final architecture:

```text
                    Internet
                       |
                       v
                    CDN/WAF
                       |
                +------+------+
                |             |
                v             v
             Frontend       FastAPI
                              |
                              v
                         LangGraph
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
             ChromaDB       Neo4j          S3
                |             |             |
                +-------------+-------------+
                              |
                              v
                           LLM/VLM
                              |
                              v
                         Evaluation
                              |
                              v
                       Observability
```

---

# 47. Development Rules

For every feature create:

```text
1. Concept
2. Minimal implementation
3. Unit test
4. Integration test
5. Example
6. Failure case
7. Evaluation
8. Documentation
```

Example for BM25:

```text
Learn BM25
 ↓
Implement BM25
 ↓
Test exact keyword query
 ↓
Test semantic query
 ↓
Compare with vector search
 ↓
Measure Recall@K
 ↓
Document findings
 ↓
Move to Hybrid Search
```

---

# 48. What NOT to Do

Do not start with:

```text
LangChain
LangGraph
ChromaDB
Neo4j
S3
React
Docker
LLM
```

all at once.

You will get:

```text
500 lines of code
 ↓
something works
 ↓
you don't know why
 ↓
debugging becomes impossible
```

Instead:

```text
Pure Python
 ↓
Understand
 ↓
Test
 ↓
Add library
 ↓
Test
 ↓
Move forward
```

---

# 49. Learning Philosophy

For every component ask:

### What problem does it solve?

### What happens internally?

### What are its limitations?

### When should I NOT use it?

### How can I measure whether it helped?

For example:

BM25:

```text
Problem:
Exact keyword matching

Strength:
Rare/exact terms

Weakness:
Poor semantic understanding
```

Vector:

```text
Problem:
Semantic similarity

Strength:
Meaning

Weakness:
Exact terms / identifiers can be weaker
```

Hybrid:

```text
Combines both.
```

Reranker:

```text
Improves ordering of retrieved candidates.
```

HyDE:

```text
Improves query-document semantic alignment.
```

Graph RAG:

```text
Handles entity relationships.
```

Multimodal RAG:

```text
Handles information that cannot be represented adequately as plain text.
```

Agentic routing:

```text
Chooses which retrieval strategy should be used.
```

---

# 50. Git Strategy

Create one branch per phase.

```text
main

phase/01-basic-rag
phase/02-s3
phase/03-chunking
phase/04-bm25
phase/05-hybrid
phase/06-reranking
phase/07-query-rewriting
phase/08-hyde
phase/09-parent-child
phase/10-metadata
phase/11-contextual
phase/12-multimodal
phase/13-graph-rag
phase/14-agent-router
phase/15-evaluation
phase/16-fastapi
phase/17-frontend
phase/18-production
```

Each phase should end with:

```text
Working code
Tests
README update
Evaluation results
Git commit
```

---

# 51. Commit Examples

```text
feat: implement basic PDF ingestion

feat: add recursive chunking

feat: add vector retrieval

feat: implement BM25 retrieval

feat: add reciprocal rank fusion

feat: add cross encoder reranking

feat: implement query rewriting

feat: implement HyDE retrieval

feat: add parent child retrieval

feat: add metadata filtering

feat: add contextual retrieval

feat: add image extraction

feat: add multimodal retrieval

feat: add graph retrieval

feat: implement LangGraph router

feat: add RAG evaluation pipeline
```

---

# 52. Final Resume Description

Project:

**Multimodal Agentic RAG Platform**

Possible final resume bullets:

- Built a production-oriented multimodal RAG system supporting **text, images, tables, and document-level metadata**, with S3-backed asset storage and CDN-based delivery.

- Engineered a **hybrid retrieval pipeline** combining BM25, dense embeddings, reciprocal-rank fusion, cross-encoder reranking, query rewriting, HyDE, parent-child retrieval, and contextual retrieval.

- Developed an **agentic LangGraph router** that dynamically selects text, multimodal, table, graph, or hybrid retrieval strategies based on query intent.

- Implemented **Graph RAG with Neo4j** to retrieve entity relationships alongside semantic document context.

- Built an evaluation framework measuring **Recall@K, MRR, NDCG, faithfulness, context relevance, answer relevance, and citation accuracy**, including retrieval ablation experiments.

Only include metrics in the resume after you actually measure them.

---

# 53. Final Skill Outcome

After completing this project, you should be able to explain:

```text
What is RAG?
        ↓
Why vector search?
        ↓
Why BM25?
        ↓
Why hybrid search?
        ↓
Why reranking?
        ↓
Why query rewriting?
        ↓
Why HyDE?
        ↓
Why parent-child retrieval?
        ↓
Why metadata filtering?
        ↓
Why contextual retrieval?
        ↓
Why Graph RAG?
        ↓
Why Multimodal RAG?
        ↓
Why Agentic Routing?
        ↓
How do we evaluate RAG?
        ↓
How do we deploy it?
        ↓
How do we secure it?
        ↓
How do we optimize latency/cost?
```

The end goal is not just:

> "I know LangChain/LangGraph."

It should be:

> **"I understand how modern retrieval systems work internally and can design, implement, evaluate, and deploy one."**

---

# 54. YOUR FIRST MILESTONE

Do NOT start S3, Graph RAG, LangGraph, or multimodal processing yet.

Start here:

```text
PHASE 1

PDF
 ↓
PyMuPDF
 ↓
Extract text
 ↓
Chunk
 ↓
Embedding
 ↓
ChromaDB
 ↓
Similarity Search
 ↓
LLM
 ↓
Answer + Page Citation
```

Your first success condition:

```text
Upload:
    one PDF

Ask:
    5 different questions

System:
    retrieves relevant chunks

Output:
    answer + page numbers
```

Then manually inspect **every retrieved chunk**.

Once this works and you understand it, move to **BM25 → Hybrid Search → Reranking**.

Only after the text retrieval pipeline is solid should you add the multimodal layer.
