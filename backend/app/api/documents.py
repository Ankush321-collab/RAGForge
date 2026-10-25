import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.core.security import validate_upload, validate_upload_size
from backend.app.graph.entities import EntityExtractor
from backend.app.graph.relationships import RelationshipExtractor
from backend.app.ingestion.chunker import chunk_parent_child
from backend.app.ingestion.loader import load_document
from backend.app.retrieval.vector import (
    SentenceTransformerEmbeddings,
    chunks_to_documents,
    get_vector_store,
)
from backend.app.storage.s3 import S3Storage

logger = logging.getLogger("ragforge.documents")
router = APIRouter(prefix="/documents", tags=["documents"])
storage = S3Storage()

DATA_DOCS_DIR = Path("data/documents")
DATA_DOCS_DIR.mkdir(parents=True, exist_ok=True)


class DocumentUploadResponse(BaseModel):
    document_id: str
    object_key: str
    filename: str
    content_type: str
    size_bytes: int
    num_pages: int = 1
    num_chunks: int = 0
    num_images: int = 0
    status: str = "indexed"


class PresignedUrlResponse(BaseModel):
    document_id: str
    object_key: str
    url: str
    expires_in: int


class GraphNode(BaseModel):
    id: str
    label: str
    type: str


class GraphLink(BaseModel):
    source: str
    target: str
    label: str


class DocumentGraphResponse(BaseModel):
    document_id: str
    nodes: List[GraphNode]
    links: List[GraphLink]


def _ensure_default_corpus():
    """Ensure default corpora (oops_notes.pdf and os_notes.pdf) are indexed and available."""
    corpora = [
        ("default_oops_corpus", Path("data/raw/oops_notes.pdf"), "oops_notes.pdf"),
        ("default_os_corpus", Path("data/raw/os_notes.pdf"), "os_notes.pdf"),
    ]

    for default_id, raw_pdf, filename in corpora:
        meta_path = DATA_DOCS_DIR / default_id / "metadata.json"
        if not meta_path.exists() and raw_pdf.exists():
            try:
                target_dir = DATA_DOCS_DIR / default_id
                target_dir.mkdir(parents=True, exist_ok=True)
                doc_data = load_document(raw_pdf)
                pages = doc_data["pages"]

                parents, children = chunk_parent_child(
                    pages,
                    parent_size=1000,
                    child_size=300,
                    child_overlap=50,
                    document_id=default_id,
                )

                for c in children:
                    c["document_id"] = default_id
                    c["filename"] = filename

                vs = get_vector_store()
                child_docs = chunks_to_documents(children)
                if child_docs:
                    emb_fn = SentenceTransformerEmbeddings()
                    texts = [d.page_content for d in child_docs]
                    embeddings = emb_fn.embed_documents(texts)
                    ids = [f"{default_id}_{c.get('chunk_id', i)}" for i, c in enumerate(children)]
                    vs._collection.upsert(
                        ids=ids,
                        documents=texts,
                        metadatas=[d.metadata for d in child_docs],
                        embeddings=embeddings,
                    )

                # Extract sample entities & relationships
                full_text = " ".join([p["text"] for p in pages[:12]])
                ee = EntityExtractor()
                re = RelationshipExtractor()
                entities = ee.extract_entities(full_text)
                relationships = re.extract_relationships(full_text, entities)

                meta = {
                    "document_id": default_id,
                    "filename": filename,
                    "object_key": f"documents/{default_id}/{filename}",
                    "content_type": "application/pdf",
                    "size_bytes": raw_pdf.stat().st_size,
                    "num_pages": len(pages),
                    "num_chunks": len(children),
                    "num_images": 2,
                    "status": "indexed",
                    "entities": [e.to_dict() for e in entities],
                    "relationships": [r.to_dict() for r in relationships],
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                (target_dir / "chunks.json").write_text(json.dumps(children[:50], indent=2), encoding="utf-8")
                logger.info("Auto-indexed %s with %d pages and %d chunks", filename, len(pages), len(children))
            except Exception as e:
                logger.warning("Could not auto-index %s: %s", filename, e)


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload any PDF, persist to S3/local storage, and automatically index for RAG."""
    filename = validate_upload(file)
    content = await file.read()
    validate_upload_size(len(content))

    document_id = str(uuid4())[:8]
    object_key = storage.build_object_key(document_id, filename)
    storage.upload_bytes(content, object_key, file.content_type or "application/pdf")

    # Save to local disk for PyMuPDF / pypdf parsing
    doc_dir = DATA_DOCS_DIR / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = doc_dir / filename
    pdf_path.write_bytes(content)

    # 1. Parse pages
    try:
        parsed_doc = load_document(pdf_path)
        pages = parsed_doc.get("pages", [])
    except Exception as e:
        logger.error("Failed to parse PDF %s: %s", filename, e)
        pages = [{"page_number": 1, "text": ""}]

    # 2. Chunking (Parent-Child)
    try:
        parents, children = chunk_parent_child(
            pages,
            parent_size=1000,
            child_size=300,
            child_overlap=50,
            document_id=document_id,
        )
    except Exception as e:
        logger.error("Failed to chunk PDF %s: %s", filename, e)
        parents, children = [], []

    # Tag metadata
    for c in children:
        c["document_id"] = document_id
        c["filename"] = filename
    for p in parents:
        p["document_id"] = document_id
        p["filename"] = filename

    # 3. Embed & Upsert into Chroma Vector Store
    if children:
        try:
            vs = get_vector_store()
            child_docs = chunks_to_documents(children)
            emb_fn = SentenceTransformerEmbeddings()
            texts = [d.page_content for d in child_docs]
            embeddings = emb_fn.embed_documents(texts)
            ids = [f"{document_id}_{c.get('chunk_id', i)}" for i, c in enumerate(children)]
            vs._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=[d.metadata for d in child_docs],
                embeddings=embeddings,
            )
            logger.info("Indexed %d chunks for document %s in Chroma", len(children), document_id)
        except Exception as e:
            logger.error("Vector store indexing failed for %s: %s", document_id, e)

    # 4. Extract figures & images if possible
    images = []
    try:
        from backend.app.multimodal.image_extractor import PDFImageExtractor
        extractor = PDFImageExtractor(storage=storage)
        images = extractor.extract_images_from_pdf(pdf_path, document_id=document_id)
    except Exception as e:
        logger.warning("Image extraction notice for %s: %s", document_id, e)

    # 5. Extract Entities & Relationships for Graph RAG
    entities = []
    relationships = []
    try:
        full_text = " ".join([p["text"] for p in pages[:10]])
        ee = EntityExtractor()
        re = RelationshipExtractor()
        entities = ee.extract_entities(full_text)
        relationships = re.extract_relationships(full_text, entities)
    except Exception as e:
        logger.warning("Graph extraction notice for %s: %s", document_id, e)

    # 6. Save metadata manifest
    meta = {
        "document_id": document_id,
        "object_key": object_key,
        "filename": filename,
        "content_type": file.content_type or "application/pdf",
        "size_bytes": len(content),
        "num_pages": len(pages),
        "num_chunks": len(children),
        "num_images": len(images),
        "status": "indexed",
        "entities": [e.to_dict() for e in entities],
        "relationships": [r.to_dict() for r in relationships],
    }
    (doc_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (doc_dir / "chunks.json").write_text(json.dumps(children[:50], indent=2), encoding="utf-8")

    return DocumentUploadResponse(
        document_id=document_id,
        object_key=object_key,
        filename=filename,
        content_type=file.content_type or "application/pdf",
        size_bytes=len(content),
        num_pages=len(pages),
        num_chunks=len(children),
        num_images=len(images),
        status="indexed",
    )


@router.get("", response_model=List[DocumentUploadResponse])
def list_documents() -> List[DocumentUploadResponse]:
    """List all indexed documents available for RAG queries."""
    _ensure_default_corpus()
    docs = []

    if DATA_DOCS_DIR.exists():
        for d_id_dir in sorted(DATA_DOCS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d_id_dir.is_dir():
                meta_file = d_id_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        data = json.loads(meta_file.read_text(encoding="utf-8"))
                        docs.append(DocumentUploadResponse(
                            document_id=data["document_id"],
                            object_key=data.get("object_key", f"documents/{data['document_id']}/{data['filename']}"),
                            filename=data["filename"],
                            content_type=data.get("content_type", "application/pdf"),
                            size_bytes=data.get("size_bytes", 0),
                            num_pages=data.get("num_pages", 1),
                            num_chunks=data.get("num_chunks", 0),
                            num_images=data.get("num_images", 0),
                            status=data.get("status", "indexed"),
                        ))
                    except Exception:
                        pass

    return docs


@router.get("/{document_id}/graph", response_model=DocumentGraphResponse)
def get_document_graph(document_id: str) -> DocumentGraphResponse:
    """Return Knowledge Graph entity nodes and directed relationship edges for a document."""
    meta_file = DATA_DOCS_DIR / document_id / "metadata.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    data = json.loads(meta_file.read_text(encoding="utf-8"))
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    nodes = [
        GraphNode(
            id=e["name"],
            label=e["name"],
            type=e.get("entity_type", "Concept")
        )
        for e in entities
    ]

    links = [
        GraphLink(
            source=r["source"],
            target=r["target"],
            label=r.get("relation_type", "RELATED_TO")
        )
        for r in relationships
    ]

    return DocumentGraphResponse(
        document_id=document_id,
        nodes=nodes,
        links=links
    )


@router.get("/{document_id}/chunks")
def get_document_chunks(document_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return sample child chunks with page numbers and text for previewing."""
    chunks_file = DATA_DOCS_DIR / document_id / "chunks.json"
    if not chunks_file.exists():
        return []
    try:
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        return chunks[:limit]
    except Exception:
        return []


@router.get("/{document_id}/url", response_model=PresignedUrlResponse)
def get_document_url(document_id: str) -> PresignedUrlResponse:
    object_key = storage.find_document_key(document_id)
    if object_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    expiry = get_settings().s3_presigned_url_expiry
    return PresignedUrlResponse(
        document_id=document_id,
        object_key=object_key,
        url=storage.generate_presigned_url(object_key, expiry),
        expires_in=expiry,
    )


@router.get("/download/{object_key:path}")
def download_document(object_key: str) -> Response:
    try:
        content = storage.download_bytes(object_key)
        media_type = "application/pdf" if object_key.endswith(".pdf") else "application/octet-stream"
        if object_key.endswith(".png"):
            media_type = "image/png"
        elif object_key.endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        return Response(content=content, media_type=media_type)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Object not found")
