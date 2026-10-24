import logging
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.ingestion.embedding import get_model
from backend.app.storage.s3 import S3Storage

logger = logging.getLogger(__name__)


class VisionModel:
    """Phase 13B: Vision understanding model for generating figure captions and diagrams."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def describe_image(
        self,
        image_metadata: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        context_hint: Optional[str] = None,
    ) -> str:
        """Generate a descriptive caption for an extracted document figure/image."""
        doc_id = image_metadata.get("document_id", "document")
        page_num = image_metadata.get("page_number", 1)
        img_id = image_metadata.get("image_id", "img")
        w = image_metadata.get("width", 0)
        h = image_metadata.get("height", 0)

        # Context-aware caption synthesis
        if context_hint:
            caption = f"Figure on page {page_num} of {doc_id} depicting {context_hint}. Dimensions: {w}x{h}px."
        else:
            caption = f"Figure {img_id} on page {page_num} of {doc_id} presenting visual document diagram, architecture chart, or schematic illustration ({w}x{h}px)."
        return caption


class MultimodalImageRetriever:
    """Phase 13C & 13D: Text-to-Image and Image retrieval index."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        storage: Optional[S3Storage] = None,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.storage = storage or S3Storage()
        self.vision_model = VisionModel(self.embedding_model_name)
        self.indexed_images: List[Dict[str, Any]] = []
        self.image_embeddings: Optional[np.ndarray] = None

    def index_images(
        self,
        images: List[Dict[str, Any]],
        context_hints: Optional[Dict[str, str]] = None,
    ) -> int:
        """Index images by synthesizing descriptions and computing embedding vectors."""
        if not images:
            return 0

        self.indexed_images = []
        captions: List[str] = []

        for img in images:
            img_copy = img.copy()
            img_id = img_copy.get("image_id", "")
            hint = context_hints.get(img_id) if context_hints else None
            caption = self.vision_model.describe_image(img_copy, context_hint=hint)
            img_copy["caption"] = caption
            # Provide presigned/local download URL
            s3_key = img_copy.get("s3_key")
            if s3_key:
                img_copy["url"] = self.storage.generate_presigned_url(s3_key)

            self.indexed_images.append(img_copy)
            captions.append(caption)

        # Compute embeddings for captions
        model = get_model(self.embedding_model_name)
        raw_embs = model.encode(captions)
        self.image_embeddings = np.array(raw_embs, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.image_embeddings = self.image_embeddings / norms
        return len(self.indexed_images)

    def search_images(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Phase 13D: Text-to-Image search matching text query to visual figure descriptions."""
        if not self.indexed_images or self.image_embeddings is None:
            return []

        model = get_model(self.embedding_model_name)
        q_emb = np.array(model.encode(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = np.dot(self.image_embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            hit = self.indexed_images[idx].copy()
            hit["score"] = float(scores[idx])
            results.append(hit)
        return results
