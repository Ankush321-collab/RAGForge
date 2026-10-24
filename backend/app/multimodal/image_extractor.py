import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pymupdf  # PyMuPDF
from PIL import Image

from backend.app.storage.s3 import S3Storage

logger = logging.getLogger(__name__)


class PDFImageExtractor:
    """Phase 13A: Extract and store embedded document figures/images."""

    def __init__(self, storage: Optional[S3Storage] = None) -> None:
        self.storage = storage or S3Storage()

    def extract_images_from_pdf(
        self,
        pdf_path: Union[str, Path],
        document_id: str = "doc",
        min_width: int = 40,
        min_height: int = 40,
    ) -> List[Dict[str, Any]]:
        """Extract all embedded raster images from each page of a PDF document."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        extracted_images: List[Dict[str, Any]] = []
        doc = pymupdf.open(str(pdf_path))

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list, start=1):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                image_ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip tiny icons / decorative artifacts
                if width < min_width or height < min_height:
                    continue

                image_id = f"img_p{page_num}_{img_idx}"
                object_key = f"documents/{document_id}/images/{image_id}.{image_ext}"

                # Store image into S3 or local fallback
                try:
                    self.storage.upload_bytes(
                        content=image_bytes,
                        object_key=object_key,
                        content_type=f"image/{image_ext}",
                    )
                except Exception as e:
                    logger.warning("Failed to store extracted image %s: %s", object_key, e)

                extracted_images.append({
                    "image_id": image_id,
                    "document_id": document_id,
                    "page_number": page_num,
                    "s3_key": object_key,
                    "format": image_ext,
                    "width": width,
                    "height": height,
                    "type": "figure",
                })

        doc.close()
        return extracted_images
