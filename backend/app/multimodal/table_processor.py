import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pdfplumber

logger = logging.getLogger(__name__)


class TableProcessor:
    """Phase 13E: Structured Table Extraction and Tabular Retrieval."""

    @staticmethod
    def table_to_markdown(headers: List[str], rows: List[List[Any]]) -> str:
        """Format table headers and rows as a clean Markdown table string."""
        if not headers and not rows:
            return ""

        clean_headers = [str(h or "").strip() for h in headers]
        col_count = len(clean_headers) if clean_headers else (len(rows[0]) if rows else 0)
        if not clean_headers:
            clean_headers = [f"Col_{i+1}" for i in range(col_count)]

        header_line = "| " + " | ".join(clean_headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(clean_headers)) + " |"

        row_lines = []
        for row in rows:
            clean_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
            # Pad or truncate row to match header columns
            if len(clean_row) < len(clean_headers):
                clean_row.extend([""] * (len(clean_headers) - len(clean_row)))
            row_lines.append("| " + " | ".join(clean_row[:len(clean_headers)]) + " |")

        return "\n".join([header_line, sep_line] + row_lines)

    def extract_tables_from_pdf(
        self,
        pdf_path: Union[str, Path],
        document_id: str = "doc",
    ) -> List[Dict[str, Any]]:
        """Extract all structured tables across pages in a PDF file."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        extracted_tables: List[Dict[str, Any]] = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    for t_idx, raw_table in enumerate(tables, start=1):
                        if not raw_table or len(raw_table) < 2:
                            continue

                        headers = raw_table[0]
                        rows = raw_table[1:]
                        markdown = self.table_to_markdown(headers, rows)

                        table_id = f"table_p{page_idx}_{t_idx}"
                        extracted_tables.append({
                            "table_id": table_id,
                            "document_id": document_id,
                            "page_number": page_idx,
                            "headers": headers,
                            "rows": rows,
                            "markdown": markdown,
                            "text": f"Table on Page {page_idx}:\n{markdown}",
                            "type": "table",
                        })
        except Exception as e:
            logger.warning("Table extraction encountered an error on %s: %s", pdf_path, e)

        return extracted_tables

    def search_tables(
        self,
        query: str,
        tables: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Rank tables based on keyword overlaps in columns, headers, and values."""
        if not tables:
            return []

        query_terms = set(query.lower().split())
        scored_tables = []

        for tbl in tables:
            score = 0.0
            # Check headers match
            for h in tbl.get("headers", []):
                if any(qt in str(h).lower() for qt in query_terms):
                    score += 2.0

            # Check rows / markdown match
            md_text = tbl.get("markdown", "").lower()
            for qt in query_terms:
                if qt in md_text:
                    score += 1.0

            if score > 0:
                t_copy = tbl.copy()
                t_copy["score"] = score
                scored_tables.append(t_copy)

        scored_tables.sort(key=lambda x: x["score"], reverse=True)
        return scored_tables[:top_k]
