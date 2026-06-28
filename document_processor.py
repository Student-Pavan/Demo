from __future__ import annotations

from functools import lru_cache
from typing import List

import fitz
import torch
from PIL import Image
from transformers import pipeline


class DocumentProcessor:
    """Extract text from plain text, images, and PDFs using free HuggingFace models."""

    # Free OCR model available on HuggingFace hub — no API key required
    DEFAULT_OCR_MODEL = "microsoft/trocr-base-printed"
    FALLBACK_OCR_MODEL = "microsoft/trocr-small-printed"

    def __init__(
        self,
        ocr_model_name: str = DEFAULT_OCR_MODEL,
        max_pdf_pages: int = 10,
        min_pdf_text_chars: int = 80,
    ) -> None:
        self.ocr_model_name = ocr_model_name
        self.max_pdf_pages = max_pdf_pages
        self.min_pdf_text_chars = min_pdf_text_chars

    @property
    def ocr_pipeline(self):
        return _get_ocr_pipeline(self.ocr_model_name)

    def extract_from_text(self, text: str) -> str:
        return " ".join((text or "").split())

    def extract_from_image(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            # Resize large images for better OCR performance
            max_dim = 1024
            w, h = rgb.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            result = self.ocr_pipeline(rgb)
        return " ".join(item["generated_text"].strip() for item in result).strip()

    def extract_from_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        pages: List[str] = []

        try:
            page_count = min(len(doc), self.max_pdf_pages)
            for idx in range(page_count):
                page = doc.load_page(idx)
                page_text = " ".join(page.get_text("text").split())

                if len(page_text) >= self.min_pdf_text_chars:
                    pages.append(page_text)
                    continue

                # OCR fallback for scanned pages
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_result = self.ocr_pipeline(img)
                extracted = " ".join(item["generated_text"].strip() for item in ocr_result).strip()
                if extracted:
                    pages.append(extracted)
        finally:
            doc.close()

        return "\n".join(pages).strip()

    def detect_and_extract(
        self,
        input_type: str,
        text: str = "",
        image_path: str | None = None,
        pdf_path: str | None = None,
    ) -> str:
        if input_type == "Text":
            return self.extract_from_text(text)
        if input_type == "Image":
            return self.extract_from_image(image_path) if image_path else ""
        if input_type == "PDF":
            return self.extract_from_pdf(pdf_path) if pdf_path else ""
        return ""


@lru_cache(maxsize=2)
def _get_ocr_pipeline(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "image-to-text",
        model=model_name,
        device=device,
    )
