"""Probe PDF structure to determine vector vs raster content."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageProbe:
    """Probe results for a single page."""

    page_number: int
    width: float
    height: float
    text_block_count: int
    text_char_count: int
    drawing_count: int  # Vector path count
    image_count: int
    image_area_ratio: float  # Fraction of page area covered by images
    is_vector: bool
    is_raster: bool
    is_mixed: bool


@dataclass
class ProbeResult:
    """Probe results for an entire PDF."""

    path: str
    page_count: int
    pages: list[PageProbe] = field(default_factory=list)
    overall_vector: bool = False
    overall_raster: bool = False
    overall_mixed: bool = False

    def summary(self) -> str:
        lines = [
            f"PDF: {self.path}",
            f"Pages: {self.page_count}",
            f"Content type: {'VECTOR' if self.overall_vector else 'RASTER' if self.overall_raster else 'MIXED'}",
            "",
        ]
        for p in self.pages:
            lines.append(
                f"  Page {p.page_number}: "
                f"text_blocks={p.text_block_count}, chars={p.text_char_count}, "
                f"drawings={p.drawing_count}, images={p.image_count}, "
                f"img_area={p.image_area_ratio:.1%} → "
                f"{'vector' if p.is_vector else 'raster' if p.is_raster else 'mixed'}"
            )
        return "\n".join(lines)


def probe_pdf(pdf_path: str | Path) -> ProbeResult:
    """Probe a PDF to determine if it contains vector or raster content.

    Vector PDFs have extractable text objects and drawing paths.
    Raster PDFs contain embedded images with little/no vector text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ProbeResult with per-page and overall analysis.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    result = ProbeResult(path=str(pdf_path), page_count=len(doc))

    vector_pages = 0
    raster_pages = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        rect = page.rect
        page_area = rect.width * rect.height

        # Extract text blocks
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        text_blocks = text_dict.get("blocks", [])
        text_block_count = sum(1 for b in text_blocks if b.get("type") == 0)

        # Count characters
        char_count = 0
        for block in text_blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        char_count += len(span.get("text", ""))

        # Extract vector drawings
        drawings = page.get_drawings()
        drawing_count = len(drawings)

        # Extract images
        images = page.get_images(full=True)
        image_count = len(images)

        # Calculate image area coverage
        image_area = 0.0
        for img in images:
            xref = img[0]
            try:
                img_rects = page.get_image_rects(xref)
                for ir in img_rects:
                    image_area += ir.width * ir.height
            except Exception:
                pass

        image_area_ratio = image_area / page_area if page_area > 0 else 0.0

        # Classify the page
        has_substantial_text = char_count > 20
        has_substantial_drawings = drawing_count > 10
        has_large_images = image_area_ratio > 0.5

        is_vector = (has_substantial_text or has_substantial_drawings) and not has_large_images
        is_raster = has_large_images and not has_substantial_text
        is_mixed = has_large_images and has_substantial_text

        if is_vector:
            vector_pages += 1
        elif is_raster:
            raster_pages += 1

        page_probe = PageProbe(
            page_number=page_num + 1,
            width=rect.width,
            height=rect.height,
            text_block_count=text_block_count,
            text_char_count=char_count,
            drawing_count=drawing_count,
            image_count=image_count,
            image_area_ratio=image_area_ratio,
            is_vector=is_vector,
            is_raster=is_raster,
            is_mixed=is_mixed,
        )
        result.pages.append(page_probe)

    doc.close()

    # Overall classification
    total = len(result.pages)
    if total == 0:
        result.overall_vector = False
        result.overall_raster = False
    elif vector_pages == total:
        result.overall_vector = True
    elif raster_pages == total:
        result.overall_raster = True
    else:
        result.overall_mixed = True

    logger.info(result.summary())
    return result
