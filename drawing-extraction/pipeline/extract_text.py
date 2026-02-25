"""Extract text elements with position, rotation, and value from PDF pages."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from config import ExtractionConfig

logger = logging.getLogger(__name__)


@dataclass
class TextElement:
    """A text element extracted from a drawing."""

    text: str
    x: float  # Center x position in PDF coordinates
    y: float  # Center y position in PDF coordinates
    rotation: float  # Degrees, counter-clockwise from horizontal
    font_size: float
    font_name: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_number: int
    view_index: int | None = None  # Which detail view this belongs to
    category: str | None = None  # Classified type (see classify_text)

    @property
    def is_numeric(self) -> bool:
        """Check if this text is a numeric dimension value."""
        cleaned = self.text.strip().replace('"', '').replace("'", "")
        # Match numbers like 12, 12.5, 12.500, 1/2, 3/4, 12-3/8
        return bool(re.match(
            r'^[\d]+(?:[./][\d]+)?(?:\s*-\s*[\d]+/[\d]+)?$',
            cleaned
        ))

    @property
    def numeric_value(self) -> float | None:
        """Parse the text as a numeric value, handling fractions."""
        cleaned = self.text.strip().replace('"', '').replace("'", "").replace(" ", "")
        try:
            # Simple decimal: 12.500
            return float(cleaned)
        except ValueError:
            pass

        # Mixed fraction: 12-3/8
        match = re.match(r'^([\d]+)-([\d]+)/([\d]+)$', cleaned)
        if match:
            whole = int(match.group(1))
            num = int(match.group(2))
            den = int(match.group(3))
            return whole + num / den

        # Simple fraction: 3/8
        match = re.match(r'^([\d]+)/([\d]+)$', cleaned)
        if match:
            return int(match.group(1)) / int(match.group(2))

        return None


def _compute_rotation_from_matrix(matrix: tuple) -> float:
    """Compute rotation angle from a transformation matrix.

    The matrix is (a, b, c, d, e, f) representing:
        [a  b  0]
        [c  d  0]
        [e  f  1]

    Rotation angle = atan2(b, a) in degrees.
    """
    a, b = matrix[0], matrix[1]
    angle_rad = math.atan2(b, a)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def _snap_rotation(angle: float, tolerance: float) -> float:
    """Snap rotation to nearest cardinal angle if within tolerance."""
    cardinals = [0, 90, 180, 270, -90, -180, -270, 360]
    for c in cardinals:
        if abs(angle - c) < tolerance:
            return c % 360
    return round(angle, 1)


def classify_text(text: str, config: ExtractionConfig) -> str:
    """Classify a text element into a category.

    Categories:
        - "dimension": Numeric dimension value
        - "steel_shape": Steel shape callout (HSS, W, L, etc.)
        - "piece_mark": Assembly identifier (C-01, B-14, etc.)
        - "section_cut": Section indicator (A-A, B-B, etc.)
        - "weld_symbol": Weld specification
        - "bolt_spec": Bolt specification
        - "note": General annotation/note
        - "unknown": Unclassified
    """
    cleaned = text.strip()

    if not cleaned:
        return "unknown"

    # Steel shape callouts
    for pattern in config.steel_shape_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return "steel_shape"

    # Piece marks
    for pattern in config.piece_mark_patterns:
        if re.match(pattern, cleaned):
            return "piece_mark"

    # Section cuts (A-A, B-B, etc.)
    if re.match(r'^[A-Z]\s*-\s*[A-Z]$', cleaned):
        return "section_cut"

    # Bolt specs
    if re.search(r'\d+/\d+"\s*(?:A325|A490|A307)', cleaned, re.IGNORECASE):
        return "bolt_spec"
    if re.search(r'\d+[xX×]\s*\d+/\d+"\s*(?:bolt|A325|A490)', cleaned, re.IGNORECASE):
        return "bolt_spec"

    # Weld symbols (simplified detection)
    if re.search(r'(?:E70|E60|FILLET|CJP|PJP)', cleaned, re.IGNORECASE):
        return "weld_symbol"

    # Degree symbol → angular dimension
    if '°' in cleaned and re.search(r'\d', cleaned):
        return "dimension"

    # Numeric dimension
    cleaned_nodim = cleaned.replace('"', '').replace("'", "").replace(" ", "")
    if re.match(r'^[\d]+(?:[./][\d]+)?(?:-[\d]+/[\d]+)?$', cleaned_nodim):
        return "dimension"

    # General notes (multi-word text)
    if len(cleaned.split()) > 2:
        return "note"

    return "unknown"


def extract_text(
    page: fitz.Page,
    page_number: int,
    config: ExtractionConfig | None = None,
    clip_rect: fitz.Rect | None = None,
    view_index: int | None = None,
) -> list[TextElement]:
    """Extract all text elements from a page or clipped region.

    Uses PyMuPDF's detailed text extraction to get per-span information
    including transformation matrices for rotation detection.

    Args:
        page: A PyMuPDF page object.
        page_number: 1-based page number.
        config: Extraction configuration.
        clip_rect: Optional region to restrict extraction to.
        view_index: Index of the view region this belongs to.

    Returns:
        List of TextElement objects with position, rotation, and classification.
    """
    if config is None:
        config = ExtractionConfig()

    flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
    text_dict = page.get_text("rawdict", clip=clip_rect, flags=flags)

    elements = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip image blocks
            continue

        for line in block.get("lines", []):
            line_dir = line.get("dir", (1.0, 0.0))
            line_angle = math.degrees(math.atan2(line_dir[1], line_dir[0]))

            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue

                font_size = span.get("size", 0)
                if font_size < config.min_text_size:
                    continue

                font_name = span.get("font", "")
                bbox = span.get("bbox", (0, 0, 0, 0))

                # Compute rotation from the line direction
                rotation = _snap_rotation(line_angle, config.rotation_snap_tolerance)

                # Center position
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                category = classify_text(text, config)

                elem = TextElement(
                    text=text,
                    x=cx,
                    y=cy,
                    rotation=rotation,
                    font_size=font_size,
                    font_name=font_name,
                    bbox=bbox,
                    page_number=page_number,
                    view_index=view_index,
                    category=category,
                )
                elements.append(elem)

    logger.info(
        f"Page {page_number}"
        f"{f', view {view_index}' if view_index is not None else ''}"
        f": extracted {len(elements)} text elements"
    )

    # Log category breakdown
    cats = {}
    for e in elements:
        cats[e.category] = cats.get(e.category, 0) + 1
    for cat, count in sorted(cats.items()):
        logger.debug(f"    {cat}: {count}")

    return elements


def extract_text_from_pdf(
    pdf_path: str | Path,
    config: ExtractionConfig | None = None,
) -> dict[int, list[TextElement]]:
    """Extract text from all pages of a PDF.

    Args:
        pdf_path: Path to the PDF.
        config: Extraction configuration.

    Returns:
        Dict mapping page number (1-based) to list of TextElement objects.
    """
    if config is None:
        config = ExtractionConfig()

    doc = fitz.open(str(pdf_path))
    all_text: dict[int, list[TextElement]] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        elements = extract_text(page, page_num + 1, config)
        all_text[page_num + 1] = elements

    doc.close()
    return all_text
