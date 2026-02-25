"""Sheet decomposition — detect detail view bounding boxes and crop regions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from config import ExtractionConfig

logger = logging.getLogger(__name__)


@dataclass
class ViewRegion:
    """A detected detail view region on a drawing page."""

    page_number: int
    view_index: int
    rect: fitz.Rect  # Bounding rectangle in PDF coordinates
    label: str | None = None  # View title label if detected (e.g., "FRONT VIEW")
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    def __post_init__(self):
        self.x0 = self.rect.x0
        self.y0 = self.rect.y0
        self.x1 = self.rect.x1
        self.y1 = self.rect.y1

    @property
    def width(self) -> float:
        return self.rect.width

    @property
    def height(self) -> float:
        return self.rect.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


def _find_rectangles_from_drawings(page: fitz.Page, config: ExtractionConfig) -> list[fitz.Rect]:
    """Find rectangles from vector drawing paths.

    Looks for closed rectangular paths that could be detail view borders.
    """
    drawings = page.get_drawings()
    rectangles = []

    for path in drawings:
        items = path.get("items", [])
        if not items:
            continue

        # Collect all points in this path
        points = []
        for item in items:
            op = item[0]
            if op == "l":  # line
                points.append((item[1].x, item[1].y))
                points.append((item[2].x, item[2].y))
            elif op == "re":  # rectangle operator
                r = item[1]
                if isinstance(r, fitz.Rect):
                    if (r.width >= config.min_view_width and
                            r.height >= config.min_view_height):
                        rectangles.append(r)
                continue
            elif op == "c":  # curve — skip
                continue

        # Check if the points form a rectangle (4 corners, closed path)
        if len(points) >= 8:  # 4 line segments = 8 points
            unique_points = _deduplicate_points(points, config.border_tolerance)
            if len(unique_points) == 4:
                rect = _points_to_rect(unique_points)
                if (rect and
                        rect.width >= config.min_view_width and
                        rect.height >= config.min_view_height):
                    rectangles.append(rect)

    return rectangles


def _find_rectangles_from_lines(page: fitz.Page, config: ExtractionConfig) -> list[fitz.Rect]:
    """Find rectangles by looking for axis-aligned line groups forming closed boxes.

    Fallback approach when rectangle operators aren't used in the PDF.
    """
    drawings = page.get_drawings()
    h_lines = []  # Horizontal lines
    v_lines = []  # Vertical lines
    tol = config.border_tolerance

    for path in drawings:
        for item in path.get("items", []):
            if item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            dx = abs(p2.x - p1.x)
            dy = abs(p2.y - p1.y)

            if dy < tol and dx >= config.min_view_width * 0.5:
                # Horizontal line
                h_lines.append((min(p1.x, p2.x), max(p1.x, p2.x), (p1.y + p2.y) / 2))
            elif dx < tol and dy >= config.min_view_height * 0.5:
                # Vertical line
                v_lines.append((min(p1.y, p2.y), max(p1.y, p2.y), (p1.x + p2.x) / 2))

    # Try to match horizontal pairs with vertical pairs to form rectangles
    rectangles = []
    used_h = set()
    used_v = set()

    for i, (hx0_a, hx1_a, hy_a) in enumerate(h_lines):
        for j, (hx0_b, hx1_b, hy_b) in enumerate(h_lines):
            if j <= i or j in used_h or i in used_h:
                continue
            # Check if these two horizontal lines have similar x-extent
            if abs(hx0_a - hx0_b) > tol or abs(hx1_a - hx1_b) > tol:
                continue
            h_span = hx1_a - hx0_a
            v_span = abs(hy_b - hy_a)
            if h_span < config.min_view_width or v_span < config.min_view_height:
                continue

            # Look for matching vertical lines
            top_y = min(hy_a, hy_b)
            bot_y = max(hy_a, hy_b)
            left_found = False
            right_found = False

            for k, (vy0, vy1, vx) in enumerate(v_lines):
                if k in used_v:
                    continue
                if abs(vy0 - top_y) < tol and abs(vy1 - bot_y) < tol:
                    if abs(vx - hx0_a) < tol:
                        left_found = True
                    elif abs(vx - hx1_a) < tol:
                        right_found = True

            if left_found and right_found:
                rect = fitz.Rect(hx0_a, top_y, hx1_a, bot_y)
                rectangles.append(rect)
                used_h.add(i)
                used_h.add(j)

    return rectangles


def _deduplicate_points(
    points: list[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    """Remove near-duplicate points."""
    unique = []
    for px, py in points:
        is_dup = False
        for ux, uy in unique:
            if abs(px - ux) < tolerance and abs(py - uy) < tolerance:
                is_dup = True
                break
        if not is_dup:
            unique.append((px, py))
    return unique


def _points_to_rect(points: list[tuple[float, float]]) -> fitz.Rect | None:
    """Convert 4 corner points to a Rect if they form an axis-aligned rectangle."""
    if len(points) != 4:
        return None

    xs = sorted(set(round(p[0], 1) for p in points))
    ys = sorted(set(round(p[1], 1) for p in points))

    # An axis-aligned rectangle has exactly 2 unique x values and 2 unique y values
    if len(xs) == 2 and len(ys) == 2:
        return fitz.Rect(xs[0], ys[0], xs[1], ys[1])

    return None


def _filter_nested_rects(rects: list[fitz.Rect], tolerance: float = 5.0) -> list[fitz.Rect]:
    """Remove rectangles that are fully contained within larger ones.

    Keep only the outermost view borders (discard inner subdivision lines).
    Also remove near-duplicate rectangles.
    """
    if not rects:
        return []

    # Sort by area descending
    rects_sorted = sorted(rects, key=lambda r: r.width * r.height, reverse=True)
    keep = []

    for rect in rects_sorted:
        is_contained = False
        is_duplicate = False
        for kept in keep:
            # Check near-duplicate
            if (abs(rect.x0 - kept.x0) < tolerance and
                    abs(rect.y0 - kept.y0) < tolerance and
                    abs(rect.x1 - kept.x1) < tolerance and
                    abs(rect.y1 - kept.y1) < tolerance):
                is_duplicate = True
                break
            # Check containment (with tolerance)
            if (rect.x0 >= kept.x0 - tolerance and
                    rect.y0 >= kept.y0 - tolerance and
                    rect.x1 <= kept.x1 + tolerance and
                    rect.y1 <= kept.y1 + tolerance):
                is_contained = True
                break
        if not is_contained and not is_duplicate:
            keep.append(rect)

    return keep


def _detect_view_label(page: fitz.Page, rect: fitz.Rect) -> str | None:
    """Try to find a view title label near the top or bottom edge of a view box.

    Common patterns: "FRONT VIEW", "SIDE VIEW", "SECTION A-A", "DETAIL 1"
    """
    import re

    label_patterns = [
        r"(?:FRONT|SIDE|TOP|BOTTOM|REAR|LEFT|RIGHT)\s*(?:VIEW|ELEVATION)?",
        r"SECTION\s+[A-Z]-[A-Z]",
        r"DETAIL\s+\d+",
        r"VIEW\s+[A-Z]",
        r"PLAN\s*(?:VIEW)?",
        r"ELEVATION",
    ]
    combined_pattern = "|".join(f"({p})" for p in label_patterns)

    # Search in a strip above the rectangle top edge
    search_strip = fitz.Rect(rect.x0, rect.y0 - 20, rect.x1, rect.y0 + 5)
    text_above = page.get_text("text", clip=search_strip).strip()

    # Also search just inside the top of the rectangle
    search_strip_inside = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + 25)
    text_inside = page.get_text("text", clip=search_strip_inside).strip()

    # Search below the rectangle
    search_strip_below = fitz.Rect(rect.x0, rect.y1 - 5, rect.x1, rect.y1 + 20)
    text_below = page.get_text("text", clip=search_strip_below).strip()

    for text in [text_above, text_inside, text_below]:
        match = re.search(combined_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip().upper()

    return None


def decompose_page(
    page: fitz.Page,
    page_number: int,
    config: ExtractionConfig | None = None,
) -> list[ViewRegion]:
    """Detect detail view bounding boxes on a single page.

    Tries vector rectangle detection first, falls back to line-matching.

    Args:
        page: A PyMuPDF page object.
        page_number: 1-based page number.
        config: Extraction configuration.

    Returns:
        List of detected ViewRegion objects.
    """
    if config is None:
        config = ExtractionConfig()

    # Strategy 1: Look for rectangle drawing operators
    rects = _find_rectangles_from_drawings(page, config)

    # Strategy 2: Fallback to matching line pairs
    if len(rects) < 2:
        line_rects = _find_rectangles_from_lines(page, config)
        rects.extend(line_rects)

    # Filter out page border (full-page rectangle) and title block
    page_rect = page.rect
    margin = 10.0
    filtered = []
    for r in rects:
        # Skip if it's essentially the full page
        if (abs(r.x0 - page_rect.x0) < margin and
                abs(r.y0 - page_rect.y0) < margin and
                abs(r.x1 - page_rect.x1) < margin and
                abs(r.y1 - page_rect.y1) < margin):
            continue
        filtered.append(r)

    # Remove nested/duplicate rectangles
    filtered = _filter_nested_rects(filtered, config.border_tolerance)

    # Sort top-to-bottom, left-to-right
    filtered.sort(key=lambda r: (round(r.y0 / 50) * 50, r.x0))

    # Build ViewRegion objects
    views = []
    for idx, rect in enumerate(filtered):
        label = _detect_view_label(page, rect)
        view = ViewRegion(
            page_number=page_number,
            view_index=idx,
            rect=rect,
            label=label,
        )
        views.append(view)
        logger.debug(
            f"  View {idx}: ({rect.x0:.0f},{rect.y0:.0f})-({rect.x1:.0f},{rect.y1:.0f}) "
            f"label={label}"
        )

    if not views:
        # If no view boxes found, treat the whole page as a single view
        logger.info(f"  No view boxes detected on page {page_number}; using full page.")
        views.append(ViewRegion(
            page_number=page_number,
            view_index=0,
            rect=page.rect,
            label="FULL PAGE",
        ))

    logger.info(f"Page {page_number}: detected {len(views)} view regions")
    return views


def decompose_pdf(
    pdf_path: str | Path,
    config: ExtractionConfig | None = None,
) -> dict[int, list[ViewRegion]]:
    """Decompose all pages of a PDF into view regions.

    Args:
        pdf_path: Path to the PDF file.
        config: Extraction configuration.

    Returns:
        Dict mapping page number (1-based) to list of ViewRegion objects.
    """
    if config is None:
        config = ExtractionConfig()

    doc = fitz.open(str(pdf_path))
    all_views: dict[int, list[ViewRegion]] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        views = decompose_page(page, page_num + 1, config)
        all_views[page_num + 1] = views

    doc.close()
    return all_views
