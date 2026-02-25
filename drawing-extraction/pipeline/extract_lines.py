"""Extract line segments and geometric primitives from PDF pages."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from config import ExtractionConfig

logger = logging.getLogger(__name__)


@dataclass
class LineElement:
    """A line segment extracted from a drawing."""

    x0: float
    y0: float
    x1: float
    y1: float
    stroke_width: float
    color: tuple[float, ...] | None = None
    page_number: int = 0
    view_index: int | None = None
    line_type: str = "unknown"  # "dimension", "extension", "leader", "object", "border"

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def angle(self) -> float:
        """Angle in degrees from horizontal, counter-clockwise."""
        return math.degrees(math.atan2(self.y1 - self.y0, self.x1 - self.x0))

    @property
    def is_horizontal(self) -> bool:
        return abs(self.y1 - self.y0) < 1.0

    @property
    def is_vertical(self) -> bool:
        return abs(self.x1 - self.x0) < 1.0

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def distance_to_point(self, px: float, py: float) -> float:
        """Perpendicular distance from a point to this line segment."""
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return math.hypot(px - self.x0, py - self.y0)

        t = max(0, min(1, ((px - self.x0) * dx + (py - self.y0) * dy) / len_sq))
        proj_x = self.x0 + t * dx
        proj_y = self.y0 + t * dy
        return math.hypot(px - proj_x, py - proj_y)


@dataclass
class ArrowHead:
    """A detected arrowhead at the end of a line."""

    tip: tuple[float, float]
    direction: float  # Angle in degrees the arrow points toward
    line_index: int  # Index of the parent line


@dataclass
class DimensionLineGroup:
    """A group of lines forming a dimension annotation.

    Consists of:
    - dimension_line: The main line between extension lines
    - extension_lines: Two lines perpendicular to the measured feature
    - arrows: Arrowheads or tick marks at dimension line endpoints
    """

    dimension_line: LineElement
    extension_lines: list[LineElement] = field(default_factory=list)
    arrows: list[ArrowHead] = field(default_factory=list)
    measured_distance: float | None = None  # Distance between extension line endpoints


def _classify_line_by_weight(line: LineElement) -> str:
    """Initial classification based on stroke width.

    Conventions in engineering drawings:
    - Object/visible lines: thick (0.5-0.7mm → ~1.4-2.0 pts)
    - Dimension/extension lines: thin (0.25-0.35mm → ~0.7-1.0 pts)
    - Centerlines: thin with dash pattern
    - Border lines: thick or medium
    """
    w = line.stroke_width
    if w < 0.3:
        return "thin"  # Likely dimension/extension
    elif w < 0.8:
        return "medium"  # Could be dimension or light object line
    elif w < 1.5:
        return "thick"  # Object line
    else:
        return "heavy"  # Border


def extract_lines(
    page: fitz.Page,
    page_number: int,
    config: ExtractionConfig | None = None,
    clip_rect: fitz.Rect | None = None,
    view_index: int | None = None,
) -> list[LineElement]:
    """Extract all line segments from a page or clipped region.

    Args:
        page: A PyMuPDF page object.
        page_number: 1-based page number.
        config: Extraction configuration.
        clip_rect: Optional region to restrict extraction to.
        view_index: Index of the view region this belongs to.

    Returns:
        List of LineElement objects.
    """
    if config is None:
        config = ExtractionConfig()

    drawings = page.get_drawings()
    lines: list[LineElement] = []

    for path in drawings:
        stroke_width = path.get("width", 0.5)
        color = path.get("color")

        for item in path.get("items", []):
            op = item[0]

            if op == "l":  # Line segment
                p1, p2 = item[1], item[2]

                # Apply clip if specified
                if clip_rect:
                    if not (clip_rect.contains(p1) or clip_rect.contains(p2)):
                        continue

                line = LineElement(
                    x0=p1.x,
                    y0=p1.y,
                    x1=p2.x,
                    y1=p2.y,
                    stroke_width=stroke_width,
                    color=color,
                    page_number=page_number,
                    view_index=view_index,
                )
                lines.append(line)

            elif op == "re":  # Rectangle — decompose into 4 lines
                rect = item[1]
                if isinstance(rect, fitz.Rect):
                    if clip_rect and not clip_rect.intersects(rect):
                        continue

                    corners = [
                        (rect.x0, rect.y0), (rect.x1, rect.y0),
                        (rect.x1, rect.y1), (rect.x0, rect.y1),
                    ]
                    for i in range(4):
                        x0, y0 = corners[i]
                        x1, y1 = corners[(i + 1) % 4]
                        line = LineElement(
                            x0=x0, y0=y0, x1=x1, y1=y1,
                            stroke_width=stroke_width,
                            color=color,
                            page_number=page_number,
                            view_index=view_index,
                            line_type="border",
                        )
                        lines.append(line)

    logger.info(
        f"Page {page_number}"
        f"{f', view {view_index}' if view_index is not None else ''}"
        f": extracted {len(lines)} line segments"
    )
    return lines


def find_dimension_line_groups(
    lines: list[LineElement],
    text_elements: list | None = None,
    config: ExtractionConfig | None = None,
) -> list[DimensionLineGroup]:
    """Identify groups of lines that form dimension annotations.

    A dimension consists of:
    1. Two parallel extension lines perpendicular to the measured feature
    2. A dimension line connecting them (with arrows or ticks)
    3. A numeric text value near the dimension line

    This function uses heuristics based on line geometry and proximity.

    Args:
        lines: All extracted line segments.
        text_elements: Optional text elements to help confirm dimension lines.
        config: Extraction configuration.

    Returns:
        List of identified DimensionLineGroup objects.
    """
    if config is None:
        config = ExtractionConfig()

    # Separate thin lines (likely dimension/extension) from thick (object) lines
    thin_lines = []
    for line in lines:
        weight = _classify_line_by_weight(line)
        if weight in ("thin", "medium") and line.line_type != "border":
            thin_lines.append(line)

    groups = []

    # Strategy: find pairs of parallel thin lines that are perpendicular
    # to a third thin line connecting them
    horizontal_thin = [l for l in thin_lines if l.is_horizontal and l.length > 5]
    vertical_thin = [l for l in thin_lines if l.is_vertical and l.length > 5]

    # Look for horizontal dimension lines with vertical extension lines
    for h_line in horizontal_thin:
        # Find vertical lines near the endpoints of this horizontal line
        left_ext = []
        right_ext = []

        for v_line in vertical_thin:
            # Check if vertical line is near the left end
            d_left = math.hypot(v_line.x0 - h_line.x0, 0)
            if d_left < config.extension_line_max_gap:
                # Check vertical overlap
                v_top = min(v_line.y0, v_line.y1)
                v_bot = max(v_line.y0, v_line.y1)
                if v_top <= h_line.y0 + 2 and v_bot >= h_line.y0 - 2:
                    left_ext.append(v_line)

            # Check if vertical line is near the right end
            d_right = math.hypot(v_line.x0 - h_line.x1, 0)
            if d_right < config.extension_line_max_gap:
                v_top = min(v_line.y0, v_line.y1)
                v_bot = max(v_line.y0, v_line.y1)
                if v_top <= h_line.y0 + 2 and v_bot >= h_line.y0 - 2:
                    right_ext.append(v_line)

        if left_ext and right_ext:
            h_line.line_type = "dimension"
            group = DimensionLineGroup(
                dimension_line=h_line,
                extension_lines=[left_ext[0], right_ext[0]],
                measured_distance=h_line.length,
            )
            for ext in group.extension_lines:
                ext.line_type = "extension"
            groups.append(group)

    # Look for vertical dimension lines with horizontal extension lines
    for v_line in vertical_thin:
        top_ext = []
        bottom_ext = []

        for h_line in horizontal_thin:
            d_top = math.hypot(0, h_line.y0 - v_line.y0)
            if d_top < config.extension_line_max_gap:
                h_left = min(h_line.x0, h_line.x1)
                h_right = max(h_line.x0, h_line.x1)
                if h_left <= v_line.x0 + 2 and h_right >= v_line.x0 - 2:
                    top_ext.append(h_line)

            d_bottom = math.hypot(0, h_line.y0 - v_line.y1)
            if d_bottom < config.extension_line_max_gap:
                h_left = min(h_line.x0, h_line.x1)
                h_right = max(h_line.x0, h_line.x1)
                if h_left <= v_line.x0 + 2 and h_right >= v_line.x0 - 2:
                    bottom_ext.append(h_line)

        if top_ext and bottom_ext:
            v_line.line_type = "dimension"
            group = DimensionLineGroup(
                dimension_line=v_line,
                extension_lines=[top_ext[0], bottom_ext[0]],
                measured_distance=v_line.length,
            )
            for ext in group.extension_lines:
                ext.line_type = "extension"
            groups.append(group)

    logger.info(f"Found {len(groups)} dimension line groups")
    return groups


def extract_lines_from_pdf(
    pdf_path: str | Path,
    config: ExtractionConfig | None = None,
) -> dict[int, list[LineElement]]:
    """Extract lines from all pages of a PDF.

    Args:
        pdf_path: Path to the PDF.
        config: Extraction configuration.

    Returns:
        Dict mapping page number (1-based) to list of LineElement objects.
    """
    if config is None:
        config = ExtractionConfig()

    doc = fitz.open(str(pdf_path))
    all_lines: dict[int, list[LineElement]] = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        line_elements = extract_lines(page, page_num + 1, config)
        all_lines[page_num + 1] = line_elements

    doc.close()
    return all_lines
