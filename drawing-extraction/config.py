"""Configuration for the drawing extraction pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionConfig:
    """Pipeline configuration."""

    # Input
    input_pdf: Path = Path("input.pdf")
    template_row: Path | None = None  # JSON file with completed first row

    # Output
    output_dir: Path = Path("output")

    # PDF rendering
    render_dpi: int = 300  # DPI for rasterization when needed

    # View detection
    min_view_width: float = 50.0  # Minimum view box width in points
    min_view_height: float = 50.0  # Minimum view box height in points
    border_tolerance: float = 2.0  # Tolerance for rectangle detection (points)

    # Text extraction
    min_text_size: float = 4.0  # Minimum font size to extract
    rotation_snap_tolerance: float = 5.0  # Degrees tolerance for snapping to 0/90/180/270

    # Dimension association
    dimension_line_proximity: float = 15.0  # Max distance (pts) between text and dimension line
    leader_line_proximity: float = 20.0  # Max distance (pts) for leader line association
    extension_line_max_gap: float = 10.0  # Max gap between extension line and object edge

    # Overlay annotation
    circle_radius: float = 12.0  # Radius of attention marker circles (points)
    circle_stroke_width: float = 2.0  # Stroke width for circles
    tag_font_size: float = 7.0  # Font size for info tags
    tag_offset: float = 20.0  # Offset of info tag from circle center
    tag_bg_opacity: float = 0.85  # Background opacity for info tags

    # Confidence thresholds
    high_confidence: float = 0.85
    medium_confidence: float = 0.60

    # Colors (R, G, B) normalized 0-1
    color_high: tuple = (0.0, 0.7, 0.0)  # Green
    color_medium: tuple = (0.9, 0.7, 0.0)  # Yellow/amber
    color_low: tuple = (0.9, 0.0, 0.0)  # Red
    color_default: tuple = (0.9, 0.0, 0.0)  # Red (default circle color)

    # Known steel shape patterns
    steel_shape_patterns: list = field(default_factory=lambda: [
        r"HSS\s*\d+[xX×]\d+[xX×][\d/]+",  # HSS 6x6x3/8
        r"W\d+[xX×]\d+",  # W12x26
        r"L\d+[xX×]\d+[xX×][\d/]+",  # L4x4x3/8
        r"C\d+[xX×][\d.]+",  # C10x25
        r"WT\d+[xX×][\d.]+",  # WT6x25
        r"S\d+[xX×][\d.]+",  # S12x35
        r"HP\d+[xX×]\d+",  # HP12x53
        r"MC\d+[xX×][\d.]+",  # MC10x28.5
        r"PL\s*[\d/]+\s*[xX×]\s*\d+",  # PL 3/4 x 12
    ])

    # Known piece mark patterns
    piece_mark_patterns: list = field(default_factory=lambda: [
        r"[A-Z]{1,3}-\d{1,3}",  # C-01, BM-14, etc.
    ])
