"""PDF annotation overlay — mark up original drawings with extraction results."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from config import ExtractionConfig
from pipeline.correlate import ObjectRecord, PropertyValue

logger = logging.getLogger(__name__)


def _confidence_color(confidence: str, config: ExtractionConfig) -> tuple[float, float, float]:
    """Get the annotation color based on confidence level."""
    if confidence == "high":
        return config.color_high
    elif confidence == "medium":
        return config.color_medium
    return config.color_low


def _add_circle_annotation(
    page: fitz.Page,
    center_x: float,
    center_y: float,
    radius: float,
    color: tuple[float, float, float],
    stroke_width: float,
):
    """Add a circle annotation (unfilled, colored perimeter) at a position."""
    rect = fitz.Rect(
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )
    annot = page.add_circle_annot(rect)
    annot.set_border(width=stroke_width)
    annot.set_colors(stroke=color)
    annot.update()
    return annot


def _add_info_tag(
    page: fitz.Page,
    x: float,
    y: float,
    value: str,
    property_name: str,
    object_id: str,
    color: tuple[float, float, float],
    config: ExtractionConfig,
):
    """Add an info tag text annotation near a circle.

    The tag contains: value, property name, and object ID.
    """
    tag_text = f"{value}\n{property_name}\n[{object_id}]"

    # Calculate tag dimensions
    font_size = config.tag_font_size
    line_height = font_size * 1.3
    lines = tag_text.split("\n")
    tag_width = max(len(line) for line in lines) * font_size * 0.55 + 8
    tag_height = len(lines) * line_height + 6

    # Position the tag offset from the circle
    tag_x = x + config.tag_offset
    tag_y = y - tag_height / 2

    # Ensure tag stays within page bounds
    page_rect = page.rect
    if tag_x + tag_width > page_rect.x1 - 5:
        tag_x = x - config.tag_offset - tag_width
    if tag_y < page_rect.y0 + 5:
        tag_y = page_rect.y0 + 5
    if tag_y + tag_height > page_rect.y1 - 5:
        tag_y = page_rect.y1 - 5 - tag_height

    tag_rect = fitz.Rect(tag_x, tag_y, tag_x + tag_width, tag_y + tag_height)

    # Draw background rectangle
    bg_color = (1.0, 1.0, 1.0)  # White background
    shape = page.new_shape()
    shape.draw_rect(tag_rect)
    shape.finish(color=color, fill=bg_color, width=0.5, fill_opacity=config.tag_bg_opacity)

    # Draw text lines
    text_x = tag_x + 4
    text_y = tag_y + font_size + 2

    for line_text in lines:
        shape.insert_text(
            fitz.Point(text_x, text_y),
            line_text,
            fontsize=font_size,
            color=color,
        )
        text_y += line_height

    shape.commit()

    return tag_rect


def _add_connector_line(
    page: fitz.Page,
    circle_x: float,
    circle_y: float,
    tag_rect: fitz.Rect,
    color: tuple[float, float, float],
    config: ExtractionConfig,
):
    """Draw a thin connector line from the circle to the info tag."""
    # Connect from circle center to nearest edge of the tag
    tag_cx = (tag_rect.x0 + tag_rect.x1) / 2
    tag_cy = (tag_rect.y0 + tag_rect.y1) / 2

    # Find the nearest point on the tag border
    if tag_rect.x0 > circle_x:
        connect_x = tag_rect.x0
    elif tag_rect.x1 < circle_x:
        connect_x = tag_rect.x1
    else:
        connect_x = circle_x

    if tag_rect.y0 > circle_y:
        connect_y = tag_rect.y0
    elif tag_rect.y1 < circle_y:
        connect_y = tag_rect.y1
    else:
        connect_y = circle_y

    # Only draw connector if tag is far enough from circle
    dx = connect_x - circle_x
    dy = connect_y - circle_y
    dist = (dx * dx + dy * dy) ** 0.5

    if dist > config.circle_radius + 5:
        shape = page.new_shape()
        shape.draw_line(
            fitz.Point(circle_x, circle_y),
            fitz.Point(connect_x, connect_y),
        )
        shape.finish(color=color, width=0.5, dashes="[2 2]")
        shape.commit()


def annotate_pdf(
    input_pdf: str | Path,
    output_pdf: str | Path,
    records: list[ObjectRecord],
    config: ExtractionConfig | None = None,
) -> Path:
    """Generate an annotated PDF with extraction result overlays.

    For each extracted property, adds:
    1. A colored circle around the source annotation location
    2. An info tag with the value, property name, and object ID
    3. A connector line between them

    Color coding:
    - Green: High confidence
    - Yellow/amber: Medium confidence
    - Red: Low confidence

    Args:
        input_pdf: Path to the original PDF.
        output_pdf: Path to write the annotated PDF.
        records: Extraction results to overlay.
        config: Extraction configuration.

    Returns:
        Path to the output PDF.
    """
    if config is None:
        config = ExtractionConfig()

    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(input_pdf))

    annotation_count = 0

    for record in records:
        for prop_name, prop in record.properties.items():
            page_num = prop.page - 1  # Convert to 0-based
            if page_num < 0 or page_num >= len(doc):
                logger.warning(
                    f"Property {prop_name} for {record.object_id} references "
                    f"page {prop.page} but PDF only has {len(doc)} pages"
                )
                continue

            page = doc[page_num]
            color = _confidence_color(prop.confidence, config)

            # 1. Add attention circle
            _add_circle_annotation(
                page,
                prop.position[0],
                prop.position[1],
                config.circle_radius,
                color,
                config.circle_stroke_width,
            )

            # 2. Add info tag
            tag_rect = _add_info_tag(
                page,
                prop.position[0],
                prop.position[1],
                prop.value,
                prop_name,
                record.object_id,
                color,
                config,
            )

            # 3. Add connector line
            _add_connector_line(
                page,
                prop.position[0],
                prop.position[1],
                tag_rect,
                color,
                config,
            )

            annotation_count += 1

    doc.save(str(output_pdf))
    doc.close()

    logger.info(f"Annotated PDF saved to {output_pdf} ({annotation_count} annotations)")
    return output_pdf


def annotate_pdf_as_png(
    input_pdf: str | Path,
    output_dir: str | Path,
    records: list[ObjectRecord],
    config: ExtractionConfig | None = None,
    dpi: int = 200,
) -> list[Path]:
    """Fallback: render annotated pages as PNG images using Pillow.

    Uses Pillow to draw circles, text boxes, and connectors on rasterized pages.

    Args:
        input_pdf: Path to the original PDF.
        output_dir: Directory to write PNG files.
        records: Extraction results to overlay.
        config: Extraction configuration.
        dpi: Resolution for rasterization.

    Returns:
        List of paths to output PNG files.
    """
    if config is None:
        config = ExtractionConfig()

    from PIL import Image, ImageDraw, ImageFont

    input_pdf = Path(input_pdf)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(input_pdf))
    output_paths = []

    # Group properties by page
    page_props: dict[int, list[tuple[ObjectRecord, str, PropertyValue]]] = {}
    for record in records:
        for prop_name, prop in record.properties.items():
            page_num = prop.page
            page_props.setdefault(page_num, []).append((record, prop_name, prop))

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Render page to image
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(img)

        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(config.tag_font_size * zoom))
        except (OSError, IOError):
            font = ImageFont.load_default()

        props_for_page = page_props.get(page_num + 1, [])

        for record, prop_name, prop in props_for_page:
            # Scale coordinates to image space
            sx = prop.position[0] * zoom
            sy = prop.position[1] * zoom
            radius = config.circle_radius * zoom

            # Get color as 0-255 RGB
            color_float = _confidence_color(prop.confidence, config)
            color_rgb = tuple(int(c * 255) for c in color_float)

            # Draw circle
            draw.ellipse(
                [sx - radius, sy - radius, sx + radius, sy + radius],
                outline=color_rgb,
                width=max(1, int(config.circle_stroke_width * zoom)),
            )

            # Draw info tag
            tag_text = f"{prop.value} | {prop_name} | [{record.object_id}]"
            offset = config.tag_offset * zoom

            # Text background
            bbox = draw.textbbox((0, 0), tag_text, font=font)
            tw = bbox[2] - bbox[0] + 8
            th = bbox[3] - bbox[1] + 6

            tx = sx + offset
            ty = sy - th / 2

            # Keep within image bounds
            tx = min(tx, img.width - tw - 5)
            ty = max(5, min(ty, img.height - th - 5))

            draw.rectangle([tx, ty, tx + tw, ty + th], fill=(255, 255, 255, 220), outline=color_rgb)
            draw.text((tx + 4, ty + 3), tag_text, fill=color_rgb, font=font)

            # Connector line
            draw.line([(sx, sy), (tx, ty + th / 2)], fill=color_rgb, width=1)

        output_path = output_dir / f"page_{page_num + 1}_annotated.png"
        img.save(str(output_path))
        output_paths.append(output_path)

    doc.close()
    logger.info(f"Annotated PNGs saved to {output_dir} ({len(output_paths)} pages)")
    return output_paths
