#!/usr/bin/env python3
"""
Engineering Drawing Data Extraction & Visual Validation Pipeline

Extracts structured dimensional data from steel fabrication shop drawings
and produces annotated PDFs for visual validation.

Usage:
    # Probe a PDF to check if it's vector or raster
    python main.py probe drawing.pdf

    # Run full extraction pipeline
    python main.py extract drawing.pdf --output output/

    # Extract with a template row for column matching
    python main.py extract drawing.pdf --template template_row.json

    # Generate annotated overlay from existing extraction results
    python main.py overlay drawing.pdf extraction_results.json --output annotated.pdf

    # Decompose a PDF into view regions (debug/inspect)
    python main.py decompose drawing.pdf --output output/views/

    # Extract text only (debug/inspect)
    python main.py text drawing.pdf --page 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent))

from config import ExtractionConfig
from pipeline.probe import probe_pdf
from pipeline.decompose import decompose_page, decompose_pdf
from pipeline.extract_text import extract_text, extract_text_from_pdf, TextElement
from pipeline.extract_lines import extract_lines, extract_lines_from_pdf, find_dimension_line_groups
from pipeline.associate import associate_dimensions
from pipeline.correlate import correlate_views
from pipeline.tabulate import to_json, to_csv, to_table, to_sources_table
from pipeline.overlay import annotate_pdf, annotate_pdf_as_png

logger = logging.getLogger("drawing-extraction")


def cmd_probe(args):
    """Probe a PDF to determine vector vs raster content."""
    result = probe_pdf(args.pdf)
    print(result.summary())
    return result


def cmd_decompose(args):
    """Decompose PDF pages into view regions."""
    import fitz

    config = ExtractionConfig()
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output) if args.output else Path("output/views")
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        if args.page and (page_num + 1) != args.page:
            continue

        page = doc[page_num]
        views = decompose_page(page, page_num + 1, config)

        print(f"\nPage {page_num + 1}: {len(views)} views detected")
        for view in views:
            print(f"  View {view.view_index}: "
                  f"({view.x0:.0f},{view.y0:.0f})-({view.x1:.0f},{view.y1:.0f}) "
                  f"{view.width:.0f}x{view.height:.0f} "
                  f"label={view.label or '(none)'}")

            # Render each view as a cropped PNG
            zoom = args.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            clip = view.rect
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            out_path = output_dir / f"page{page_num+1}_view{view.view_index}.png"
            pix.save(str(out_path))
            print(f"    → {out_path}")

    doc.close()


def cmd_text(args):
    """Extract and display text from a PDF."""
    import fitz

    config = ExtractionConfig()
    doc = fitz.open(str(args.pdf))

    for page_num in range(len(doc)):
        if args.page and (page_num + 1) != args.page:
            continue

        page = doc[page_num]
        elements = extract_text(page, page_num + 1, config)

        print(f"\nPage {page_num + 1}: {len(elements)} text elements")
        print(f"{'Text':<30} {'Category':<15} {'Pos (x,y)':<20} {'Rot':<8} {'Size':<6}")
        print("-" * 80)

        for elem in elements:
            text_display = elem.text[:28]
            print(f"{text_display:<30} {elem.category or 'unknown':<15} "
                  f"({elem.x:.0f},{elem.y:.0f}){'':<8} "
                  f"{elem.rotation:>5.0f}° {elem.font_size:>5.1f}")

    doc.close()


def cmd_extract(args):
    """Run the full extraction pipeline."""
    import fitz

    config = ExtractionConfig()
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template if provided
    template_columns = None
    if args.template:
        template_data = json.loads(Path(args.template).read_text())
        if "properties" in template_data:
            template_columns = list(template_data["properties"].keys())
        elif isinstance(template_data, dict):
            template_columns = [k for k in template_data.keys() if k != "object_id"]

    # Step 1: Probe
    print("=" * 60)
    print("Step 1: Probing PDF structure...")
    print("=" * 60)
    probe_result = probe_pdf(pdf_path)
    print(probe_result.summary())

    if probe_result.overall_raster:
        print("\nWARNING: PDF appears to contain raster content.")
        print("Vector text extraction may yield limited results.")
        print("Consider using LLM vision (Gemini/Claude) for raster content.")
        print("Proceeding with best-effort vector extraction...\n")

    # Step 2: Decompose
    print("\n" + "=" * 60)
    print("Step 2: Decomposing pages into view regions...")
    print("=" * 60)
    doc = fitz.open(str(pdf_path))
    all_views = {}
    all_text: list[TextElement] = []
    all_lines = []
    all_dim_groups = []
    all_associations = []

    flat_views = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        views = decompose_page(page, page_num + 1, config)
        all_views[page_num + 1] = views

        print(f"  Page {page_num + 1}: {len(views)} views")

        # Step 3: Extract text and lines per view
        for view in views:
            flat_views.append(view)

            text_elements = extract_text(
                page, page_num + 1, config,
                clip_rect=view.rect,
                view_index=view.view_index,
            )
            all_text.extend(text_elements)

            line_elements = extract_lines(
                page, page_num + 1, config,
                clip_rect=view.rect,
                view_index=view.view_index,
            )
            all_lines.extend(line_elements)

            # Step 4: Find dimension line groups
            dim_groups = find_dimension_line_groups(line_elements, text_elements, config)
            all_dim_groups.extend(dim_groups)

            # Step 5: Associate dimensions
            associations = associate_dimensions(text_elements, line_elements, dim_groups, config)
            all_associations.extend(associations)

    doc.close()

    print(f"\nExtraction summary:")
    print(f"  Total text elements: {len(all_text)}")
    print(f"  Total line segments: {len(all_lines)}")
    print(f"  Dimension groups: {len(all_dim_groups)}")
    print(f"  Associations: {len(all_associations)}")

    # Step 6: Correlate across views
    print("\n" + "=" * 60)
    print("Step 3: Correlating multi-view data...")
    print("=" * 60)
    records = correlate_views(
        all_associations, all_text, flat_views, config, template_columns
    )
    print(f"  Object records: {len(records)}")

    # Step 7: Output results
    print("\n" + "=" * 60)
    print("Step 4: Generating output...")
    print("=" * 60)

    # JSON
    json_path = output_dir / "extraction_results.json"
    to_json(records, json_path)
    print(f"  JSON → {json_path}")

    # CSV
    csv_path = output_dir / "extraction_results.csv"
    to_csv(records, csv_path)
    print(f"  CSV  → {csv_path}")

    # Sources table
    sources_path = output_dir / "extraction_sources.csv"
    sources_csv = to_sources_table(records)
    sources_path.write_text(sources_csv)
    print(f"  Sources → {sources_path}")

    # Human-readable table
    print("\n" + "=" * 60)
    print("Extraction Results Table:")
    print("=" * 60)
    print(to_table(records))
    print("\n(? = medium confidence, ?? = low confidence)")

    # Step 8: Generate annotated PDF overlay
    if not args.no_overlay:
        print("\n" + "=" * 60)
        print("Step 5: Generating annotated overlay...")
        print("=" * 60)

        overlay_pdf_path = output_dir / "annotated_drawing.pdf"
        try:
            annotate_pdf(pdf_path, overlay_pdf_path, records, config)
            print(f"  Annotated PDF → {overlay_pdf_path}")
        except Exception as e:
            logger.warning(f"PDF annotation failed: {e}. Falling back to PNG overlay.")
            png_dir = output_dir / "annotated_pages"
            annotate_pdf_as_png(pdf_path, png_dir, records, config)
            print(f"  Annotated PNGs → {png_dir}/")

    print("\nDone.")
    return records


def cmd_overlay(args):
    """Generate an annotated overlay from existing extraction results."""
    config = ExtractionConfig()
    pdf_path = Path(args.pdf)
    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else Path("output/annotated_drawing.pdf")

    # Load extraction results
    data = json.loads(results_path.read_text())
    extraction_results = data.get("extraction_results", data if isinstance(data, list) else [data])

    # Reconstruct ObjectRecord objects from JSON
    from pipeline.correlate import ObjectRecord, PropertyValue

    records = []
    for item in extraction_results:
        record = ObjectRecord(object_id=item["object_id"])
        props = item.get("properties", {})
        sources = {s["property"]: s for s in item.get("sources", [])}

        for prop_name, value in props.items():
            source = sources.get(prop_name, {})
            pos = source.get("position", [0, 0])
            record.properties[prop_name] = PropertyValue(
                name=prop_name,
                value=str(value),
                numeric_value=float(value) if isinstance(value, (int, float)) else None,
                page=source.get("page", 1),
                view_label=source.get("view"),
                position=(pos[0], pos[1]),
                rotation=source.get("rotation", 0),
                confidence=source.get("confidence", "medium"),
            )
        records.append(record)

    # Generate overlay
    if args.png:
        png_dir = output_path.parent / "annotated_pages"
        annotate_pdf_as_png(pdf_path, png_dir, records, config)
        print(f"Annotated PNGs → {png_dir}/")
    else:
        annotate_pdf(pdf_path, output_path, records, config)
        print(f"Annotated PDF → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Engineering Drawing Data Extraction & Visual Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose/debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # Probe command
    p_probe = subparsers.add_parser("probe", help="Probe PDF for vector vs raster content")
    p_probe.add_argument("pdf", help="Path to PDF file")

    # Decompose command
    p_decompose = subparsers.add_parser("decompose", help="Decompose pages into view regions")
    p_decompose.add_argument("pdf", help="Path to PDF file")
    p_decompose.add_argument("--output", "-o", help="Output directory for cropped view images")
    p_decompose.add_argument("--page", "-p", type=int, help="Process only this page number")
    p_decompose.add_argument("--dpi", type=int, default=200, help="DPI for cropped images")

    # Text command
    p_text = subparsers.add_parser("text", help="Extract and display text elements")
    p_text.add_argument("pdf", help="Path to PDF file")
    p_text.add_argument("--page", "-p", type=int, help="Process only this page number")

    # Extract command
    p_extract = subparsers.add_parser("extract", help="Run full extraction pipeline")
    p_extract.add_argument("pdf", help="Path to PDF file")
    p_extract.add_argument("--output", "-o", help="Output directory")
    p_extract.add_argument("--template", "-t", help="Template row JSON for column matching")
    p_extract.add_argument("--no-overlay", action="store_true", help="Skip annotation overlay")

    # Overlay command
    p_overlay = subparsers.add_parser("overlay", help="Generate annotated overlay from results")
    p_overlay.add_argument("pdf", help="Path to original PDF file")
    p_overlay.add_argument("results", help="Path to extraction_results.json")
    p_overlay.add_argument("--output", "-o", help="Output path for annotated PDF")
    p_overlay.add_argument("--png", action="store_true", help="Output as PNG instead of PDF")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "probe": cmd_probe,
        "decompose": cmd_decompose,
        "text": cmd_text,
        "extract": cmd_extract,
        "overlay": cmd_overlay,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
