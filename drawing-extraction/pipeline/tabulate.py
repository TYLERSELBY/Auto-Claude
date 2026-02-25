"""Output extraction results as JSON, CSV, and formatted tables."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

from pipeline.correlate import ObjectRecord

logger = logging.getLogger(__name__)


def to_json(
    records: list[ObjectRecord],
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> str:
    """Convert object records to JSON.

    Produces the full extraction format with source information.

    Args:
        records: List of ObjectRecord objects.
        output_path: Optional file path to write JSON to.
        pretty: Pretty-print with indentation.

    Returns:
        JSON string.
    """
    data = {
        "extraction_results": [rec.to_extraction_json() for rec in records],
        "summary": {
            "total_objects": len(records),
            "objects": [rec.object_id for rec in records],
        },
    }

    json_str = json.dumps(data, indent=2 if pretty else None, ensure_ascii=False)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")
        logger.info(f"Wrote JSON to {output_path}")

    return json_str


def to_csv(
    records: list[ObjectRecord],
    output_path: str | Path | None = None,
) -> str:
    """Convert object records to CSV.

    Produces a flat table with one row per object and columns for each property.

    Args:
        records: List of ObjectRecord objects.
        output_path: Optional file path to write CSV to.

    Returns:
        CSV string.
    """
    if not records:
        return ""

    # Collect all unique property names across all records
    all_props = set()
    for rec in records:
        all_props.update(rec.properties.keys())

    # Sort columns: object_id first, then alphabetically
    columns = ["object_id"] + sorted(all_props)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()

    for rec in records:
        row = rec.to_dict()
        writer.writerow(row)

    csv_str = buf.getvalue()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(csv_str, encoding="utf-8")
        logger.info(f"Wrote CSV to {output_path}")

    return csv_str


def to_table(
    records: list[ObjectRecord],
    max_col_width: int = 20,
) -> str:
    """Format object records as a human-readable ASCII table.

    Args:
        records: List of ObjectRecord objects.
        max_col_width: Maximum column width before truncation.

    Returns:
        Formatted table string.
    """
    if not records:
        return "(no records)"

    # Collect all property names
    all_props = set()
    for rec in records:
        all_props.update(rec.properties.keys())

    columns = ["object_id"] + sorted(all_props)

    # Build rows
    rows = []
    for rec in records:
        row = [rec.object_id]
        for prop in sorted(all_props):
            if prop in rec.properties:
                val = rec.properties[prop].value
                confidence = rec.properties[prop].confidence
                marker = "" if confidence == "high" else "?" if confidence == "medium" else "??"
                row.append(f"{val}{marker}")
            else:
                row.append("-")
        rows.append(row)

    # Calculate column widths
    col_widths = []
    for i, col in enumerate(columns):
        width = len(col)
        for row in rows:
            if i < len(row):
                width = max(width, len(str(row[i])))
        col_widths.append(min(width, max_col_width))

    # Build the table
    def fmt_row(vals):
        cells = []
        for i, val in enumerate(vals):
            w = col_widths[i] if i < len(col_widths) else max_col_width
            s = str(val)[:w]
            cells.append(s.ljust(w))
        return " | ".join(cells)

    lines = []
    header = fmt_row(columns)
    lines.append(header)
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def to_sources_table(records: list[ObjectRecord]) -> str:
    """Generate a detailed sources table showing where each value was extracted from.

    This is useful for the overlay system to know what to annotate.

    Args:
        records: List of ObjectRecord objects.

    Returns:
        CSV-formatted string with source details.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Object ID", "Property Name", "Extracted Value",
        "Source Location (x,y)", "Page", "View",
        "Rotation", "Confidence"
    ])

    for rec in records:
        for name, prop in sorted(rec.properties.items()):
            writer.writerow([
                rec.object_id,
                name,
                prop.value,
                f"({prop.position[0]:.1f}, {prop.position[1]:.1f})",
                prop.page,
                prop.view_label or "unknown",
                f"{prop.rotation:.0f}°",
                prop.confidence,
            ])

    return buf.getvalue()
