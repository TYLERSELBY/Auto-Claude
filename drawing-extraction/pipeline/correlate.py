"""Multi-view correlation — combine data from multiple views into object records."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from config import ExtractionConfig
from pipeline.extract_text import TextElement
from pipeline.associate import DimensionAssociation
from pipeline.decompose import ViewRegion

logger = logging.getLogger(__name__)


@dataclass
class PropertyValue:
    """A single property extracted from the drawings."""

    name: str  # Property/column name
    value: str  # Raw text value
    numeric_value: float | None = None
    unit: str = "inches"
    page: int = 0
    view_label: str | None = None
    position: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    confidence: str = "medium"
    source_association: DimensionAssociation | None = None


@dataclass
class ObjectRecord:
    """A correlated record for a single assembly/object across all views."""

    object_id: str  # Piece mark (e.g., "C-01")
    properties: dict[str, PropertyValue] = field(default_factory=dict)
    views_found_in: list[str] = field(default_factory=list)
    pages_found_on: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a flat dictionary for tabular output."""
        result = {"object_id": self.object_id}
        for name, prop in self.properties.items():
            result[name] = prop.value
        return result

    def to_extraction_json(self) -> dict:
        """Convert to the full extraction JSON format with source information."""
        props = {}
        sources = []
        for name, prop in self.properties.items():
            props[name] = prop.numeric_value if prop.numeric_value is not None else prop.value
            sources.append({
                "property": name,
                "value": prop.value,
                "page": prop.page,
                "view": prop.view_label or "unknown",
                "position": list(prop.position),
                "rotation": prop.rotation,
                "confidence": prop.confidence,
            })
        return {
            "object_id": self.object_id,
            "properties": props,
            "sources": sources,
        }


def _find_piece_marks(
    text_elements: list[TextElement],
    config: ExtractionConfig,
) -> dict[str, list[TextElement]]:
    """Find all piece mark text elements and group by mark ID.

    Returns:
        Dict mapping piece mark string to list of TextElement occurrences.
    """
    marks: dict[str, list[TextElement]] = {}
    for text in text_elements:
        if text.category == "piece_mark":
            mark = text.text.strip().upper()
            marks.setdefault(mark, []).append(text)
    return marks


def _assign_associations_to_objects(
    associations: list[DimensionAssociation],
    piece_marks: dict[str, list[TextElement]],
    views: list[ViewRegion],
) -> dict[str, list[DimensionAssociation]]:
    """Assign each association to the nearest piece mark's object.

    Uses view region containment first, then proximity as fallback.
    """
    # Build a lookup: which view contains which piece marks
    view_marks: dict[int, list[str]] = {}  # view_index → list of mark IDs
    for mark_id, mark_texts in piece_marks.items():
        for mt in mark_texts:
            for view in views:
                if (view.page_number == mt.page_number and
                        view.rect.contains(fitz_point(mt.x, mt.y))):
                    view_marks.setdefault(view.view_index, []).append(mark_id)

    # Assign associations
    result: dict[str, list[DimensionAssociation]] = {}

    for assoc in associations:
        # Find which view this association is in
        assoc_view = assoc.text_element.view_index

        # Check if a piece mark is in the same view
        if assoc_view is not None and assoc_view in view_marks:
            marks_in_view = view_marks[assoc_view]
            if marks_in_view:
                # Assign to the first mark in this view (most common case: 1 mark per view)
                mark = marks_in_view[0]
                result.setdefault(mark, []).append(assoc)
                continue

        # Fallback: find the closest piece mark by position
        best_mark = None
        best_dist = float("inf")
        for mark_id, mark_texts in piece_marks.items():
            for mt in mark_texts:
                if mt.page_number != assoc.text_element.page_number:
                    continue
                dx = mt.x - assoc.text_element.x
                dy = mt.y - assoc.text_element.y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_mark = mark_id

        if best_mark:
            result.setdefault(best_mark, []).append(assoc)
        else:
            # No piece mark found — assign to "UNKNOWN"
            result.setdefault("UNKNOWN", []).append(assoc)

    return result


def fitz_point(x: float, y: float):
    """Create a fitz.Point-like tuple for containment checking."""
    try:
        import fitz
        return fitz.Point(x, y)
    except ImportError:
        # Fallback: just return tuple (won't work with fitz.Rect.contains)
        return (x, y)


def _resolve_property_name(
    assoc: DimensionAssociation,
    template_columns: list[str] | None = None,
) -> str:
    """Determine the property/column name for an association.

    Uses the property hint if available, otherwise generates a generic name.
    If a template is provided, tries to match to template column names.
    """
    if assoc.property_hint:
        hint = assoc.property_hint

        # If we have template columns, try to match
        if template_columns:
            hint_lower = hint.lower().replace("_", " ")
            for col in template_columns:
                col_lower = col.lower().replace("_", " ")
                if hint_lower in col_lower or col_lower in hint_lower:
                    return col

        # Specific category-based naming
        if hint == "steel_shape":
            return "HSS_Size"
        elif hint == "piece_mark":
            return "Piece_Mark"
        elif hint == "bolt_spec":
            return "Bolt_Pattern"
        elif hint == "weld_symbol":
            return "Weld_Spec"
        elif hint in ("width_or_length", "height_or_depth"):
            return hint

        return hint

    # Generate from association type and position
    if assoc.association_type == "dimension_line" and assoc.dimension_group:
        dim = assoc.dimension_group.dimension_line
        if dim.is_horizontal:
            return "horizontal_dim"
        elif dim.is_vertical:
            return "vertical_dim"

    return f"value_at_{assoc.text_element.x:.0f}_{assoc.text_element.y:.0f}"


def correlate_views(
    associations: list[DimensionAssociation],
    text_elements: list[TextElement],
    views: list[ViewRegion],
    config: ExtractionConfig | None = None,
    template_columns: list[str] | None = None,
) -> list[ObjectRecord]:
    """Correlate associations across views into per-object records.

    Args:
        associations: All dimension associations from all pages/views.
        text_elements: All text elements from all pages/views.
        views: All detected view regions.
        config: Extraction configuration.
        template_columns: Optional list of property/column names from the template row.

    Returns:
        List of ObjectRecord objects, one per assembly/piece mark.
    """
    if config is None:
        config = ExtractionConfig()

    # Find all piece marks
    piece_marks = _find_piece_marks(text_elements, config)
    logger.info(f"Found {len(piece_marks)} unique piece marks: {list(piece_marks.keys())}")

    # Assign associations to objects
    object_assocs = _assign_associations_to_objects(associations, piece_marks, views)

    # Build object records
    records = []
    for mark_id, assocs in sorted(object_assocs.items()):
        record = ObjectRecord(object_id=mark_id)

        # Track which views/pages this object was found in
        view_labels = set()
        pages = set()

        for assoc in assocs:
            pages.add(assoc.text_element.page_number)
            if assoc.text_element.view_index is not None:
                # Find the view label
                for v in views:
                    if (v.page_number == assoc.text_element.page_number and
                            v.view_index == assoc.text_element.view_index):
                        if v.label:
                            view_labels.add(v.label)

            # Resolve property name
            prop_name = _resolve_property_name(assoc, template_columns)

            # If property already exists, keep higher confidence one
            if prop_name in record.properties:
                existing = record.properties[prop_name]
                if assoc.confidence_label == "high" and existing.confidence != "high":
                    pass  # Replace below
                elif existing.confidence == "high":
                    continue  # Keep existing

            prop = PropertyValue(
                name=prop_name,
                value=assoc.value,
                numeric_value=assoc.numeric_value,
                page=assoc.text_element.page_number,
                view_label=next(iter(view_labels), None),
                position=assoc.position,
                rotation=assoc.text_element.rotation,
                confidence=assoc.confidence_label,
                source_association=assoc,
            )
            record.properties[prop_name] = prop

        record.views_found_in = sorted(view_labels)
        record.pages_found_on = sorted(pages)
        records.append(record)

    logger.info(f"Correlated into {len(records)} object records")
    for rec in records:
        logger.info(f"  {rec.object_id}: {len(rec.properties)} properties from pages {rec.pages_found_on}")

    return records
