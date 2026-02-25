"""Spatial association — link dimension text to geometric features."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from config import ExtractionConfig
from pipeline.extract_text import TextElement
from pipeline.extract_lines import LineElement, DimensionLineGroup

logger = logging.getLogger(__name__)


@dataclass
class DimensionAssociation:
    """An association between a text value and a geometric feature."""

    text_element: TextElement
    value: str
    numeric_value: float | None
    association_type: str  # "dimension_line", "leader", "proximity", "callout"
    confidence: float  # 0.0 to 1.0
    dimension_group: DimensionLineGroup | None = None
    nearby_lines: list[LineElement] = field(default_factory=list)
    property_hint: str | None = None  # Guessed property name based on context

    @property
    def position(self) -> tuple[float, float]:
        return (self.text_element.x, self.text_element.y)

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.85:
            return "high"
        elif self.confidence >= 0.60:
            return "medium"
        return "low"


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _find_closest_dimension_group(
    text: TextElement,
    groups: list[DimensionLineGroup],
    max_distance: float,
) -> tuple[DimensionLineGroup | None, float]:
    """Find the dimension line group closest to a text element."""
    best_group = None
    best_dist = float("inf")

    for group in groups:
        dim_line = group.dimension_line
        dist = dim_line.distance_to_point(text.x, text.y)
        if dist < best_dist:
            best_dist = dist
            best_group = group

    if best_dist <= max_distance:
        return best_group, best_dist
    return None, best_dist


def _find_nearby_lines(
    text: TextElement,
    lines: list[LineElement],
    max_distance: float,
) -> list[tuple[LineElement, float]]:
    """Find all lines within max_distance of a text element, sorted by distance."""
    results = []
    for line in lines:
        dist = line.distance_to_point(text.x, text.y)
        if dist <= max_distance:
            results.append((line, dist))
    results.sort(key=lambda x: x[1])
    return results


def _detect_leader_line(
    text: TextElement,
    lines: list[LineElement],
    max_distance: float,
) -> LineElement | None:
    """Detect if a line acts as a leader pointing from the text to a feature.

    Leader lines typically start near the text and point away from it
    at an angle (not axis-aligned like dimension lines).
    """
    for line in lines:
        # Check if either endpoint is near the text
        d0 = _distance((line.x0, line.y0), (text.x, text.y))
        d1 = _distance((line.x1, line.y1), (text.x, text.y))

        near_dist = min(d0, d1)
        if near_dist > max_distance:
            continue

        # Leader lines are usually short-to-medium length and angled
        if line.length < 5 or line.length > 200:
            continue

        # Prefer lines that aren't strictly axis-aligned (those are more likely
        # extension or dimension lines)
        if not line.is_horizontal and not line.is_vertical:
            return line

        # Even axis-aligned lines can be leaders if they're short
        if line.length < 30:
            return line

    return None


def _guess_property_from_context(
    text: TextElement,
    dim_group: DimensionLineGroup | None,
    nearby_texts: list[TextElement],
) -> str | None:
    """Try to guess the property name from surrounding context.

    Looks at nearby callout text, the orientation of the dimension,
    and any adjacent labels.
    """
    # Check nearby text elements for property-like labels
    property_keywords = {
        "width": ["WIDTH", "W", "WIDE"],
        "height": ["HEIGHT", "H", "HT", "HIGH", "TALL"],
        "length": ["LENGTH", "L", "LG", "LONG"],
        "depth": ["DEPTH", "D", "DP", "DEEP"],
        "thickness": ["THK", "THICK", "T", "THICKNESS"],
        "offset": ["OFFSET", "OFF"],
        "spacing": ["SPACING", "SPC", "O.C.", "OC"],
        "diameter": ["DIA", "Ø", "DIAM"],
        "radius": ["RAD", "R"],
        "gap": ["GAP", "CLR", "CLEAR"],
        "edge_distance": ["EDGE", "E.D."],
    }

    for nearby in nearby_texts:
        if nearby is text:
            continue
        d = _distance((text.x, text.y), (nearby.x, nearby.y))
        if d > 50:
            continue

        upper = nearby.text.upper().strip()
        for prop, keywords in property_keywords.items():
            for kw in keywords:
                if kw in upper:
                    return prop

    # If we have a dimension group, guess from orientation
    if dim_group:
        dim_line = dim_group.dimension_line
        if dim_line.is_horizontal:
            return "width_or_length"
        elif dim_line.is_vertical:
            return "height_or_depth"

    return None


def associate_dimensions(
    text_elements: list[TextElement],
    lines: list[LineElement],
    dimension_groups: list[DimensionLineGroup],
    config: ExtractionConfig | None = None,
) -> list[DimensionAssociation]:
    """Associate dimension text with geometric features.

    Process:
    1. For each numeric text element, find the closest dimension line group
    2. If a dimension group is found within threshold, create a high-confidence association
    3. Otherwise, look for leader lines pointing to the text
    4. Fall back to proximity-based association

    Args:
        text_elements: All extracted text elements.
        lines: All extracted line segments.
        dimension_groups: Detected dimension line groups.
        config: Extraction configuration.

    Returns:
        List of DimensionAssociation objects.
    """
    if config is None:
        config = ExtractionConfig()

    associations = []
    dimension_texts = [t for t in text_elements if t.category == "dimension"]

    for text in dimension_texts:
        # Strategy 1: Match to a dimension line group
        group, group_dist = _find_closest_dimension_group(
            text, dimension_groups, config.dimension_line_proximity
        )

        if group is not None:
            confidence = max(0.5, 1.0 - (group_dist / config.dimension_line_proximity))
            prop_hint = _guess_property_from_context(text, group, text_elements)
            assoc = DimensionAssociation(
                text_element=text,
                value=text.text.strip(),
                numeric_value=text.numeric_value,
                association_type="dimension_line",
                confidence=confidence,
                dimension_group=group,
                property_hint=prop_hint,
            )
            associations.append(assoc)
            continue

        # Strategy 2: Look for a leader line
        leader = _detect_leader_line(text, lines, config.leader_line_proximity)
        if leader is not None:
            confidence = 0.70
            prop_hint = _guess_property_from_context(text, None, text_elements)
            assoc = DimensionAssociation(
                text_element=text,
                value=text.text.strip(),
                numeric_value=text.numeric_value,
                association_type="leader",
                confidence=confidence,
                nearby_lines=[leader],
                property_hint=prop_hint,
            )
            associations.append(assoc)
            continue

        # Strategy 3: Proximity-based (lowest confidence)
        nearby = _find_nearby_lines(text, lines, config.dimension_line_proximity * 2)
        if nearby:
            nearby_lines = [nl[0] for nl in nearby[:3]]
            confidence = 0.40
            prop_hint = _guess_property_from_context(text, None, text_elements)
            assoc = DimensionAssociation(
                text_element=text,
                value=text.text.strip(),
                numeric_value=text.numeric_value,
                association_type="proximity",
                confidence=confidence,
                nearby_lines=nearby_lines,
                property_hint=prop_hint,
            )
            associations.append(assoc)
        else:
            # Orphaned dimension text — still record it
            assoc = DimensionAssociation(
                text_element=text,
                value=text.text.strip(),
                numeric_value=text.numeric_value,
                association_type="proximity",
                confidence=0.20,
            )
            associations.append(assoc)

    # Also handle non-dimension callouts (steel shapes, piece marks, bolt specs)
    callout_categories = {"steel_shape", "piece_mark", "bolt_spec", "weld_symbol"}
    for text in text_elements:
        if text.category not in callout_categories:
            continue

        leader = _detect_leader_line(text, lines, config.leader_line_proximity)
        assoc_type = "leader" if leader else "callout"
        confidence = 0.80 if leader else 0.65

        assoc = DimensionAssociation(
            text_element=text,
            value=text.text.strip(),
            numeric_value=None,
            association_type=assoc_type,
            confidence=confidence,
            nearby_lines=[leader] if leader else [],
            property_hint=text.category,
        )
        associations.append(assoc)

    logger.info(f"Created {len(associations)} dimension/callout associations")

    # Log confidence distribution
    high = sum(1 for a in associations if a.confidence_label == "high")
    med = sum(1 for a in associations if a.confidence_label == "medium")
    low = sum(1 for a in associations if a.confidence_label == "low")
    logger.info(f"  Confidence: {high} high, {med} medium, {low} low")

    return associations
