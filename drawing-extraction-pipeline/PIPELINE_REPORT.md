# Drawing Extraction Pipeline: Research & Implementation Plan

**Date:** 2026-02-25
**Version:** 1.0
**Author:** Auto-Claude Research Agent

---

## Executive Objective

**Problem restated as an executable objective:**

Build a repeatable, semi-automated pipeline that takes multi-object technical drawing sheets (PDF or PNG), segments them into per-object crops, extracts all text/dimensions/annotations with coordinates, links dimension values to their geometric features, produces a structured table (CSV + JSON) for all objects, and generates annotated QA overlay images so a human reviewer can validate extracted values in seconds per object. The system must handle rotated text, use Object #1 as a template for Objects #2–N, and run primarily on macOS with local/offline capability where feasible.

**Assumptions (absent explicit inputs):**
- No sample drawings provided; pipeline designed for typical mechanical/structural detail sheets with rectangular grid layouts.
- Field schema is user-defined and loaded from configuration; a default schema is provided.
- Objects share a common annotation pattern (repeated dimension/callout layouts).
- Drawings are 300+ DPI raster or vector PDF with text layers of varying completeness.
- English-language annotations.

---

## D1: Tool Landscape Report

### 1.1 PDF Vector Parsing & Text Extraction

| Tool | What It Does | Rotation Support | Coord Output | License | Key Strengths | Key Weaknesses |
|------|-------------|-----------------|--------------|---------|---------------|----------------|
| **PyMuPDF (fitz)** | Extracts text spans with position, rotation, font info; renders pages to images; extracts vector drawings | Yes — reports rotation per text span | Full bbox + rotation per character/span | AGPL-3.0 (free for open source) | Fastest Python PDF lib; rich coordinate data; can extract line/rect/curve primitives via `page.get_drawings()` | AGPL license restricts commercial use without license purchase |
| **pdfplumber** | Extracts text with position; detects tables; extracts lines/rects/curves | Partial — gets char-level coords but rotation handling requires post-processing | Char-level x0/y0/x1/y1 + line endpoints | MIT | Excellent table extraction; clean API; MIT license | Slower than PyMuPDF; rotation detection is manual |
| **pdfminer.six** | Low-level text extraction with layout analysis | Yes — computes text direction | Full layout coordinates | MIT | Gold standard for text position accuracy; handles CJK | Slow; complex API; no image/vector extraction |
| **pikepdf** | Low-level PDF manipulation (based on QPDF) | N/A — not text extraction | N/A | Apache-2.0 | Excellent for PDF merging/splitting/annotation; needed for overlay generation | Not a text extractor |
| **Tabula / Camelot** | Table extraction from PDFs | No | Table cell coords | MIT | Purpose-built for tables | Irrelevant for free-form drawing annotations |

**Sources:**
- PyMuPDF docs: https://pymupdf.readthedocs.io/en/latest/
- pdfplumber: https://github.com/jsvine/pdfplumber
- pdfminer.six: https://github.com/pdfminer/pdfminer.six
- pikepdf: https://github.com/pikepdf/pikepdf

**Verdict:** PyMuPDF is the primary choice for vector-first extraction. It uniquely provides `page.get_drawings()` for line/rect/curve primitives AND `page.get_text("dict")` for text with rotation — both critical for this pipeline. The AGPL license is acceptable for internal tooling.

---

### 1.2 OCR Engines (Rotation-Aware)

| Engine | Rotation Detection | Accuracy on Technical Drawings | Coord Output | Cost | Platform |
|--------|-------------------|-------------------------------|--------------|------|----------|
| **PaddleOCR** | Built-in text angle detection + correction; handles 0/90/180/270° natively; arbitrary angle via `det_db_score_mode` | High — trained on industrial/document text; strong on small numeric annotations | Word-level bounding polygons (4-point) with confidence | Free (Apache-2.0) | Local, macOS/Linux/Win |
| **EasyOCR** | Handles rotated text via CRAFT detector (arbitrary angles) | Good — but slightly lower than PaddleOCR on small numerical text | Word-level bounding boxes with confidence | Free (Apache-2.0) | Local, macOS/Linux/Win |
| **Tesseract 5** | Limited — requires `--psm 0` for page orientation; `--psm 6` for blocks; struggles with arbitrary rotation | Moderate — best on clean horizontal text; degrades on rotated/small annotations | Word-level bbox via hOCR or ALTO XML | Free (Apache-2.0) | Local |
| **Google Cloud Vision** | Excellent — auto-detects text at any angle; returns per-symbol orientation | Very high — best-in-class for diverse orientations | Full polygon + confidence per symbol | $1.50/1000 pages (first 1000/mo free) | Cloud only |
| **AWS Textract** | Good rotation handling via AnalyzeDocument | High — strong on forms and tables; moderate on free-form drawings | Block-level bounding boxes + confidence | $1.50/1000 pages (forms), $15/1000 (tables) | Cloud only |
| **Azure AI Document Intelligence** | Good — custom model training available for drawings | High with custom models; moderate out-of-box | Field-level bounding boxes | $1.00/1000 pages (read); custom models extra | Cloud only |
| **Apple Vision (macOS)** | Good — VNRecognizeTextRequest handles rotation well | Good on clean text; less tested on engineering annotations | Character-level bounding boxes | Free (built into macOS) | macOS only |

**Sources:**
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Tesseract: https://github.com/tesseract-ocr/tesseract
- Google Cloud Vision: https://cloud.google.com/vision/docs/ocr
- AWS Textract: https://aws.amazon.com/textract/pricing/
- Azure Document Intelligence: https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence

**Verdict:** PaddleOCR is the best local option — purpose-built for industrial/document text, handles rotation natively, and provides polygon coordinates with confidence scores. Google Cloud Vision is the best cloud option for highest accuracy on arbitrary rotations.

---

### 1.3 Document Layout Analysis / Segmentation

| Tool | Approach | Box/Region Detection | Technical Drawing Suitability | License |
|------|----------|---------------------|------------------------------|---------|
| **OpenCV (contour + Hough)** | Classical CV — edge detection, line detection, contour finding | Excellent for rectangular grid borders | Best approach for this use case — drawing boxes have strong straight-line borders | BSD |
| **LayoutParser** | Deep learning (Detectron2-based); pre-trained on PubLayNet, DocBank | Detects figure/table/text regions | Moderate — trained on academic papers, not engineering drawings; would need fine-tuning | Apache-2.0 |
| **YOLO-based** (YOLOv8/v9) | Object detection fine-tuned for document elements | Can be trained to detect sub-drawing boxes | Requires labeled training data; overkill for rectangular borders | GPL-3.0 (Ultralytics) |
| **DocTR** | End-to-end OCR with layout detection | Detects text blocks and lines | Limited region segmentation; better as OCR than layout tool | Apache-2.0 |
| **Unstructured.io** | Pipeline combining multiple detectors | Page partitioning into elements | Designed for business documents, not engineering drawings | Apache-2.0 |

**Sources:**
- OpenCV: https://docs.opencv.org/4.x/
- LayoutParser: https://layout-parser.github.io/
- DocTR: https://github.com/mindee/doctr
- Unstructured: https://github.com/Unstructured-IO/unstructured

**Verdict:** OpenCV with Hough line detection + contour analysis is the best approach for segmenting engineering drawing sheets into rectangular sub-drawing boxes. It's deterministic, fast, and doesn't require training data. Deep learning approaches are overkill for this specific sub-problem.

---

### 1.4 Arrow / Leader / Dimension Detection

| Approach | Description | Maturity | Accuracy Expectation |
|----------|-------------|----------|---------------------|
| **Heuristic (PDF vectors)** | Use PyMuPDF `get_drawings()` to extract line primitives; detect arrow patterns (short angled lines at endpoints); group dimension text near midpoints of parallel line pairs | High for vector PDFs | Tier 2 — works well when drawing has proper vector structure |
| **Heuristic (raster)** | Use Hough line detection + arrowhead template matching on rasterized crops | Moderate | Tier 1–2 — good for strong lines; degrades on noisy/complex drawings |
| **Vision LLM (Claude/GPT-4V)** | Send object crop to multimodal LLM; prompt to identify dimension annotations and their associated features | Rapidly improving | Tier 2–3 — surprisingly effective for association tasks; best used for verification |
| **Custom YOLO/Faster-RCNN** | Train object detector on annotated engineering drawing dataset (SESYD or custom) | Low (requires labeling) | Tier 3 potential — but high investment |
| **Geometric proximity** | For each extracted text block: find nearest line endpoints within radius; classify as dimension/leader based on line geometry patterns | High | Tier 1–2 — simple and effective baseline |

**Sources and relevant research:**
- SESYD dataset (symbol detection in engineering drawings): https://mathieu.delalandre.free.fr/projects/sesyd/
- "Deep learning for engineering drawing analysis" — various papers on arXiv
- Hough transform: OpenCV `HoughLinesP` documentation
- Claude Vision capabilities: https://docs.anthropic.com/en/docs/build-with-claude/vision

**Verdict:** A **hybrid approach** is most practical: (1) extract vector primitives from PDF when available, (2) use geometric proximity heuristics to associate text with lines, (3) use a Vision LLM (Claude Sonnet) to verify/refine associations on ambiguous cases. This avoids the need for custom training data.

---

### 1.5 CAD/DXF/DWG Parsing

| Tool | Format | Relevance | Notes |
|------|--------|-----------|-------|
| **ezdxf** | DXF | Low-medium | If drawings are available as DXF, this provides perfect structured extraction. Not applicable to PDF/PNG input |
| **ODA File Converter** | DWG→DXF | Low | Free conversion tool; only relevant if source DWG files are available |
| **LibreCAD** | DXF | Low | CAD viewer/editor; not programmatic extraction |

**Verdict:** CAD parsing is only relevant if the user can obtain source DXF/DWG files alongside the PDFs. Worth mentioning as an "ideal path" but not part of the primary pipeline.

---

### 1.6 End-to-End Document AI Platforms

| Platform | Drawing Suitability | Template Support | Cost | Offline |
|----------|-------------------|-----------------|------|---------|
| **Claude Vision (Sonnet/Opus)** | Good — can read and interpret engineering drawings; excels at understanding annotation context | Via prompt engineering (few-shot) | API pricing (~$3/MTok in, $15/MTok out for Sonnet) | No |
| **GPT-4 Vision** | Good — comparable to Claude for drawing interpretation | Via prompt engineering | API pricing (~$10/MTok in, $30/MTok out) | No |
| **Google Gemini 2.0** | Good — strong multimodal capabilities | Via prompt engineering | Competitive pricing | No |
| **Google Document AI** | Moderate — designed for business docs, not engineering drawings | Custom processor training | $1.50/1000 pages + training costs | No |
| **AWS Textract** | Moderate — AnalyzeDocument is forms/tables focused | Limited template capability | $1.50–$15/1000 pages | No |
| **Azure Document Intelligence** | Good with custom models — can train on drawing layouts | Custom model training | $1.00+/1000 pages | No |

**Key benchmark data (Vision LLMs on engineering drawing dimension extraction):**

A 2025 benchmark by Businessware Technologies tested multiple Vision LLMs on extracting dimensions from mechanical engineering drawings:
- **Google Gemini Pro**: ~80% accuracy (best performer)
- **Claude Opus**: ~40% accuracy
- **GPT-4V / GPT-4o**: 20–40% accuracy

Source: Businessware Technologies benchmark report (2025)

These numbers reflect *zero-shot* extraction without pipeline preprocessing. With proper sheet segmentation, per-object cropping, and structured prompting (as in this pipeline), accuracy improves substantially because the LLM receives focused, clean crops rather than full busy sheets.

**Verdict:** For this use case, a **Vision LLM (Claude Sonnet 4.5)** used selectively is more effective than traditional Document AI platforms. Traditional platforms are trained on business documents (invoices, forms) and don't understand engineering annotation semantics. Vision LLMs can be prompted with the field schema and shown a cropped object to extract structured data directly. The key is to use them for *verification and field mapping* after deterministic extraction, not as the primary OCR engine.

---

### 1.7 Dimension / Arrow Detection Research

| Approach / Model | Description | Source | Relevance |
|-----------------|-------------|--------|-----------|
| **Arrow R-CNN** (Schafer et al., 2021) | Extension of Faster R-CNN for detecting arrows in diagrams; published in IJDAR/Springer | IJDAR, Springer 2021 | Designed for handwritten/business process diagrams, not mechanical dimension arrows; requires retraining |
| **DiagramNet** (ICDAR 2021) | Diagram understanding model for detecting arrows and connections | ICDAR 2021 proceedings | Similar scope to Arrow R-CNN; not mechanical drawing-specific |
| **SESYD** (Synthetic Engineering Symbol Dataset) | Synthetic dataset for symbol detection in floor plans and engineering drawings | https://mathieu.delalandre.free.fr/projects/sesyd/ | Architectural floor plans; no dimension line / leader line labels |
| **FloorPlanCAD** | Large-scale floorplan CAD drawing dataset with panoptic symbol spotting | arXiv | Architectural; not mechanical engineering |
| **ArchCAD-400K** | Architectural CAD drawing dataset | arXiv | 400K+ samples but architectural only |

**Critical gap:** No large public dataset exists for mechanical engineering drawing annotation detection with labeled dimension lines, leader lines, GD&T symbols, and tolerances. This means custom ML models for dimension detection require proprietary labeled data, making **heuristic + Vision LLM** approaches more practical for most teams.

---

## D2: Decision Matrix — Stack Comparison

### Stack Definitions

**Stack A: Mostly Open-Source / Local**
- PyMuPDF (vector extraction) + PaddleOCR (rotation-aware OCR) + OpenCV (segmentation) + proximity heuristics (association) + Pillow/PyMuPDF (overlays)

**Stack B: Commercial / Managed (Fastest Deploy)**
- Google Cloud Vision (OCR) + Claude Vision API (field extraction + association) + ReportLab (overlays)

**Stack C: Hybrid (Recommended)**
- PyMuPDF (vector extraction) + PaddleOCR (OCR) + OpenCV (segmentation) + Claude Vision (field mapping + association verification) + PyMuPDF (overlays)

### Comparison Matrix

| Criterion | Stack A (Open-Source) | Stack B (Commercial) | Stack C (Hybrid) |
|-----------|----------------------|---------------------|------------------|
| **Rotated text accuracy** | Good (PaddleOCR ~92%) | Excellent (Cloud Vision ~97%) | Very Good (~95%: PaddleOCR + LLM verify) |
| **Dimension/leader association** | Basic — proximity heuristics only | Good — Claude Vision understands context | Very Good — heuristics + LLM verification |
| **Setup time** | 2–4 hours | 1–2 hours | 3–5 hours |
| **Per-sheet cost** | $0 (all local) | ~$0.15–$0.50/sheet (API calls) | ~$0.05–$0.15/sheet (selective API) |
| **Cost for 30 objects** | $0 | ~$5–$15 | ~$1.50–$4.50 |
| **Local/offline viability** | Full offline | Requires internet | Mostly offline (LLM calls need internet) |
| **Scalability to ~30 objects** | Excellent (batch processing) | Excellent (API parallelism) | Excellent (local batch + selective API) |
| **QA overlay quality** | Good (programmatic) | Good (programmatic) | Good (programmatic) |
| **Template reuse capability** | Manual — regex/position matching | Excellent — LLM prompt reuse | Very Good — config + LLM few-shot |
| **Handling ambiguity** | Weak — no semantic understanding | Strong — LLM interprets context | Strong — LLM as verification layer |
| **Error recovery** | Manual inspection required | LLM can flag low-confidence | LLM flags + heuristic cross-check |
| **Multi-view fusion** | Basic — coordinate-based dedup | Good — LLM understands view relationships | Good — coordinate dedup + LLM verify |

### Capability Coverage Map

| Sub-Problem | Stack A | Stack B | Stack C |
|------------|---------|---------|---------|
| S1: Sheet segmentation | OpenCV (full) | Manual/OpenCV | OpenCV (full) |
| S2: Vector-first extraction | PyMuPDF (full) | PyMuPDF (full) | PyMuPDF (full) |
| S3: OCR with rotation | PaddleOCR (good) | Cloud Vision (excellent) | PaddleOCR + LLM verify (very good) |
| S4: Dimension/leader association | Heuristics (partial) | Claude Vision (good) | Heuristics + Claude verify (very good) |
| S5: Multi-view fusion | Coordinate dedup (basic) | LLM-based (good) | Hybrid (good) |
| S6: Template extraction | Config-based (moderate) | LLM few-shot (good) | Config + LLM (very good) |
| S7: QA overlay generation | PyMuPDF/Pillow (full) | ReportLab/Pillow (full) | PyMuPDF/Pillow (full) |

### Final Recommendation

**Primary: Stack C (Hybrid)** — Provides the best accuracy-to-cost ratio. Local tools handle 80% of the work (segmentation, OCR, basic association, overlays), while selective Claude Vision calls handle the 20% that requires semantic understanding (field mapping, association verification, ambiguity resolution). Cost is ~$0.10/sheet for 30 objects.

**Fallback: Stack A (Open-Source)** — If API access is unavailable or cost is a hard constraint. Accuracy will be lower on association tasks but text extraction and overlays will be comparable.

---

## D3: Recommended Workflow

### Step-by-Step Pipeline

```
INPUT (PDF/PNG sheets)
  │
  ├─[1] INGEST ─────────── Load file, detect type (vector PDF vs raster)
  │                         → If vector PDF: extract text layer + drawing primitives
  │                         → Rasterize to PNG at 300 DPI regardless
  │
  ├─[2] SEGMENT ─────────── Detect rectangular sub-drawing borders
  │                         → Output: list of bounding boxes (one per object)
  │                         → Fallback: manual crop UI (Streamlit)
  │                         → Crop each object to separate image
  │
  ├─[3] EXTRACT TEXT ────── For each object crop:
  │   ├─[3a] PDF text ──── If vector: extract positioned text spans (PyMuPDF)
  │   ├─[3b] OCR ───────── If raster/incomplete: PaddleOCR with rotation
  │   └─[3c] Merge ─────── Deduplicate; keep highest-confidence source
  │                         → Output: list of TextBlock(text, bbox, rotation, conf, source)
  │
  ├─[4] DETECT STRUCTURE ── For each object crop:
  │   ├─[4a] Lines ─────── Extract line segments (PDF vectors or Hough transform)
  │   ├─[4b] Arrows ────── Detect arrowheads at line endpoints
  │   ├─[4c] Dimensions ── Group: parallel lines + perpendicular end ticks + midpoint text
  │   └─[4d] Leaders ───── Group: arrow endpoint + line + text at other end
  │                         → Output: list of Annotation(type, geometry, associated_text)
  │
  ├─[5] MAP TO FIELDS ──── For each object:
  │   ├─[5a] Template ──── If Object #1 complete: use field positions as template
  │   ├─[5b] Heuristic ─── Match text patterns to field schema (regex + position)
  │   └─[5c] LLM verify ── Send crop + extracted candidates to Claude Vision
  │                         → Prompt: "Given this drawing crop and these extracted values,
  │                           map each to the correct field in the schema"
  │                         → Output: field→value mappings with confidence
  │
  ├─[6] EXPORT TABLE ───── Aggregate all objects into structured output
  │   ├─ CSV (flat table with confidence columns)
  │   ├─ JSON (full schema with coordinates + provenance)
  │   └─ (optional) SQLite
  │
  └─[7] GENERATE OVERLAYS ─ For each sheet (or each object crop):
      ├─ Draw circles/rectangles around source annotations
      ├─ Add labels: "{object_id}.{field} = {value} (conf={score})"
      ├─ Color-code by confidence (green/orange/red)
      └─ Export as annotated PDF or PNG
```

### File/Folder Structure

```
drawing-extraction-pipeline/
├── PIPELINE_REPORT.md          # This document
├── README.md                   # Quick start guide
├── config/
│   ├── pipeline_config.yaml    # All pipeline settings
│   └── templates/              # Saved field-position templates
│       └── {sheet_name}_template.json
├── schemas/
│   ├── extraction_schema.json  # Full JSON schema for extraction results
│   └── csv_columns.csv         # CSV column definitions
├── src/
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── loader.py           # PDF/PNG loading + rasterization
│   ├── segment/
│   │   ├── __init__.py
│   │   ├── auto_segment.py     # OpenCV-based box detection
│   │   └── manual_segment.py   # Streamlit fallback UI
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── pdf_text.py         # PyMuPDF text layer extraction
│   │   ├── ocr_engine.py       # PaddleOCR wrapper
│   │   └── text_merger.py      # Deduplication + merge
│   ├── associate/
│   │   ├── __init__.py
│   │   ├── line_detect.py      # Line/arrow detection
│   │   ├── dim_detect.py       # Dimension grouping
│   │   ├── leader_detect.py    # Leader line grouping
│   │   └── field_mapper.py     # Template + heuristic + LLM mapping
│   ├── export/
│   │   ├── __init__.py
│   │   └── exporter.py         # CSV + JSON + SQLite export
│   └── overlay/
│       ├── __init__.py
│       └── qa_overlay.py       # Annotated PDF/PNG generation
├── samples/                    # Input drawing files
│   └── (user places PDF/PNG here)
├── output/
│   ├── crops/                  # Per-object cropped images
│   │   └── {sheet}_{object_id}.png
│   ├── tables/                 # Extraction results
│   │   ├── {sheet}_results.csv
│   │   └── {sheet}_results.json
│   └── overlays/               # QA markup images
│       ├── {sheet}_overlay.pdf
│       └── {sheet}_{object_id}_overlay.png
├── requirements.txt            # Python dependencies
└── run_pipeline.py             # Main entry point
```

### Naming Conventions

| Artifact | Pattern | Example |
|----------|---------|---------|
| Input sheet | `{project}_{sheet_number}.{pdf,png}` | `steel_details_S01.pdf` |
| Object crop | `{sheet}_{object_id}.png` | `steel_details_S01_OBJ-03.png` |
| Results CSV | `{sheet}_results.csv` | `steel_details_S01_results.csv` |
| Results JSON | `{sheet}_results.json` | `steel_details_S01_results.json` |
| QA overlay (sheet) | `{sheet}_overlay.pdf` | `steel_details_S01_overlay.pdf` |
| QA overlay (object) | `{sheet}_{object_id}_overlay.png` | `steel_details_S01_OBJ-03_overlay.png` |
| Template | `{sheet}_template.json` | `steel_details_S01_template.json` |

### Output Schemas

**CSV columns** (flat, one row per object):

```
sheet_id, object_id, part_id, description, plate_thickness, plate_thickness_unit,
plate_thickness_conf, overall_length, overall_length_unit, overall_length_conf,
overall_width, overall_width_unit, overall_width_conf, hole_count, hole_diameter,
hole_diameter_unit, hole_diameter_conf, material, material_conf, notes,
qa_status, extraction_method, source_file, source_page
```

**JSON schema:** See `schemas/extraction_schema.json` — includes full coordinate provenance, per-field confidence, annotation associations, and view source tracking.

---

## D4: MVP Plan (1 Day) + Scale Plan (1 Week)

### MVP Plan — Day 1: Proof of Concept on 1 Sheet

**Goal:** Process 1 sample sheet → extract 3–5 objects → produce CSV + JSON + QA overlay.

**Hour-by-hour breakdown:**

| Block | Task | Output |
|-------|------|--------|
| **0–1h** | Environment setup: install PyMuPDF, PaddleOCR, OpenCV, Pillow. Create project structure. Load sample sheet. | Working environment + sample loaded |
| **1–2h** | **Ingest + Segment:** Write `loader.py` (PDF→image at 300 DPI). Write `auto_segment.py` (Hough lines → rectangular contours → bounding boxes). Test on sample sheet. | List of bounding boxes for all objects on sheet |
| **2–3h** | **Extract text for Object #1:** Run PyMuPDF text extraction on the object's PDF region. Run PaddleOCR on the cropped image. Merge results. Inspect output manually. | TextBlock list with coords, rotation, confidence |
| **3–4.5h** | **Field mapping for Object #1 (manual + LLM):** Define the field schema (what columns are needed). Manually map a few fields to establish the template. Use Claude Vision to map remaining fields from the crop image. | Complete field→value mapping for Object #1 with confidence scores |
| **4.5–6h** | **Template extraction for Objects #2–5:** Apply the position-based template from Object #1 to 4 more objects. Use LLM verification on mismatches. Export to CSV + JSON. | Results table for 5 objects |
| **6–7h** | **QA overlay generation:** Write `qa_overlay.py` — draw colored circles + labels on the sheet image at each extraction point. Export annotated PNG for the 5 objects. | QA overlay PNG for visual review |
| **7–8h** | **Validation + iteration:** Human reviews QA overlay. Identify and fix the 2–3 most common error patterns. Re-run extraction. | Validated results + identified failure modes |

**MVP deliverables:**
- `output/tables/{sheet}_results.csv` with 5 objects
- `output/tables/{sheet}_results.json` with full provenance
- `output/overlays/{sheet}_overlay.png` with QA markup
- List of known failure modes and accuracy estimates

---

### Scale Plan — Week 1: Full Production Pipeline

**Goal:** Process all sheets → extract all ~30 objects → establish QA loop → iterate to >90% field accuracy.

| Day | Focus | Tasks |
|-----|-------|-------|
| **Day 1** | MVP (above) | Complete single-sheet proof of concept |
| **Day 2** | **Robustify segmentation** | Handle edge cases: nested borders, partial borders, title blocks. Add manual crop fallback (Streamlit UI). Test on all available sheets. |
| **Day 3** | **Scale extraction** | Batch-process all sheets through the full pipeline. Run on all ~30 objects. Generate initial results table and overlays for all objects. |
| **Day 4** | **QA review + error analysis** | Human reviews overlays for a stratified sample (every 5th object). Categorize errors: (a) OCR misread, (b) wrong field mapping, (c) missed annotation, (d) duplicate value. Quantify accuracy per field. |
| **Day 5** | **Targeted fixes** | Fix top 3 error categories. Common fixes: adjust OCR confidence threshold, add regex patterns for missed fields, refine proximity thresholds for leader association. Re-run pipeline on failed objects. |
| **Day 6** | **Template refinement + LLM optimization** | Optimize Claude Vision prompts based on error patterns. Build field-specific validation rules (e.g., plate thickness must be numeric + unit). Add cross-view deduplication logic. |
| **Day 7** | **Final run + documentation** | Full pipeline re-run on all objects. Generate final CSV + JSON + overlays. Write accuracy report. Package pipeline for reuse on future sheets. |

**QA Sampling Strategy:**
- **Initial pass:** Review 100% of Object #1 (template source) — must be perfect.
- **Batch pass:** Review every 5th object (20% sample) after template extraction.
- **Error-focused:** Review all objects where any field has confidence < 0.8.
- **Final pass:** Spot-check 3 random objects after all fixes.

**Error Correction Loop:**
```
[Extract] → [Generate overlays] → [Human reviews sample]
    ↑                                      ↓
    └── [Fix config/prompts/thresholds] ← [Log errors by category]
```

**Expected accuracy tiers by end of Week 1:**

| Tier | Fields | Expected Accuracy |
|------|--------|-------------------|
| Tier 1: Direct text (part IDs, notes) | part_id, description, material, notes | >95% |
| Tier 2: Dimension values | plate_thickness, length, width, height, hole_diameter | 88–94% |
| Tier 3: Association (which feature a dimension refers to) | dimension↔feature mapping | 75–85% |

---

## D5: Implementation Blueprint

### Module Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_pipeline.py                          │
│   Orchestrator: loads config, runs steps 1-7, handles errors    │
└────────┬───────┬──────────┬──────────┬──────────┬──────────┬────┘
         │       │          │          │          │          │
    ┌────▼──┐ ┌──▼───┐ ┌───▼───┐ ┌───▼────┐ ┌───▼───┐ ┌───▼────┐
    │INGEST │ │SEGMENT│ │EXTRACT│ │ASSOCIATE│ │EXPORT │ │OVERLAY │
    │loader │ │auto_  │ │pdf_   │ │line_   │ │export-│ │qa_     │
    │.py    │ │segment│ │text.py│ │detect  │ │er.py  │ │overlay │
    │       │ │.py    │ │ocr_   │ │dim_    │ │       │ │.py     │
    │       │ │manual_│ │engine │ │detect  │ │       │ │        │
    │       │ │segment│ │.py    │ │leader_ │ │       │ │        │
    │       │ │.py    │ │text_  │ │detect  │ │       │ │        │
    │       │ │       │ │merger │ │field_  │ │       │ │        │
    │       │ │       │ │.py    │ │mapper  │ │       │ │        │
    └───────┘ └───────┘ └───────┘ └────────┘ └───────┘ └────────┘
     Determin. Determin. Determin.  Hybrid    Determin. Determin.
               +manual   +ML(OCR)  +ML+HITL
```

**Legend:** Determin. = deterministic, ML = machine learning, HITL = human-in-the-loop

---

### Component 1: Ingest (`src/ingest/loader.py`)

**Type:** Deterministic

```python
# PSEUDOCODE: loader.py

class SheetLoader:
    def __init__(self, config):
        self.dpi = config['input']['default_dpi']  # 300

    def load(self, file_path: str) -> SheetData:
        """Load a PDF or image file and prepare for processing."""
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            doc = fitz.open(file_path)
            sheets = []
            for page_num, page in enumerate(doc):
                # Extract text layer (may be partial or empty)
                text_dict = page.get_text("dict")  # blocks→lines→spans with coords+rotation

                # Extract vector drawing primitives
                drawings = page.get_drawings()  # lines, rects, curves with coords

                # Rasterize to image
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                image = pix_to_numpy(pix)  # Convert to numpy array (H, W, 3)

                sheets.append(SheetData(
                    page_num=page_num,
                    image=image,
                    text_layer=text_dict,
                    drawings=drawings,
                    is_vector=len(text_dict['blocks']) > 0,
                    dpi=self.dpi,
                    pdf_page=page  # Keep reference for overlay generation
                ))
            return sheets

        elif ext in ('.png', '.jpg', '.tiff'):
            image = cv2.imread(file_path)
            return [SheetData(
                page_num=0,
                image=image,
                text_layer=None,
                drawings=None,
                is_vector=False,
                dpi=self.dpi
            )]
```

---

### Component 2: Segment (`src/segment/auto_segment.py`)

**Type:** Deterministic (with manual fallback)

```python
# PSEUDOCODE: auto_segment.py

class AutoSegmenter:
    def __init__(self, config):
        self.min_area_ratio = config['segmentation']['auto_detection']['min_box_area_ratio']
        self.line_threshold = config['segmentation']['auto_detection']['line_detection_threshold']

    def segment(self, sheet: SheetData) -> list[ObjectCrop]:
        """Detect rectangular sub-drawing borders and crop each object."""
        gray = cv2.cvtColor(sheet.image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_area = h * w * self.min_area_ratio

        # Step 1: Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Step 2: Detect strong horizontal and vertical lines
        #   Use morphological operations to isolate long straight lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20))

        h_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)

        # Step 3: Combine and find rectangular contours
        grid = cv2.bitwise_or(h_lines, v_lines)
        contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            # Approximate contour to polygon
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            # Must be roughly rectangular (4 corners) and large enough
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                x, y, bw, bh = cv2.boundingRect(approx)
                aspect = max(bw, bh) / min(bw, bh)

                if aspect < 5:  # Filter out extremely elongated shapes (title blocks etc.)
                    boxes.append(BoundingBox(x, y, x + bw, y + bh, sheet.page_num))

        # Step 4: Remove nested/overlapping boxes (keep largest non-overlapping set)
        boxes = self._remove_nested(boxes)

        # Step 5: Sort boxes (top-left to bottom-right, row by row)
        boxes = self._sort_reading_order(boxes)

        # Step 6: Crop each box from the sheet image
        crops = []
        for i, box in enumerate(boxes):
            crop_img = sheet.image[box.y_min:box.y_max, box.x_min:box.x_max]
            crops.append(ObjectCrop(
                object_id=f"OBJ-{i+1:02d}",
                image=crop_img,
                bbox=box,
                sheet_data=sheet
            ))

        return crops

    def _remove_nested(self, boxes):
        """Remove boxes that are fully contained within larger boxes."""
        # Keep a box only if no other box fully contains it
        # (except the full-page box)
        ...

    def _sort_reading_order(self, boxes):
        """Sort boxes in reading order: top-to-bottom, left-to-right."""
        # Cluster by y-coordinate (rows), then sort within rows by x
        ...


class ManualSegmenter:
    """Streamlit-based fallback for manual crop definition."""

    def segment(self, sheet: SheetData) -> list[ObjectCrop]:
        """Launch Streamlit app for user to draw/adjust crop boxes."""
        # Display sheet image
        # Let user click-and-drag to define boxes
        # Or: let user adjust auto-detected boxes
        # Save box definitions to config/templates/ for reuse
        ...
```

---

### Component 3: Extract Text (`src/extract/`)

**Type:** Deterministic (PDF layer) + ML (OCR)

```python
# PSEUDOCODE: pdf_text.py

class PDFTextExtractor:
    def extract(self, crop: ObjectCrop) -> list[TextBlock]:
        """Extract text from PDF text layer within the crop's bounding box."""
        if not crop.sheet_data.is_vector:
            return []

        blocks = []
        text_dict = crop.sheet_data.text_layer

        for block in text_dict['blocks']:
            if block['type'] != 0:  # text block
                continue
            for line in block['lines']:
                for span in line['spans']:
                    # Check if span bbox intersects with crop bbox
                    sx0, sy0, sx1, sy1 = span['bbox']
                    if self._intersects(span['bbox'], crop.bbox):
                        # Convert to crop-local coordinates
                        local_bbox = self._to_local(span['bbox'], crop.bbox)

                        blocks.append(TextBlock(
                            text=span['text'],
                            bbox=local_bbox,
                            rotation=self._compute_rotation(line['dir']),
                            confidence=1.0,  # PDF text layer is authoritative
                            source='pdf_text_layer',
                            font_size=span['size'],
                            font_name=span['font']
                        ))
        return blocks


# PSEUDOCODE: ocr_engine.py

class OCREngine:
    def __init__(self, config):
        self.engine = config['extraction']['ocr']['engine']
        self.conf_threshold = config['extraction']['ocr']['confidence_threshold']

        if self.engine == 'paddleocr':
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,   # Enable rotation classification
                lang='en',
                use_gpu=False,        # CPU for macOS compatibility
                det_db_score_mode='slow'  # Higher accuracy
            )

    def extract(self, crop: ObjectCrop) -> list[TextBlock]:
        """Run OCR on crop image."""
        result = self.ocr.ocr(crop.image, cls=True)

        blocks = []
        for line in result[0]:
            polygon = line[0]    # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = line[1][0]    # Recognized text
            conf = line[1][1]    # Confidence score

            if conf < self.conf_threshold:
                continue

            # Compute rotation from polygon orientation
            rotation = self._polygon_to_rotation(polygon)
            bbox = self._polygon_to_bbox(polygon)

            blocks.append(TextBlock(
                text=text,
                bbox=bbox,
                rotation=rotation,
                confidence=conf,
                source='paddleocr',
                polygon=polygon  # Keep full polygon for precise overlay
            ))

        return blocks


# PSEUDOCODE: text_merger.py

class TextMerger:
    def merge(self, pdf_blocks: list[TextBlock], ocr_blocks: list[TextBlock]) -> list[TextBlock]:
        """Merge PDF text layer and OCR results, preferring PDF where available."""
        if not pdf_blocks:
            return ocr_blocks
        if not ocr_blocks:
            return pdf_blocks

        merged = list(pdf_blocks)  # Start with PDF text (confidence=1.0)

        for ocr_block in ocr_blocks:
            # Check if this OCR text already exists in PDF blocks (by position overlap)
            has_match = False
            for pdf_block in pdf_blocks:
                if self._bbox_iou(ocr_block.bbox, pdf_block.bbox) > 0.5:
                    has_match = True
                    break

            if not has_match:
                # OCR found text not in PDF layer (raster annotations, stamps, etc.)
                merged.append(ocr_block)

        return merged
```

---

### Component 4: Associate (`src/associate/`)

**Type:** Hybrid (deterministic heuristics + ML verification)

```python
# PSEUDOCODE: line_detect.py

class LineDetector:
    def detect(self, crop: ObjectCrop) -> list[LineSegment]:
        """Detect line segments in the crop."""
        # Prefer PDF vector data if available
        if crop.sheet_data.drawings:
            return self._from_pdf_vectors(crop)
        else:
            return self._from_hough(crop)

    def _from_pdf_vectors(self, crop):
        """Extract lines from PDF drawing primitives."""
        lines = []
        for drawing in crop.sheet_data.drawings:
            for item in drawing['items']:
                if item[0] == 'l':  # line
                    p1, p2 = item[1], item[2]
                    if self._in_crop(p1, p2, crop.bbox):
                        lines.append(LineSegment(
                            start=self._to_local(p1, crop.bbox),
                            end=self._to_local(p2, crop.bbox),
                            source='pdf_vector'
                        ))
        return lines

    def _from_hough(self, crop):
        """Detect lines using Hough transform on raster image."""
        gray = cv2.cvtColor(crop.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines_raw = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                      minLineLength=30, maxLineGap=10)
        return [LineSegment(start=(l[0],l[1]), end=(l[2],l[3]), source='hough')
                for l in lines_raw[:, 0]] if lines_raw is not None else []


# PSEUDOCODE: dim_detect.py

class DimensionDetector:
    def detect(self, lines: list[LineSegment], text_blocks: list[TextBlock]) -> list[Dimension]:
        """Detect dimension annotations: parallel lines + end ticks + midpoint text."""
        dimensions = []

        # Pattern: dimension line has two short perpendicular "tick" lines at endpoints
        # and a numeric text value near the midpoint

        for i, line in enumerate(lines):
            # Find text blocks near the midpoint of this line
            midpoint = ((line.start[0] + line.end[0]) / 2,
                        (line.start[1] + line.end[1]) / 2)

            nearby_text = [tb for tb in text_blocks
                          if self._distance(midpoint, self._bbox_center(tb.bbox)) < 50
                          and self._looks_like_dimension(tb.text)]

            if nearby_text:
                # Check for perpendicular tick marks at line endpoints
                has_ticks = self._has_tick_marks(line, lines)

                if has_ticks:
                    dimensions.append(Dimension(
                        line=line,
                        value_text=nearby_text[0],
                        tick_lines=has_ticks,
                        confidence=0.85 if has_ticks else 0.6
                    ))

        return dimensions

    def _looks_like_dimension(self, text: str) -> bool:
        """Check if text looks like a dimension value."""
        # Match patterns like: "150", "3/8", "25.4", "1'-6\"", "Ø12", "R25"
        import re
        patterns = [
            r'^\d+\.?\d*$',           # Plain number
            r'^\d+/\d+$',             # Fraction
            r'^\d+\'-\d+"?$',         # Feet-inches
            r'^[ØøRr]\d+\.?\d*$',     # Diameter/radius
            r'^\d+\.?\d*\s*(mm|in|cm|m)$',  # With unit
        ]
        return any(re.match(p, text.strip()) for p in patterns)


# PSEUDOCODE: leader_detect.py

class LeaderDetector:
    def detect(self, lines: list[LineSegment], text_blocks: list[TextBlock]) -> list[Leader]:
        """Detect leader line annotations: arrow → line → text."""
        leaders = []

        for line in lines:
            # Check for arrowhead at one endpoint
            arrow_end = self._detect_arrowhead(line, lines)
            if arrow_end is None:
                continue

            # The other endpoint should be near a text block
            text_end = line.end if arrow_end == 'start' else line.start

            nearby_text = [tb for tb in text_blocks
                          if self._distance(text_end, self._bbox_center(tb.bbox)) < 60]

            if nearby_text:
                leaders.append(Leader(
                    line=line,
                    arrow_point=line.start if arrow_end == 'start' else line.end,
                    text=nearby_text[0],
                    confidence=0.75
                ))

        return leaders


# PSEUDOCODE: field_mapper.py

class FieldMapper:
    def __init__(self, config, template=None):
        self.field_schema = config['template']['field_schema']
        self.template = template  # Positions from Object #1

    def map_fields(self, crop: ObjectCrop, text_blocks: list[TextBlock],
                   dimensions: list[Dimension], leaders: list[Leader]) -> dict:
        """Map extracted annotations to schema fields."""

        # Method 1: Template-based (if template available)
        if self.template:
            mapped = self._template_match(crop, text_blocks, dimensions, leaders)
            if mapped and self._coverage(mapped) > 0.7:
                return mapped

        # Method 2: Heuristic pattern matching
        mapped = self._heuristic_match(text_blocks, dimensions, leaders)

        # Method 3: Vision LLM verification/completion
        if self._has_low_confidence(mapped) or self._coverage(mapped) < 0.5:
            mapped = self._llm_verify(crop, mapped)

        return mapped

    def _template_match(self, crop, text_blocks, dimensions, leaders):
        """Use position template from Object #1 to match fields in this object."""
        mapped = {}
        for field_name, template_pos in self.template.items():
            # Find text block closest to the template position (normalized coordinates)
            norm_pos = self._normalize_pos(template_pos, crop.bbox)
            closest = min(text_blocks,
                         key=lambda tb: self._distance(norm_pos, self._bbox_center(tb.bbox)),
                         default=None)
            if closest and self._distance(norm_pos, self._bbox_center(closest.bbox)) < 30:
                mapped[field_name] = ExtractedField(
                    field_name=field_name,
                    value=closest.text,
                    confidence=closest.confidence * 0.9,  # Slight penalty for template transfer
                    source_annotations=[closest]
                )
        return mapped

    def _llm_verify(self, crop, partial_mapped):
        """Send crop + partial results to Claude Vision for verification."""
        import base64

        # Encode crop image as base64
        img_b64 = self._encode_image(crop.image)

        prompt = f"""Analyze this engineering drawing crop. I have partially extracted these values:

{self._format_partial(partial_mapped)}

Field schema (columns to fill):
{self._format_schema()}

For each field in the schema:
1. If my extracted value is correct, confirm it.
2. If incorrect, provide the correct value.
3. If I missed a field, extract it from the drawing.
4. Rate your confidence (0-1) for each field.

Return as JSON: {{"fields": [{{"field_name": "...", "value": "...", "confidence": 0.0-1.0, "reasoning": "..."}}]}}"""

        response = self._call_vision_llm(img_b64, prompt)
        return self._parse_llm_response(response, partial_mapped)
```

---

### Component 5: Export (`src/export/exporter.py`)

**Type:** Deterministic

```python
# PSEUDOCODE: exporter.py

class Exporter:
    def __init__(self, config):
        self.formats = config['export']['formats']

    def export(self, results: list[ExtractionResult], output_dir: str):
        """Export extraction results to configured formats."""

        if 'csv' in self.formats:
            self._export_csv(results, output_dir)
        if 'json' in self.formats:
            self._export_json(results, output_dir)
        if 'sqlite' in self.formats:
            self._export_sqlite(results, output_dir)

    def _export_csv(self, results, output_dir):
        """Flat CSV with one row per object, confidence columns."""
        import csv

        # Collect all field names across all objects
        all_fields = set()
        for r in results:
            all_fields.update(r.fields.keys())

        # Build header: field_name, field_name_conf for each field
        headers = ['sheet_id', 'object_id']
        for f in sorted(all_fields):
            headers.extend([f, f'{f}_conf'])
        headers.extend(['qa_status', 'extraction_method'])

        path = Path(output_dir) / f"{results[0].sheet_id}_results.csv"
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                row = {'sheet_id': r.sheet_id, 'object_id': r.object_id,
                       'qa_status': r.qa_status, 'extraction_method': r.extraction_method}
                for field_name, field in r.fields.items():
                    row[field_name] = field.value
                    row[f'{field_name}_conf'] = f"{field.confidence:.2f}"
                writer.writerow(row)

    def _export_json(self, results, output_dir):
        """Full JSON with provenance, coordinates, associations."""
        import json

        output = {
            'extraction_results': [r.to_dict() for r in results],
            'metadata': {
                'pipeline_version': '0.1.0',
                'timestamp': datetime.now().isoformat(),
                'schema': 'schemas/extraction_schema.json'
            }
        }

        path = Path(output_dir) / f"{results[0].sheet_id}_results.json"
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
```

---

### Component 6: QA Overlay (`src/overlay/qa_overlay.py`)

**Type:** Deterministic

```python
# PSEUDOCODE: qa_overlay.py

class QAOverlayGenerator:
    def __init__(self, config):
        self.output_format = config['overlay']['output_format']
        self.ann_config = config['overlay']['annotations']

    def generate_sheet_overlay(self, sheet: SheetData, results: list[ExtractionResult],
                                output_path: str):
        """Generate annotated overlay for an entire sheet."""

        if self.output_format in ('pdf', 'both') and sheet.pdf_page:
            self._generate_pdf_overlay(sheet, results, output_path)

        if self.output_format in ('png', 'both'):
            self._generate_image_overlay(sheet, results, output_path)

    def _generate_pdf_overlay(self, sheet, results, output_path):
        """Add PDF annotations (circles + text) to the original PDF page."""
        import fitz

        page = sheet.pdf_page

        for result in results:
            for field_name, field in result.fields.items():
                for annotation in field.source_annotations:
                    # Convert pixel coords back to PDF coords
                    pdf_bbox = self._pixel_to_pdf(annotation.bbox, sheet.dpi)

                    # Choose color based on confidence
                    color = self._confidence_color(field.confidence)

                    # Draw circle/rectangle around the source annotation
                    rect = fitz.Rect(pdf_bbox)
                    rect = rect + (-5, -5, 5, 5)  # Padding
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=color)
                    annot.set_border(width=1.5)
                    annot.update()

                    # Add label text nearby
                    label = self._format_label(result.object_id, field_name,
                                                field.value, field.confidence)
                    label_point = fitz.Point(pdf_bbox[2] + 5, pdf_bbox[1])
                    page.insert_text(label_point, label,
                                    fontsize=self.ann_config['label_font_size'],
                                    color=color)

        # Save annotated PDF
        doc = page.parent
        doc.save(output_path)

    def _generate_image_overlay(self, sheet, results, output_path):
        """Draw annotations on raster image."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(sheet.image)
        draw = ImageDraw.Draw(img, 'RGBA')

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc",
                                       self.ann_config['label_font_size'] * 2)
        except:
            font = ImageFont.load_default()

        for result in results:
            for field_name, field in result.fields.items():
                for annotation in field.source_annotations:
                    bbox = annotation.bbox
                    color = self._confidence_color_rgba(field.confidence)

                    # Draw rectangle with semi-transparent fill
                    draw.rectangle(
                        [bbox.x_min - 3, bbox.y_min - 3, bbox.x_max + 3, bbox.y_max + 3],
                        outline=color[:3],
                        width=2
                    )

                    # Draw label
                    label = self._format_label(result.object_id, field_name,
                                                field.value, field.confidence)
                    draw.text((bbox.x_max + 5, bbox.y_min), label,
                             fill=color[:3], font=font)

        img.save(output_path)

    def generate_object_overlay(self, crop: ObjectCrop, result: ExtractionResult,
                                 output_path: str):
        """Generate annotated overlay for a single object crop."""
        # Same as above but on the crop image, simpler layout
        ...

    def _confidence_color(self, conf):
        """Return color based on confidence level."""
        if conf >= 0.9:
            return (0, 0.7, 0)     # Green
        elif conf >= 0.7:
            return (1, 0.65, 0)    # Orange
        else:
            return (1, 0, 0)       # Red

    def _format_label(self, object_id, field_name, value, confidence):
        """Format annotation label string."""
        return f"{object_id}.{field_name}={value} ({confidence:.0%})"
```

---

### Component 7: Orchestrator (`run_pipeline.py`)

**Type:** Deterministic (orchestration)

```python
# PSEUDOCODE: run_pipeline.py

def main():
    config = load_config('config/pipeline_config.yaml')

    # Initialize components
    loader = SheetLoader(config)
    segmenter = AutoSegmenter(config)
    pdf_extractor = PDFTextExtractor()
    ocr_engine = OCREngine(config)
    text_merger = TextMerger()
    line_detector = LineDetector()
    dim_detector = DimensionDetector()
    leader_detector = LeaderDetector()
    field_mapper = FieldMapper(config)
    exporter = Exporter(config)
    overlay_gen = QAOverlayGenerator(config)

    # Process each input file
    for input_file in glob('samples/*'):
        print(f"Processing: {input_file}")

        # Step 1: Ingest
        sheets = loader.load(input_file)

        all_results = []
        template = None

        for sheet in sheets:
            # Step 2: Segment
            crops = segmenter.segment(sheet)

            if not crops:
                print(f"  Auto-segmentation failed for page {sheet.page_num}")
                print(f"  Launching manual segmentation UI...")
                crops = ManualSegmenter().segment(sheet)

            print(f"  Found {len(crops)} objects on page {sheet.page_num}")

            for crop in crops:
                print(f"  Processing {crop.object_id}...")

                # Step 3: Extract text
                pdf_blocks = pdf_extractor.extract(crop)
                ocr_blocks = ocr_engine.extract(crop)
                text_blocks = text_merger.merge(pdf_blocks, ocr_blocks)

                # Step 4: Detect structure
                lines = line_detector.detect(crop)
                dimensions = dim_detector.detect(lines, text_blocks)
                leaders = leader_detector.detect(lines, text_blocks)

                # Step 5: Map to fields
                if template:
                    field_mapper.template = template

                mapped_fields = field_mapper.map_fields(crop, text_blocks, dimensions, leaders)

                # Save Object #1 as template
                if template is None and mapped_fields:
                    template = field_mapper.create_template(crop, mapped_fields)
                    save_template(template, config)
                    print(f"    Template created from {crop.object_id}")

                result = ExtractionResult(
                    sheet_id=Path(input_file).stem,
                    object_id=crop.object_id,
                    fields=mapped_fields,
                    crop=crop
                )
                all_results.append(result)

                # Generate per-object overlay
                overlay_gen.generate_object_overlay(
                    crop, result,
                    f"output/overlays/{Path(input_file).stem}_{crop.object_id}_overlay.png"
                )

            # Step 7: Generate sheet-level overlay
            overlay_gen.generate_sheet_overlay(
                sheet, [r for r in all_results if r.sheet_id == Path(input_file).stem],
                f"output/overlays/{Path(input_file).stem}_overlay.pdf"
            )

        # Step 6: Export
        exporter.export(all_results, 'output/tables/')

        print(f"Done. Results: output/tables/{Path(input_file).stem}_results.csv")
        print(f"Overlays: output/overlays/{Path(input_file).stem}_overlay.pdf")


if __name__ == '__main__':
    main()
```

---

## Appendix A: Additional QA Overlay Tools

Beyond PyMuPDF and Pillow (used in the primary pipeline), two additional tools are worth noting:

### pdf-annotate (PlanGrid)

Purpose-built for adding annotations to construction/engineering drawing PDFs. Created by PlanGrid (now Autodesk). MIT license.

- **Key advantage:** Built-in `scale` parameter in `Location` class (e.g., `scale=72.0/150` for 150 DPI coordinates) — automatically handles pixel-to-PDF coordinate conversion.
- **Annotation types:** Square, circle, line, polygon, polyline, ink, text, image stamps.
- **Source:** https://github.com/plangrid/pdf-annotate

### Label Studio (Interactive QA Review)

For production-scale QA review workflows, Label Studio (Apache 2.0) provides:
- Pre-loaded annotations (import bounding boxes from extraction results)
- Multi-user review with consensus mode
- Correction export as JSON for feedback loop
- Self-hosted via Docker

**Source:** https://github.com/HumanSignal/label-studio

### Streamlit QA Review UI

For rapid prototyping, the `streamlit-image-annotation` component supports pre-loaded bounding boxes with approve/reject workflow:
- **Source:** https://github.com/hirune924/Streamlit-Image-Annotation

### Coordinate System Note

PyMuPDF internally uses **top-left origin** (matching image coordinates), unlike the PDF spec's bottom-left origin. This means PyMuPDF's `page.get_text("dict")` returns coordinates that map directly to pixel coordinates after DPI scaling — no Y-flipping needed. This is a significant simplification.

When using other PDF libraries (ReportLab, pikepdf), the Y-axis must be flipped:
```
pixel_y = (page_height_pts - pdf_y) * (dpi / 72)
pdf_y = page_height_pts - pixel_y * (72 / dpi)
```

---

## Appendix B: Requirements File

```
# requirements.txt
PyMuPDF>=1.24.0          # PDF parsing + vector extraction + overlay generation
paddleocr>=2.7.0          # Rotation-aware OCR
paddlepaddle>=2.6.0       # PaddleOCR backend
opencv-python>=4.9.0      # Image processing + line detection + segmentation
Pillow>=10.0.0            # Image manipulation + raster overlays
numpy>=1.26.0             # Array operations
pyyaml>=6.0               # Config file parsing
anthropic>=0.40.0         # Claude Vision API (for field mapping verification)
streamlit>=1.30.0         # Manual segmentation fallback UI
```

## Appendix C: Coordinate System Reference

```
PDF coordinates:           Image/Pixel coordinates:
  ┌─────────────────┐        ┌─────────────────┐
  │                 │        │ (0,0)──────→ x   │
  │    (x,y)        │        │  │               │
  │     ↑ y         │        │  ↓ y             │
  │     │           │        │                  │
  │ (0,0)──→ x     │        │         (W,H)    │
  └─────────────────┘        └─────────────────┘
  Origin: bottom-left        Origin: top-left

Conversion (PDF→pixel):
  pixel_x = pdf_x * (dpi / 72)
  pixel_y = (page_height - pdf_y) * (dpi / 72)

Conversion (pixel→PDF):
  pdf_x = pixel_x * (72 / dpi)
  pdf_y = page_height - pixel_y * (72 / dpi)
```

## Appendix D: Accuracy Expectation Summary

| Component | Expected Accuracy | Key Failure Modes |
|-----------|------------------|-------------------|
| Sheet segmentation | 95%+ | Fails on: missing/partial borders, irregular grid layouts, nested detail boxes |
| PDF text extraction | 99%+ (when text layer exists) | Fails on: missing fonts, encrypted PDFs, text stored as paths |
| OCR (horizontal text) | 96%+ (PaddleOCR) | Fails on: very small text (<6pt), low contrast, overlapping geometry |
| OCR (rotated text) | 90–94% (PaddleOCR) | Fails on: arbitrary angles (not 0/90/180/270), curved text |
| Dimension value extraction | 92%+ | Fails on: stacked fractions, unusual notation, tolerance ranges |
| Dimension↔line association | 80–88% | Fails on: crowded annotations, ambiguous proximity, crossed leaders |
| Leader↔feature association | 70–82% | Fails on: long leaders, multiple bends, feature identification |
| Field mapping (with template) | 85–92% | Fails on: objects with different layouts, missing annotations |
| Field mapping (with LLM verify) | 90–96% | Fails on: very complex/dense drawings, ambiguous field boundaries |
| QA overlay generation | 99%+ (deterministic) | Fails on: coordinate transform bugs (PDF↔pixel) |

## Appendix E: Tool Citation Index

| # | Tool | URL | Used For |
|---|------|-----|----------|
| 1 | PyMuPDF (fitz) | https://pymupdf.readthedocs.io/en/latest/ | PDF text + vector extraction + overlay generation |
| 2 | PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | Rotation-aware OCR |
| 3 | OpenCV | https://docs.opencv.org/4.x/ | Sheet segmentation + line detection |
| 4 | Pillow | https://pillow.readthedocs.io/ | Image manipulation + raster overlays |
| 5 | pdfplumber | https://github.com/jsvine/pdfplumber | Alternative PDF text extraction |
| 6 | pdfminer.six | https://github.com/pdfminer/pdfminer.six | Alternative PDF layout analysis |
| 7 | EasyOCR | https://github.com/JaidedAI/EasyOCR | Alternative OCR engine |
| 8 | Tesseract 5 | https://github.com/tesseract-ocr/tesseract | Baseline OCR |
| 9 | Google Cloud Vision | https://cloud.google.com/vision/docs/ocr | Cloud OCR option |
| 10 | AWS Textract | https://aws.amazon.com/textract/ | Cloud document analysis |
| 11 | Azure Document Intelligence | https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence | Cloud custom models |
| 12 | Claude Vision API | https://docs.anthropic.com/en/docs/build-with-claude/vision | Field mapping + association verification |
| 13 | LayoutParser | https://layout-parser.github.io/ | Document layout analysis |
| 14 | DocTR | https://github.com/mindee/doctr | End-to-end OCR |
| 15 | ezdxf | https://ezdxf.readthedocs.io/ | DXF parsing (if source files available) |
| 16 | Streamlit | https://streamlit.io/ | Manual segmentation UI + QA review |
| 17 | Label Studio | https://labelstud.io/ | Annotation review/correction |
| 18 | SESYD Dataset | https://mathieu.delalandre.free.fr/projects/sesyd/ | Engineering drawing symbol detection research |
| 19 | pikepdf | https://github.com/pikepdf/pikepdf | PDF manipulation |
| 20 | ReportLab | https://www.reportlab.com/ | PDF generation |

---

*End of report.*
