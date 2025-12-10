# Methods and Tools: Rule‑Based Shelf Life and Defect Severity

This document explains the current rule‑based logic that powers shelf‑life estimation and severity scoring in BeanScan (as implemented in `lib/pages/results_page.dart`). It replaces older LSTM/ML formulas with deterministic, interpretable rules.

## Pipeline Overview
1) **Normalize detections**: Standardize detection fields (`defect_type`, `confidence`, coordinates, image dims).  
2) **Filter non‑defects**: `good_beans` are excluded from counts/percentages and never raise severity. If only good beans remain, severity becomes `normal`, defect% = 0, total defects = 0.  
3) **Summarize defects**: Merge backend summary, shelf-life payload, and normalized detections into a single summary (totals, type counts, defect%, severity).  
4) **Severity resolution**: Pick the worst of backend severity/category, computed severity from defect%, and rank floors from defect types.  
5) **Weighted defect score**: Convert defect types/counts into a 0–1 weighted score using a defect-weight table.  
6) **Shelf-life adjustment**: Scale baseline predicted days by a decay multiplier derived from the weighted score (and defect%). Recompute months from adjusted days.  
7) **Status/category**: Derive status from adjusted days and resolved severity; apply same logic to PDF export for parity with UI.

## Detection Normalization and Counting
- Accepts detections in various shapes (bbox maps/lists, `x/y/width/height`, `coordinates`, etc.).  
- Extracts/derives `x1,y1,x2,y2`, normalizes `defect_type`, rescales confidence to 0–1, and carries image dimensions when present.  
- Two views of detections are kept: full list (for overlays) and filtered list (excluding `good_beans`) for counts/percentages.

## Defect Types, Ranks, and Weights
- **Severity rank (high→low)**: Fully Black > Insect‑Damaged > Broken/Cut > Roasted. Rank sets minimum severity/defect% floors.  
- **Weight table (for shelf‑life decay)**: Fully Black 1.0, Insect 0.7, Broken/Cut 0.5, Roasted 0.2 (underscores/hyphens/spacing tolerated). Any unknown defect defaults to weight 0.3.  
- **Weighted score** = (Σ weight × count) / total_defects (good beans excluded), clamped to 0–1. This captures both mix and severity of defects.

## Defect Percentage and Severity
- **Defect% sources (in order)**: backend `defect_percentage` → shelf-life `defect_percentage` → detection-based estimator (confidence/count-driven) → fallback heuristic.  
- **Rank floors**: High-rank defects enforce a minimum defect% and severity.  
- **Severity resolution**: Worst of (a) backend severity/category (normalized), (b) computed from defect%, (c) rank-derived severity. Zero-defect (or good-bean-only) cases are forced to `normal`.

## Shelf-Life Adjustment (Rule-Based Decay)
- **Baseline days**: backend `predicted_days` if provided; if only good beans, fallback to 45 days; otherwise baseline defaults to bean type (Arabica 30, Liberica 28, Excelsa 26, Robusta 25, Other 20).  
- **Decay multiplier (linear)** from weighted score: score 0 → 1.0×, score 1 → 0.3× (floor 0.25×). Defect% can further clamp the multiplier when no weighted score is used.  
- **Adjusted days**: `adjusted_days = max(7, baseline_days * multiplier)`.  
- **Estimated months**: always recomputed from adjusted days (`adjusted_days / 30`, 1 decimal).  
- **Category/Status**: Derived from adjusted days when no explicit category exists; severity feeds the label mapping (Normal, Excellent, Warning, Critical).

## Good-Bean Handling
- `good_beans` are ignored for defect counts, percentages, and weighted score.  
- If all detections are good beans: defect% = 0, severity = `normal`, predicted days fallback (45), months recomputed, category “Excellent,” and status resolves accordingly.

## Parity Between UI and PDF
- The same summary, weighting, severity resolution, and shelf-life adjustment are applied when rendering the on-screen info card and the PDF export to avoid mismatches.

## Inputs and Outputs
- **Inputs**: detections (type, confidence, optional counts/sizes), optional backend summary and shelf-life payload (predicted_days, estimated_months, defect_percentage, severity/category).  
- **Outputs**: adjusted predicted days/months, resolved severity/status, defect counts/types, defect percentage, and confidence score for display and export.

## Tuning Knobs
- Update `_defectWeights` to change the relative impact of defect types.  
- Adjust the linear decay in `_shelfLifeMultiplierFromScore` (change slope/floor) for harsher or softer penalties.  
- Modify rank floors if a defect type should force a higher minimum severity/defect%.  
- Change the good-bean fallback days/months if “excellent” baseline should differ.
