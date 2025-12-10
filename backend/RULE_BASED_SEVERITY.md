# Rule-based severity and shelf-life logic (table)

Location: `backend/ml/custom_models.py` → `RuleBasedShelfLife.predict_shelf_life` (used by `/api/v1/scan` in `backend/api/scan_routes_custom.py`).

## Inputs
- `defect_sequence`: list of defects. Each item may include `type`, `confidence`, `count`. If not provided, it is derived from detections (type, confidence, count=1).
- `bean_type`: string (Arabica, Robusta, Liberica, Excelsa, Other).
- `confidence_threshold` (optional): minimum confidence for “certain” classification (default 0.7).

## Computation steps
1. **Base shelf life (days)**: look up per bean type (default “other” = 1095 to match the clean specialty baseline). Current baselines: Arabica 1095; Liberica 1095; Excelsa 900; Robusta 720; Other 1095.
2. **Defect score**: for each defect, compute `impact = weight * confidence * count`; sum to `total_defect_score`.
   - Weights (higher = worse): insect_damage 8.0; quaker 7.0; nugget 5.0; discoloration 6.0; physical_damage/broken/cut/chip/crack 4.0; shell 3.0; under_roast 2.0; roasted_beans 7.0; unknown/default 1.0.
   - Track `defect_counts`, `total_detected`, and `cumulative_confidence`. Skip only “clean” tokens (good/healthy/clean/background/no_defect).
3. **Defect percentage**: `normalized_score = min(total_defect_score / 45.0, 1.5)` then `defect_percentage = clamp(normalized_score * 100, 0, 100)`.
4. **Scenario selection** (table-driven):
   - Roasted-bean defect present → “roasted_beans” scenario (7–12 months, ↓50–70%).
   - Else if multiple defect categories present → “general defects (mixed)” (12–24 months, ↓10–60%).
   - Else if black defect present → “fully black” (10–14 months, ↓40–60%).
   - Else if insect present → “insect damage” (12–16 months, ↓30–50%).
   - Else if broken/cut/chip/crack/physical present → “broken/cut” (18–27 months, ↓10–25%).
   - Else clean: “Perfect green beans” → 24–36 months (specialty) or 12–24 months (commodity), category `normal`.
5. **Place estimate inside range**: use defect_percentage to push toward lower bound (position = 0.25 + 0.7 * pct; clamp). Roasted defect also applies a 50% cap of its upper bound.
6. **Confidence**: start from scenario base confidence (clean 0.95/0.92; mixed 0.82; fully black 0.70; insect 0.75; broken 0.82; roasted 0.68), subtract intensity and small multi-defect penalties, scale by average detection confidence, clamp to 0.2–0.96, and ensure `confidence_threshold` floor (with “Uncertain” if bumped).
7. **Quality/categorization**: mild/normal → Excellent / Grade A; moderate → Good (<=40%) or Warning (>40%) and Grade B/C; severe → Critical / Grade D.

## Outputs
Dictionary with:
- `predicted_days`, `estimated_months`, `estimated_months_range`, `base_shelf_life`
- `category`, `quality_grade`, `severity`, `severity_position`, `profile_used`
- `confidence`, `defect_percentage`, `defect_score`, `defect_counts`, `total_defects_detected`, `average_detection_confidence`

## Where used
- `backend/api/scan_routes_custom.py` `/api/v1/scan`: builds a defect sequence from detections, calls `predict_shelf_life`, and returns/stores severity, defect_percentage, shelf-life days/months, and confidence.
- The legacy router `backend/api/scan_routes.py` is not mounted and does not use this rule-based model.

## Quick tuning guide
- Adjust `defect_weights` if certain detections should count more/less.
- Tweak the band definitions (`percent_range`, peaks, `month_edges`, `conf_edges`) to reshape severity and shelf-life scaling.
- Update base shelf life per bean type in `base_shelf_life_days` to reflect storage targets.
