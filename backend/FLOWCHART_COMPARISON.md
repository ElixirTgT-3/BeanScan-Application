# Backend scan flow vs provided flowchart

## What the backend actually does (`/api/v1/scan` in `backend/api/scan_routes_custom.py`)
1. Validate upload: accepts common image content types or extensions; reads bytes; writes a temp copy plus a static copy for serving.
2. Preprocess: converts to RGB, resizes to 224×224, normalizes with ImageNet means/stds.
3. Run ensemble (`ml/custom_models.py`):
   - Bean type classification (MobileNetV3-based CNN with light TTA).
   - Defect detection via Faster R-CNN (or MobileNet defect classifier fallback); results are coordinate-normalized back to the original image size.
   - Rule-based shelf-life model builds a defect sequence (type, confidence, count=1) and produces `defect_percentage`, severity band, confidence, and estimated days/months.
   - Health score = bean confidence minus defect penalties; separate from shelf-life severity.
4. Database work (Supabase): optional user resolution/creation by `device_id`, bean type lookup/insert, image row insert, per-defect rows (area, coords; `defect_percentage` often 0 because detectors do not supply it), shelf-life row insert, history row with `healthy_percent` and `defective_percent` (driven by shelf-life `defect_percentage`).
5. Response: returns history/image IDs, image URL, detections with normalized coordinates, defect summary, shelf-life payload (includes `defect_percentage` and severity), and health score.

## How that compares to the flowchart
- **Preprocess (resize/normalize)**: matches the chart.
- **Detection stage**: uses Faster R-CNN-style detector (RPN/ROI heads) or classifier fallback; comparable to the chart’s “Fast R-CNN processing,” but the downstream metric differs.
- **Defect % calculation**: chart uses summed bbox area ÷ image area × 100; backend’s live path uses rule-based weighted defect scores → `defect_percentage = clamp(min(total_defect_score/45, 1.5) * 100)`. Bounding-box area is not used to derive the final percentage.
- **Severity mapping**: chart has explicit branches (0%→Normal, 1–22 Mild, 18–78 Moderate, 70–100 Severe). Backend now returns **Normal** when no defects are detected (0%); otherwise it picks a band (mild/moderate/severe) from the rule-based percentage.
- **Result handling**: backend stores image/defect/shelf-life/history rows and returns JSON; the flowchart’s “Display result” and “Store defect data” steps are covered, with extra user-resolution logic not shown in the diagram.
- **Legacy/unused path**: `backend/api/scan_routes.py` (not mounted in `backend/main.py`) does compute defect_percentage from bbox area, closer to the chart, but it is not the active route.

## Bottom line
The active backend flow follows the same high-level shape (ingest → preprocess → detect → summarize → store) but diverges on how defect percentage and severity are derived (rule-based weighted scoring instead of bbox-area percentage, and no separate “Normal” branch for 0%).***
