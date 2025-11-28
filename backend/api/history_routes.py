from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any, Iterable, Tuple
from datetime import datetime, timedelta
import os
from PIL import Image

from database.supabase_client import supabase, HISTORY_TABLE, BEAN_IMAGE_TABLE, BEAN_TYPE_TABLE, DEFECT_TABLE, SHELF_LIFE_TABLE, USER_TABLE

router = APIRouter()

STATIC_IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "images"))
DEFAULT_MODEL_INPUT_WIDTH = 224.0
DEFAULT_MODEL_INPUT_HEIGHT = 224.0


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            return float(cleaned)
    except (TypeError, ValueError):
        return None
    return None


def _load_image_dimensions(image_record: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not image_record:
        return None
    filename = image_record.get("image_path") or image_record.get("filename")
    if not filename:
        return None
    image_path = os.path.join(STATIC_IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        return None
    try:
        with Image.open(image_path) as img:
            return float(img.width), float(img.height)
    except Exception:
        return None


def _rescale_history_detections(
    detections: Iterable[Dict[str, Any]],
    image_width: float,
    image_height: float,
) -> List[Dict[str, Any]]:
    if not detections:
        return []

    image_width = float(image_width)
    image_height = float(image_height)
    scaled = []

    for detection in detections:
        if not isinstance(detection, dict):
            scaled.append(detection)
            continue

        det = dict(detection)
        
        # Check if detection already has image dimensions matching target size
        # If so, coordinates are likely already in the correct size
        existing_image_width = _safe_float(det.get("image_width")) or _safe_float((det.get("image_size") or {}).get("width"))
        existing_image_height = _safe_float(det.get("image_height")) or _safe_float((det.get("image_size") or {}).get("height"))
        
        # If coordinates are already in target image size, skip rescaling
        if (existing_image_width and abs(existing_image_width - image_width) < 1.0 and
            existing_image_height and abs(existing_image_height - image_height) < 1.0):
            # Coordinates are already in correct size, just ensure image dimensions are set
            print(f"[DEBUG] Detection coordinates already in target size ({image_width}x{image_height}), skipping rescale")
            det["image_width"] = float(image_width)
            det["image_height"] = float(image_height)
            det["image_size"] = {
                "width": float(image_width),
                "height": float(image_height)
            }
            scaled.append(det)
            continue
        
        model_input_width = (
            _safe_float(det.get("model_input_width")) or
            _safe_float(det.get("image_width")) or
            _safe_float((det.get("image_size") or {}).get("width")) or
            DEFAULT_MODEL_INPUT_WIDTH
        )
        model_input_height = (
            _safe_float(det.get("model_input_height")) or
            _safe_float(det.get("image_height")) or
            _safe_float((det.get("image_size") or {}).get("height")) or
            DEFAULT_MODEL_INPUT_HEIGHT
        )

        coords_source = det.get("defect_coordinates") or det.get("coordinates") or {}
        if not isinstance(coords_source, dict):
            coords_source = {}

        x1 = _safe_float(coords_source.get("x1") or coords_source.get("left") or coords_source.get("xmin") or coords_source.get("x"))
        y1 = _safe_float(coords_source.get("y1") or coords_source.get("top") or coords_source.get("ymin") or coords_source.get("y"))
        x2 = _safe_float(coords_source.get("x2") or coords_source.get("right") or coords_source.get("xmax"))
        y2 = _safe_float(coords_source.get("y2") or coords_source.get("bottom") or coords_source.get("ymax"))
        width_val = _safe_float(coords_source.get("width") or coords_source.get("w"))
        height_val = _safe_float(coords_source.get("height") or coords_source.get("h"))

        if x2 is None and x1 is not None and width_val is not None:
            x2 = x1 + width_val
        if y2 is None and y1 is not None and height_val is not None:
            y2 = y1 + height_val

        scale_x = 1.0
        scale_y = 1.0
        if model_input_width and abs(model_input_width - image_width) > 1.0:
            scale_x = image_width / model_input_width
            print(f"[DEBUG] Scaling X: {model_input_width} -> {image_width} (scale={scale_x:.3f})")
        if model_input_height and abs(model_input_height - image_height) > 1.0:
            scale_y = image_height / model_input_height
            print(f"[DEBUG] Scaling Y: {model_input_height} -> {image_height} (scale={scale_y:.3f})")
        if scale_x == 1.0 and scale_y == 1.0:
            print(f"[DEBUG] No scaling needed: model_input={model_input_width}x{model_input_height}, target={image_width}x{image_height}")

        def _scale_val(value: Optional[float], scale: float) -> Optional[float]:
            if value is None:
                return None
            return float(value * scale)

        x1 = _scale_val(x1, scale_x)
        y1 = _scale_val(y1, scale_y)
        x2 = _scale_val(x2, scale_x)
        y2 = _scale_val(y2, scale_y)

        coords = {}
        if x1 is not None:
            coords["x1"] = x1
        if y1 is not None:
            coords["y1"] = y1
        if x2 is not None:
            coords["x2"] = x2
        if y2 is not None:
            coords["y2"] = y2
        if x1 is not None and x2 is not None:
            coords["width"] = float(max(0.0, x2 - x1))
        if y1 is not None and y2 is not None:
            coords["height"] = float(max(0.0, y2 - y1))

        bbox = det.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bbox = [
                _scale_val(_safe_float(bbox[0]), scale_x),
                _scale_val(_safe_float(bbox[1]), scale_y),
                _scale_val(_safe_float(bbox[2]), scale_x),
                _scale_val(_safe_float(bbox[3]), scale_y),
            ]

        area = _safe_float(det.get("area"))
        if area is not None and (scale_x != 1.0 or scale_y != 1.0):
            area = float(area * scale_x * scale_y)

        det["coordinates"] = coords if coords else det.get("coordinates")
        det["defect_coordinates"] = coords if coords else det.get("defect_coordinates")
        if bbox:
            det["bbox"] = bbox
        if area is not None:
            det["area"] = area

        det["image_width"] = float(image_width)
        det["image_height"] = float(image_height)
        det["image_size"] = {
            "width": float(image_width),
            "height": float(image_height)
        }
        det["model_input_width"] = float(model_input_width or DEFAULT_MODEL_INPUT_WIDTH)
        det["model_input_height"] = float(model_input_height or DEFAULT_MODEL_INPUT_HEIGHT)
        det["model_input_size"] = {
            "width": float(model_input_width or DEFAULT_MODEL_INPUT_WIDTH),
            "height": float(model_input_height or DEFAULT_MODEL_INPUT_HEIGHT)
        }
        if scale_x != 1.0 or scale_y != 1.0:
            det["scaled_from_model_input"] = True
            det["coordinate_space"] = "original_image"

        scaled.append(det)

    return scaled

@router.get("/history")
async def get_scan_history(
    user_id: Optional[int] = None,
    device_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get scan history with optional filtering
    """
    try:
        print(f"[DEBUG] History request - device_id: {device_id}, user_id: {user_id}")
        
        # Resolve user by device_id if provided and user_id is not
        resolved_user_id = user_id
        if device_id and not user_id:
            try:
                print(f"[DEBUG] Looking up user by device_id: {device_id}")
                user_lookup = supabase.table(USER_TABLE).select("user_id").eq("Name", device_id).limit(1).execute()
                print(f"[DEBUG] User lookup result: {user_lookup.data}")
                if user_lookup.data:
                    resolved_user_id = user_lookup.data[0]["user_id"]
                    print(f"[DEBUG] Resolved user_id: {resolved_user_id}")
                else:
                    print(f"[DEBUG] No user found for device_id: {device_id}")
                    # If no user found, this might be a new device - return empty for now
                    # The user will be created when they do their first scan
                    return JSONResponse(content={
                        "scans": [],
                        "total": 0,
                        "limit": limit,
                        "offset": offset,
                        "message": "No history found for this device. Scan a bean to create your first record."
                    })
            except Exception as e:
                print(f"[DEBUG] Error during user lookup: {e}")
                resolved_user_id = user_id

        print(f"[DEBUG] Final resolved_user_id: {resolved_user_id}")

        # Build query
        query = supabase.table(HISTORY_TABLE).select("*")
        
        # Apply filters
        if resolved_user_id:
            query = query.eq("user_id", resolved_user_id)
            print(f"[DEBUG] Filtering by user_id: {resolved_user_id}")
        else:
            print(f"[DEBUG] No user_id filter applied - returning all records")
        
        if start_date:
            query = query.gte("created_at", start_date)
        
        if end_date:
            query = query.lte("created_at", end_date)
        
        # Apply pagination and ordering
        query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
        
        result = query.execute()
        
        return JSONResponse(content={
            "scans": result.data,
            "total": len(result.data),
            "limit": limit,
            "offset": offset
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/by-device/{device_id}")
async def get_history_by_device(
    device_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        user_lookup = supabase.table(USER_TABLE).select("user_id").eq("Name", device_id).limit(1).execute()
        if not user_lookup.data:
            return JSONResponse(content={
                "scans": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            })
        uid = user_lookup.data[0]["user_id"]
        result = (
            supabase
            .table(HISTORY_TABLE)
            .select("*")
            .eq("user_id", uid)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return JSONResponse(content={
            "scans": result.data,
            "total": len(result.data),
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{history_id}")
async def get_scan_details(history_id: int):
    """
    Get detailed information about a specific scan
    """
    try:
        print(f"[DEBUG] get_scan_details called for history_id={history_id}")
        # Get history record
        history_result = supabase.table(HISTORY_TABLE).select("*").eq("history_id", history_id).execute()
        print(f"[DEBUG] History result: found={len(history_result.data) if history_result.data else 0} records")
        
        if not history_result.data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        try:
            history = history_result.data[0]
            print(f"[DEBUG] History record type: {type(history)}")
            print(f"[DEBUG] History record keys: {list(history.keys()) if isinstance(history, dict) else 'not a dict'}")
            
            # Get related data
            image_id = history.get("image_id")
            bean_type_id = history.get("bean_type_id")
            print(f"[DEBUG] Extracted image_id={image_id}, bean_type_id={bean_type_id}")
        except Exception as history_err:
            print(f"[ERROR] Failed to process history record: {history_err}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to process history record: {str(history_err)}")
        
        if not image_id:
            print(f"[ERROR] History record missing image_id. History data: {history}")
            raise HTTPException(status_code=500, detail="History record missing image_id")
        
        # bean_type_id is optional - some history records may not have it
        if bean_type_id:
            print(f"[DEBUG] Fetching image_id={image_id}, bean_type_id={bean_type_id}")
        else:
            print(f"[DEBUG] Fetching image_id={image_id}, bean_type_id=None (optional)")
        
        image_result = supabase.table(BEAN_IMAGE_TABLE).select("*").eq("image_id", image_id).execute()
        print(f"[DEBUG] Image result: found={len(image_result.data) if image_result.data else 0} records")

        if not image_result.data:
            print(f"[ERROR] Image not found for image_id={image_id}")
            raise HTTPException(status_code=500, detail=f"Image not found for image_id={image_id}")
        
        # Only fetch bean_type if bean_type_id is provided
        bean_type_data = None
        if bean_type_id:
            bean_type_result = supabase.table(BEAN_TYPE_TABLE).select("*").eq("bean_type_id", bean_type_id).execute()
            print(f"[DEBUG] Bean type result: found={len(bean_type_result.data) if bean_type_result.data else 0} records")
            if bean_type_result.data:
                bean_type_data = bean_type_result.data[0]
            else:
                print(f"[WARNING] Bean type not found for bean_type_id={bean_type_id}, returning None")
        else:
            print(f"[DEBUG] No bean_type_id provided, bean_type will be None in response")

        image_dims = _load_image_dimensions(image_result.data[0] if image_result.data else None)

        # Build defect_detection payload (detections + summary)
        defects_result = supabase.table(DEFECT_TABLE).select("*").eq("image_id", image_id).execute()
        detections = []
        defect_types_counts = {}
        defect_percentage_total = 0.0
        for d in defects_result.data or []:
            try:
                coords = d.get("defect_coordinates") or {}
                if not isinstance(coords, dict):
                    coords = {}
                
                # Build detection with coordinates
                detection = {
                    "defect_type": d.get("defect_type"),
                    "confidence": None,
                    "coordinates": {
                        "x1": float(coords.get("x1", 0.0) or 0.0),
                        "y1": float(coords.get("y1", 0.0) or 0.0),
                        "x2": float(coords.get("x2", 0.0) or 0.0),
                        "y2": float(coords.get("y2", 0.0) or 0.0),
                    },
                    "area": d.get("defect_area"),
                    "defect_percentage": float(d.get("defect_percentage", 0.0) or 0.0),
                }
                
                # Add image dimensions to detection so rescaling function knows the coordinate space
                # Coordinates in database are likely already in original image size, but we'll let
                # the rescaling function check and only scale if needed
                if image_dims:
                    detection["image_width"] = float(image_dims[0])
                    detection["image_height"] = float(image_dims[1])
                    detection["image_size"] = {
                        "width": float(image_dims[0]),
                        "height": float(image_dims[1]),
                    }
                
                detections.append(detection)
                t = d.get("defect_type") or "unknown"
                defect_types_counts[t] = defect_types_counts.get(t, 0) + 1
                defect_percentage_total += float(d.get("defect_percentage", 0.0) or 0.0)
            except Exception as det_err:
                print(f"[ERROR] Failed to process detection: {det_err}, detection data: {d}")
                continue

        # Only rescale if coordinates appear to be in model input size (not original image size)
        # The rescaling function will check if coordinates are already in target size
        if image_dims and detections:
            try:
                detections = _rescale_history_detections(detections, image_dims[0], image_dims[1])
            except Exception as rescale_err:
                print(f"[ERROR] Failed to rescale detections: {rescale_err}")
                # Continue with unscaled detections

        total_defects = len(detections)
        avg_defect_percentage = (defect_percentage_total / total_defects) if total_defects > 0 else 0.0

        def derive_quality_grade(pct: float) -> str:
            if pct < 10: return "A"
            if pct < 20: return "B"
            if pct < 35: return "C"
            if pct < 50: return "D"
            return "F"

        defect_detection_payload = {
            "detections": detections,
            "summary": {
                "quality_grade": derive_quality_grade(avg_defect_percentage),
                "total_defects": total_defects,
                "defect_percentage": avg_defect_percentage,
                "defect_types": defect_types_counts,
            }
        }
        if image_dims:
            defect_detection_payload["image_dimensions"] = {
                "width": image_dims[0],
                "height": image_dims[1],
            }
            defect_detection_payload["model_input_dimensions"] = {
                "width": DEFAULT_MODEL_INPUT_WIDTH,
                "height": DEFAULT_MODEL_INPUT_HEIGHT,
            }

        # Shelf life enrichment
        shelf_life_payload = None
        if history.get("shelf_life_id"):
            shelf_life_result = supabase.table(SHELF_LIFE_TABLE).select("*").eq("shelf_life_id", history["shelf_life_id"]).execute()
            if shelf_life_result.data:
                sl = shelf_life_result.data[0]
                days = int(sl.get("predicted_days", 0) or 0)
                def derive_category(d: int) -> str:
                    if d >= 30: return "Excellent"
                    if d >= 20: return "Good"
                    if d >= 10: return "Warning"
                    if d > 0: return "Critical"
                    return "Unknown"
                sl_enriched = dict(sl)
                sl_enriched.setdefault("confidence_score", sl.get("confidence_score", 0.0))
                sl_enriched.setdefault("category", derive_category(days))
                shelf_life_payload = sl_enriched

        print(f"[DEBUG] Preparing response for history_id={history_id}")
        response_data = {
            "history": history,
            "image": image_result.data[0],
            "bean_type": bean_type_data,  # Can be None if bean_type_id was missing
            "defect_detection": defect_detection_payload,
            "shelf_life": shelf_life_payload
        }
        print(f"[DEBUG] Response prepared successfully, returning JSONResponse")
        return JSONResponse(content=response_data)
        
    except HTTPException as http_err:
        # Re-raise HTTP exceptions as-is
        print(f"[ERROR] HTTPException in get_scan_details for history_id={history_id}: {http_err.detail}")
        raise
    except Exception as e:
        import traceback
        import sys
        error_trace = traceback.format_exc()
        error_msg = f"[ERROR] get_scan_details failed for history_id={history_id}: {e}"
        print(error_msg, file=sys.stderr, flush=True)
        print(f"[ERROR] Traceback: {error_trace}", file=sys.stderr, flush=True)
        print(error_msg, flush=True)
        print(f"[ERROR] Traceback: {error_trace}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to get scan details: {str(e)}")

@router.delete("/history/{history_id}")
async def delete_scan(history_id: int, user_id: Optional[int] = None):
    """
    Delete a scan (only if user owns it)
    """
    try:
        # Check if scan exists and user has permission
        history_result = supabase.table(HISTORY_TABLE).select("user_id").eq("history_id", history_id).execute()
        
        if not history_result.data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        if user_id and history_result.data[0]["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this scan")
        
        # Delete history record (cascading will handle related records)
        try:
            supabase.table(HISTORY_TABLE).delete().eq("history_id", history_id).execute()
        except Exception as db_error:
            raise HTTPException(status_code=500, detail=f"Database error: {db_error}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Scan deleted successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/stats")
async def get_scan_statistics(
    user_id: Optional[int] = None,
    days: int = Query(30, ge=1, le=365)
):
    """
    Get scan statistics for a period
    """
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Build query
        query = supabase.table(HISTORY_TABLE).select("bean_type_id, confidence_score, created_at, healthy_percent, defective_percent")
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        query = query.gte("created_at", start_date.isoformat())
        
        result = query.execute()
        
        if not result.data:
            return JSONResponse(content={
                "total_scans": 0,
                "bean_types": {},
                "average_confidence": 0,
                "average_healthy_percent": 0,
                "average_defective_percent": 0,
                "period_days": days
            })
        
        # Calculate statistics
        total_scans = len(result.data)
        bean_types = {}
        total_confidence = 0
        total_healthy = 0
        total_defective = 0
        
        for scan in result.data:
            if scan.get("bean_type_id"):
                # Get bean type name
                bean_type_result = supabase.table(BEAN_TYPE_TABLE).select("type_name").eq("bean_type_id", scan["bean_type_id"]).execute()
                if bean_type_result.data:
                    bean_type = bean_type_result.data[0]["type_name"]
                    bean_types[bean_type] = bean_types.get(bean_type, 0) + 1
            
            total_confidence += scan.get("confidence_score", 0)
            total_healthy += scan.get("healthy_percent", 0)
            total_defective += scan.get("defective_percent", 0)
        
        average_confidence = total_confidence / total_scans if total_scans > 0 else 0
        average_healthy = total_healthy / total_scans if total_scans > 0 else 0
        average_defective = total_defective / total_scans if total_scans > 0 else 0
        
        return JSONResponse(content={
            "total_scans": total_scans,
            "bean_types": bean_types,
            "average_confidence": round(average_confidence, 3),
            "average_healthy_percent": round(average_healthy, 2),
            "average_defective_percent": round(average_defective, 2),
            "period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/export")
async def export_scan_history(
    user_id: Optional[int] = None,
    format: str = Query("json", regex="^(json|csv)$"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Export scan history in different formats
    """
    try:
        # Build query with joins for comprehensive data
        query = supabase.table(HISTORY_TABLE).select("""
            history_id,
            created_at,
            confidence_score,
            healthy_percent,
            defective_percent,
            notes,
            user_id,
            image_id,
            bean_type_id,
            defect_id,
            shelf_life_id
        """)
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        if start_date:
            query = query.gte("created_at", start_date)
        
        if end_date:
            query = query.lte("created_at", end_date)
        
        result = query.execute()
        
        if format == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            if result.data:
                writer = csv.DictWriter(output, fieldnames=result.data[0].keys())
                writer.writeheader()
                writer.writerows(result.data)
            
            return JSONResponse(
                content={"csv_data": output.getvalue()},
                headers={"Content-Type": "text/csv"}
            )
        else:
            # Return JSON format
            return JSONResponse(content={
                "scans": result.data,
                "total": len(result.data),
                "export_format": format
            })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/summary")
async def get_scan_summary(
    user_id: Optional[int] = None,
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get scan summary with user names and bean types
    """
    try:
        # Use the scan_summary view
        query = supabase.from_("scan_summary").select("*")
        
        if user_id:
            # We need to filter by user_id, but the view doesn't have it
            # So we'll get the data and filter in Python
            result = query.execute()
            filtered_data = [scan for scan in result.data if scan.get("user_id") == user_id]
            result.data = filtered_data[:limit]
        else:
            result = query.limit(limit).execute()
        
        return JSONResponse(content={
            "summary": result.data,
            "total": len(result.data)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users")
async def get_users():
    """
    Get all users (admin only)
    """
    try:
        result = supabase.table(USER_TABLE).select("user_id, Name, role, location, created_at").order("Name").execute()
        return JSONResponse(content={"users": result.data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
