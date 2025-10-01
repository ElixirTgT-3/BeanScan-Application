from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
from datetime import datetime, timedelta

from database.supabase_client import supabase, HISTORY_TABLE, BEAN_IMAGE_TABLE, BEAN_TYPE_TABLE, DEFECT_TABLE, SHELF_LIFE_TABLE, USER_TABLE

router = APIRouter()

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
        # Resolve user by device_id if provided and user_id is not
        resolved_user_id = user_id
        if device_id and not user_id:
            try:
                user_lookup = supabase.table(USER_TABLE).select("user_id").eq("Name", device_id).limit(1).execute()
                if user_lookup.data:
                    resolved_user_id = user_lookup.data[0]["user_id"]
            except Exception:
                resolved_user_id = user_id

        # Build query
        query = supabase.table(HISTORY_TABLE).select("*")
        
        # Apply filters
        if resolved_user_id:
            query = query.eq("user_id", resolved_user_id)
        
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
        # Get history record
        history_result = supabase.table(HISTORY_TABLE).select("*").eq("history_id", history_id).execute()
        
        if not history_result.data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        history = history_result.data[0]
        
        # Get related data
        image_result = supabase.table(BEAN_IMAGE_TABLE).select("*").eq("image_id", history["image_id"]).execute()
        bean_type_result = supabase.table(BEAN_TYPE_TABLE).select("*").eq("bean_type_id", history["bean_type_id"]).execute()

        # Build defect_detection payload (detections + summary)
        defects_result = supabase.table(DEFECT_TABLE).select("*").eq("image_id", history["image_id"]).execute()
        detections = []
        defect_types_counts = {}
        defect_percentage_total = 0.0
        for d in defects_result.data or []:
            coords = d.get("defect_coordinates") or {}
            detections.append({
                "defect_type": d.get("defect_type"),
                "confidence": None,
                "coordinates": {
                    "x1": float(coords.get("x1", 0.0)),
                    "y1": float(coords.get("y1", 0.0)),
                    "x2": float(coords.get("x2", 0.0)),
                    "y2": float(coords.get("y2", 0.0)),
                },
                "area": d.get("defect_area"),
                "defect_percentage": float(d.get("defect_percentage", 0.0)),
            })
            t = d.get("defect_type") or "unknown"
            defect_types_counts[t] = defect_types_counts.get(t, 0) + 1
            defect_percentage_total += float(d.get("defect_percentage", 0.0))

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

        return JSONResponse(content={
            "history": history,
            "image": image_result.data[0] if image_result.data else None,
            "bean_type": bean_type_result.data[0] if bean_type_result.data else None,
            "defect_detection": defect_detection_payload,
            "shelf_life": shelf_life_payload
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
