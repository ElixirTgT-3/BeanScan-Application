from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import os
import uuid
from datetime import datetime

from ml.bean_classifier import create_bean_classifier
from database.supabase_client import supabase, BEAN_IMAGE_TABLE, BEAN_TYPE_TABLE, DEFECT_TABLE, SHELF_LIFE_TABLE, HISTORY_TABLE

router = APIRouter()

# Initialize classifier
classifier = create_bean_classifier()

@router.post("/scan")
async def scan_bean_image(
    image: UploadFile = File(...),
    user_id: Optional[int] = None,
    location: Optional[str] = None
):
    """
    Scan and classify a bean image
    """
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Generate unique filename
        file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Save image temporarily (you might want to save to cloud storage instead)
        temp_path = f"temp/{filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        try:
            # Classify the image
            prediction_result = classifier.predict(temp_path)
            
            if not prediction_result["success"]:
                raise HTTPException(status_code=500, detail=f"Classification failed: {prediction_result['error']}")
            
            # Get or create bean type
            bean_type_name = prediction_result["predicted_class"]
            bean_type_result = supabase.table(BEAN_TYPE_TABLE).select("bean_type_id").eq("type_name", bean_type_name).execute()
            
            if bean_type_result.data:
                bean_type_id = bean_type_result.data[0]["bean_type_id"]
            else:
                # Create new bean type if it doesn't exist
                new_bean_type = supabase.table(BEAN_TYPE_TABLE).insert({
                    "type_name": bean_type_name,
                    "description": f"Auto-detected bean type: {bean_type_name}"
                }).execute()
                bean_type_id = new_bean_type.data[0]["bean_type_id"]
            
            # Prepare image data
            image_data = {
                "user_id": user_id,
                "bean_type_id": bean_type_id,
                "image_path": filename,
                "image_url": f"/images/{filename}",
                "file_size": len(image_bytes),
                "image_format": file_extension.upper(),
                "capture_date": datetime.utcnow().isoformat()
            }
            
            # Insert image record
            image_result = supabase.table(BEAN_IMAGE_TABLE).insert(image_data).execute()
            image_id = image_result.data[0]["image_id"]
            
            # Create defect record (if confidence is low, consider it a defect)
            defect_id = None
            if prediction_result["confidence"] < 0.8:
                defect_data = {
                    "image_id": image_id,
                    "defect_type": "low_confidence_classification",
                    "severity_level": "low",
                    "defect_percentage": (1 - prediction_result["confidence"]) * 100
                }
                defect_result = supabase.table(DEFECT_TABLE).insert(defect_data).execute()
                defect_id = defect_result.data[0]["defect_id"]
            
            # Create shelf life prediction
            shelf_life_data = {
                "image_id": image_id,
                "bean_type_id": bean_type_id,
                "defect_id": defect_id,
                "predicted_days": 30,  # Default prediction, can be enhanced with ML
                "confidence_score": prediction_result["confidence"],
                "storage_conditions": {"temperature": "room_temp", "humidity": "low"}
            }
            
            shelf_life_result = supabase.table(SHELF_LIFE_TABLE).insert(shelf_life_data).execute()
            shelf_life_id = shelf_life_result.data[0]["shelf_life_id"]
            
            # Calculate healthy vs defective percentages
            healthy_percent = prediction_result["confidence"] * 100
            defective_percent = (1 - prediction_result["confidence"]) * 100
            
            # Create history record
            history_data = {
                "user_id": user_id,
                "image_id": image_id,
                "shelf_life_id": shelf_life_id,
                "bean_type_id": bean_type_id,
                "defect_id": defect_id,
                "healthy_percent": round(healthy_percent, 2),
                "defective_percent": round(defective_percent, 2),
                "confidence_score": prediction_result["confidence"],
                "notes": f"Auto-scanned bean image. Predicted: {bean_type_name}"
            }
            
            history_result = supabase.table(HISTORY_TABLE).insert(history_data).execute()
            history_id = history_result.data[0]["history_id"]
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return JSONResponse(content={
                "success": True,
                "history_id": history_id,
                "image_id": image_id,
                "prediction": prediction_result,
                "bean_type_id": bean_type_id,
                "healthy_percent": healthy_percent,
                "defective_percent": defective_percent,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scan/{history_id}")
async def get_scan_result(history_id: int):
    """
    Get scan result by history ID
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
        defect_result = None
        shelf_life_result = None
        
        if history.get("defect_id"):
            defect_result = supabase.table(DEFECT_TABLE).select("*").eq("defect_id", history["defect_id"]).execute()
        
        if history.get("shelf_life_id"):
            shelf_life_result = supabase.table(SHELF_LIFE_TABLE).select("*").eq("shelf_life_id", history["shelf_life_id"]).execute()
        
        return JSONResponse(content={
            "history": history,
            "image": image_result.data[0] if image_result.data else None,
            "bean_type": bean_type_result.data[0] if bean_type_result.data else None,
            "defect": defect_result.data[0] if defect_result and defect_result.data else None,
            "shelf_life": shelf_life_result.data[0] if shelf_life_result and shelf_life_result.data else None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scan/status/{history_id}")
async def get_scan_status(history_id: int):
    """
    Get scan processing status
    """
    try:
        history_result = supabase.table(HISTORY_TABLE).select("history_id, created_at, confidence_score, healthy_percent, defective_percent").eq("history_id", history_id).execute()
        
        if not history_result.data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        return JSONResponse(content={
            "history_id": history_id,
            "status": "completed",
            "result": history_result.data[0]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bean-types")
async def get_bean_types():
    """
    Get all available bean types
    """
    try:
        result = supabase.table(BEAN_TYPE_TABLE).select("*").order("type_name").execute()
        return JSONResponse(content={"bean_types": result.data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
