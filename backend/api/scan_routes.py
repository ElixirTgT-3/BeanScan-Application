from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import os
import uuid
import logging
from datetime import datetime

from ml.bean_classifier import create_bean_classifier
from ml.defect_detector import create_defect_detector
from database.supabase_client import supabase, BEAN_IMAGE_TABLE, BEAN_TYPE_TABLE, DEFECT_TABLE, SHELF_LIFE_TABLE, HISTORY_TABLE

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize models
classifier = create_bean_classifier()
try:
    defect_detector = create_defect_detector()
    logger.info("Defect detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize defect detector: {e}")
    defect_detector = None

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
        logger.info(f"Received scan request for image: {image.filename}")
        logger.info(f"Content type: {image.content_type}")
        # Validate image file - be more flexible with content types
        if image.content_type and not image.content_type.startswith('image/'):
            # Check if it's a common image file extension instead
            if image.filename:
                file_ext = image.filename.lower().split('.')[-1]
                if file_ext not in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']:
                    raise HTTPException(status_code=400, detail="File must be an image")
            else:
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
            logger.info(f"Starting image classification for {temp_path}")
            prediction_result = classifier.predict(temp_path)
            logger.info(f"Classification result: {prediction_result}")
            
            if not prediction_result["success"]:
                logger.error(f"Classification failed: {prediction_result['error']}")
                raise HTTPException(status_code=500, detail=f"Classification failed: {prediction_result['error']}")
            
            # Detect defects
            if defect_detector is not None:
                try:
                    defect_result = defect_detector.detect_defects(temp_path)
                except Exception as e:
                    logger.error(f"Defect detection failed: {e}")
                    defect_result = {
                        'success': False,
                        'error': str(e),
                        'detections': [],
                        'summary': {
                            'total_defects': 0,
                            'defect_types': {},
                            'defect_percentage': 0,
                            'quality_score': 1.0,
                            'quality_grade': 'Unknown'
                        }
                    }
            else:
                logger.warning("Defect detector not available, skipping defect detection")
                defect_result = {
                    'success': False,
                    'error': 'Defect detector not available',
                    'detections': [],
                    'summary': {
                        'total_defects': 0,
                        'defect_types': {},
                        'defect_percentage': 0,
                        'quality_score': 1.0,
                        'quality_grade': 'Unknown'
                    }
                }
            
            # Get or create bean type
            bean_type_name = prediction_result["predicted_class"]
            logger.info(f"Looking up bean type: {bean_type_name}")
            bean_type_result = supabase.table(BEAN_TYPE_TABLE).select("bean_type_id").eq("type_name", bean_type_name).execute()
            logger.info(f"Bean type lookup result: {bean_type_result.data}")
            
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
            
            # Create defect records based on actual defect detection
            defect_id = None
            if defect_result["success"] and defect_result["summary"]["total_defects"] > 0:
                # Create a summary defect record
                defect_percentage = defect_result["summary"]["defect_percentage"]
                if defect_percentage < 5:
                    severity = "low"
                elif defect_percentage < 15:
                    severity = "medium"
                elif defect_percentage < 30:
                    severity = "high"
                else:
                    severity = "critical"
                
                defect_data = {
                    "image_id": image_id,
                    "defect_type": "multiple_defects",
                    "severity_level": severity,
                    "defect_area": defect_percentage,
                    "defect_percentage": defect_percentage,
                    "defect_coordinates": defect_result["detections"]  # Store detection coordinates
                }
                defect_result_db = supabase.table(DEFECT_TABLE).insert(defect_data).execute()
                defect_id = defect_result_db.data[0]["defect_id"]
            elif prediction_result["confidence"] < 0.8:
                # Fallback to low confidence classification
                confidence_percentage = (1 - prediction_result["confidence"]) * 100
                defect_data = {
                    "image_id": image_id,
                    "defect_type": "low_confidence_classification",
                    "severity_level": "low",
                    "defect_area": confidence_percentage,
                    "defect_percentage": confidence_percentage,
                    "defect_coordinates": None
                }
                defect_result_db = supabase.table(DEFECT_TABLE).insert(defect_data).execute()
                defect_id = defect_result_db.data[0]["defect_id"]
            
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
            if defect_result["success"]:
                healthy_percent = defect_result["summary"]["quality_score"] * 100
                defective_percent = 100 - healthy_percent
            else:
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
                try:
                    os.remove(temp_path)
                except PermissionError:
                    # File might be locked on Windows, try again later
                    logger.warning(f"Could not delete temp file {temp_path}, will be cleaned up later")
            
            return JSONResponse(content={
                "success": True,
                "history_id": history_id,
                "image_id": image_id,
                "prediction": prediction_result,
                "defect_detection": defect_result,
                "bean_type_id": bean_type_id,
                "healthy_percent": healthy_percent,
                "defective_percent": defective_percent,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except PermissionError:
                    logger.warning(f"Could not delete temp file {temp_path}")
            raise e
            
    except Exception as e:
        logger.error(f"Unexpected error in scan_bean_image: {e}", exc_info=True)
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

@router.post("/detect-defects")
async def detect_defects_only(
    image: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    """
    Detect defects in a coffee bean image without classification
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
        
        # Save image temporarily
        temp_path = f"temp/{filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        try:
            # Detect defects
            if defect_detector is not None:
                defect_result = defect_detector.detect_defects(temp_path, confidence_threshold)
            else:
                defect_result = {
                    'success': False,
                    'error': 'Defect detector not available',
                    'detections': [],
                    'summary': {
                        'total_defects': 0,
                        'defect_types': {},
                        'defect_percentage': 0,
                        'quality_score': 1.0,
                        'quality_grade': 'Unknown'
                    }
                }
            
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except PermissionError:
                    logger.warning(f"Could not delete temp file {temp_path}")
            
            return JSONResponse(content=defect_result)
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except PermissionError:
                    logger.warning(f"Could not delete temp file {temp_path}")
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-scan")
async def test_scan_endpoint(image: UploadFile = File(...)):
    """
    Test endpoint to debug scan issues
    """
    try:
        logger.info(f"Test scan received image: {image.filename}")
        logger.info(f"Content type: {image.content_type}")
        
        # Validate image file - be more flexible with content types
        if image.content_type and not image.content_type.startswith('image/'):
            # Check if it's a common image file extension instead
            if image.filename:
                file_ext = image.filename.lower().split('.')[-1]
                if file_ext not in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']:
                    return JSONResponse(content={
                        "success": False,
                        "error": "File must be an image"
                    }, status_code=400)
            else:
                return JSONResponse(content={
                    "success": False,
                    "error": "File must be an image"
                }, status_code=400)
        
        # Read image bytes
        image_bytes = await image.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        # Generate unique filename
        file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        filename = f"test_{uuid.uuid4()}.{file_extension}"
        
        # Save image temporarily
        temp_path = f"temp/{filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        logger.info(f"Image saved to: {temp_path}")
        
        # Test image loading
        from PIL import Image
        with Image.open(temp_path) as test_image:
            logger.info(f"Image loaded successfully: {test_image.size}")
            image_size = test_image.size
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return JSONResponse(content={
            "success": True,
            "message": "Test scan completed successfully",
            "image_info": {
                "filename": image.filename,
                "content_type": image.content_type,
                "size": len(image_bytes),
                "dimensions": image_size
            }
        })
        
    except Exception as e:
        logger.error(f"Test scan error: {e}", exc_info=True)
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

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
