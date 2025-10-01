from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import os
import uuid
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from ml.custom_models import BeanScanEnsemble, create_models
from database.supabase_client import supabase, BEAN_IMAGE_TABLE, BEAN_TYPE_TABLE, DEFECT_TABLE, SHELF_LIFE_TABLE, HISTORY_TABLE

def _estimate_bean_count(image_path, defect_detection, health_score):
    """
    Estimate the number of beans in the image using computer vision techniques.
    This uses contour detection and blob analysis to count individual beans.
    """
    try:
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return _fallback_bean_count(defect_detection, health_score)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # Apply threshold to create binary image
        # Use adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio
        bean_contours = []
        min_area = 500  # Minimum area for a bean
        max_area = 5000  # Maximum area for a bean
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Check aspect ratio (beans are roughly oval)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:  # Beans are roughly oval
                    bean_contours.append(contour)
        
        # Count the filtered contours
        bean_count = len(bean_contours)
        
        # If we get a reasonable count, use it
        if 5 <= bean_count <= 100:
            confidence = "high" if len(bean_contours) > 10 else "medium"
            return {
                "estimated_count": bean_count,
                "confidence": confidence,
                "method": "computer_vision"
            }
        else:
            # Fall back to simpler method if CV gives unreasonable results
            return _fallback_bean_count(defect_detection, health_score)
            
    except Exception as e:
        print(f"Error in bean counting: {e}")
        return _fallback_bean_count(defect_detection, health_score)

def _fallback_bean_count(defect_detection, health_score):
    """
    Fallback bean counting method when computer vision fails.
    """
    # Simple estimation based on image analysis heuristics
    base_count = 15  # Start with a reasonable base
    
    # Adjust based on health score (higher health might mean more beans visible)
    health_percentage = health_score.get('percentage', 50)
    if health_percentage > 80:
        base_count += 10
    elif health_percentage > 60:
        base_count += 5
    
    # Add some variation based on defect count (more defects might mean more beans)
    if defect_detection:
        base_count += len(defect_detection) * 2
    
    return {
        "estimated_count": min(max(base_count, 10), 50),
        "confidence": "low",
        "method": "fallback"
    }

router = APIRouter()

# Initialize custom models
print("🚀 Loading custom deep learning models...")
models = create_models(device='cpu')  # Use 'cuda' if GPU available
ensemble = models['ensemble']
print("✅ Models loaded successfully!")

# Image transformations for the models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

@router.post("/scan")
async def scan_bean_image(
    image: UploadFile = File(...),
    user_id: Optional[int] = None,
    location: Optional[str] = None,
    device_id: Optional[str] = None
):
    """
    Scan and analyze a bean image using custom deep learning models
    """
    try:
        # Validate image file - check both content type and file extension
        is_valid_image = False
        
        # Check content type
        if image.content_type and image.content_type.startswith('image/'):
            is_valid_image = True
        # Check file extension as fallback (common with Flutter uploads)
        elif image.filename:
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
            file_ext = image.filename.lower().split('.')[-1] if '.' in image.filename else ''
            if f'.{file_ext}' in valid_extensions:
                is_valid_image = True
                print(f"✅ Valid image by extension: {image.filename}")
        
        if not is_valid_image:
            print(f"❌ Invalid content type: {image.content_type}")
            print(f"❌ Filename: {image.filename}")
            raise HTTPException(status_code=400, detail=f"File must be an image. Received content type: {image.content_type}, filename: {image.filename}")
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Generate unique filename
        file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Save image temporarily and also copy to static/images for serving
        temp_path = f"temp/{filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        # Copy to static folder served by FastAPI
        try:
            static_dir = os.path.join(os.path.dirname(__file__), "..", "static", "images")
            static_dir = os.path.abspath(static_dir)
            os.makedirs(static_dir, exist_ok=True)
            static_path = os.path.join(static_dir, filename)
            with open(static_path, "wb") as sf:
                sf.write(image_bytes)
            public_image_url = f"/images/{filename}"
        except Exception:
            public_image_url = f"/images/{filename}"
        
        try:
            # Load and preprocess image for models
            image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension
            
            # Run complete analysis using ensemble model
            print("🔍 Running deep learning analysis...")
            try:
                analysis_results = ensemble.forward(image_tensor)
                print("✅ Ensemble analysis completed")
            except Exception as e:
                print(f"❌ Ensemble analysis failed: {e}")
                raise e
            
            # Extract results
            bean_classification = analysis_results['bean_classification']
            defect_detection = analysis_results['defect_detection']
            health_score = analysis_results['health_score']
            
            print(f"📊 Analysis complete: Bean={bean_classification[0]['class']}, Health={health_score['grade']}")
            
            # Resolve user by device_id when provided (no login flow)
            resolved_user_id = user_id
            try:
                if device_id and not user_id:
                    # Prefer explicit device_id column if present; fallback to Name
                    user_lookup = None
                    try:
                        user_lookup = supabase.table("User").select("user_id").eq("device_id", device_id).limit(1).execute()
                    except Exception:
                        user_lookup = None
                    if not (user_lookup and user_lookup.data):
                        user_lookup = supabase.table("User").select("user_id").eq("Name", device_id).limit(1).execute()

                    if user_lookup and user_lookup.data:
                        resolved_user_id = user_lookup.data[0]["user_id"]
                    else:
                        # Create a lightweight user row for this device
                        payload = {
                            "Name": device_id,
                            "role": "user",
                            "location": location or None
                        }
                        # Try to set device_id column if it exists
                        try:
                            payload["device_id"] = device_id
                        except Exception:
                            pass
                        new_user = supabase.table("User").insert(payload).execute()
                        if new_user.data:
                            resolved_user_id = new_user.data[0]["user_id"]
            except Exception as _user_err:
                # Proceed without user if mapping fails
                resolved_user_id = user_id

            # Get or create bean type
            bean_type_name = bean_classification[0]['class']
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
                "user_id": resolved_user_id,
                "bean_type_id": bean_type_id,
                "image_path": filename,
                "image_url": public_image_url,
                "file_size": len(image_bytes),
                "image_format": file_extension.upper(),
                "capture_date": datetime.utcnow().isoformat()
            }
            
            # Insert image record
            image_result = supabase.table(BEAN_IMAGE_TABLE).insert(image_data).execute()
            image_id = image_result.data[0]["image_id"]
            
            # Create defect records
            defect_id = None
            if defect_detection:
                for defect in defect_detection:
                    defect_data = {
                        "image_id": image_id,
                        "defect_type": defect['defect_type'],
                        "severity_level": defect['severity_level'] if 'severity_level' in defect else 'medium',
                        "defect_area": defect['area'],
                        "defect_percentage": defect['defect_percentage'] if 'defect_percentage' in defect else 0,
                        "defect_coordinates": defect['coordinates']
                    }
                    defect_result = supabase.table(DEFECT_TABLE).insert(defect_data).execute()
                    if defect_id is None:
                        defect_id = defect_result.data[0]["defect_id"]
            
            # Create shelf life prediction using rule-based model
            shelf_life_model = models['shelf_life_model']
            
            # Prepare defect sequence for rule-based prediction
            defect_sequence = []
            if defect_detection:
                for defect in defect_detection:
                    defect_sequence.append({
                        'type': defect.get('type', 'unknown'),
                        'confidence': defect.get('confidence', 0.5),
                        'count': 1
                    })
            
            # Get shelf life prediction
            shelf_life_prediction = shelf_life_model.predict_shelf_life(
                defect_sequence, 
                bean_type_name
            )
            
            shelf_life_data = {
                "image_id": image_id,
                "bean_type_id": bean_type_id,
                "defect_id": defect_id,
                "predicted_days": shelf_life_prediction['predicted_days'],
                "confidence_score": shelf_life_prediction['confidence'],
                "storage_conditions": {"temperature": "room_temp", "humidity": "low"}
            }
            
            shelf_life_result = supabase.table(SHELF_LIFE_TABLE).insert(shelf_life_data).execute()
            shelf_life_id = shelf_life_result.data[0]["shelf_life_id"]
            
            # Bean counting removed from response/data model
            
            # Calculate percentages for history
            healthy_percent = health_score['percentage']
            defective_percent = 100 - healthy_percent
            
            # Create history record
            history_data = {
                "user_id": resolved_user_id,
                "image_id": image_id,
                "shelf_life_id": shelf_life_id,
                "bean_type_id": bean_type_id,
                "defect_id": defect_id,
                "healthy_percent": healthy_percent,
                "defective_percent": defective_percent,
                "confidence_score": bean_classification[0]['confidence'],
                "notes": f"Health Grade: {health_score['grade']}, Defects: {len(defect_detection)}"
            }
            
            history_result = supabase.table(HISTORY_TABLE).insert(history_data).execute()
            history_id = history_result.data[0]["history_id"]
            
            # Clean up temp file
            os.remove(temp_path)
            
            return JSONResponse(content={
                "success": True,
                "history_id": history_id,
                "data": {
                    "prediction": {
                        "predicted_class": bean_type_name,
                        "confidence": bean_classification[0]['confidence'],
                        "all_probabilities": bean_classification[0]['probabilities']
                    },
                    "defect_detection": {
                        "detections": defect_detection,
                        "summary": {
                            "total_defects": len(defect_detection),
                            "defect_types": {},
                            "defect_percentage": 0.0,
                            "quality_score": health_score.get('percentage', 0.0),
                            "quality_grade": health_score.get('grade', 'F')
                        }
                    },
                    # bean_count removed
                    "health_score": health_score,
                    "shelf_life": {
                        "predicted_days": shelf_life_data["predicted_days"],
                        "confidence": shelf_life_data["confidence_score"],
                        "category": shelf_life_prediction['category'],
                        "defect_score": shelf_life_prediction['defect_score'],
                        "defect_counts": shelf_life_prediction['defect_counts'],
                        "base_shelf_life": shelf_life_prediction['base_shelf_life']
                    }
                },
                "message": "Bean analysis completed successfully"
            })
            
        except Exception as e:
            import traceback
            print(f"❌ Error in analysis: {e}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
            
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else "Unknown error"
        print(f"❌ Error in scan_bean_image: {error_msg}")
        print(f"❌ Error type: {type(e).__name__}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Scan failed: {error_msg}")

@router.get("/scan/{history_id}")
async def get_scan_result(history_id: int):
    """
    Get detailed scan results by history ID
    """
    try:
        # Get history record with all related data
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
            "scan_id": history_id,
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
        history_result = supabase.table(HISTORY_TABLE).select("history_id, created_at, confidence_score").eq("history_id", history_id).execute()
        
        if not history_result.data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        history = history_result.data[0]
        
        return JSONResponse(content={
            "history_id": history_id,
            "status": "completed",
            "completed_at": history["created_at"],
            "confidence": history["confidence_score"]
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

@router.post("/scan/advanced")
async def advanced_bean_analysis(
    image: UploadFile = File(...),
    user_id: Optional[int] = None,
    include_defect_analysis: bool = True,
    include_shelf_life: bool = True
):
    """
    Advanced bean analysis with optional components
    """
    try:
        # Validate image file - check both content type and file extension
        is_valid_image = False
        
        # Check content type
        if image.content_type and image.content_type.startswith('image/'):
            is_valid_image = True
        # Check file extension as fallback (common with Flutter uploads)
        elif image.filename:
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
            file_ext = image.filename.lower().split('.')[-1] if '.' in image.filename else ''
            if f'.{file_ext}' in valid_extensions:
                is_valid_image = True
                print(f"✅ Valid image by extension: {image.filename}")
        
        if not is_valid_image:
            print(f"❌ Invalid content type: {image.content_type}")
            print(f"❌ Filename: {image.filename}")
            raise HTTPException(status_code=400, detail=f"File must be an image. Received content type: {image.content_type}, filename: {image.filename}")
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Load and preprocess image
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image_pil).unsqueeze(0)
        
        # Run analysis based on options
        results = {}
        
        # Always run bean classification
        bean_results = models['cnn'].predict(image_tensor)
        results['bean_classification'] = bean_results
        
        # Optional defect detection
        if include_defect_analysis:
            defect_results = models['defect_detector'].detect_defects(image_tensor)
            results['defect_detection'] = defect_results
        
        # Optional shelf life prediction
        if include_shelf_life:
            # Use rule-based shelf life prediction
            shelf_life_model = models['shelf_life_model']
            
            # Prepare defect sequence from detected defects
            defect_sequence = []
            if 'defect_detection' in results and results['defect_detection']:
                for defect in results['defect_detection']:
                    defect_sequence.append({
                        'type': defect.get('type', 'unknown'),
                        'confidence': defect.get('confidence', 0.5),
                        'count': 1
                    })
            
            # Get bean type for prediction
            bean_type = results.get('bean_classification', [{}])[0].get('class', 'Arabica') if results.get('bean_classification') else 'Arabica'
            
            shelf_life_results = shelf_life_model.predict_shelf_life(defect_sequence, bean_type)
            results['shelf_life_prediction'] = shelf_life_results
        
        return JSONResponse(content={
            "success": True,
            "analysis": results,
            "message": "Advanced analysis completed"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
