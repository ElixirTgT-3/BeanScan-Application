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
    location: Optional[str] = None
):
    """
    Scan and analyze a bean image using custom deep learning models
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
            # Load and preprocess image for models
            image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension
            
            # Run complete analysis using ensemble model
            print("🔍 Running deep learning analysis...")
            analysis_results = ensemble.forward(image_tensor)
            
            # Extract results
            bean_classification = analysis_results['bean_classification']
            defect_detection = analysis_results['defect_detection']
            health_score = analysis_results['health_score']
            
            print(f"📊 Analysis complete: Bean={bean_classification[0]['class']}, Health={health_score['grade']}")
            
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
            
            # Create shelf life prediction using LSTM
            shelf_life_data = {
                "image_id": image_id,
                "bean_type_id": bean_type_id,
                "defect_id": defect_id,
                "predicted_days": 30,  # Default, can be enhanced with LSTM
                "confidence_score": bean_classification[0]['confidence'],
                "storage_conditions": {"temperature": "room_temp", "humidity": "low"}
            }
            
            shelf_life_result = supabase.table(SHELF_LIFE_TABLE).insert(shelf_life_data).execute()
            shelf_life_id = shelf_life_result.data[0]["shelf_life_id"]
            
            # Calculate percentages for history
            healthy_percent = health_score['percentage']
            defective_percent = 100 - healthy_percent
            
            # Create history record
            history_data = {
                "user_id": user_id,
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
                "analysis": {
                    "bean_type": {
                        "name": bean_type_name,
                        "confidence": bean_classification[0]['confidence'],
                        "probabilities": bean_classification[0]['probabilities']
                    },
                    "defects": defect_detection,
                    "health_score": health_score,
                    "shelf_life": {
                        "predicted_days": shelf_life_data["predicted_days"],
                        "confidence": shelf_life_data["confidence_score"]
                    }
                },
                "message": "Bean analysis completed successfully"
            })
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
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
            # Create dummy sequence for demonstration
            dummy_sequence = torch.randn(1, 10, 64)  # 10 time steps, 64 features
            shelf_life_results = models['lstm'].predict_shelf_life(dummy_sequence)
            results['shelf_life_prediction'] = shelf_life_results
        
        return JSONResponse(content={
            "success": True,
            "analysis": results,
            "message": "Advanced analysis completed"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
