from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from predict_bean import predict_bean_type

# Create FastAPI app
app = FastAPI(
    title="BeanScan Test API",
    description="Test API for BeanScan with trained CNN model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "BeanScan Test API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BeanScan Test API"}

@app.post("/api/predict")
async def predict_bean(
    image: UploadFile = File(...)
):
    """
    Predict bean type from uploaded image
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            # Read and write image content
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Make prediction
            result = predict_bean_type(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return {
                "success": True,
                "prediction": result['predicted_class'],
                "confidence": result['confidence'],
                "all_probabilities": result['all_probabilities']
            }
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting BeanScan Test API...")
    print("📡 API will be available at: http://localhost:8000")
    print("🔍 Test endpoint: POST http://localhost:8000/api/predict")
    print("📊 Health check: GET http://localhost:8000/health")
    
    uvicorn.run(
        "test_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
