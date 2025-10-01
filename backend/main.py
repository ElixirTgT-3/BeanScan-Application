from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from dotenv import load_dotenv

# Import our modules
from api import scan_routes_custom, history_routes
from database.supabase_client import get_supabase_client
from ml.bean_classifier import BeanClassifier

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="BeanScan API",
    description="Backend API for BeanScan application with PyTorch ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scan_routes_custom.router, prefix="/api/v1", tags=["scan"])
app.include_router(history_routes.router, prefix="/api/v1", tags=["history"])

# Serve images from local static directory
static_images_dir = os.path.join(os.path.dirname(__file__), "static", "images")
os.makedirs(static_images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=static_images_dir), name="images")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "BeanScan API is running!"}

# Respond 200 to HEAD probes too
@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BeanScan API"}

@app.head("/health")
async def health_head():
    return Response(status_code=200)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )
