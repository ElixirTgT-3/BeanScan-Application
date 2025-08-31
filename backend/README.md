# BeanScan Backend API

A Python-based backend for the BeanScan application featuring PyTorch deep learning models and Supabase database integration.

## 🚀 Features

- **FastAPI** REST API with automatic documentation
- **PyTorch** deep learning models for bean classification
- **Supabase** database integration with real-time capabilities
- **Image processing** and analysis
- **User authentication** and authorization
- **Scan history** management
- **Statistics** and analytics

## 🏗️ Architecture

```
backend/
├── api/                 # API routes and endpoints
├── database/           # Database client and schema
├── ml/                # Machine learning models
├── models/            # Trained model files
├── utils/             # Utility functions
├── main.py            # FastAPI application entry point
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 📋 Prerequisites

- Python 3.8+
- PyTorch 2.1.0+
- Supabase account and project
- Git

## 🛠️ Installation

### 1. Clone and Setup

```bash
cd backend
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy the environment template and configure your Supabase credentials:

```bash
copy env_example.txt .env
```

Edit `.env` with your actual values:
```env
# Supabase Configuration
SUPABASE_URL=https://qnunuwncpizyaettalol.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudW51d25jcGl6eWFldHRhbG9sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTY2MzMyNDgsImV4cCI6MjA3MjIwOTI0OH0.5rjAO2DGjcR0neYjvh0JXim9FOLWdOosyk0rYH-KWqI
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudW51d25jcGl6eWFldHRhbG9sIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NjYzMzI0OCwiZXhwIjoyMDcyMjA5MjQ4fQ.cN5EdvVbpxhAD8nTI2QuXKMk4wTHqn2q1RW-ssLlrJ8

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Model Configuration
MODEL_PATH=./models/bean_classifier.pth
DEVICE=cpu
```

## 🗄️ Database Setup

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Create a new project
3. Note your project URL and API keys

### 2. Run Database Schema

1. Go to your Supabase project dashboard
2. Navigate to SQL Editor
3. Copy and paste the contents of `database/schema.sql`
4. Execute the script

### 3. Configure RLS Policies

The schema includes Row Level Security (RLS) policies for data protection. Ensure your Supabase authentication is properly configured.

## 🚀 Running the Application

### Development Mode

```bash
python main.py
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### With Auto-reload

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Documentation

Once running, visit:
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔌 API Endpoints

### Scan Endpoints
- `POST /api/v1/scan` - Upload and classify bean image
- `GET /api/v1/scan/{history_id}` - Get scan result
- `GET /api/v1/scan/status/{history_id}` - Get scan status
- `GET /api/v1/bean-types` - Get available bean types

### History Endpoints
- `GET /api/v1/history` - Get scan history
- `GET /api/v1/history/{history_id}` - Get scan details
- `DELETE /api/v1/history/{history_id}` - Delete scan
- `GET /api/v1/history/stats` - Get statistics
- `GET /api/v1/history/export` - Export data
- `GET /api/v1/history/summary` - Get scan summary
- `GET /api/v1/users` - Get user list (admin)

## 🤖 Machine Learning

### Model Architecture

The bean classifier uses a **ResNet18** backbone with custom classification layers:

- **Input**: 224x224 RGB images
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Classification**: Custom fully connected layers
- **Output**: 5 bean types (Arabica, Robusta, Liberica, Excelsa, Other)

### Training

To train your own model:

1. Prepare your dataset
2. Use the training script (coming soon)
3. Save the model to `models/` directory
4. Update the model path in `.env`

### Inference

The model automatically loads when the API starts and can classify images in real-time.

## 🔐 Authentication

The API supports Supabase authentication:

1. **Public endpoints**: Health check, basic info
2. **Protected endpoints**: User-specific data, scan history
3. **Admin endpoints**: Model management, analytics

## 📊 Database Schema

### Tables

- **User**: User profiles with roles (admin, user, analyst)
- **Bean_Type**: Coffee bean classifications
- **Bean_Image**: Uploaded bean images with metadata
- **Defect**: Detected defects with severity levels
- **Shelf_Life**: Shelf life predictions based on images/defects
- **History**: Central record linking all scan data

### Relationships

- Users can have multiple images and scans
- Each image can have multiple defects
- Images are linked to bean types and shelf life predictions
- History table centralizes all scan information

## 🧪 Testing

### Run Tests

```bash
pytest
```

### Test Coverage

```bash
pytest --cov=.
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Ensure all required environment variables are set in production:
- Database credentials
- API configuration
- Model paths
- Security settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the API documentation
2. Review the logs
3. Check Supabase dashboard
4. Create an issue in the repository

## 🔮 Future Enhancements

- [ ] Model training pipeline
- [ ] Real-time notifications
- [ ] Advanced analytics
- [ ] Multi-language support
- [ ] Mobile app integration
- [ ] Cloud deployment scripts
