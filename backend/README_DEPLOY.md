# BeanScan Backend Deployment (Render)

This guide shows how to deploy the FastAPI backend online so the app works without starting a local server.

## 1) Requirements
- Render account
- GitHub repo containing this project

## 2) What’s included
- `backend/Dockerfile` for containerizing FastAPI
- `backend/render.yaml` for Render one‑click deployment
- `backend/.dockerignore` to keep images small

## 3) Steps to deploy on Render
1. Push/commit the latest code to your GitHub.
2. In Render, create a new Web Service from your GitHub repo.
3. When asked for the root directory, set it to `backend/` (Render reads `render.yaml` there).
4. Choose plan (Starter is fine for testing). Keep port 8000.
5. Add environment variables in Render dashboard:
   - `API_HOST=0.0.0.0`
   - `API_PORT=8000`
   - `DEBUG=False`
   - `SUPABASE_URL` (your Supabase project URL)
   - `SUPABASE_ANON_KEY` or `SUPABASE_SERVICE_ROLE_KEY` (as needed)
6. Deploy. After it’s live, you’ll get a public URL like `https://beanscan-api.onrender.com`.

Health check: `GET https://<your-service>/health` should return 200.

## 4) Point the Flutter app to the hosted API
The Flutter app reads base URL from dart‑define values in `lib/utils/api_service.dart`:
- `API_BASE_URL`
- `ANDROID_API_BASE_URL`

Example commands:

```
flutter run --dart-define=API_BASE_URL=https://<your-service> \
           --dart-define=ANDROID_API_BASE_URL=https://<your-service>
```

For release builds:
```
flutter build apk --dart-define=API_BASE_URL=https://<your-service> \
                  --dart-define=ANDROID_API_BASE_URL=https://<your-service>
```

Note: The app’s health check probes `/health` and then uses `/api/v1/...` endpoints.

## 5) Notes on models and performance
- Model files under `backend/models/` are included in the image.
- This Dockerfile installs CPU Torch. For GPU hosting, adjust the base image and Torch versions accordingly.

## 6) CORS
CORS is currently permissive (`*`) in `backend/main.py`. For production, restrict `allow_origins` to your app origins.
