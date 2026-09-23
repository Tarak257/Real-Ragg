import os
import sys
import uvicorn

# Reconfigure stdout to UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    print("🚀 Starting Dedicated FastAPI Backend Server on http://127.0.0.1:8001 ...")
    print("📖 OpenAPI Swagger Docs available at: http://127.0.0.1:8001/docs")
    print("📘 ReDoc API Docs available at: http://127.0.0.1:8001/redoc")
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8001,
        reload=False
    )
