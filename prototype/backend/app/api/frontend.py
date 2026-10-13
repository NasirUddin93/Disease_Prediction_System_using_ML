"""
Static file serving — mounts the frontend/index.html via FastAPI.
Add this to main.py to serve the frontend directly from the backend.
"""
from pathlib import Path
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"


def mount_frontend(app):
    """Mount the frontend directory and add a root route."""

    if FRONTEND_DIR.exists():
        # Serve static assets
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(str(FRONTEND_DIR / "index.html"))
    else:
        @app.get("/", include_in_schema=False)
        async def no_frontend():
            return {"message": "CDSS API is running. Frontend not found.", "docs": "/docs"}
