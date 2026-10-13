"""
conftest.py — ensures the model registry is loaded before any test runs.
The FastAPI TestClient triggers the lifespan context, but the registry
must be loaded with the correct absolute path from the test environment.
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure registry is loaded with the real artifacts before the app is imported
from app.services.model_registry import registry
from app.core.config import settings


def pytest_configure(config):
    """Load model artifacts once before the entire test session."""
    if not registry.is_ready:
        try:
            registry.load(settings.ARTIFACTS_DIR)
            print(f"\n[conftest] Loaded artifacts from: {settings.ARTIFACTS_DIR}")
        except Exception as e:
            print(f"\n[conftest] WARNING — Could not load artifacts: {e}")
            print(f"           ARTIFACTS_DIR = {settings.ARTIFACTS_DIR}")
