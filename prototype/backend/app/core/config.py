from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Clinical Decision Support System (CDSS) Prototype"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Artifact paths
    # ROOT_DIR = prototype/backend/app/core/../../.. = prototype/
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
    MODEL_PATH: Path = ARTIFACTS_DIR / "model_pipeline.joblib"
    CALIBRATOR_PATH: Path = ARTIFACTS_DIR / "calibrator.joblib"
    EXPLAINER_PATH: Path = ARTIFACTS_DIR / "explainer.joblib"
    METADATA_PATH: Path = ARTIFACTS_DIR / "model_metadata.json"
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8181", "http://127.0.0.1:8181", "*"]

settings = Settings()
