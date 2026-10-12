"""
Utility functions for logging, file handling, and export for the Disease Prediction Research Suite.
"""

import os
import sys
import json
import logging
from pathlib import Path

def setup_logger(name: str = "research_logger", log_file: str = None) -> logging.Logger:
    """Configures a standardized logger for console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if re-initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger

def ensure_dir(dir_path: str | Path) -> Path:
    """Ensures that a directory exists, creating parents if needed."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(data: dict, filepath: str | Path, indent: int = 2) -> None:
    """Saves a dictionary as a formatted JSON file."""
    ensure_dir(Path(filepath).parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
