import joblib
import logging
from pathlib import Path
from prop_analyzer import config as cfg

def save_artifacts(prop_cat, artifacts):
    """Saves the XGBoost FGA Regressor and Classifier artifacts."""
    model_path = cfg.MODEL_DIR / f"model_{prop_cat}.pkl"
    try:
        joblib.dump(artifacts, model_path)
        logging.info(f"Saved artifacts for {prop_cat} to {model_path.name}")
    except Exception as e:
        logging.error(f"Failed to save model for {prop_cat}: {e}")

def load_artifacts(prop_cat):
    """Loads the artifacts and validates the FGA Engine keys."""
    model_path = cfg.MODEL_DIR / f"model_{prop_cat}.pkl"
    if not model_path.exists():
        logging.warning(f"Model file not found for {prop_cat} at {model_path}")
        return None
        
    try:
        artifacts = joblib.load(model_path)
        
        # Check for the NEW keys we created in training.py
        required = ['scaler', 'features', 'regressor', 'classifier']
        missing = [k for k in required if k not in artifacts]
        
        if missing:
            logging.warning(f"Model file for {prop_cat} missing expected keys: {missing}")
            
        return artifacts
    except Exception as e:
        logging.error(f"Failed to load model for {prop_cat}: {e}")
        return None