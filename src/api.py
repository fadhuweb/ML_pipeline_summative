import os
import shutil
import time
import zipfile
from typing import Optional, List
import warnings

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# Project modules
from src.prediction import predict_image_from_path
from src.model import load_trained_model, retrain_model
from src.preprocessing import create_data_generators

# Directories
UPLOAD_DIR = os.environ.get(
    "UPLOAD_DIR",
    r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\new_uploads"
)
MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\models"
)
TEST_DIR = os.environ.get(
    "TEST_DIR",
    r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\test"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="Chest X-ray Pneumonia API")

# Global variable to store selected model
SELECTED_MODEL = None


# -----------------------
# Utility helpers
# -----------------------
def _save_upload_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    upload_file.file.close()


def _extract_zip_to_folder(zip_path: str, target_folder: str):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_folder)
    os.remove(zip_path)


def _clear_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _get_latest_model_file():
    """Get the most recent model file from MODELS_DIR."""
    if not os.path.exists(MODELS_DIR):
        raise HTTPException(
            status_code=404, 
            detail=f"Models directory does not exist: {MODELS_DIR}"
        )
    
    all_files = os.listdir(MODELS_DIR)
    model_files = [
        f for f in all_files 
        if os.path.isfile(os.path.join(MODELS_DIR, f)) 
        and (f.endswith(".keras") or f.endswith(".h5"))
    ]
    
    if not model_files:
        raise HTTPException(
            status_code=404, 
            detail=f"No .keras or .h5 model files found in {MODELS_DIR}"
        )
    
    model_files_with_time = [
        (f, os.path.getmtime(os.path.join(MODELS_DIR, f))) 
        for f in model_files
    ]
    model_files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    latest_model = model_files_with_time[0][0]
    full_path = os.path.join(MODELS_DIR, latest_model)
    
    return full_path


def _get_model_to_use():
    """Get the model to use (selected or latest)"""
    global SELECTED_MODEL
    
    if SELECTED_MODEL:
        model_path = os.path.join(MODELS_DIR, SELECTED_MODEL)
        if os.path.exists(model_path):
            return model_path
        else:
            # Selected model doesn't exist, fall back to latest
            SELECTED_MODEL = None
    
    return _get_latest_model_file()


# -----------------------
# Health check
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


# -----------------------
# List available models
# -----------------------
@app.get("/models")
def list_models():
    """Get list of all available models"""
    global SELECTED_MODEL
    
    try:
        all_files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
        model_files = [
            f for f in all_files 
            if os.path.isfile(os.path.join(MODELS_DIR, f)) 
            and (f.endswith(".keras") or f.endswith(".h5"))
        ]
        
        latest_model_path = _get_latest_model_file() if model_files else None
        
        return {
            "models_directory": MODELS_DIR,
            "all_files": all_files,
            "valid_model_files": model_files,
            "selected_latest_model": latest_model_path,
            "currently_selected": SELECTED_MODEL
        }
    except Exception as e:
        return {
            "error": str(e),
            "models_directory": MODELS_DIR
        }


# -----------------------
# Select model
# -----------------------
@app.post("/select-model")
def select_model(model_name: str):
    """Select a specific model to use for all operations"""
    global SELECTED_MODEL
    
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    if not (model_name.endswith(".keras") or model_name.endswith(".h5")):
        raise HTTPException(status_code=400, detail="Invalid model format")
    
    SELECTED_MODEL = model_name
    
    return {
        "status": "success",
        "selected_model": SELECTED_MODEL,
        "message": f"Now using '{model_name}' for all operations"
    }


# -----------------------
# Download model
# -----------------------
@app.get("/download-model/{model_name}")
def download_model(model_name: str):
    """Download a specific model file"""
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        model_path,
        filename=model_name,
        media_type="application/octet-stream"
    )


# -----------------------
# Predict single image
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a single image file and get prediction.
    Uses selected model or latest model.
    """
    model_file = _get_model_to_use()
    print(f"Using model: {model_file}")

    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_file}")
    
    if not (model_file.endswith(".keras") or model_file.endswith(".h5")):
        raise HTTPException(status_code=400, detail=f"Invalid model format")

    # Save uploaded file temporarily
    tmp_folder = os.path.join("tmp", str(int(time.time() * 1000)))
    os.makedirs(tmp_folder, exist_ok=True)
    tmp_path = os.path.join(tmp_folder, file.filename)
    
    # CRITICAL FIX: Reset file pointer before reading
    await file.seek(0)
    
    # Save the file properly
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Load model and predict
    try:
        model = load_trained_model(model_file)
        result = predict_image_from_path(model, tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        try:
            shutil.rmtree(tmp_folder)
        except Exception:
            pass

    return JSONResponse(content=result)


# -----------------------
# Upload bulk files for retraining
# -----------------------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Accept multiple files or zipped folders for retraining"""
    _clear_folder(UPLOAD_DIR)
    saved = []
    for upload_file in files:
        dest = os.path.join(UPLOAD_DIR, upload_file.filename)
        _save_upload_file(upload_file, dest)
        saved.append(upload_file.filename)
        if upload_file.filename.lower().endswith(".zip"):
            _extract_zip_to_folder(dest, UPLOAD_DIR)
    return {"status": "ok", "saved": saved, "upload_folder": UPLOAD_DIR}


# -----------------------
# Trigger retraining
# -----------------------
@app.post("/retrain")
def trigger_retrain(epochs: int = 5, batch_size: int = 32, fine_tune: bool = True):
    """
    Trigger retraining using files in UPLOAD_DIR.
    Uses selected model as base or latest model.
    """
    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="No uploaded data found. Please POST to /upload first.")

    model_path, history = retrain_model(
        new_data_folder=UPLOAD_DIR,
        original_train_folder=r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\train",
        batch_size=batch_size,
        epochs=epochs,
        output_directory=MODELS_DIR,
        fine_tune=fine_tune
    )

    summary = {k: v[-1] if isinstance(v, list) else v for k, v in history.history.items()}
    return {"status": "completed", "model_path": model_path, "summary": summary}


# -----------------------
# Model metrics endpoint
# -----------------------
@app.get("/metrics")
def metrics(batch_size: int = 64):
    """
    Evaluate model on test set.
    Uses selected model or latest model.
    Returns: loss, accuracy, precision, recall, and AUC only.
    """
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    
    model_path = _get_model_to_use()
    print(f"Evaluating model: {model_path}")

    # Load the model
    try:
        model = load_trained_model(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Create test data generator
    try:
        _, _, test_gen = create_data_generators(
            train_dir=r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\train",
            test_dir=TEST_DIR,
            img_size=(224, 224),
            batch_size=batch_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create test data generator: {str(e)}")

    # Evaluate model
    try:
        print("Calculating metrics...")
        results = model.evaluate(test_gen, verbose=1)
        
        if isinstance(results, (list, tuple)):
            loss = round(float(results[0]), 4)
            accuracy = round(float(results[1]), 4) if len(results) > 1 else None
        else:
            loss = round(float(results), 4)
            accuracy = None
        
        test_gen.reset()
        
        y_pred_probs = model.predict(test_gen, verbose=1)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        
        precision = round(float(precision_score(y_true, y_pred)), 4)
        recall = round(float(recall_score(y_true, y_pred)), 4)
        auc = round(float(roc_auc_score(y_true, y_pred_probs)), 4)
        
        if accuracy is None:
            accuracy = round(float(np.mean(y_pred == y_true)), 4)
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)