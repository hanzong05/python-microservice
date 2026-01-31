"""
Geotechnical ML Training Pipeline API - PRODUCTION VERSION
FastAPI service for Tarlac Liquefaction Prediction System

UPDATED: Models loaded directly from Supabase Storage into memory (no local files)
UPGRADED: Integration with upgraded ANN model architecture (256-128-64)
"""

# ============================================================================
# IMPORTS
# ============================================================================
import json
import io
from datetime import datetime, timedelta
import asyncio
import sys
import uvicorn
from typing import Optional, Dict, List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, BackgroundTasks
import os
from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Geo-ML Training API",
    description="ML Training Pipeline for Liquefaction Prediction System",
    version="3.0.0-UPGRADED"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline status tracker
pipeline_status = {
    "is_running": False,
    "current_step": None,
    "progress": 0,
    "start_time": None,
    "end_time": None,
    "error": None,
    "steps_completed": [],
    "logs": []
}

# Global cache for models loaded directly into memory
_model_cache = {
    'scaler': None,
    'liquefaction': None,
    'settlement': None,
    'bearing_capacity': None,
    'metadata': None,
    'last_loaded': None
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PipelineConfig(BaseModel):
    """Configuration for training pipeline"""
    scripts_directory: Optional[str] = None
    run_data_cleaning: Optional[bool] = True
    run_data_preparation: Optional[bool] = True
    run_etl: Optional[bool] = True
    run_feature_engineering: Optional[bool] = True
    run_model_training: Optional[bool] = True


class StepStatus(BaseModel):
    """Status of individual pipeline step"""
    step_name: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    latitude: float
    longitude: float
    features: Optional[Dict[str, float]] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_message(step: str, message: str):
    """Add log message to pipeline status"""
    global pipeline_status
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "message": message
    }
    pipeline_status["logs"].append(log_entry)
    print(f"[{step}] {message}")


def find_script(script_name: str, custom_dir: Optional[str] = None) -> Optional[Path]:
    """Search for script in multiple locations"""
    api_file_location = Path(__file__).resolve()
    search_paths = []

    if custom_dir:
        search_paths.append(Path(custom_dir))

    search_paths.extend([
        api_file_location.parent.parent,
        api_file_location.parent,
        api_file_location.parent.parent.parent,
        api_file_location.parent.parent / "scripts",
        api_file_location.parent.parent / "ml_scripts",
        api_file_location.parent.parent / "training_scripts",
        api_file_location.parent / "scripts",
    ])

    log_message("SCRIPT_SEARCH", f"Searching for: {script_name}")
    log_message("SCRIPT_SEARCH", f"API location: {api_file_location}")

    for search_path in search_paths:
        script_path = search_path / script_name
        log_message("SCRIPT_SEARCH", f"Checking: {script_path}")
        if script_path.exists():
            log_message("SCRIPT_SEARCH",
                        f"[FOUND] Script found at: {script_path}")
            return script_path

    log_message("SCRIPT_SEARCH",
                f"[NOT FOUND] Script not found in any location")
    return None


async def run_script(script_name: str, step_name: str, progress: int,
                     custom_script_dir: Optional[str] = None) -> tuple:
    """Run a Python training script asynchronously"""
    global pipeline_status

    pipeline_status["current_step"] = step_name
    pipeline_status["progress"] = progress
    log_message(step_name, f"Starting {step_name}...")

    script_path = find_script(script_name, custom_script_dir)

    if script_path is None:
        error_msg = f"Script not found: {script_name}"
        log_message(step_name, f"[ERROR] {error_msg}")
        log_message(step_name, "Use GET /diagnostics to see search locations")
        return False, error_msg

    try:
        log_message(step_name, f"Running script: {script_path}")
        log_message(
            step_name, f"Working directory: {script_path.parent.parent}")

        script_env = os.environ.copy()
        script_env['PYTHONIOENCODING'] = 'utf-8'

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent.parent),
            env=script_env
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=1800  # 30 minutes
        )

        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""

        log_message(step_name, f"Return code: {process.returncode}")
        if stdout_str:
            log_message(step_name, f"STDOUT:\n{stdout_str[:1000]}")
        if stderr_str:
            log_message(step_name, f"STDERR:\n{stderr_str[:1000]}")

        if process.returncode == 0:
            log_message(
                step_name, f"[SUCCESS] {step_name} completed successfully")
            pipeline_status["steps_completed"].append(step_name)
            return True, stdout_str
        else:
            error_msg = stderr_str or stdout_str or f"Process exited with code {process.returncode}"
            log_message(step_name, f"[FAILED] {error_msg[:500]}")
            pipeline_status["error"] = error_msg
            return False, error_msg

    except asyncio.TimeoutError:
        error_msg = f"{step_name} timed out after 30 minutes"
        log_message(step_name, f"[TIMEOUT] {error_msg}")
        pipeline_status["error"] = error_msg
        return False, error_msg

    except Exception as e:
        error_msg = str(e)
        log_message(step_name, f"[ERROR] {error_msg}")
        pipeline_status["error"] = error_msg
        return False, error_msg


# ============================================================================
# MODEL LOADING FROM SUPABASE STORAGE (DIRECT IN-MEMORY)
# ============================================================================

def load_models_from_supabase_direct():
    """
    Load UPGRADED trained models DIRECTLY from Supabase Storage into memory
    NO local file system required - uses BytesIO for in-memory loading

    UPGRADED: Now loads models trained with 256-128-64 architecture

    Returns:
        dict: Loaded model objects (NOT file paths)
    """
    global _model_cache

    # Return cached models if available (cache for 1 hour)
    if _model_cache['scaler'] is not None:
        if _model_cache['last_loaded'] and \
           datetime.now() - _model_cache['last_loaded'] < timedelta(hours=1):
            print(
                f"✓ Using cached UPGRADED models (loaded at {_model_cache['last_loaded'].strftime('%H:%M:%S')})")
            return _model_cache

    print("📥 Loading UPGRADED models directly from Supabase Storage into memory...")
    from supabase_client import get_supabase_client

    client = get_supabase_client()
    if not client:
        raise Exception("Failed to connect to Supabase")

    model_files = {
        'scaler': 'ml_models/scaler.pkl',
        'liquefaction': 'ml_models/ann_liquefaction.pkl',
        'settlement': 'ml_models/ann_settlement.pkl',
        'bearing_capacity': 'ml_models/ann_bearing_capacity.pkl',
        'metadata': 'ml_models/ann_metadata.json'
    }

    loaded_models = {}

    for model_name, storage_path in model_files.items():
        try:
            print(f"  Loading {model_name} from {storage_path}...")

            # Download file bytes from Supabase Storage
            file_data = client.storage.from_(
                'geotechnical-data').download(storage_path)

            if model_name == 'metadata':
                # JSON metadata - decode and parse
                metadata = json.loads(file_data.decode('utf-8'))
                loaded_models[model_name] = metadata

                # Display model version info
                version = metadata.get('version', 'unknown')
                architecture = metadata.get('model_architecture', {})
                print(f"  ✓ Loaded {model_name} (version: {version})")
                if architecture:
                    hidden_layers = architecture.get('hidden_layers', [])
                    print(f"    Architecture: {hidden_layers}")
            else:
                # Pickle files - load directly from bytes using BytesIO
                loaded_models[model_name] = joblib.load(io.BytesIO(file_data))
                print(f"  ✓ Loaded {model_name} into memory")

        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}")
            raise Exception(f"Failed to load {model_name} from Supabase: {e}")

    # Update cache
    _model_cache = loaded_models
    _model_cache['last_loaded'] = datetime.now()

    # Display loaded model information
    metadata = loaded_models.get('metadata', {})
    num_features = metadata.get('num_features', 'unknown')
    enhancements = metadata.get('enhancements', [])

    print(f"✓ All UPGRADED models loaded successfully into memory!")
    print(f"  Features: {num_features}")
    if enhancements:
        print(f"  Enhancements: {len(enhancements)} improvements")
        for enhancement in enhancements[:3]:  # Show first 3
            print(f"    - {enhancement}")

    return _model_cache


def clear_model_cache():
    """Clear the model cache to force reload from Supabase Storage"""
    global _model_cache
    _model_cache = {
        'scaler': None,
        'liquefaction': None,
        'settlement': None,
        'bearing_capacity': None,
        'metadata': None,
        'last_loaded': None
    }
    print("✓ Model cache cleared - will reload UPGRADED models on next prediction")
    return {"status": "success", "message": "Model cache cleared"}


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

async def run_full_pipeline(config: PipelineConfig):
    """Run the complete ML training pipeline"""
    global pipeline_status

    pipeline_status = {
        "is_running": True,
        "current_step": None,
        "progress": 0,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "error": None,
        "steps_completed": [],
        "logs": []
    }

    log_message(
        "PIPELINE", "[START] Starting UPGRADED ML Training Pipeline...")

    try:
        script_dir = config.scripts_directory

        if config.run_data_cleaning:
            success, output = await run_script("01_data_cleaning.py", "Data Cleaning", 10, script_dir)
            if not success:
                raise Exception(f"Data Cleaning failed: {output}")

        if config.run_data_preparation:
            success, output = await run_script("01b_ml_data_preparation.py", "Data Preparation", 30, script_dir)
            if not success:
                raise Exception(f"Data Preparation failed: {output}")

        if config.run_etl:
            success, output = await run_script("02_etl_to_supabase.py", "ETL Pipeline", 50, script_dir)
            if not success:
                raise Exception(f"ETL Pipeline failed: {output}")
            log_message(
                "ETL", "[REMINDER] Run PostGIS location update in Supabase SQL Editor:")
            log_message(
                "ETL", "UPDATE boreholes SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)")

        if config.run_feature_engineering:
            success, output = await run_script("03_feature_engineering.py", "Feature Engineering", 70, script_dir)
            if not success:
                raise Exception(f"Feature Engineering failed: {output}")

        if config.run_model_training:
            success, output = await run_script("04_model_training.py", "UPGRADED ANN Training", 90, script_dir)
            if not success:
                raise Exception(f"UPGRADED ANN Training failed: {output}")

            # Clear model cache after training to load new models
            clear_model_cache()
            log_message(
                "PIPELINE", "[INFO] Model cache cleared - UPGRADED models will be loaded on next prediction")

        pipeline_status["progress"] = 100
        pipeline_status["end_time"] = datetime.now().isoformat()
        pipeline_status["is_running"] = False
        log_message(
            "PIPELINE", "[SUCCESS] UPGRADED Pipeline completed successfully!")

    except Exception as e:
        pipeline_status["error"] = str(e)
        pipeline_status["is_running"] = False
        pipeline_status["end_time"] = datetime.now().isoformat()
        log_message("PIPELINE", f"[FAILED] Pipeline failed: {str(e)}")


# ============================================================================
# API ENDPOINTS - GENERAL
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "[OK] Geo-ML Training API is running!",
        "status": "operational",
        "version": "3.0.0-UPGRADED",
        "model_info": {
            "architecture": "UPGRADED ANN (256-128-64 neurons)",
            "version": "upgraded",
            "enhancements": [
                "Spatial features from PostGIS",
                "Class weight balancing",
                "Cross-validation",
                "ROC-AUC metrics",
                "Feature importance analysis"
            ]
        },
        "features": {
            "direct_memory_loading": "Models loaded from Supabase Storage into memory",
            "no_local_files": "No file system dependencies",
            "model_caching": "1-hour in-memory cache for fast predictions",
            "upgraded_architecture": "Enhanced 256-128-64 neural network"
        },
        "endpoints": {
            "GET /diagnostics": "Show script locations and system info",
            "POST /pipeline/start": "Start the ML training pipeline",
            "GET /pipeline/status": "Check pipeline status",
            "GET /pipeline/logs": "Get pipeline logs",
            "POST /pipeline/stop": "Stop running pipeline",
            "POST /predict": "Predict liquefaction at coordinates",
            "GET /predict-by-location": "Predict by lat/lon",
            "GET /nearest-borehole": "Find nearest borehole data",
            "POST /models/clear-cache": "Clear model cache (after retraining)",
            "GET /models/cache-status": "Check model cache status",
            "GET /models/info": "Get UPGRADED model information"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": "3.0.0-UPGRADED"
    }


@app.get("/diagnostics")
async def get_diagnostics():
    """Get diagnostic information about script locations"""
    api_file = Path(__file__).resolve()

    def list_py_files(directory: Path) -> List[str]:
        if not directory.exists():
            return []
        return [f.name for f in directory.glob("*.py")]

    diagnostics = {
        "api_file_location": str(api_file),
        "current_directory": str(Path.cwd()),
        "parent_directory": str(api_file.parent.parent),
        "search_locations": {
            "api_directory": {
                "path": str(api_file.parent),
                "exists": api_file.parent.exists(),
                "python_files": list_py_files(api_file.parent)
            },
            "parent_directory": {
                "path": str(api_file.parent.parent),
                "exists": api_file.parent.parent.exists(),
                "python_files": list_py_files(api_file.parent.parent)
            },
            "two_levels_up": {
                "path": str(api_file.parent.parent.parent),
                "exists": api_file.parent.parent.parent.exists(),
                "python_files": list_py_files(api_file.parent.parent.parent)
            }
        },
        "required_scripts": [
            "01_data_cleaning.py",
            "01b_ml_data_preparation.py",
            "02_etl_to_supabase.py",
            "03_feature_engineering.py",
            "04_model_training.py"
        ],
        "found_scripts": {}
    }

    for script_name in diagnostics["required_scripts"]:
        found_path = find_script(script_name)
        diagnostics["found_scripts"][script_name] = {
            "found": found_path is not None,
            "location": str(found_path) if found_path else None
        }

    return diagnostics


# ============================================================================
# API ENDPOINTS - PIPELINE CONTROL
# ============================================================================

@app.post("/pipeline/start")
async def start_pipeline(
    background_tasks: BackgroundTasks,
    config: Optional[PipelineConfig] = None
):
    """Start the complete ML training pipeline"""
    global pipeline_status

    if pipeline_status["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running. Check /pipeline/status for progress."
        )

    if config is None:
        config = PipelineConfig()

    background_tasks.add_task(run_full_pipeline, config)

    return {
        "status": "started",
        "message": "UPGRADED ML training pipeline started in background",
        "estimated_duration": "15-40 minutes",
        "check_status": "/pipeline/status",
        "scripts_directory": config.scripts_directory or "auto-detect",
        "model_version": "upgraded"
    }


@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get current status of the training pipeline"""
    global pipeline_status

    return {
        "is_running": pipeline_status["is_running"],
        "current_step": pipeline_status["current_step"],
        "progress": pipeline_status["progress"],
        "start_time": pipeline_status["start_time"],
        "end_time": pipeline_status["end_time"],
        "steps_completed": pipeline_status["steps_completed"],
        "error": pipeline_status["error"],
        "total_logs": len(pipeline_status["logs"])
    }


@app.get("/pipeline/logs")
async def get_pipeline_logs(limit: Optional[int] = 50):
    """Get recent pipeline logs"""
    global pipeline_status
    logs = pipeline_status["logs"][-limit:]
    return {
        "logs": logs,
        "total_logs": len(pipeline_status["logs"]),
        "showing": len(logs)
    }


@app.post("/pipeline/stop")
async def stop_pipeline():
    """Stop the running pipeline"""
    global pipeline_status

    if not pipeline_status["is_running"]:
        raise HTTPException(
            status_code=400,
            detail="No pipeline is currently running"
        )

    pipeline_status["is_running"] = False
    pipeline_status["error"] = "Pipeline stopped by user"
    pipeline_status["end_time"] = datetime.now().isoformat()
    log_message("PIPELINE", "[STOP] Pipeline stopped by user")

    return {
        "status": "stopped",
        "message": "Pipeline stop requested"
    }


# ============================================================================
# API ENDPOINTS - INDIVIDUAL TRAINING STEPS
# ============================================================================

@app.post("/train/step1-data-cleaning")
async def run_data_cleaning(background_tasks: BackgroundTasks, scripts_directory: Optional[str] = None):
    """Run only Step 1: Data Cleaning"""
    if pipeline_status["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    async def run_step():
        pipeline_status["is_running"] = True
        await run_script("01_data_cleaning.py", "Data Cleaning", 100, scripts_directory)
        pipeline_status["is_running"] = False

    background_tasks.add_task(run_step)
    return {"status": "started", "step": "Data Cleaning"}


@app.post("/train/step2-data-prep")
async def run_data_preparation(background_tasks: BackgroundTasks, scripts_directory: Optional[str] = None):
    """Run only Step 2: Data Preparation"""
    if pipeline_status["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    async def run_step():
        pipeline_status["is_running"] = True
        await run_script("01b_ml_data_preparation.py", "Data Preparation", 100, scripts_directory)
        pipeline_status["is_running"] = False

    background_tasks.add_task(run_step)
    return {"status": "started", "step": "Data Preparation"}


@app.post("/train/step3-etl")
async def run_etl(background_tasks: BackgroundTasks, scripts_directory: Optional[str] = None):
    """Run only Step 3: ETL Pipeline"""
    if pipeline_status["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    async def run_step():
        pipeline_status["is_running"] = True
        await run_script("02_etl_to_supabase.py", "ETL Pipeline", 100, scripts_directory)
        pipeline_status["is_running"] = False

    background_tasks.add_task(run_step)
    return {"status": "started", "step": "ETL Pipeline"}


@app.post("/train/step4-feature-eng")
async def run_feature_engineering(background_tasks: BackgroundTasks, scripts_directory: Optional[str] = None):
    """Run only Step 4: Feature Engineering"""
    if pipeline_status["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    async def run_step():
        pipeline_status["is_running"] = True
        await run_script("03_feature_engineering.py", "Feature Engineering", 100, scripts_directory)
        pipeline_status["is_running"] = False

    background_tasks.add_task(run_step)
    return {"status": "started", "step": "Feature Engineering"}


@app.post("/train/step5-model-training")
async def run_model_training(background_tasks: BackgroundTasks, scripts_directory: Optional[str] = None):
    """Run only Step 5: UPGRADED Model Training"""
    if pipeline_status["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    async def run_step():
        pipeline_status["is_running"] = True
        await run_script("04_model_training.py", "UPGRADED Model Training", 100, scripts_directory)
        pipeline_status["is_running"] = False
        clear_model_cache()  # Clear cache after training

    background_tasks.add_task(run_step)
    return {"status": "started", "step": "UPGRADED Model Training"}


# ============================================================================
# API ENDPOINTS - MODEL MANAGEMENT
# ============================================================================

@app.post("/models/clear-cache")
async def clear_models_cache():
    """
    Clear the model cache to force reload from Supabase Storage
    Use this after retraining models
    """
    return clear_model_cache()


@app.get("/models/cache-status")
async def get_cache_status():
    """Check if models are currently cached in memory"""
    global _model_cache

    is_cached = _model_cache['scaler'] is not None
    last_loaded = _model_cache['last_loaded'].isoformat(
    ) if _model_cache['last_loaded'] else None

    return {
        "cached": is_cached,
        "last_loaded": last_loaded,
        "models_in_cache": [k for k in _model_cache.keys() if k != 'last_loaded' and _model_cache[k] is not None],
        "model_version": "upgraded" if is_cached else "unknown"
    }


@app.get("/models/info")
async def get_model_info():
    """Get information about the UPGRADED models"""
    try:
        # Load models to get metadata
        models = load_models_from_supabase_direct()
        metadata = models.get('metadata', {})

        return {
            "version": metadata.get('version', 'unknown'),
            "architecture": metadata.get('model_architecture', {}),
            "num_features": metadata.get('num_features', 0),
            "training_samples": metadata.get('training_samples', 0),
            "enhancements": metadata.get('enhancements', []),
            "results": metadata.get('results', {}),
            "timestamp": metadata.get('timestamp', None),
            "feature_importance_available": bool(metadata.get('feature_importance', {}))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model information: {str(e)}"
        )


# ============================================================================
# API ENDPOINTS - PREDICTION
# ============================================================================

@app.get("/nearest-borehole")
async def get_nearest_borehole(latitude: float, longitude: float):
    """Find the nearest borehole and return soil parameters"""
    try:
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        if not client:
            raise HTTPException(
                status_code=503, detail="Database connection failed")

        boreholes_response = client.table('boreholes').select(
            'id, borehole_id, latitude, longitude'
        ).execute()

        if not boreholes_response.data:
            raise HTTPException(
                status_code=404, detail="No boreholes found in database")

        import math
        min_distance = float('inf')
        nearest_borehole = None

        for borehole in boreholes_response.data:
            if borehole['latitude'] is None or borehole['longitude'] is None:
                continue

            lat_diff = float(borehole['latitude']) - latitude
            lon_diff = float(borehole['longitude']) - longitude
            distance = math.sqrt(lat_diff**2 + lon_diff**2)

            if distance < min_distance:
                min_distance = distance
                nearest_borehole = borehole

        if not nearest_borehole:
            raise HTTPException(
                status_code=404, detail="No valid boreholes found")

        layers_response = client.table('soil_layers').select(
            'spt_n60, unit_weight, csr, groundwater_depth_m, fines_content, layer_number'
        ).eq('borehole_id', nearest_borehole['id']).order('layer_number', desc=False).execute()

        if not layers_response.data:
            raise HTTPException(
                status_code=404, detail="No soil data for nearest borehole")

        soil_data = layers_response.data

        # Helper function to safely extract and average values
        def safe_avg(key, default):
            values = [float(s[key])
                      for s in soil_data if s.get(key) is not None]
            return round(np.nanmean(values), 2) if values else default

        avg_spt = safe_avg('spt_n60', 8.0)
        avg_weight = safe_avg('unit_weight', 17.8)
        avg_csr = round(safe_avg('csr', 0.28), 4)
        avg_gwl = safe_avg('groundwater_depth_m', 3.0)
        avg_fines = safe_avg('fines_content', 20.0)
        crr = round(0.1 + 0.0048 * avg_spt, 4)

        return {
            "success": True,
            "nearest_borehole": {
                "id": nearest_borehole['id'],
                "borehole_id": nearest_borehole['borehole_id'],
                "distance_km": round(min_distance * 111, 2),
                "latitude": float(nearest_borehole['latitude']),
                "longitude": float(nearest_borehole['longitude'])
            },
            "soil_parameters": {
                "spt_n60": avg_spt,
                "unit_weight": avg_weight,
                "csr": avg_csr,
                "crr": crr,
                "gwl": avg_gwl,
                "fines_percent": avg_fines
            },
            "data_quality": {
                "total_layers": len(soil_data),
                "layers_with_spt": sum(1 for s in soil_data if s.get('spt_n60') is not None),
                "layers_with_unit_weight": sum(1 for s in soil_data if s.get('unit_weight') is not None),
                "layers_with_csr": sum(1 for s in soil_data if s.get('csr') is not None),
                "layers_with_gwl": sum(1 for s in soil_data if s.get('groundwater_depth_m') is not None),
                "layers_with_fines": sum(1 for s in soil_data if s.get('fines_content') is not None)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error finding nearest borehole: {str(e)}")


@app.get("/debug/borehole/{borehole_id}")
async def debug_borehole_data(borehole_id: str):
    """
    Debug endpoint to check what data exists for a specific borehole
    Example: /debug/borehole/BH-001
    """
    try:
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        if not client:
            raise HTTPException(
                status_code=503, detail="Database connection failed")

        # Find the borehole
        borehole_response = client.table('boreholes').select(
            '*').eq('borehole_id', borehole_id).execute()

        if not borehole_response.data:
            raise HTTPException(
                status_code=404, detail=f"Borehole {borehole_id} not found")

        borehole = borehole_response.data[0]

        # Get all soil layers
        layers_response = client.table('soil_layers').select(
            '*').eq('borehole_id', borehole['id']).execute()

        if not layers_response.data:
            return {
                "borehole": borehole,
                "message": "No soil layers found for this borehole"
            }

        # Analyze data availability
        layers = layers_response.data
        analysis = {
            "borehole_info": {
                "id": borehole['id'],
                "borehole_id": borehole['borehole_id'],
                "latitude": borehole.get('latitude'),
                "longitude": borehole.get('longitude'),
                "total_depth": borehole.get('depth_total_m')
            },
            "layers_count": len(layers),
            "data_availability": {
                "spt_n60": sum(1 for l in layers if l.get('spt_n60') is not None),
                "unit_weight": sum(1 for l in layers if l.get('unit_weight') is not None),
                "csr": sum(1 for l in layers if l.get('csr') is not None),
                "groundwater_depth_m": sum(1 for l in layers if l.get('groundwater_depth_m') is not None),
                "fines_content": sum(1 for l in layers if l.get('fines_content') is not None),
                "liquefaction": sum(1 for l in layers if l.get('liquefaction') is not None),
                "settlement_cm": sum(1 for l in layers if l.get('settlement_cm') is not None),
                "bearing_capacity_kpa": sum(1 for l in layers if l.get('bearing_capacity_kpa') is not None)
            },
            "sample_layer": layers[0] if layers else None,
            "all_column_names": list(layers[0].keys()) if layers else []
        }

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")


@app.post("/predict")
async def predict_liquefaction(request: PredictionRequest):
    """
    Predict liquefaction potential and impacts at a given location

    UPDATED: Loads UPGRADED models directly from Supabase Storage into memory (no local files)
    UPGRADED: Uses enhanced ANN architecture (256-128-64) with improved features
    """
    try:
        import traceback

        # Load UPGRADED models directly from Supabase Storage into memory
        try:
            models = load_models_from_supabase_direct()

            # Extract models from cache (these are actual model objects, not paths)
            scaler = models['scaler']
            liq_model = models['liquefaction']
            settlement_model = models['settlement']
            bearing_model = models['bearing_capacity']
            metadata = models['metadata']

            feature_names = metadata.get('feature_names', [])
            if not feature_names:
                raise Exception("No feature names found in metadata")
            print(
                f"✓ Using {len(feature_names)} features for prediction (UPGRADED model)")

        except Exception as e:
            print(f"Error loading UPGRADED models from Supabase: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load UPGRADED ML models from Supabase Storage: {str(e)}"
            )

        # Fetch soil data from nearest borehole
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        if not client:
            raise HTTPException(
                status_code=503, detail="Database connection failed")

        boreholes_response = client.table('boreholes').select(
            'id, borehole_id, latitude, longitude'
        ).execute()

        if not boreholes_response.data:
            raise HTTPException(
                status_code=404, detail="No boreholes found in database")

        # Find nearest borehole
        import math
        min_distance = float('inf')
        nearest_borehole = None

        for borehole in boreholes_response.data:
            if borehole['latitude'] is None or borehole['longitude'] is None:
                continue

            lat_diff = float(borehole['latitude']) - request.latitude
            lon_diff = float(borehole['longitude']) - request.longitude
            distance = math.sqrt(lat_diff**2 + lon_diff**2)

            if distance < min_distance:
                min_distance = distance
                nearest_borehole = borehole

        if not nearest_borehole:
            raise HTTPException(
                status_code=404, detail="No valid boreholes found")

        # Get soil layers data
        layers_response = client.table('soil_layers').select('*').eq(
            'borehole_id', nearest_borehole['id']
        ).order('layer_number', desc=False).execute()

        if not layers_response.data:
            raise HTTPException(
                status_code=404, detail="No soil data for nearest borehole")

        print(
            f"Found {len(layers_response.data)} soil layers for borehole {nearest_borehole['borehole_id']}")

        # Debug: Show first layer's available columns and sample data
        if layers_response.data:
            first_layer = layers_response.data[0]
            available_columns = list(first_layer.keys())
            print(
                f"Available columns in soil_layers: {', '.join(available_columns[:15])}...")
            print(f"Sample data from first layer:")
            print(f"  spt_n60: {first_layer.get('spt_n60')}")
            print(f"  unit_weight: {first_layer.get('unit_weight')}")
            print(f"  csr: {first_layer.get('csr')}")
            print(
                f"  groundwater_depth_m: {first_layer.get('groundwater_depth_m')}")
            print(f"  fines_content: {first_layer.get('fines_content')}")

        # Prepare feature vector
        soil_data = layers_response.data
        feature_vector = []

        # FIRST: Extract actual soil parameters for display (from soil_layers table)
        soil_params_display = {}

        # Helper function to safely extract numeric values
        def extract_values(key):
            values = []
            for s in soil_data:
                val = s.get(key)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass
            return values

        # Extract SPT N60
        spt_values = extract_values('spt_n60')
        soil_params_display['spt_n60'] = round(
            np.nanmean(spt_values), 2) if spt_values else 8.0

        # Extract Unit Weight
        weight_values = extract_values('unit_weight')
        soil_params_display['unit_weight'] = round(
            np.nanmean(weight_values), 2) if weight_values else 17.8

        # Extract CSR (Cyclic Stress Ratio)
        csr_values = extract_values('csr')
        soil_params_display['csr'] = round(
            np.nanmean(csr_values), 4) if csr_values else 0.28

        # Extract Groundwater Depth
        gwl_values = extract_values('groundwater_depth_m')
        soil_params_display['gwl'] = round(
            np.nanmean(gwl_values), 2) if gwl_values else 3.0

        # Extract Fines Content
        fines_values = extract_values('fines_content')
        soil_params_display['fines_percent'] = round(
            np.nanmean(fines_values), 2) if fines_values else 20.0

        # Calculate CRR from SPT
        soil_params_display['crr'] = round(
            0.1 + 0.0048 * soil_params_display['spt_n60'], 4)

        # Debug output
        print(f"\n=== Extracted Soil Parameters ===")
        print(
            f"  SPT N60: {soil_params_display['spt_n60']} (from {len(spt_values)} values)")
        print(
            f"  Unit Weight: {soil_params_display['unit_weight']} kN/m³ (from {len(weight_values)} values)")
        print(
            f"  CSR: {soil_params_display['csr']} (from {len(csr_values)} values)")
        print(f"  CRR: {soil_params_display['crr']}")
        print(
            f"  GWL: {soil_params_display['gwl']} m (from {len(gwl_values)} values)")
        print(
            f"  Fines: {soil_params_display['fines_percent']}% (from {len(fines_values)} values)")
        print(f"================================\n")

        # SECOND: Build feature vector for model prediction
        for feature_name in feature_names:
            values = []
            for soil_layer in soil_data:
                if feature_name in soil_layer and soil_layer[feature_name] is not None:
                    try:
                        values.append(float(soil_layer[feature_name]))
                    except (ValueError, TypeError):
                        pass

            if values:
                feature_value = np.nanmean(values)
            else:
                # Try alternative column names
                alt_names = {
                    'gwl': 'groundwater_depth_m',
                    'fines_percent': 'fines_content'
                }
                alt_name = alt_names.get(feature_name, feature_name)
                if alt_name in soil_data[0] if soil_data else {}:
                    values = [float(s[alt_name]) for s in soil_data
                              if alt_name in s and s[alt_name] is not None]
                    feature_value = np.nanmean(values) if values else 0.0
                else:
                    feature_value = 0.0

            feature_vector.append(feature_value)

        # Convert to numpy array
        input_features = np.array([feature_vector])
        print(f"Input features shape: {input_features.shape}")

        # Scale features using the loaded scaler
        input_scaled = scaler.transform(input_features)

        # Make predictions using UPGRADED loaded models
        liq_probability = liq_model.predict_proba(input_scaled)[0][1]
        liq_prediction = liq_model.predict(input_scaled)[0]
        settlement_pred = settlement_model.predict(input_scaled)[0]
        bearing_post = bearing_model.predict(input_scaled)[0]

        # Calculate derived values
        bearing_pre = bearing_post * 2.8
        capacity_reduction = ((bearing_pre - bearing_post) / bearing_pre) * 100

        # Determine risk level
        if liq_probability >= 0.75:
            risk_level = "HIGH"
            severity = "Severe"
        elif liq_probability >= 0.50:
            risk_level = "MEDIUM"
            severity = "Moderate"
        else:
            risk_level = "LOW"
            severity = "Minor"

        # Generate recommendations
        recommendations = []
        if risk_level == "HIGH":
            recommendations = [
                "Detailed geotechnical investigation required",
                "Deep foundation system implementation recommended",
                "Soil densification treatment strongly advised",
                "Post-liquefaction design considerations essential"
            ]
        elif risk_level == "MEDIUM":
            recommendations = [
                "Standard geotechnical investigation recommended",
                "Shallow to medium depth foundation design",
                "Soil improvement methods advisable",
                "Regular monitoring plan implementation"
            ]
        else:
            recommendations = [
                "Routine geotechnical survey sufficient",
                "Standard foundation design applicable",
                "Annual monitoring recommended"
            ]

        return {
            "model_info": {
                "version": "upgraded",
                "architecture": "ANN (256-128-64)",
                "features_used": len(feature_names)
            },
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "nearest_borehole_distance_km": round(min_distance * 111, 2)
            },
            "risk_assessment": {
                "risk_level": risk_level,
                "probability": round(liq_probability * 100, 1),
                "severity": severity,
                "confidence": "High" if metadata.get('results', {}).get('liquefaction', {}).get('test', {}).get('roc_auc', 0) > 0.8 else "Medium"
            },
            "soil_parameters": {
                "spt_n60": soil_params_display['spt_n60'],
                "unit_weight": soil_params_display['unit_weight'],
                "csr": soil_params_display['csr'],
                "crr": soil_params_display['crr'],
                "gwl": soil_params_display['gwl'],
                "fines_percent": soil_params_display['fines_percent'],
                "source": "Averaged from nearest borehole soil layers"
            },
            "settlement": {
                "predicted_cm": round(settlement_pred, 2),
                "severity": "Severe" if settlement_pred > 10 else "Moderate" if settlement_pred > 5 else "Minor"
            },
            "bearing_capacity": {
                "pre_liquefaction_kpa": round(bearing_pre, 2),
                "post_liquefaction_kpa": round(bearing_post, 2),
                "capacity_reduction_percent": round(capacity_reduction, 1)
            },
            "recommendations": recommendations
        }

    except ImportError as ie:
        raise HTTPException(status_code=500, detail=f"Import error: {str(ie)}")
    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict-by-location")
async def predict_by_location(latitude: float, longitude: float):
    """Predict using only latitude & longitude (UPGRADED model)"""
    try:
        pred_req = PredictionRequest(
            latitude=latitude,
            longitude=longitude,
            features=None
        )
        return await predict_liquefaction(pred_req)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("Predict-by-location error:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Prediction by location failed: {str(e)}")

@app.get("/debug/env")
async def debug_environment():
    """Debug endpoint to check environment variables (DO NOT USE IN PRODUCTION WITH REAL KEYS)"""
    return {
        "supabase_url_present": bool(os.getenv("SUPABASE_URL")),
        "supabase_url_value": os.getenv("SUPABASE_URL", "NOT SET")[:50] + "..." if os.getenv("SUPABASE_URL") else "NOT SET",
        "supabase_key_present": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
        "supabase_key_length": len(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")) if os.getenv("SUPABASE_SERVICE_ROLE_KEY") else 0,
        "all_env_vars": list(os.environ.keys())
    }

@app.get("/debug/supabase-test")
async def test_supabase_connection():
    """Test if Supabase connection works"""
    try:
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        
        if not client:
            return {
                "success": False,
                "error": "Client is None - check supabase_client.py logs"
            }
        
        # Try to access the storage bucket
        try:
            # Test storage access
            files = client.storage.from_('geotechnical-data').list('ml_models')
            
            return {
                "success": True,
                "message": "Supabase connection successful",
                "storage_accessible": True,
                "files_in_ml_models": [f['name'] for f in files] if files else []
            }
        except Exception as storage_error:
            return {
                "success": True,
                "message": "Database connected but storage error",
                "storage_accessible": False,
                "storage_error": str(storage_error)
            }
            
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
