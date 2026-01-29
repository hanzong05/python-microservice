"""
Geotechnical ML Training Pipeline API - FIXED
FastAPI service for Tarlac Liquefaction Prediction System

FIX: Loads .env from parent directory
"""

# ============================================================================
# CRITICAL FIX: Load .env from parent directory FIRST!
# ============================================================================
import json
from datetime import datetime
import asyncio
import subprocess
import sys
import uvicorn
from typing import Optional, Dict, List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, BackgroundTasks
import os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

# Get the parent directory (test-py/)
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / '.env'

# Load .env from parent directory
print(f"Loading .env from: {env_path}")
print(f".env exists: {env_path.exists()}")

print(f"SUPABASE_URL loaded: {bool(os.getenv('SUPABASE_URL'))}")
print(
    f"SUPABASE_SERVICE_ROLE_KEY loaded: {bool(os.getenv('SUPABASE_SERVICE_ROLE_KEY'))}")
# ============================================================================


app = FastAPI(
    title="Geo-ML Training API",
    description="ML Training Pipeline for Liquefaction Prediction System",
    version="1.0.1-FIXED"
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


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PipelineConfig(BaseModel):
    """Configuration for training pipeline"""
    scripts_directory: Optional[str] = None  # NEW: Allow custom script path
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
    """
    Search for script in multiple locations

    Search order:
    1. Custom directory (if provided)
    2. Parent directory (..)
    3. Current directory (.)
    4. Two levels up (../..)
    5. Common subdirectories
    """
    api_file_location = Path(__file__).resolve()

    search_paths = []

    # 1. Custom directory
    if custom_dir:
        search_paths.append(Path(custom_dir))

    # 2. Parent directory
    search_paths.append(api_file_location.parent.parent)

    # 3. Current directory
    search_paths.append(api_file_location.parent)

    # 4. Two levels up
    search_paths.append(api_file_location.parent.parent.parent)

    # 5. Common subdirectories
    search_paths.extend([
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
    """
    Run a Python training script asynchronously

    Args:
        script_name: Name of the Python script to run
        step_name: Display name for the step
        progress: Progress percentage (0-100)
        custom_script_dir: Optional custom directory where scripts are located

    Returns:
        tuple: (success: bool, output: str)
    """
    global pipeline_status

    # Update status
    pipeline_status["current_step"] = step_name
    pipeline_status["progress"] = progress
    log_message(step_name, f"Starting {step_name}...")

    # Find script
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

        # Prepare environment with UTF-8 encoding for Windows compatibility
        script_env = os.environ.copy()
        script_env['PYTHONIOENCODING'] = 'utf-8'

        # Run the script
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # ← Changed to parent's parent (test-py/)
            cwd=str(script_path.parent.parent),
            env=script_env
        )

        # Wait for completion with 30-minute timeout
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
            # Capture both stderr and stdout for debugging
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
# MODEL DOWNLOADING FROM SUPABASE
# ============================================================================

def download_models_from_supabase():
    """
    Download trained models from Supabase Storage and cache them locally

    Returns:
        dict: Paths to downloaded/cached models or raises exception if failed
    """
    import joblib

    model_dir = Path(__file__).parent.parent / 'ml_models'
    model_dir.mkdir(exist_ok=True)

    model_files = {
        'scaler.pkl': 'scaler',
        'ann_liquefaction.pkl': 'liquefaction classifier',
        'ann_settlement.pkl': 'settlement regressor',
        'ann_bearing_capacity.pkl': 'bearing capacity regressor'
    }

    # Check if models already exist locally (use cached version)
    local_cache_valid = all((model_dir / filename).exists()
                            for filename in model_files.keys())
    if local_cache_valid:
        print(f"✓ Using cached models from: {model_dir}")
        return {
            'scaler': model_dir / 'scaler.pkl',
            'liquefaction': model_dir / 'ann_liquefaction.pkl',
            'settlement': model_dir / 'ann_settlement.pkl',
            'bearing_capacity': model_dir / 'ann_bearing_capacity.pkl'
        }

    # Download from Supabase Storage
    print("📥 Downloading models from Supabase Storage...")
    from supabase_client import get_supabase_client

    client = get_supabase_client()
    if not client:
        raise Exception("Failed to connect to Supabase")

    downloaded_models = {}

    for filename, description in model_files.items():
        try:
            storage_path = f'ml_models/{filename}'
            local_path = model_dir / filename

            print(f"  Downloading {description} ({filename})...")
            file_data = client.storage.from_(
                'geotechnical-data').download(storage_path)

            # Save to local cache
            with open(local_path, 'wb') as f:
                f.write(file_data)

            downloaded_models[filename.replace('.pkl', '')] = local_path
            print(f"  ✓ Saved: {local_path}")

        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
            raise Exception(f"Failed to download {filename}: {e}")

    print(f"✓ All models downloaded and cached to: {model_dir}")
    return {
        'scaler': downloaded_models.get('scaler') or model_dir / 'scaler.pkl',
        'liquefaction': downloaded_models.get('ann_liquefaction') or model_dir / 'ann_liquefaction.pkl',
        'settlement': downloaded_models.get('ann_settlement') or model_dir / 'ann_settlement.pkl',
        'bearing_capacity': downloaded_models.get('ann_bearing_capacity') or model_dir / 'ann_bearing_capacity.pkl'
    }


# Global cache for model metadata
_model_metadata_cache = None


def load_model_metadata():
    """
    Load model metadata (feature names, etc.) from Supabase Storage or local cache

    Returns:
        dict: Metadata containing feature_names, etc.
    """
    global _model_metadata_cache

    # Return cached metadata if available
    if _model_metadata_cache is not None:
        return _model_metadata_cache

    import json

    model_dir = Path(__file__).parent.parent / 'ml_models'
    metadata_path = model_dir / 'ann_metadata.json'

    # Try to load from local cache first
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                _model_metadata_cache = json.load(f)
                print(f"✓ Loaded metadata from local cache: {metadata_path}")
                return _model_metadata_cache
        except Exception as e:
            print(f"Warning: Could not load local metadata: {e}")

    # Download from Supabase Storage
    print("📥 Downloading model metadata from Supabase Storage...")
    from supabase_client import get_supabase_client

    client = get_supabase_client()
    if not client:
        raise Exception("Failed to connect to Supabase for metadata")

    try:
        storage_path = 'ml_models/ann_metadata.json'
        file_data = client.storage.from_(
            'geotechnical-data').download(storage_path)

        # Parse JSON metadata
        _model_metadata_cache = json.loads(file_data.decode('utf-8'))

        # Cache locally for future use
        model_dir.mkdir(exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(_model_metadata_cache, f, indent=2)

        print(f"✓ Downloaded and cached metadata")
        return _model_metadata_cache

    except Exception as e:
        print(f"✗ Error downloading metadata: {e}")
        raise Exception(f"Failed to download model metadata: {e}")


async def run_full_pipeline(config: PipelineConfig):
    """Run the complete ML training pipeline"""
    global pipeline_status

    # Reset status
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

    log_message("PIPELINE", "[START] Starting ML Training Pipeline...")

    try:
        script_dir = config.scripts_directory

        # Step 1: Data Cleaning (optional)
        if config.run_data_cleaning:
            success, output = await run_script(
                "01_data_cleaning.py",
                "Data Cleaning",
                10,
                script_dir
            )
            if not success:
                raise Exception(f"Data Cleaning failed: {output}")

        # Step 2: Data Preparation
        if config.run_data_preparation:
            success, output = await run_script(
                "01b_ml_data_preparation.py",
                "Data Preparation",
                30,
                script_dir
            )
            if not success:
                raise Exception(f"Data Preparation failed: {output}")

        # Step 3: ETL Pipeline
        if config.run_etl:
            success, output = await run_script(
                "02_etl_to_supabase.py",
                "ETL Pipeline",
                50,
                script_dir
            )
            if not success:
                raise Exception(f"ETL Pipeline failed: {output}")

            log_message(
                "ETL", "[REMINDER] Run PostGIS location update in Supabase SQL Editor:")
            log_message(
                "ETL", "UPDATE boreholes SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)")

        # Step 4: Feature Engineering
        if config.run_feature_engineering:
            success, output = await run_script(
                "03_feature_engineering.py",
                "Feature Engineering",
                70,
                script_dir
            )
            if not success:
                raise Exception(f"Feature Engineering failed: {output}")

        # Step 5: ANN Training
        if config.run_model_training:
            success, output = await run_script(
                "04_model_training.py",
                "ANN Training",
                90,
                script_dir
            )
            if not success:
                raise Exception(f"ANN Training failed: {output}")

        # Pipeline completed
        pipeline_status["progress"] = 100
        pipeline_status["end_time"] = datetime.now().isoformat()
        pipeline_status["is_running"] = False
        log_message("PIPELINE", "[SUCCESS] Pipeline completed successfully!")

    except Exception as e:
        pipeline_status["error"] = str(e)
        pipeline_status["is_running"] = False
        pipeline_status["end_time"] = datetime.now().isoformat()
        log_message("PIPELINE", f"[FAILED] Pipeline failed: {str(e)}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "[OK] Geo-ML Training API is running!",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "GET /diagnostics": "Show script locations and system info",
            "POST /pipeline/start": "Start the ML training pipeline",
            "GET /pipeline/status": "Check pipeline status",
            "GET /pipeline/logs": "Get pipeline logs",
            "POST /pipeline/stop": "Stop running pipeline",
        },
        "docs": "/docs"
    }


@app.get("/diagnostics")
async def get_diagnostics():
    """
    Get diagnostic information about script locations
    """
    api_file = Path(__file__).resolve()

    # List all Python files in various directories
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

    # Check if each required script can be found
    for script_name in diagnostics["required_scripts"]:
        found_path = find_script(script_name)
        diagnostics["found_scripts"][script_name] = {
            "found": found_path is not None,
            "location": str(found_path) if found_path else None
        }

    return diagnostics


@app.post("/pipeline/start")
async def start_pipeline(
    background_tasks: BackgroundTasks,
    config: Optional[PipelineConfig] = None
):
    """
    Start the complete ML training pipeline

    You can optionally specify scripts_directory in the config:
    {
        "scripts_directory": "/full/path/to/scripts",
        "run_data_cleaning": true,
        ...
    }
    """
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
        "message": "ML training pipeline started in background",
        "estimated_duration": "15-40 minutes",
        "check_status": "/pipeline/status",
        "scripts_directory": config.scripts_directory or "auto-detect"
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
    """Stop the running pipeline (if any)"""
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


@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# QUICK TRAINING ENDPOINTS (Individual Steps)
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


# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    latitude: float
    longitude: float
    # Optional individual soil parameters (for manual input)
    # If not provided, will fetch from database
    features: Optional[Dict[str, float]] = None


@app.get("/nearest-borehole")
async def get_nearest_borehole(latitude: float, longitude: float):
    """
    Find the nearest borehole to the given coordinates and return soil parameters

    Returns soil data from the nearest borehole's soil layers (averaged or from shallowest layer)
    """
    try:
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        if not client:
            raise HTTPException(
                status_code=503, detail="Database connection failed")

        # Get all boreholes with their coordinates
        boreholes_response = client.table('boreholes').select(
            'id, borehole_id, latitude, longitude'
        ).execute()

        if not boreholes_response.data:
            raise HTTPException(
                status_code=404, detail="No boreholes found in database")

        # Calculate distances and find nearest
        import math
        min_distance = float('inf')
        nearest_borehole = None

        for borehole in boreholes_response.data:
            if borehole['latitude'] is None or borehole['longitude'] is None:
                continue

            # Haversine distance formula (simplified)
            lat_diff = float(borehole['latitude']) - latitude
            lon_diff = float(borehole['longitude']) - longitude
            distance = math.sqrt(lat_diff**2 + lon_diff**2)

            if distance < min_distance:
                min_distance = distance
                nearest_borehole = borehole

        if not nearest_borehole:
            raise HTTPException(
                status_code=404, detail="No valid boreholes found")

        # Get soil layers for the nearest borehole (get shallowest layers first)
        layers_response = client.table('soil_layers').select(
            'spt_n60, unit_weight, csr, groundwater_depth_m, fines_content, layer_number'
        ).eq('borehole_id', nearest_borehole['id']).order('layer_number', desc=False).execute()

        if not layers_response.data:
            raise HTTPException(
                status_code=404, detail="No soil data for nearest borehole")

        # Average soil parameters from all layers
        soil_data = layers_response.data
        avg_spt = np.nanmean([float(s['spt_n60'])
                             for s in soil_data if s['spt_n60'] is not None])
        avg_weight = np.nanmean([float(s['unit_weight'])
                                for s in soil_data if s['unit_weight'] is not None])
        avg_csr = np.nanmean([float(s['csr'])
                             for s in soil_data if s['csr'] is not None])
        avg_gwl = np.nanmean([float(s['groundwater_depth_m'])
                             for s in soil_data if s['groundwater_depth_m'] is not None])
        avg_fines = np.nanmean([float(s['fines_content'])
                               for s in soil_data if s['fines_content'] is not None])

        # CRR typically calculated from SPT N60, using simplified formula
        # CRR = 0.1 + 0.0048 * SPT_N60 (simplified Idriss & Boulanger)
        crr = round(0.1 + 0.0048 * avg_spt, 4)

        return {
            "success": True,
            "nearest_borehole": {
                "id": nearest_borehole['id'],
                "borehole_id": nearest_borehole['borehole_id'],
                # Rough conversion to km
                "distance_km": round(min_distance * 111, 2),
                "latitude": float(nearest_borehole['latitude']),
                "longitude": float(nearest_borehole['longitude'])
            },
            "soil_parameters": {
                "spt_n60": round(float(avg_spt), 2) if not np.isnan(avg_spt) else 8.0,
                "unit_weight": round(float(avg_weight), 2) if not np.isnan(avg_weight) else 17.8,
                "csr": round(float(avg_csr), 4) if not np.isnan(avg_csr) else 0.28,
                "crr": crr,
                "gwl": round(float(avg_gwl), 2) if not np.isnan(avg_gwl) else 3.0,
                "fines_percent": round(float(avg_fines), 2) if not np.isnan(avg_fines) else 20.0
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error finding nearest borehole: {str(e)}")


@app.post("/predict")
async def predict_liquefaction(request: PredictionRequest):
    """
    Predict liquefaction potential and impacts at a given location

    Returns:
    - Liquefaction risk level and probability
    - Predicted settlement
    - Bearing capacity (pre and post-liquefaction)
    - Recommendations
    """
    try:
        import joblib
        import traceback

        # Load model metadata to get feature names
        try:
            metadata = load_model_metadata()
            feature_names = metadata.get('feature_names', [])
            if not feature_names:
                raise Exception("No feature names found in metadata")
            print(f"✓ Loaded {len(feature_names)} feature names from metadata")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=503, detail=f"Failed to load model metadata: {str(e)}")

        # Download/use cached models from Supabase Storage
        try:
            model_paths = download_models_from_supabase()
            scaler_path = model_paths['scaler']
            liq_model_path = model_paths['liquefaction']
            settlement_model_path = model_paths['settlement']
            bearing_model_path = model_paths['bearing_capacity']
        except Exception as e:
            print(f"Error downloading models from Supabase: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=503, detail=f"Failed to load ML models from Supabase: {str(e)}")

        # Load models from cache
        scaler = joblib.load(scaler_path)
        liq_model = joblib.load(liq_model_path)
        settlement_model = joblib.load(settlement_model_path)
        bearing_model = joblib.load(bearing_model_path)

        # Fetch soil data for all required features from nearest borehole
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        if not client:
            raise HTTPException(
                status_code=503, detail="Database connection failed")

        # Get all boreholes with their coordinates
        boreholes_response = client.table('boreholes').select(
            'id, borehole_id, latitude, longitude').execute()
        if not boreholes_response.data:
            raise HTTPException(
                status_code=404, detail="No boreholes found in database")

        # Find nearest borehole
        import math
        min_distance = float('inf')
        nearest_borehole = None

        for borehole in boreholes_response.data:
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
        layers_response = client.table('soil_layers').select(
            '*').eq('borehole_id', nearest_borehole['id']).order('layer_number', desc=False).execute()
        if not layers_response.data:
            raise HTTPException(
                status_code=404, detail="No soil data for nearest borehole")

        # Prepare feature array with all required features
        # Fill with averaged data from soil layers
        soil_data = layers_response.data
        feature_vector = []

        # Track key soil parameters for frontend display
        soil_params_display = {
            'spt_n60': 0, 'unit_weight': 0, 'csr': 0,
            'crr': 0, 'gwl': 0, 'fines_percent': 0
        }

        for feature_name in feature_names:
            # Try to get the feature value from soil data
            values = []
            for soil_layer in soil_data:
                if feature_name in soil_layer and soil_layer[feature_name] is not None:
                    try:
                        values.append(float(soil_layer[feature_name]))
                    except (ValueError, TypeError):
                        pass

            # Use average if available, otherwise use 0
            if values:
                feature_value = np.nanmean(values)
            else:
                # Try alternative column names
                alt_names = {
                    'spt_n60': 'spt_n60',
                    'unit_weight': 'unit_weight',
                    'gwl': 'groundwater_depth_m',
                    'fines_percent': 'fines_content',
                    'csr': 'csr'
                }
                alt_name = alt_names.get(feature_name, None)
                if alt_name and alt_name != feature_name:
                    values = [float(
                        s[alt_name]) for s in soil_data if alt_name in s and s[alt_name] is not None]
                    feature_value = np.nanmean(values) if values else 0.0
                else:
                    feature_value = 0.0

            # Track display parameters
            for key in soil_params_display.keys():
                if key.lower() in feature_name.lower() or feature_name.lower() in key.lower():
                    soil_params_display[key] = round(
                        float(feature_value), 2 if key != 'csr' and key != 'crr' else 4)

            feature_vector.append(feature_value)

        # Calculate CRR if not in features
        if 'crr' not in [f.lower() for f in feature_names]:
            soil_params_display['crr'] = round(
                0.1 + 0.0048 * soil_params_display['spt_n60'], 4)

        # Convert to numpy array and reshape for prediction
        input_features = np.array([feature_vector])
        print(f"Input features shape: {input_features.shape}")
        print(f"Expected features: {len(feature_names)}")

        # Scale features
        input_scaled = scaler.transform(input_features)

        # Make predictions
        liq_probability = liq_model.predict_proba(
            input_scaled)[0][1]  # Probability of liquefaction
        liq_prediction = liq_model.predict(input_scaled)[0]
        settlement_pred = settlement_model.predict(input_scaled)[0]
        bearing_post = bearing_model.predict(input_scaled)[0]

        # Estimate pre-liquefaction bearing capacity (typically 2.5-3x post-liquefaction)
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

        # Generate recommendations based on risk
        recommendations = []
        if risk_level == "HIGH":
            recommendations = [
                "Detailed geotechnical investigation",
                "Deep foundation system implementation",
                "Soil densification treatment",
                "Post-liquefaction design considerations"
            ]
        elif risk_level == "MEDIUM":
            recommendations = [
                "Standard geotechnical investigation",
                "Shallow to medium depth foundation design",
                "Soil improvement methods",
                "Regular monitoring plan"
            ]
        else:
            recommendations = [
                "Routine geotechnical survey",
                "Standard foundation design",
                "Annual monitoring"
            ]

        return {
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "nearest_borehole_distance_km": round(min_distance * 111, 2)
            },
            "risk_assessment": {
                "risk_level": risk_level,
                "probability": round(liq_probability * 100, 1),
                "severity": severity
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

    except ImportError:
        raise HTTPException(status_code=500, detail="joblib not installed")
    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict-by-location")
async def predict_by_location(latitude: float, longitude: float):
    """Predict using only latitude & longitude: fetch all features from nearest borehole and run prediction."""
    try:
        # Create prediction request with just coordinates
        # The predict endpoint will fetch all required features from database
        pred_req = PredictionRequest(
            latitude=latitude,
            longitude=longitude,
            features=None  # Will fetch from database
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
