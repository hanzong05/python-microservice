"""
Geotechnical ML Training Pipeline API - WITH DIAGNOSTICS
FastAPI service for Tarlac Liquefaction Prediction System

This version includes diagnostics to find where scripts are located.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
import os
import sys
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime
import json

app = FastAPI(
    title="Geo-ML Training API",
    description="ML Training Pipeline for Liquefaction Prediction System",
    version="1.0.0"
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

        # Run the script
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent),
            env=os.environ.copy()  # ← ADD THIS LINE!
        )

        # Wait for completion with 30-minute timeout
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=1800  # 30 minutes
        )

        if process.returncode == 0:
            log_message(
                step_name, f"[SUCCESS] {step_name} completed successfully")
            pipeline_status["steps_completed"].append(step_name)
            return True, stdout.decode()
        else:
            error_msg = stderr.decode() if stderr else "Unknown error"
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


@app.post("/train/step5-ann-training")
async def run_ann_training(background_tasks: BackgroundTasks, scripts_directory: Optional[str] = None):
    """Run only Step 5: ANN Training"""
    if pipeline_status["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    async def run_step():
        pipeline_status["is_running"] = True
        await run_script("04_model_training.py", "ANN Training", 100, scripts_directory)
        pipeline_status["is_running"] = False

    background_tasks.add_task(run_step)
    return {"status": "started", "step": "ANN Training"}


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
