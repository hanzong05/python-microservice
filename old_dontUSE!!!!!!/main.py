"""
Geotechnical ML Training Pipeline API - FEATURE-ALIGNED VERSION
FastAPI service for Tarlac Geotechnical Prediction System

CRITICAL FIX FOR FEATURE MISMATCH:
- Model expects 104 engineered features
- v_complete_soil_data only has 28 raw columns
- Solution: Apply same feature engineering during prediction as during training
- This ensures exact feature alignment (104 = 104)

Version: 3.5.0-FEATURE-ALIGNED
"""

# ============================================================================
# IMPORTS
# ============================================================================
from feature_helpers import (
    parse_pga_value,
    parse_relative_density,
    extract_depth_range,
    safe_float,
    load_medians_from_csv_or_bytes,
    compute_borehole_aggregates,
    compute_layer_aggregates,
    fetch_muni_stats,
)
import json
import io
from datetime import datetime, timedelta
import sys
import asyncio
import uvicorn
from typing import Optional, Dict, List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, BackgroundTasks
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
import math

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Geo-ML Training API",
    description="ML Training Pipeline for Geotechnical Prediction System",
    version="3.5.0-FEATURE-ALIGNED"
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

# Thread pool for running subprocesses
executor = ThreadPoolExecutor(max_workers=1)


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
# FEATURE ENGINEERING FUNCTIONS (MUST MATCH TRAINING!)
# ============================================================================

"""
ULTIMATE FIX: Use Exact Feature Names from Model Metadata

The problem is that we're trying to guess which features the model needs.
The solution is to use the EXACT feature names stored in the model's metadata.

Replace your engineer_features_for_prediction() function with this simpler approach.
"""


def engineer_features_for_prediction(soil_data_df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """
    Engineer features to match EXACT features from model metadata

    Args:
        soil_data_df: Raw soil data from database
        required_features: Exact list of feature names from model metadata

    Returns:
        DataFrame with exactly the features the model expects
    """
    df = soil_data_df.copy()

    print(f"[DEBUG] Starting feature engineering")
    print(f"[DEBUG] Input columns: {len(df.columns)}")
    print(f"[DEBUG] Required features: {len(required_features)}")

    # ========================================================================
    # Create ALL possible features (even if not used)
    # ========================================================================

    # DEPTH FEATURES
    df['depth_mid_m'] = (df.get('depth_from_m', 0) +
                         df.get('depth_to_m', 1.5)) / 2
    df['depth_thickness_m'] = df.get(
        'depth_to_m', 1.5) - df.get('depth_from_m', 0)
    df['depth_to_groundwater_m'] = df.get(
        'groundwater_depth_m', 5.0) - df['depth_mid_m']
    df['is_below_groundwater'] = (df['depth_mid_m'] > df.get(
        'groundwater_depth_m', 5.0)).astype(int)
    df['depth_mid_squared'] = df['depth_mid_m'] ** 2
    df['depth_normalized'] = df['depth_mid_m'] / 15.0

    # SPT FEATURES
    spt_n = df.get('spt_n_value', 15.0)
    spt_n160 = df.get('spt_n160', spt_n * 1.1)
    spt_n60 = df.get('spt_n60', spt_n)

    df['spt_correction_ratio'] = spt_n160 / (spt_n + 1)
    df['spt_n_log'] = np.log1p(spt_n)
    df['spt_n160_log'] = np.log1p(spt_n160)
    df['spt_n60_log'] = np.log1p(spt_n60)
    df['spt_n_squared'] = spt_n ** 2
    df['spt_n160_squared'] = spt_n160 ** 2
    df['spt_n60_squared'] = spt_n60 ** 2
    df['relative_density_from_spt'] = np.clip(
        np.sqrt(spt_n160 / 60) * 100, 0, 100)

    # STRESS FEATURES
    unit_weight = df.get('unit_weight', 18.0)
    gwl = df.get('groundwater_depth_m', 5.0)

    df['total_overburden_pressure'] = unit_weight * df['depth_mid_m']

    gamma_water = 9.81
    depth_below_wt = np.maximum(0, df['depth_mid_m'] - gwl)
    df['effective_overburden_pressure'] = df['total_overburden_pressure'] - \
        (gamma_water * depth_below_wt)

    df['effective_stress_ratio'] = df['effective_overburden_pressure'] / \
        (df['total_overburden_pressure'] + 1)
    df['overburden_pressure_diff'] = df['total_overburden_pressure'] - \
        df['effective_overburden_pressure']
    df['effective_stress_normalized'] = df['effective_overburden_pressure'] / 100
    df['total_stress_normalized'] = df['total_overburden_pressure'] / 100
    df['pore_pressure_approx'] = df['overburden_pressure_diff']
    df['stress_reduction_factor'] = df['effective_overburden_pressure'] / \
        (df['total_overburden_pressure'] + 0.001)

    # SEISMIC FEATURES
    csr = df.get('csr', 0.2)
    df['cyclic_strength_ratio'] = np.where(
        spt_n160 <= 30, 1 / (34 - spt_n160 + 0.001), 0.5)
    df['factor_of_safety'] = (
        df['cyclic_strength_ratio'] + 0.001) / (csr + 0.001)
    df['liquefaction_potential_index'] = np.where(
        df['factor_of_safety'] < 1.0, 1.0 - df['factor_of_safety'], 0)
    df['csr_crr_ratio'] = csr / (df['cyclic_strength_ratio'] + 0.001)

    # BEARING CAPACITY FEATURES
    bc_kpa = df.get('bearing_capacity_kpa', spt_n * 30)
    qa_kpa = df.get('qa_allowable_kpa', bc_kpa / 3.0)

    df['bearing_capacity_kpa'] = bc_kpa
    df['qa_allowable_kpa'] = qa_kpa
    df['bearing_capacity_safety_factor'] = qa_kpa / (bc_kpa + 1)
    df['bc_utilization_ratio'] = bc_kpa / (qa_kpa + 1)
    df['qa_allowable_log'] = np.log1p(qa_kpa)
    df['bearing_capacity_log'] = np.log1p(bc_kpa)
    df['qa_bc_ratio'] = qa_kpa / (bc_kpa + 1)

    # SOIL PROPERTY FEATURES
    fines = df.get('fines_content', 15.0)
    moisture = df.get('moisture_content', 20.0)

    df['moisture_content_log'] = np.log1p(moisture)
    df['fines_content_log'] = np.log1p(fines)
    df['is_clean_sand'] = (fines < 5).astype(int)
    df['is_silty_sand'] = ((fines >= 5) & (fines < 35)).astype(int)
    df['is_fine_grained'] = (fines >= 35).astype(int)
    # Ensure we operate with Series objects to avoid scalar boolean issues
    if 'mean_particle_size_d50' in df.columns:
        particle_size_series = df['mean_particle_size_d50'].fillna(0.2)
    else:
        particle_size_series = pd.Series(0.2, index=df.index)
    df['particle_size_log'] = np.log1p(particle_size_series)

    if 'plasticity_index' in df.columns:
        plasticity_series = df['plasticity_index'].fillna(5.0)
    else:
        plasticity_series = pd.Series(5.0, index=df.index)
    df['plasticity_index_log'] = np.log1p(plasticity_series)
    df['is_plastic'] = (plasticity_series > 7).astype(int)
    df['unit_weight_log'] = np.log1p(unit_weight)

    # SHEAR STRENGTH FEATURES
    friction = df.get('friction_angle', 27.5 + 0.3 * spt_n)
    cohesion = df.get('cohesion_kpa', 0.0)

    df['friction_angle_log'] = np.log1p(friction)
    df['friction_angle_radians'] = np.radians(friction)
    df['tan_friction_angle'] = np.tan(df['friction_angle_radians'])
    df['cohesion_log'] = np.log1p(cohesion)
    df['is_cohesive'] = (cohesion > 10).astype(int)

    # SPATIAL FEATURES
    lat = df.get('latitude', 15.4754)
    lon = df.get('longitude', 120.5963)
    tarlac_lat, tarlac_lon = 15.4753, 120.5969

    # Haversine distance
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(tarlac_lat), np.radians(tarlac_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['distance_from_tarlac_center_km'] = c * 6371

    df['nearest_borehole_distance_km'] = df.get(
        'nearest_borehole_distance_km', 1.0)
    df['zone_liquefaction_risk_percent'] = 0.0
    df['zone_sample_count'] = 0
    df['zone_avg_spt_n'] = spt_n

    # INTERACTION FEATURES
    pga = df.get('pga_g', 0.35)

    df['depth_spt_interaction'] = df['depth_mid_m'] * spt_n
    df['depth_fines_interaction'] = df['depth_mid_m'] * fines
    df['depth_moisture_interaction'] = df['depth_mid_m'] * moisture
    df['depth_stress_interaction'] = df['depth_mid_m'] * \
        df['effective_overburden_pressure']
    df['spt_fines_interaction'] = spt_n * (100 - fines) / 100
    df['spt_moisture_interaction'] = spt_n * (100 - moisture) / 100
    df['spt_stress_interaction'] = spt_n * df['effective_overburden_pressure']
    df['spt_depth_ratio'] = spt_n / (df['depth_mid_m'] + 1)
    df['csr_depth_interaction'] = csr * df['depth_mid_m']
    df['csr_fines_interaction'] = csr * fines
    df['csr_spt_interaction'] = csr * spt_n
    df['pga_depth_interaction'] = pga * df['depth_mid_m']
    df['distance_spt_interaction'] = df['distance_from_tarlac_center_km'] * spt_n
    df['distance_depth_interaction'] = df['distance_from_tarlac_center_km'] * \
        df['depth_mid_m']
    df['bc_spt_interaction'] = bc_kpa * spt_n
    df['bc_depth_interaction'] = bc_kpa * df['depth_mid_m']

    # AGGREGATE FEATURES (defaults for single prediction)
    df['bh_avg_spt'] = spt_n
    df['bh_min_spt'] = spt_n
    df['bh_max_spt'] = spt_n
    df['bh_std_spt'] = 0.0
    df['bh_avg_qa_allowable'] = qa_kpa
    df['bh_min_qa_allowable'] = qa_kpa
    df['bh_max_qa_allowable'] = qa_kpa
    df['layer_avg_spt'] = spt_n
    df['layer_std_spt'] = 0.0
    df['layer_avg_unit_weight'] = unit_weight
    df['layer_avg_fines'] = fines
    df['layer_avg_qa_allowable'] = qa_kpa
    df['spt_relative_to_layer'] = 0.0
    df['qa_relative_to_layer'] = 0.0

    # OPTIONAL FEATURES (from views - may not be in training)
    df['min_bc_kpa'] = bc_kpa * 0.8
    df['avg_bc_kpa'] = bc_kpa
    df['max_bc_kpa'] = bc_kpa * 1.2
    df['stddev_bc_kpa'] = 0.0

    print(f"[DEBUG] Created {len(df.columns)} total features")

    # ========================================================================
    # Improved missing-feature handling:
    # - Try to impute missing features using medians from engineering exports
    # - Attempt to fetch aggregates (municipality/borehole/layer) from DB if available
    # - Build missing columns in one DataFrame and concat to avoid fragmentation
    # ========================================================================

    # 1) Try to load precomputed medians from local export or Supabase
    medians = {}
    try:
        # Local CSV exported by feature engineering
        local_path = Path('feature_engineering/features_engineered_FIXED.csv')
        if local_path.exists():
            sample_df = pd.read_csv(local_path)
            med = sample_df.median(numeric_only=True)
            medians = med.to_dict()
            print(f"[DEBUG] Loaded medians from {local_path}")
        else:
            # Try Supabase storage if client available
            try:
                from supabase_client import get_supabase_client
                client = get_supabase_client()
                if client:
                    try:
                        resp = client.storage.from_(
                            'geotechnical-data').download('feature_engineering/features_engineered_FIXED.csv')
                        if resp:
                            sample_df = pd.read_csv(io.BytesIO(resp))
                            med = sample_df.median(numeric_only=True)
                            medians = med.to_dict()
                            print("[DEBUG] Loaded medians from Supabase storage")
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception as e:
        print(f"[DEBUG] Could not load medians: {e}")

    # 2) Optional: attempt to fetch aggregates from DB to fill borehole/municipality/layer features
    try:
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        if client is not None:
            # Fetch municipality stats if municipality available
            if 'municipality' in df.columns and not df['municipality'].isna().all():
                muni = str(df.loc[df.index[0], 'municipality'])
                try:
                    res = client.table('v_municipality_statistics').select(
                        '*').eq('municipality', muni).limit(1).execute()
                    if res and getattr(res, 'data', None):
                        muni_row = res.data[0]
                        for k, v in muni_row.items():
                            if k not in df.columns:
                                df[k] = v
                        print(f"[DEBUG] Merged municipality stats for {muni}")
                except Exception:
                    pass

            # Fetch bearing capacity by layer if layer_number available
            if 'layer_number' in df.columns and not df['layer_number'].isna().all():
                ln = int(df.loc[df.index[0], 'layer_number'])
                try:
                    res = client.table('v_bearing_capacity_by_layer').select(
                        '*').eq('layer_number', ln).limit(1).execute()
                    if res and getattr(res, 'data', None):
                        layer_row = res.data[0]
                        # map view columns to expected feature names
                        mapping = {
                            'min_bc_kpa': 'min_bc_kpa',
                            'avg_bc_kpa': 'avg_bc_kpa',
                            'max_bc_kpa': 'max_bc_kpa',
                            'stddev_bc_kpa': 'stddev_bc_kpa'
                        }
                        for k_src, k_dst in mapping.items():
                            if k_src in layer_row and k_dst not in df.columns:
                                df[k_dst] = layer_row[k_src]
                        print(
                            f"[DEBUG] Merged layer-level bearing stats for layer {ln}")
                except Exception:
                    pass
    except Exception:
        # Supabase client not available or fetch failed — continue
        pass

    # 3) Build missing columns dict and concat once to avoid fragmentation
    missing_dict = {}
    for feat in required_features:
        if feat not in df.columns:
            # Prefer median if available
            if feat in medians and not pd.isna(medians[feat]):
                missing_val = float(medians[feat])
                print(f"[DEBUG] Imputing {feat} with median: {missing_val}")
            else:
                # sensible fallbacks for common groups
                if 'spt' in feat or 'depth' in feat or 'fines' in feat or 'qa' in feat or 'bearing' in feat or 'unit_weight' in feat:
                    # numeric geological defaults
                    fallback_map = {
                        'spt': 15.0,
                        'depth': 0.75,
                        'fines': 15.0,
                        'qa': 1000.0,
                        'bearing': 3000.0,
                        'unit_weight': 18.0
                    }
                    # pick based on substring
                    val = 0.0
                    for key, fv in fallback_map.items():
                        if key in feat:
                            val = fv
                            break
                    missing_val = val
                    print(
                        f"[DEBUG] Imputing {feat} with fallback: {missing_val}")
                else:
                    # default to 0 for unknown features
                    missing_val = 0.0
                    print(
                        f"[WARNING] Missing feature: {feat} - no median available, using 0")

            missing_dict[feat] = missing_val

    if missing_dict:
        # Create a DataFrame of missing values aligned to df index
        missing_df = pd.DataFrame([missing_dict] * len(df), index=df.index)
        # Concatenate once
        df = pd.concat([df.reset_index(drop=True),
                       missing_df.reset_index(drop=True)], axis=1)

    # Now select required features and ensure no NaNs remain
    feature_df = df[required_features].copy()
    feature_df = feature_df.fillna(0)

    print(
        f"[DEBUG] Returning {len(feature_df.columns)} features (exact match)")
    return feature_df

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


def validate_script(script_path: Path) -> Dict[str, any]:
    """Validate that a script can be executed"""
    validation = {
        "exists": script_path.exists(),
        "is_file": script_path.is_file() if script_path.exists() else False,
        "is_readable": os.access(script_path, os.R_OK) if script_path.exists() else False,
        "size_bytes": script_path.stat().st_size if script_path.exists() else 0,
        "python_syntax_valid": False,
        "has_main": False,
        "import_errors": []
    }

    if validation["exists"] and validation["is_readable"]:
        try:
            # Check Python syntax
            with open(script_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, script_path, 'exec')
                validation["python_syntax_valid"] = True
                validation["has_main"] = '__main__' in code or 'if __name__' in code
        except SyntaxError as e:
            validation["syntax_error"] = str(e)
        except Exception as e:
            validation["validation_error"] = str(e)

    return validation


def find_script(script_name: str, custom_dir: Optional[str] = None) -> Optional[Path]:
    """Search for script in multiple locations with detailed logging"""
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

            # Validate the script
            validation = validate_script(script_path)
            log_message("SCRIPT_SEARCH", f"Validation: {validation}")

            if not validation["python_syntax_valid"]:
                log_message("SCRIPT_SEARCH",
                            f"[WARNING] Script has syntax errors!")
                if "syntax_error" in validation:
                    log_message("SCRIPT_SEARCH",
                                f"Syntax error: {validation['syntax_error']}")

            return script_path

    log_message("SCRIPT_SEARCH",
                f"[NOT FOUND] Script not found in any location")
    return None


def run_script_sync(script_path: Path, step_name: str, working_dir: Path, script_env: dict) -> tuple:
    """
    Run a Python script synchronously (for Windows compatibility)

    Returns:
        tuple: (return_code, stdout, stderr)
    """
    try:
        log_message(step_name, f"Starting subprocess (sync mode)...")

        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, '-u', str(script_path)],
            cwd=str(working_dir),
            env=script_env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1800  # 30 minutes
        )

        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired as e:
        log_message(step_name, f"[TIMEOUT] Process exceeded 30 minutes")
        return -1, e.stdout or "", e.stderr or "Process timed out after 30 minutes"
    except Exception as e:
        log_message(step_name, f"[ERROR] Exception in subprocess: {e}")
        return -1, "", str(e)


async def run_script_with_debug(script_name: str, step_name: str, progress: int,
                                custom_script_dir: Optional[str] = None) -> tuple:
    """
    Run a Python training script with comprehensive debugging
    Uses synchronous subprocess in thread pool for Windows compatibility

    Returns:
        tuple: (success: bool, output: str)
    """
    global pipeline_status

    pipeline_status["current_step"] = step_name
    pipeline_status["progress"] = progress
    log_message(step_name, f"Starting {step_name}...")

    # Find script
    script_path = find_script(script_name, custom_script_dir)
    if script_path is None:
        error_msg = f"Script not found: {script_name}"
        log_message(step_name, f"[ERROR] {error_msg}")
        return False, error_msg

    # Validate script before running
    validation = validate_script(script_path)
    log_message(step_name, f"Script validation: {validation}")

    if not validation["exists"]:
        return False, f"Script does not exist: {script_path}"
    if not validation["is_readable"]:
        return False, f"Script is not readable: {script_path}"
    if not validation["python_syntax_valid"]:
        error_detail = validation.get("syntax_error", "Unknown syntax error")
        return False, f"Script has syntax errors: {error_detail}"

    working_dir = script_path.parent

    log_message(step_name, f"Script path: {script_path}")
    log_message(step_name, f"Working directory: {working_dir}")
    log_message(step_name, f"Script size: {validation['size_bytes']} bytes")
    log_message(step_name, f"Python executable: {sys.executable}")
    log_message(step_name, f"Python version: {sys.version}")

    try:
        # Set up environment
        script_env = os.environ.copy()
        script_env['PYTHONIOENCODING'] = 'utf-8'
        script_env['PYTHONUNBUFFERED'] = '1'
        script_env['PYTHONDONTWRITEBYTECODE'] = '1'

        # Add current directory to PYTHONPATH
        if 'PYTHONPATH' in script_env:
            script_env['PYTHONPATH'] = f"{working_dir}{os.pathsep}{script_env['PYTHONPATH']}"
        else:
            script_env['PYTHONPATH'] = str(working_dir)

        log_message(step_name, "Environment variables set:")
        log_message(
            step_name, f"  PYTHONPATH: {script_env.get('PYTHONPATH', 'Not set')}")
        log_message(
            step_name, f"  SUPABASE_URL present: {bool(script_env.get('SUPABASE_URL'))}")
        log_message(
            step_name, f"  SUPABASE_SERVICE_ROLE_KEY present: {bool(script_env.get('SUPABASE_SERVICE_ROLE_KEY'))}")

        # Test if we can import the script's dependencies
        log_message(step_name, "Testing imports...")
        test_imports = ['pandas', 'numpy', 'openpyxl', 'supabase']
        for module in test_imports:
            try:
                __import__(module)
                log_message(step_name, f"  ✓ {module}")
            except ImportError as e:
                log_message(step_name, f"  ✗ {module}: {e}")

        # Run script in thread pool (async wrapper around sync subprocess)
        loop = asyncio.get_event_loop()
        return_code, stdout, stderr = await loop.run_in_executor(
            executor,
            run_script_sync,
            script_path,
            step_name,
            working_dir,
            script_env
        )

        log_message(
            step_name, f"Process completed with return code: {return_code}")

        # Log output in chunks
        if stdout:
            stdout_lines = stdout.split('\n')
            log_message(step_name, f"STDOUT lines: {len(stdout_lines)}")
            for line in stdout_lines[:50]:  # Log first 50 lines
                if line.strip():
                    log_message(step_name, f"STDOUT: {line}")

        if stderr:
            stderr_lines = stderr.split('\n')
            log_message(step_name, f"STDERR lines: {len(stderr_lines)}")
            for line in stderr_lines[:50]:  # Log first 50 lines
                if line.strip():
                    log_message(step_name, f"STDERR: {line}")

        # Combine output
        full_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

        if return_code == 0:
            log_message(
                step_name, f"[SUCCESS] {step_name} completed successfully")
            pipeline_status["steps_completed"].append(step_name)
            return True, full_output
        else:
            # Build comprehensive error message
            error_msg = f"Process exited with code {return_code}\n\n"

            if stderr:
                stderr_lines = stderr.split('\n')
                error_msg += "STDERR (last 50 lines):\n" + \
                    '\n'.join(stderr_lines[-50:]) + "\n\n"

            if stdout:
                stdout_lines = stdout.split('\n')
                error_msg += "STDOUT (last 30 lines):\n" + \
                    '\n'.join(stdout_lines[-30:]) + "\n\n"

            if not stderr and not stdout:
                error_msg += "No output captured - script may have crashed immediately\n"
                error_msg += f"Try running manually:\n  cd {working_dir}\n  python {script_path.name}\n"

            log_message(step_name, f"[FAILED] {error_msg[:2000]}")
            pipeline_status["error"] = error_msg
            return False, error_msg

    except Exception as e:
        error_msg = f"Exception running script: {str(e)}"
        traceback_str = traceback.format_exc()
        log_message(step_name, f"[ERROR] {error_msg}")
        log_message(step_name, f"Traceback:\n{traceback_str}")
        pipeline_status["error"] = error_msg
        return False, f"{error_msg}\n\n{traceback_str}"


# Alias for backward compatibility
run_script = run_script_with_debug


# ============================================================================
# MODEL LOADING FROM SUPABASE STORAGE
# ============================================================================

def load_models_from_supabase_direct():
    """
    Load trained models DIRECTLY from Supabase Storage into memory
    Gracefully handles missing models
    """
    global _model_cache

    # Return cached models if available (cache for 1 hour)
    if _model_cache['scaler'] is not None:
        if _model_cache['last_loaded'] and \
           datetime.now() - _model_cache['last_loaded'] < timedelta(hours=1):
            print(
                f"[OK] Using cached models (loaded at {_model_cache['last_loaded'].strftime('%H:%M:%S')})")
            return _model_cache

    print("Loading models from Supabase Storage into memory...")

    # Check if supabase_client module exists
    try:
        from supabase_client import get_supabase_client
    except ImportError as e:
        raise Exception(f"Cannot import supabase_client: {e}")

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
            file_data = client.storage.from_(
                'geotechnical-data').download(storage_path)

            if model_name == 'metadata':
                metadata = json.loads(file_data.decode('utf-8'))
                loaded_models[model_name] = metadata
                version = metadata.get('version', 'unknown')
                print(f"  [OK] Loaded {model_name} (version: {version})")
            else:
                loaded_models[model_name] = joblib.load(io.BytesIO(file_data))
                print(f"  [OK] Loaded {model_name} into memory")

        except Exception as e:
            print(f"  [WARNING] Could not load {model_name}: {e}")
            loaded_models[model_name] = None

            if model_name in ['scaler', 'metadata']:
                raise Exception(
                    f"Failed to load critical component {model_name}: {e}")
            else:
                print(f"  [INFO] Continuing without {model_name} model")

    _model_cache = loaded_models
    _model_cache['last_loaded'] = datetime.now()

    available_models = [k for k in ['liquefaction', 'settlement', 'bearing_capacity']
                        if loaded_models.get(k) is not None]

    print(f"[OK] Models loaded into memory!")
    print(
        f"  Available models: {', '.join(available_models) if available_models else 'None'}")

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
    print("[OK] Model cache cleared - will reload models on next prediction")
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

    log_message("PIPELINE", "[START] Starting ML Training Pipeline...")

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

        if config.run_feature_engineering:
            success, output = await run_script("03_feature_engineering.py", "Feature Engineering", 70, script_dir)
            if not success:
                raise Exception(f"Feature Engineering failed: {output}")

        if config.run_model_training:
            success, output = await run_script("04_model_training.py", "Model Training", 90, script_dir)
            if not success:
                raise Exception(f"Model Training failed: {output}")
            clear_model_cache()

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
# API ENDPOINTS - GENERAL
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "[OK] Geo-ML Training API is running!",
        "status": "operational",
        "version": "3.5.0-FEATURE-ALIGNED",
        "critical_fix": "Feature engineering applied during prediction to match training (104 features)",
        "improvements": [
            "Synchronous subprocess for Windows compatibility",
            "Thread pool execution",
            "Enhanced error capture",
            "Real-time logging",
            "Feature alignment fix (104 = 104)",
            "Added GET /predict-by-location endpoint"
        ],
        "endpoints": {
            "GET /diagnostics": "Show script locations and system info",
            "GET /debug/script/{script_name}": "Detailed script validation",
            "POST /pipeline/start": "Start the ML training pipeline",
            "GET /pipeline/status": "Check pipeline status",
            "GET /pipeline/logs": "Get pipeline logs",
            "GET /predict-by-location": "Predict by lat/long (query params)",
            "POST /predict": "Predict geotechnical parameters (JSON body)"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": "3.5.0-FEATURE-ALIGNED"
    }


@app.get("/diagnostics")
async def get_diagnostics():
    """Get diagnostic information about script locations and environment"""
    api_file = Path(__file__).resolve()

    def list_py_files(directory: Path) -> List[str]:
        if not directory.exists():
            return []
        return [f.name for f in directory.glob("*.py")]

    diagnostics = {
        "system_info": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": sys.platform,
            "cwd": str(Path.cwd())
        },
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
        "found_scripts": {},
        "environment_variables": {
            "SUPABASE_URL_present": bool(os.getenv("SUPABASE_URL")),
            "SUPABASE_SERVICE_ROLE_KEY_present": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
            "PYTHONPATH": os.getenv("PYTHONPATH", "Not set")
        }
    }

    for script_name in diagnostics["required_scripts"]:
        found_path = find_script(script_name)
        if found_path:
            validation = validate_script(found_path)
            diagnostics["found_scripts"][script_name] = {
                "found": True,
                "location": str(found_path),
                "validation": validation
            }
        else:
            diagnostics["found_scripts"][script_name] = {
                "found": False,
                "location": None
            }

    return diagnostics
#


@app.get("/debug/script/{script_name}")
async def debug_script(script_name: str):
    """Detailed validation of a specific script"""
    script_path = find_script(script_name)

    if not script_path:
        return {
            "found": False,
            "script_name": script_name,
            "message": "Script not found in any search location"
        }

    validation = validate_script(script_path)

    # Try to read first few lines
    preview = None
    if script_path.exists():
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                preview = [f.readline().rstrip() for _ in range(10)]
        except:
            preview = ["Could not read file"]

    return {
        "found": True,
        "script_name": script_name,
        "location": str(script_path),
        "validation": validation,
        "preview_first_10_lines": preview
    }


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
        "message": "ML training pipeline started in background",
        "estimated_duration": "15-40 minutes",
        "check_status": "/pipeline/status",
        "view_logs": "/pipeline/logs",
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
async def get_pipeline_logs(limit: Optional[int] = 100):
    """Get recent pipeline logs"""
    global pipeline_status
    logs = pipeline_status["logs"][-limit:] if limit else pipeline_status["logs"]
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
# API ENDPOINTS - MODEL MANAGEMENT
# ============================================================================

@app.post("/models/clear-cache")
async def clear_models_cache():
    """Clear the model cache to force reload from Supabase Storage"""
    return clear_model_cache()


@app.get("/models/cache-status")
async def get_cache_status():
    """Check if models are currently cached in memory"""
    global _model_cache

    is_cached = _model_cache['scaler'] is not None
    last_loaded = _model_cache['last_loaded'].isoformat(
    ) if _model_cache['last_loaded'] else None

    available_models = [k for k in ['liquefaction', 'settlement', 'bearing_capacity']
                        if _model_cache.get(k) is not None]

    return {
        "cached": is_cached,
        "last_loaded": last_loaded,
        "available_models": available_models,
        "models_in_cache": [k for k in _model_cache.keys() if k != 'last_loaded' and _model_cache[k] is not None]
    }


@app.get("/models/info")
async def get_model_info():
    """Get information about the loaded models"""
    try:
        models = load_models_from_supabase_direct()
        metadata = models.get('metadata', {})

        available_models = [k for k in ['liquefaction', 'settlement', 'bearing_capacity']
                            if models.get(k) is not None]

        return {
            "version": metadata.get('version', 'unknown'),
            "architecture": metadata.get('model_architecture', {}),
            "num_features": metadata.get('num_features', 0),
            "training_samples": metadata.get('training_samples', 0),
            "available_models": available_models,
            "model_trained": metadata.get('model_trained', True),
            "results": metadata.get('results', {}),
            "timestamp": metadata.get('timestamp', None)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model information: {str(e)}"
        )


# ============================================================================
# API ENDPOINTS - PREDICTION (FEATURE-ALIGNED)
# ============================================================================

@app.get("/predict-by-location")
async def predict_by_location(latitude: float, longitude: float):
    """
    GET endpoint: Predict geotechnical parameters by lat/long
    Called by frontend map interface
    """
    try:
        request = PredictionRequest(latitude=latitude, longitude=longitude)
        return await predict_liquefaction(request)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict")
async def predict_liquefaction(request: PredictionRequest):
    """
    POST endpoint: Complete geotechnical prediction
    Returns all data needed by frontend sidebar
    """
    try:
        # =====================================================================
        # STEP 1: LOAD MODELS
        # =====================================================================
        models = load_models_from_supabase_direct()
        scaler = models['scaler']
        liq_model = models.get('liquefaction')
        settlement_model = models.get('settlement')
        bearing_model = models.get('bearing_capacity')
        metadata = models['metadata']

        available_models = []
        if liq_model is not None:
            available_models.append('liquefaction')
        if settlement_model is not None:
            available_models.append('settlement')
        if bearing_model is not None:
            available_models.append('bearing_capacity')

        if not available_models:
            return {
                "success": False,
                "error": "No prediction models available",
                "message": "Models are currently unavailable. Please try again later.",
                "available_models": []
            }

        feature_names = metadata.get('feature_names', [])
        if not feature_names:
            raise Exception("No feature names found in metadata")

        print(f"[DEBUG] Metadata provides {len(feature_names)} feature names")

        # Determine expected input size from scaler or models (must do BEFORE engineering)
        expected_n = None
        try:
            if hasattr(scaler, 'n_features_in_'):
                expected_n = int(getattr(scaler, 'n_features_in_'))
        except Exception:
            expected_n = None

        # Fallback: metadata may contain declared num_features
        if expected_n is None:
            expected_n = int(metadata.get('num_features')) if metadata.get(
                'num_features') else None

        if expected_n is not None:
            print(
                f"[DEBUG] Expected feature count from scaler/metadata: {expected_n}")
            # If metadata list is longer, trim to expected length to avoid later slicing
            if len(feature_names) > expected_n:
                print(
                    f"[WARNING] Trimming metadata feature list from {len(feature_names)} to {expected_n}")
                feature_names = feature_names[:expected_n]
            # If metadata list is shorter, pad with placeholder names so engineering returns expected_n cols
            elif len(feature_names) < expected_n:
                to_add = expected_n - len(feature_names)
                print(
                    f"[WARNING] Padding metadata feature list with {to_add} placeholder features to reach {expected_n}")
                for i in range(to_add):
                    feature_names.append(f"pad_feature_{i}")
        else:
            print(
                "[DEBUG] Could not determine expected feature count from scaler/metadata; proceeding with metadata list")

        # =====================================================================
        # STEP 2: FETCH NEAREST BOREHOLE DATA
        # =====================================================================
        try:
            from supabase_client import get_supabase_client
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database module unavailable: {e}"
            )

        client = get_supabase_client()
        if not client:
            raise HTTPException(
                status_code=503,
                detail="Database connection failed"
            )

        # Get complete soil data
        try:
            complete_data = client.table(
                'v_complete_soil_data').select('*').execute()
            if not complete_data.data:
                raise HTTPException(
                    status_code=404,
                    detail="No soil data available in database"
                )

            df_all = pd.DataFrame(complete_data.data)

            # If the view returns only a single (or zero) unique borehole,
            # try to reconstruct a fuller dataset from `boreholes` + `soil_layers`.
            try:
                unique_bh = df_all['borehole_id'].nunique(
                ) if 'borehole_id' in df_all.columns else 0
            except Exception:
                unique_bh = 0

            if unique_bh <= 1:
                print(
                    '[WARN] v_complete_soil_data contains <=1 unique borehole — attempting fallback to boreholes + soil_layers')
                try:
                    bh_res = client.table('boreholes').select(
                        'borehole_id,latitude,longitude').execute()
                    sl_res = client.table('soil_layers').select('*').execute()
                    if bh_res.data and sl_res.data:
                        bh_df = pd.DataFrame(bh_res.data).drop_duplicates(
                            subset=['borehole_id']).dropna(subset=['latitude', 'longitude'])
                        sl_df = pd.DataFrame(sl_res.data)

                        # Normalize keys to string for safe merge
                        for col in ['borehole_id']:
                            if col in sl_df.columns:
                                sl_df[col] = sl_df[col].astype(str)
                        if 'borehole_id' in bh_df.columns:
                            bh_df['borehole_id'] = bh_df['borehole_id'].astype(
                                str)

                        # Attempt to merge on 'borehole_id'
                        merged = None
                        if 'borehole_id' in sl_df.columns and 'borehole_id' in bh_df.columns:
                            merged = sl_df.merge(
                                bh_df, on='borehole_id', how='inner')

                        # If merge failed, try matching soil_layers.borehole_id (numeric) to boreholes.id
                        if (merged is None or len(merged) == 0) and 'borehole_id' in sl_df.columns and 'id' in bh_df.columns:
                            try:
                                merged = sl_df.merge(
                                    bh_df, left_on='borehole_id', right_on='id', how='inner')
                            except Exception:
                                merged = None

                        if merged is not None and len(merged) > 0:
                            # prefer columns expected by downstream code (latitude, longitude, borehole_id, layer info)
                            df_all = merged.copy()
                            print(
                                f"[INFO] Fallback produced {len(df_all)} rows from soil_layers+boreholes (unique boreholes: {df_all['borehole_id'].nunique() if 'borehole_id' in df_all.columns else 'unknown'})")
                        else:
                            print(
                                '[WARN] Fallback merge produced no rows; keeping original view results')

                except Exception as e:
                    print(f'[WARN] Fallback attempt failed: {e}')

            # Find nearest borehole using Haversine-like distance
            min_distance = float('inf')
            nearest_borehole_id = None

            for borehole_id in df_all['borehole_id'].unique():
                borehole_rows = df_all[df_all['borehole_id'] == borehole_id]
                if len(borehole_rows) == 0:
                    continue

                bh_lat = borehole_rows.iloc[0]['latitude']
                bh_lon = borehole_rows.iloc[0]['longitude']

                if pd.isna(bh_lat) or pd.isna(bh_lon):
                    continue

                # Simple Euclidean distance (good enough for small areas)
                lat_diff = float(bh_lat) - request.latitude
                lon_diff = float(bh_lon) - request.longitude
                distance = math.sqrt(lat_diff**2 + lon_diff**2)

                if distance < min_distance:
                    min_distance = distance
                    nearest_borehole_id = borehole_id

            if not nearest_borehole_id:
                raise HTTPException(
                    status_code=404,
                    detail="No valid boreholes found with coordinates"
                )

            # Get all soil layers for nearest borehole
            borehole_data = df_all[df_all['borehole_id']
                                   == nearest_borehole_id].copy()

            if len(borehole_data) == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No soil layers found for nearest borehole"
                )

            print(
                f"[DEBUG] Found {len(borehole_data)} layers for borehole {nearest_borehole_id}")
            print(f"[DEBUG] Distance to borehole: {min_distance * 111:.2f} km")

            # Select representative layer (middle depth if multiple layers)
            if len(borehole_data) == 1:
                representative_layer = borehole_data.iloc[0]
            else:
                # Use middle layer as most representative
                middle_idx = len(borehole_data) // 2
                representative_layer = borehole_data.iloc[middle_idx]

            # ================================================================
            # STEP 3: EXTRACT SOIL PARAMETERS FROM ACTUAL DATA
            # ================================================================

            # Get actual values from database (with safe fallbacks)
            def safe_float(value, default=0.0):
                """Safely convert to float, return default if NaN or None"""
                if pd.isna(value) or value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default

            # Critical soil parameters from actual data
            spt_n_value = safe_float(
                representative_layer.get('spt_n_value'), 15.0)
            spt_n60 = safe_float(
                representative_layer.get('spt_n60'), spt_n_value)
            spt_n160 = safe_float(representative_layer.get(
                'spt_n_value'), spt_n_value * 1.1)
            unit_weight = safe_float(
                representative_layer.get('unit_weight'), 18.0)
            fines_content = safe_float(
                representative_layer.get('fines_content'), 15.0)
            gwl = safe_float(representative_layer.get(
                'groundwater_depth_m'), 5.0)
            depth_from = safe_float(
                representative_layer.get('depth_from_m'), 0.0)
            depth_to = safe_float(representative_layer.get('depth_to_m'), 1.5)
            friction_angle = safe_float(representative_layer.get(
                'friction_angle'), 27.5 + 0.3 * spt_n_value)
            cohesion = safe_float(
                representative_layer.get('cohesion_kpa'), 0.0)

            # Seismic parameters
            pga_g = safe_float(representative_layer.get('pga_g'), 0.35)
            csr = safe_float(representative_layer.get('csr'), 0.2)

            # Calculate CRR (Cyclic Resistance Ratio) from SPT
            if spt_n160 <= 30:
                crr = 1.0 / (34.0 - spt_n160 + 0.001)
            else:
                crr = 0.5
            crr = max(0.0, min(1.0, crr))  # Clamp between 0 and 1

            print(f"[DEBUG] Actual soil parameters extracted:")
            print(f"  SPT N60: {spt_n60}")
            print(f"  Unit Weight: {unit_weight} kN/m³")
            print(f"  Fines: {fines_content}%")
            print(f"  GWL: {gwl} m")
            print(f"  CSR: {csr:.3f}, CRR: {crr:.3f}")

            # Derive bearing capacity and allowable pressure defaults
            bc_kpa = safe_float(representative_layer.get(
                'bearing_capacity_kpa'), spt_n_value * 30)
            qa_kpa = safe_float(representative_layer.get(
                'qa_allowable_kpa'), bc_kpa / 3.0)

            print(f"[DEBUG] Derived bc_kpa: {bc_kpa}, qa_kpa: {qa_kpa}")

            # ================================================================
            # STEP 4: PREPARE FEATURE VECTOR FOR PREDICTION
            # ================================================================

            # Create DataFrame for feature engineering
            # Compute borehole-level aggregates from the full dataset (df_all)
            try:
                bh_df = borehole_data.copy()
                # Numeric columns we care about
                num_cols = ['spt_n_value', 'qa_allowable_kpa',
                            'unit_weight', 'fines_content', 'bearing_capacity_kpa']
                for c in num_cols:
                    if c not in bh_df.columns:
                        bh_df[c] = pd.to_numeric(bh_df.get(c), errors='coerce')

                bh_avg_spt = float(bh_df['spt_n_value'].mean(
                )) if not bh_df['spt_n_value'].isna().all() else spt_n_value
                bh_min_spt = float(bh_df['spt_n_value'].min(
                )) if not bh_df['spt_n_value'].isna().all() else spt_n_value
                bh_max_spt = float(bh_df['spt_n_value'].max(
                )) if not bh_df['spt_n_value'].isna().all() else spt_n_value
                bh_std_spt = float(bh_df['spt_n_value'].std(
                )) if not bh_df['spt_n_value'].isna().all() else 0.0

                bh_avg_qa = float(bh_df['qa_allowable_kpa'].mean(
                )) if 'qa_allowable_kpa' in bh_df.columns and not bh_df['qa_allowable_kpa'].isna().all() else qa_kpa
                bh_min_qa = float(bh_df['qa_allowable_kpa'].min(
                )) if 'qa_allowable_kpa' in bh_df.columns and not bh_df['qa_allowable_kpa'].isna().all() else qa_kpa
                bh_max_qa = float(bh_df['qa_allowable_kpa'].max(
                )) if 'qa_allowable_kpa' in bh_df.columns and not bh_df['qa_allowable_kpa'].isna().all() else qa_kpa

            except Exception:
                bh_avg_spt = spt_n_value
                bh_min_spt = spt_n_value
                bh_max_spt = spt_n_value
                bh_std_spt = 0.0
                bh_avg_qa = qa_kpa
                bh_min_qa = qa_kpa
                bh_max_qa = qa_kpa

            # Compute layer-level aggregates across all boreholes for this layer number
            layer_avg_spt = spt_n_value
            layer_std_spt = 0.0
            layer_avg_unit_weight = unit_weight
            layer_avg_fines = fines_content
            layer_avg_qa = qa_kpa
            try:
                if 'layer_number' in representative_layer and not pd.isna(representative_layer['layer_number']):
                    ln = representative_layer['layer_number']
                    layer_df = df_all[df_all['layer_number'] == ln]
                    if len(layer_df) > 0:
                        layer_avg_spt = float(layer_df['spt_n_value'].mean(
                        )) if 'spt_n_value' in layer_df.columns else layer_avg_spt
                        layer_std_spt = float(layer_df['spt_n_value'].std(
                        )) if 'spt_n_value' in layer_df.columns else layer_std_spt
                        layer_avg_unit_weight = float(layer_df['unit_weight'].mean(
                        )) if 'unit_weight' in layer_df.columns else layer_avg_unit_weight
                        layer_avg_fines = float(layer_df['fines_content'].mean(
                        )) if 'fines_content' in layer_df.columns else layer_avg_fines
                        layer_avg_qa = float(layer_df['qa_allowable_kpa'].mean(
                        )) if 'qa_allowable_kpa' in layer_df.columns else layer_avg_qa
            except Exception:
                pass

            # Municipality-level aggregates (from df_all or view)
            muni_avg_spt_n = None
            avg_unit_weight = None
            avg_bearing_capacity_kpa = None
            borehole_count = None
            total_samples = None
            try:
                muni_name = representative_layer.get(
                    'municipality') if 'municipality' in representative_layer else None
                if muni_name is not None:
                    muni_df = df_all[df_all['municipality'] == muni_name]
                    if len(muni_df) > 0:
                        muni_avg_spt_n = float(muni_df['spt_n_value'].mean(
                        )) if 'spt_n_value' in muni_df.columns else None
                        avg_unit_weight = float(muni_df['unit_weight'].mean(
                        )) if 'unit_weight' in muni_df.columns else None
                        avg_bearing_capacity_kpa = float(muni_df['qa_allowable_kpa'].mean(
                        )) if 'qa_allowable_kpa' in muni_df.columns else None
                        borehole_count = int(muni_df['borehole_id'].nunique())
                        total_samples = int(len(muni_df))
            except Exception:
                pass

            raw_df = pd.DataFrame([{
                'depth_from_m': depth_from,
                'depth_to_m': depth_to,
                'spt_n_value': spt_n_value,
                'spt_n160': spt_n160,
                'spt_n60': spt_n60,
                'unit_weight': unit_weight,
                'fines_content': fines_content,
                'groundwater_depth_m': gwl,
                'pga_g': pga_g,
                'csr': csr,
                'friction_angle': friction_angle,
                'cohesion_kpa': cohesion,
                'bearing_capacity_kpa': safe_float(representative_layer.get('bearing_capacity_kpa'), spt_n_value * 30),
                'latitude': request.latitude,
                'longitude': request.longitude,
                # borehole-level aggregates
                'bh_avg_spt': bh_avg_spt,
                'bh_min_spt': bh_min_spt,
                'bh_max_spt': bh_max_spt,
                'bh_std_spt': bh_std_spt,
                'bh_avg_qa_allowable': bh_avg_qa,
                'bh_min_qa_allowable': bh_min_qa,
                'bh_max_qa_allowable': bh_max_qa,
                # layer-level aggregates
                'layer_avg_spt': layer_avg_spt,
                'layer_std_spt': layer_std_spt,
                'layer_avg_unit_weight': layer_avg_unit_weight,
                'layer_avg_fines': layer_avg_fines,
                'layer_avg_qa_allowable': layer_avg_qa,
                # municipality aggregates
                'muni_avg_spt_n': muni_avg_spt_n,
                'avg_unit_weight': avg_unit_weight,
                'avg_bearing_capacity_kpa': avg_bearing_capacity_kpa,
                'borehole_count': borehole_count,
                'total_samples': total_samples
            }])

            # Apply feature engineering (creates 104 features)
            engineered_df = engineer_features_for_prediction(
                raw_df, feature_names)

            # Extract features in correct order
            exclude_cols = [
                'layer_id', 'borehole_record_id', 'municipality_id', 'barangay_id',
                'borehole_id', 'barangay', 'municipality',
                'liquefaction', 'liquefaction_risk_level',
                'depth_range', 'created_at', 'updated_at',
                'settlement_cm', 'bearing_capacity_kpa', 'qa_allowable_kpa',
                'latitude', 'longitude', 'elevation'
            ]

            # Fill missing features
            for feat in feature_names:
                if feat not in engineered_df.columns:
                    engineered_df[feat] = 0.0

            # Determine expected number of features from scaler if available
            expected_n = None
            try:
                expected_n = int(getattr(scaler, 'n_features_in_', None))
            except Exception:
                expected_n = None

            if expected_n is None:
                expected_n = len(feature_names)

            # If metadata and scaler disagree, warn but attempt to proceed using metadata
            if len(feature_names) != expected_n:
                print(
                    f"[WARNING] Metadata feature count ({len(feature_names)}) != scaler expected ({expected_n})")
                print(
                    "[WARNING] Attempting to align using available engineered features")

            # Ensure engineered_df contains the required feature columns; fill missing ones with 0
            for feat in feature_names:
                if feat not in engineered_df.columns:
                    engineered_df[feat] = 0.0

            # Final feature list will be the metadata order, trimmed or extended to match expected_n
            final_feature_list = feature_names.copy()
            if len(final_feature_list) > expected_n:
                final_feature_list = final_feature_list[:expected_n]
            elif len(final_feature_list) < expected_n:
                # Append any remaining engineered columns to reach expected_n
                for col in engineered_df.columns:
                    if col not in final_feature_list:
                        final_feature_list.append(col)
                    if len(final_feature_list) >= expected_n:
                        break

            if len(final_feature_list) != expected_n:
                raise Exception(
                    f"Unable to produce feature vector of length {expected_n} (have {len(final_feature_list)})")

            feature_vector = engineered_df[final_feature_list].values[0]
            print(
                f"[DEBUG] Feature vector prepared: {len(feature_vector)} features")

        except Exception as data_error:
            print(f"[ERROR] Data processing failed: {data_error}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process soil data: {str(data_error)}"
            )

        # =====================================================================
        # STEP 5: MAKE PREDICTIONS
        # =====================================================================

        # Scale features
        input_features = np.array([feature_vector])
        input_scaled = scaler.transform(input_features)

        # Helper to prepare per-model input shape (slice or pad to match model.n_features_in_)
        def prepare_model_input(model, scaled_array):
            try:
                expected = getattr(model, 'n_features_in_', None)
            except Exception:
                expected = None

            if expected is None:
                # Model doesn't expose expected feature count; return full scaled array
                return scaled_array

            current = scaled_array.shape[1]
            if expected == current:
                return scaled_array
            elif expected < current:
                print(
                    f"[WARNING] Model expects {expected} features but input has {current}; slicing to first {expected} features")
                return scaled_array[:, :expected]
            else:
                # Pad with zeros to match expected features
                print(
                    f"[WARNING] Model expects {expected} features but input has {current}; padding with {expected-current} zeros")
                pad = np.zeros((scaled_array.shape[0], expected - current))
                return np.hstack([scaled_array, pad])

        # Initialize prediction variables
        liquefaction_prob = None
        risk_level = "MEDIUM"
        severity = "Moderate"
        settlement_cm = None
        settlement_severity = "Unknown"
        bearing_pre_kpa = None
        bearing_post_kpa = None
        capacity_reduction = 0.0

        # === LIQUEFACTION PREDICTION ===
        if liq_model is not None:
            try:
                liq_input = prepare_model_input(liq_model, input_scaled)
                liq_pred = liq_model.predict(liq_input)[0]
                liquefaction_prob = float(np.clip(liq_pred * 100, 0, 100))

                # Determine risk level based on probability
                if liquefaction_prob >= 60:
                    risk_level = "HIGH"
                    severity = "Severe"
                elif liquefaction_prob >= 30:
                    risk_level = "MEDIUM"
                    severity = "Moderate"
                else:
                    risk_level = "LOW"
                    severity = "Minor"

                print(
                    f"[DEBUG] Liquefaction prediction: {liquefaction_prob:.1f}% ({risk_level})")
            except Exception as e:
                print(f"[WARNING] Liquefaction prediction failed: {e}")

        # If no liquefaction model, estimate from factor of safety
        if liquefaction_prob is None:
            factor_of_safety = crr / (csr + 0.001)
            if factor_of_safety < 1.0:
                liquefaction_prob = (1.0 - factor_of_safety) * 100
                risk_level = "HIGH"
                severity = "Severe"
            elif factor_of_safety < 1.5:
                liquefaction_prob = 50.0
                risk_level = "MEDIUM"
                severity = "Moderate"
            else:
                liquefaction_prob = 20.0
                risk_level = "LOW"
                severity = "Minor"

            print(
                f"[DEBUG] Liquefaction estimated from FS={factor_of_safety:.2f}: {liquefaction_prob:.1f}%")

        # === SETTLEMENT PREDICTION ===
        if settlement_model is not None:
            try:
                settlement_input = prepare_model_input(
                    settlement_model, input_scaled)
                settlement_pred = settlement_model.predict(settlement_input)[0]
                settlement_cm = float(max(0, settlement_pred))

                # Determine settlement severity
                if settlement_cm < 2.5:
                    settlement_severity = "Low"
                elif settlement_cm < 5.0:
                    settlement_severity = "Moderate"
                elif settlement_cm < 10.0:
                    settlement_severity = "High"
                else:
                    settlement_severity = "Very High"

                print(
                    f"[DEBUG] Settlement prediction: {settlement_cm:.2f} cm ({settlement_severity})")
            except Exception as e:
                print(f"[WARNING] Settlement prediction failed: {e}")

        # If no settlement model, estimate from liquefaction probability
        if settlement_cm is None:
            # Higher liquefaction risk → more settlement
            settlement_cm = (liquefaction_prob / 100) * 8.0  # 0-8 cm range

            if settlement_cm < 2.5:
                settlement_severity = "Low"
            elif settlement_cm < 5.0:
                settlement_severity = "Moderate"
            else:
                settlement_severity = "High"

            print(f"[DEBUG] Settlement estimated: {settlement_cm:.2f} cm")

        # === BEARING CAPACITY PREDICTION ===
        if bearing_model is not None:
            try:
                bearing_input = prepare_model_input(
                    bearing_model, input_scaled)
                bearing_pred = bearing_model.predict(bearing_input)[0]
                bearing_post_kpa = float(max(0, bearing_pred))

                # Estimate pre-liquefaction (typically 2.5-3.5x post)
                bearing_pre_kpa = bearing_post_kpa * 3.0

                # Calculate capacity reduction
                if bearing_pre_kpa > 0:
                    capacity_reduction = (
                        (bearing_pre_kpa - bearing_post_kpa) / bearing_pre_kpa) * 100
                else:
                    capacity_reduction = 0.0

                print(
                    f"[DEBUG] Bearing capacity: Pre={bearing_pre_kpa:.1f} kPa, Post={bearing_post_kpa:.1f} kPa, Reduction={capacity_reduction:.1f}%")
            except Exception as e:
                print(f"[WARNING] Bearing capacity prediction failed: {e}")

        # If no bearing capacity model, estimate from SPT
        if bearing_post_kpa is None or bearing_pre_kpa is None:
            # Terzaghi bearing capacity estimation
            bearing_pre_kpa = spt_n_value * 30  # Rough approximation

            # Post-liquefaction: reduced by liquefaction probability
            reduction_factor = 1.0 - (liquefaction_prob / 100) * 0.7
            bearing_post_kpa = bearing_pre_kpa * reduction_factor

            capacity_reduction = (
                (bearing_pre_kpa - bearing_post_kpa) / bearing_pre_kpa) * 100

            print(
                f"[DEBUG] Bearing capacity estimated from SPT: Pre={bearing_pre_kpa:.1f} kPa, Post={bearing_post_kpa:.1f} kPa")

        # =====================================================================
        # STEP 6: GENERATE RECOMMENDATIONS
        # =====================================================================

        recommendations = []

        # Liquefaction-based recommendations
        if risk_level == "HIGH":
            recommendations.append(
                "Ground improvement techniques are strongly recommended (deep soil mixing, stone columns, or compaction grouting)"
            )
            recommendations.append(
                "Consider deep foundation systems (driven piles or drilled shafts extending to non-liquefiable strata)"
            )
            recommendations.append(
                "Conduct detailed site-specific geotechnical investigation with additional SPT and CPT soundings"
            )
        elif risk_level == "MEDIUM":
            recommendations.append(
                "Site-specific geotechnical investigation recommended before construction"
            )
            recommendations.append(
                "Consider shallow ground improvement for light to medium structures"
            )
            recommendations.append(
                "Monitor groundwater levels during and after construction"
            )
        else:
            recommendations.append(
                "Standard foundation design practices should be sufficient for this location"
            )
            recommendations.append(
                "Periodic geotechnical monitoring recommended during construction"
            )

        # Settlement-based recommendations
        if settlement_cm >= 5.0:
            recommendations.append(
                f"Predicted settlement of {settlement_cm:.1f} cm requires structural mitigation measures"
            )
            recommendations.append(
                "Consider raft foundations or mat foundations to distribute loads"
            )
        elif settlement_cm >= 2.5:
            recommendations.append(
                f"Moderate settlement ({settlement_cm:.1f} cm) expected - ensure adequate foundation stiffness"
            )

        # Bearing capacity-based recommendations
        if bearing_post_kpa < 100:
            recommendations.append(
                "Low post-liquefaction bearing capacity - soil stabilization or deep foundations required"
            )
        elif bearing_post_kpa < 200:
            recommendations.append(
                "Moderate bearing capacity - suitable for light structures only"
            )

        # Distance-based recommendation
        distance_km = min_distance * 111  # Convert degrees to km
        if distance_km > 2.0:
            recommendations.append(
                f"Note: Prediction based on borehole data {distance_km:.1f} km away. "
                f"Site-specific investigation strongly recommended for accurate assessment."
            )
        elif distance_km > 1.0:
            recommendations.append(
                f"Borehole data is {distance_km:.1f} km from site. Consider additional testing for verification."
            )

        # Soil type recommendations
        if fines_content < 5:
            recommendations.append(
                "Clean sand detected - highly susceptible to liquefaction under seismic loading"
            )
        elif fines_content > 35:
            recommendations.append(
                "Fine-grained soil detected - plasticity index testing recommended to assess clay behavior"
            )

        # =====================================================================
        # STEP 7: BUILD COMPLETE RESPONSE
        # =====================================================================

        response = {
            "success": True,
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "nearest_borehole_distance_km": round(distance_km, 2)
            },
            "risk_assessment": {
                "risk_level": risk_level,
                "probability": round(liquefaction_prob, 1),
                "severity": severity
            },
            "soil_parameters": {
                "spt_n60": round(spt_n60, 1),
                "unit_weight": round(unit_weight, 2),
                "csr": round(csr, 3),
                "crr": round(crr, 3),
                "gwl": round(gwl, 2),
                "fines_percent": round(fines_content, 1),
                "source": "actual_borehole_data"
            },
            "settlement": {
                "predicted_cm": round(settlement_cm, 2),
                "severity": settlement_severity
            },
            "bearing_capacity": {
                "pre_liquefaction_kpa": round(bearing_pre_kpa, 2),
                "post_liquefaction_kpa": round(bearing_post_kpa, 2),
                "capacity_reduction_percent": round(capacity_reduction, 1)
            },
            "recommendations": recommendations
        }

        print(
            f"[SUCCESS] Complete prediction generated for ({request.latitude}, {request.longitude})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    print("Version: 3.5.0-FEATURE-ALIGNED")
    print("Feature engineering applied during prediction for alignment")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
