#!/usr/bin/env python3
"""
Enhanced API with Spatial Interpolation
Uses multiple boreholes with inverse distance weighting for location-specific predictions
"""

import os
import io
import json
import sys
import asyncio
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from supabase import create_client
    from sklearn.preprocessing import StandardScaler
    import joblib
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("[ERROR] Required dependencies not installed")

try:
    from pipeline import GeotechnicalPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"[WARNING] Pipeline module not available: {e}")

try:
    from train_ann import MultiOutputANNTraining
    TRAINING_AVAILABLE = True
except ImportError as e:
    TRAINING_AVAILABLE = False
    print(f"[WARNING] Training module not available: {e}")

app = FastAPI(
    title="Geotechnical Prediction API - Spatial Interpolation",
    description="API with multi-borehole spatial interpolation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
_client = None
_scaler = None
_multi_model = None
_liq_model = None
_settle_model = None
_bearing_model = None
_model_metadata = None

# Pipeline and training status
_pipeline_status = {"status": "idle", "message": "", "timestamp": None}
_training_status = {"status": "idle", "message": "", "timestamp": None}

# ── Prediction cache ───────────────────────────────────────────────────────
# Key: (lat_r, lon_r, q_actual_r, magnitude_r)  — rounded to reduce key space
# Value: (result, timestamp)
_prediction_cache: dict = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour

def _cache_key(lat: float, lon: float, q_actual: float, magnitude: float) -> tuple:
    """Round inputs to 4 dp / 1 dp to group near-identical requests."""
    return (round(lat, 4), round(lon, 4), round(q_actual, 1), round(magnitude, 1))

def _cache_get(key: tuple):
    """Return cached result if still fresh, else None."""
    import time
    entry = _prediction_cache.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL_SECONDS:
        print(f"[CACHE] HIT {key}")
        return entry["result"]
    return None

def _cache_set(key: tuple, result) -> None:
    import time
    _prediction_cache[key] = {"result": result, "ts": time.time()}
    # Evict oldest entries if cache grows too large (keep last 200)
    if len(_prediction_cache) > 200:
        oldest = sorted(_prediction_cache, key=lambda k: _prediction_cache[k]["ts"])
        for k in oldest[:50]:
            del _prediction_cache[k]
    print(f"[CACHE] SET {key}  (cache size: {len(_prediction_cache)})")

# Populated dynamically from ann_metadata.json on startup
EXPECTED_FEATURES: list = []


_API_SECRET_KEY = os.getenv("API_SECRET_KEY")
if not _API_SECRET_KEY:
    raise RuntimeError("API_SECRET_KEY environment variable is not set")


async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    """Dependency: reject requests with a missing or wrong API key."""
    if x_api_key != _API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid API key.")


class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    depth_m: Optional[float] = None
    municipality: Optional[str] = None
    q_actual: Optional[float] = 50.0    # kPa — building contact pressure
    magnitude: Optional[float] = 6.5   # Mw — earthquake magnitude


class PredictionResponse(BaseModel):
    location: Dict
    risk_assessment: Dict
    soil_parameters: Dict
    settlement: Dict
    bearing_capacity: Dict
    foundation_recommendation: Optional[Dict] = None
    recommendations: List[str]
    analysis_parameters: Optional[Dict] = None
    interpolation_info: Optional[Dict] = None


def get_supabase_client():
    """Get or create Supabase client"""
    global _client
    if _client is None:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

        _client = create_client(supabase_url, supabase_key)
    return _client


def load_model():
    global _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata
    global EXPECTED_FEATURES

    if _scaler is not None:
        return _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata

    print("[INFO] Loading no-leakage models from Supabase Storage...")

    client = get_supabase_client()
    bucket_name = os.getenv("SUPABASE_STORAGE_BUCKET", "geotechnical-data")

    try:
        scaler_data = client.storage.from_(bucket_name).download(
            "models/scaler_no_leakage.pkl"
        )
        _scaler = joblib.load(io.BytesIO(scaler_data))
        print(f"  [OK] No-leakage scaler loaded")

        multi_data = client.storage.from_(bucket_name).download(
            "models/ann_multi_output_BD_no_leakage.pkl"
        )
        _multi_model = joblib.load(io.BytesIO(multi_data))
        print("  [OK] No-leakage B/D model loaded")

        _liq_model = None
        _settle_model = None
        _bearing_model = None

        metadata_data = client.storage.from_(bucket_name).download(
            "models/ann_metadata_no_leakage.json"
        )
        _model_metadata = json.loads(metadata_data.decode("utf-8"))

        EXPECTED_FEATURES = _model_metadata.get("feature_names", [])

        if not EXPECTED_FEATURES:
            raise RuntimeError("No feature_names found in ann_metadata_no_leakage.json")

        print(f"  [OK] Expected features loaded: {EXPECTED_FEATURES}")

        return _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata

    except Exception as e:
        print(f"  [ERROR] Failed to load no-leakage models: {e}")
        raise


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula (km)"""
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def find_nearest_boreholes(latitude: float, longitude: float, n: int = 5, max_distance_km: float = 100) -> List[Tuple[Dict, float]]:
    """
    Find the N nearest boreholes with their distances
    Returns: List of (borehole_data, distance_km) tuples
    """
    client = get_supabase_client()

    try:
        # Get all boreholes
        query = client.table('boreholes').select('*').execute()

        if not query.data:
            print("[WARNING] No boreholes found in database")
            return []

        # Calculate distances for all boreholes
        borehole_distances = []
        for borehole in query.data:
            bh_lat = float(borehole.get('latitude', 0))
            bh_lon = float(borehole.get('longitude', 0))

            distance_km = haversine_distance(
                latitude, longitude, bh_lat, bh_lon)

            if distance_km <= max_distance_km:
                borehole_distances.append((borehole, distance_km))

        # Sort by distance and take top N
        borehole_distances.sort(key=lambda x: x[1])
        nearest = borehole_distances[:n]

        print(
            f"[INFO] Found {len(nearest)} boreholes within {max_distance_km}km")
        for bh, dist in nearest:
            print(f"  - {bh.get('borehole_id')}: {dist:.2f} km")

        return nearest

    except Exception as e:
        print(f"[ERROR] Failed to find nearest boreholes: {e}")
        return []


def get_borehole_soil_data(borehole_id: str, depth_m: Optional[float] = None) -> List[Dict]:
    """Get soil data for a specific borehole"""
    client = get_supabase_client()

    try:
        query = client.table('v_complete_soil_data').select(
            '*').eq('borehole_id', borehole_id)

        if depth_m:
            query = query.gte('depth_from_m', depth_m -
                              2.0).lte('depth_to_m', depth_m + 2.0)

        result = query.limit(20).execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"[WARNING] Failed to query soil data for {borehole_id}: {e}")
        return []


def get_borehole_all_layers(borehole_uuid: int) -> List[Dict]:
    """Get all soil layers for a borehole by its primary key (bigint).
    Queries soil_layers directly to access all columns including cyclic_strength_ratio."""
    client = get_supabase_client()
    try:
        result = client.table('soil_layers').select(
            'layer_number, depth_from_m, depth_to_m, '
            'spt_n_value, unit_weight, fines_content, groundwater_depth_m, '
            'pga_g, csr, cyclic_strength_ratio, friction_angle, cohesion_kpa, '
            'elastic_modulus_es, liquefaction_risk_level, liquefaction'
        ).eq('borehole_id', borehole_uuid).order('layer_number').execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"[WARNING] Failed to get layers for borehole {borehole_uuid}: {e}")
        return []


def interpolate_soil_parameters(
    latitude: float,
    longitude: float,
    depth_m: Optional[float] = None,
    n_boreholes: int = 5
) -> Tuple[List[Dict], Dict]:
    """
    Interpolate soil parameters per layer using IDW.
    Returns: (interpolated_layers, interpolation_info)
    - interpolated_layers: one dict per layer_number with IDW-blended params
    - interpolation_info:  metadata about the interpolation
    """
    _RISK_ORD  = {'VERY HIGH': 5, 'HIGH': 4, 'MEDIUM': 3, 'LOW': 2, 'VERY LOW': 1}
    _RISK_PROB = {'VERY HIGH': 90.0, 'HIGH': 75.0, 'MEDIUM': 45.0, 'LOW': 15.0, 'VERY LOW': 5.0}

    nearest_boreholes = find_nearest_boreholes(latitude, longitude, n=n_boreholes)
    if not nearest_boreholes:
        print("[WARNING] No boreholes found for interpolation")
        return None, None

    # Collect all layers per borehole with their IDW weights
    borehole_data = []
    for borehole, distance_km in nearest_boreholes:
        bh_uuid  = borehole.get('id')
        bh_label = borehole.get('borehole_id')
        layers   = get_borehole_all_layers(bh_uuid)
        if not layers:
            continue

        risk_levels = [l.get('liquefaction_risk_level') for l in layers if l.get('liquefaction_risk_level')]
        worst_risk  = max(risk_levels, key=lambda r: _RISK_ORD.get(r, 0)) if risk_levels else 'LOW'
        weight = 1.0 / max(distance_km, 0.001) ** 2

        borehole_data.append({
            'borehole_id':    bh_label,
            'borehole_uuid':  bh_uuid,
            'distance_km':    distance_km,
            'weight':         weight,
            'layers':         layers,
            'risk_probability': _RISK_PROB.get(worst_risk, 15.0),
        })

    if not borehole_data:
        print("[WARNING] No soil data found in any nearby boreholes")
        return None, None

    total_weight = sum(bd['weight'] for bd in borehole_data)
    for bd in borehole_data:
        bd['norm_weight'] = bd['weight'] / total_weight

    # Site-level IDW-blended risk probability (used for overall risk blending in predict)
    site_risk_prob = sum(bd['risk_probability'] * bd['norm_weight'] for bd in borehole_data)

    # Gather all unique layer numbers across all boreholes
    all_layer_nums = sorted(set(
        l['layer_number'] for bd in borehole_data for l in bd['layers']
    ))

    def idw_field(layer_num, key, default):
        """IDW-blend a numeric field for a specific layer_number across boreholes."""
        vals = []
        for bd in borehole_data:
            lyr = next((l for l in bd['layers'] if l['layer_number'] == layer_num), None)
            if lyr and lyr.get(key) is not None:
                try:
                    vals.append((float(lyr[key]), bd['norm_weight']))
                except (TypeError, ValueError):
                    pass
        if not vals:
            return default
        total_w = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / total_w if total_w > 0 else default

    # Build one interpolated layer dict per layer_number
    interpolated_layers = []
    for ln in all_layer_nums:
        depth_from = idw_field(ln, 'depth_from_m', (ln - 1) * 1.5)
        depth_to   = idw_field(ln, 'depth_to_m',   ln * 1.5)
        thickness  = max(0.1, depth_to - depth_from)

        # Worst stored risk for this layer number across all boreholes
        layer_risks = [
            l.get('liquefaction_risk_level')
            for bd in borehole_data
            for l in bd['layers']
            if l['layer_number'] == ln and l.get('liquefaction_risk_level')
        ]
        worst_layer_risk = max(layer_risks, key=lambda r: _RISK_ORD.get(r, 0)) if layer_risks else 'LOW'

        crr_val = idw_field(ln, 'cyclic_strength_ratio', 0.0)

        interpolated_layers.append({
            'layer_number':          ln,
            'depth_from_m':          round(depth_from, 2),
            'depth_to_m':            round(depth_to, 2),
            'layer_thickness':       round(thickness, 2),
            'spt_n_value':           idw_field(ln, 'spt_n_value', 20),
            'spt_n60':               idw_field(ln, 'spt_n_value', 20),  # proxy (spt_n60 dropped from DB)
            'unit_weight':           idw_field(ln, 'unit_weight', 18),
            'fines_content':         idw_field(ln, 'fines_content', 10),
            'groundwater_depth_m':   idw_field(ln, 'groundwater_depth_m', 5),
            'pga_g':                 idw_field(ln, 'pga_g', 0.3),
            'csr':                   idw_field(ln, 'csr', 0.2),
            'crr':                   crr_val,
            'cyclic_strength_ratio': crr_val,
            'friction_angle':        idw_field(ln, 'friction_angle', 30.0),
            'cohesion_kpa':          idw_field(ln, 'cohesion_kpa', 0.0),
            'elastic_modulus_es':    idw_field(ln, 'elastic_modulus_es', 10000),
            'liquefaction_risk_level': worst_layer_risk,
            'risk_probability':      site_risk_prob,
        })

    # Conservative PGA/CSR adjustment for distant sites
    nearest_distance = borehole_data[0]['distance_km']
    if nearest_distance > 10:
        distance_factor = min(1.2, 1.0 + (nearest_distance - 10) / 100)
        for lyr in interpolated_layers:
            lyr['pga_g'] *= distance_factor
            lyr['csr']   *= distance_factor

    normalized_weights = [bd['norm_weight'] for bd in borehole_data]
    interpolation_info = {
        'method': 'Inverse Distance Weighting (IDW) — per layer',
        'power': 2,
        'boreholes_used': len(borehole_data),
        'layers_interpolated': len(interpolated_layers),
        'nearest_distance_km': round(nearest_distance, 2),
        'farthest_distance_km': round(borehole_data[-1]['distance_km'], 2),
        'borehole_contributions': [
            {
                'id': bd['borehole_id'],
                'distance_km': round(bd['distance_km'], 2),
                'weight': round(bd['norm_weight'], 3)
            }
            for bd in borehole_data
        ],
        'confidence': calculate_interpolation_confidence(nearest_distance, len(borehole_data)),
        '_nearest_borehole_uuid':  borehole_data[0].get('borehole_uuid'),
        '_nearest_borehole_label': borehole_data[0]['borehole_id'],
    }

    print(f"[INFO] Interpolated {len(interpolated_layers)} layers from {len(borehole_data)} boreholes")
    print(f"[INFO] Nearest: {borehole_data[0]['borehole_id']} ({nearest_distance:.2f} km)")
    print(f"[INFO] Confidence: {interpolation_info['confidence']}")

    return interpolated_layers, interpolation_info


def calculate_interpolation_confidence(nearest_distance_km: float, n_boreholes: int) -> str:
    """
    Calculate confidence level based on distance and number of boreholes
    """
    # Distance penalty
    if nearest_distance_km < 1:
        distance_score = 100
    elif nearest_distance_km < 5:
        distance_score = 90
    elif nearest_distance_km < 10:
        distance_score = 70
    elif nearest_distance_km < 20:
        distance_score = 50
    elif nearest_distance_km < 50:
        distance_score = 30
    else:
        distance_score = 10

    # Number of boreholes bonus
    borehole_bonus = min(20, n_boreholes * 4)

    total_score = min(100, distance_score + borehole_bonus)

    if total_score >= 80:
        return "High"
    elif total_score >= 60:
        return "Medium"
    elif total_score >= 40:
        return "Low"
    else:
        return "Very Low"


def get_nearest_borehole_db_risk(borehole_uuid, borehole_label: str) -> Optional[Dict]:
    """
    Fetch the worst-case liquefaction risk classification directly from stored soil_layers
    for a specific borehole (looked up by its UUID primary key).

    Used to override the ML model result when the query point is within 500m of a known
    borehole, so the sidebar and map popup show consistent classifications.
    """
    RISK_ORDER = {'VERY HIGH': 5, 'HIGH': 4, 'MEDIUM': 3, 'LOW': 2, 'VERY LOW': 1}
    # Map 5-level DPWH scale → 3-level ML scale used by the sidebar
    RISK_TO_ML  = {'VERY HIGH': 'HIGH',   'HIGH': 'HIGH',
                   'MEDIUM':    'MEDIUM',
                   'LOW':       'LOW',    'VERY LOW': 'LOW'}
    RISK_PROB   = {'VERY HIGH': 90.0, 'HIGH': 75.0, 'MEDIUM': 45.0, 'LOW': 15.0, 'VERY LOW': 5.0}
    RISK_SEV    = {'VERY HIGH': 'Severe', 'HIGH': 'Severe',
                   'MEDIUM': 'Moderate',
                   'LOW': 'Minor', 'VERY LOW': 'Minor'}
    try:
        client = get_supabase_client()
        result = client.table('soil_layers').select(
            'liquefaction_risk_level'
        ).eq('borehole_id', borehole_uuid).execute()

        if not result.data:
            return None

        levels = [
            l.get('liquefaction_risk_level')
            for l in result.data
            if l.get('liquefaction_risk_level')
        ]
        if not levels:
            return None

        worst = max(levels, key=lambda r: RISK_ORDER.get(r, 0))
        print(f"[INFO] DB risk override — {borehole_label}: worst layer = {worst}")
        return {
            'db_risk_level':  worst,                          # 5-level (e.g. VERY HIGH)
            'ml_risk_level':  RISK_TO_ML.get(worst, 'LOW'),  # 3-level for sidebar
            'probability':    RISK_PROB.get(worst, 50.0),
            'severity':       RISK_SEV.get(worst, 'Minor'),
            'borehole_label': borehole_label,
            'source':         f'Database — {borehole_label} (DPWH BSDS)',
        }
    except Exception as e:
        print(f"[WARNING] DB risk lookup failed for {borehole_label}: {e}")
        return None


def engineer_features_from_interpolated(
    interpolated_params: Dict,
    latitude: float,
    longitude: float,
    depth_m: Optional[float] = None
) -> pd.DataFrame:
    global EXPECTED_FEATURES

    if not EXPECTED_FEATURES:
        raise RuntimeError("EXPECTED_FEATURES is empty. Metadata was not loaded.")

    target_depth = float(depth_m if depth_m is not None else interpolated_params.get("depth_mid_m", 1.5))

    spt_n_value = float(interpolated_params.get("spt_n_value", 20.0))
    spt_n60 = float(interpolated_params.get("spt_n60", spt_n_value))
    spt_n160 = float(interpolated_params.get("spt_n160", spt_n60))

    unit_weight = float(interpolated_params.get("unit_weight", 18.0))
    fines_content = float(interpolated_params.get("fines_content", 10.0))
    groundwater_depth_m = float(interpolated_params.get("groundwater_depth_m", 5.0))

    derived = {
        "spt_n_value": spt_n_value,
        "spt_n60": spt_n60,
        "spt_n160": spt_n160,
        "unit_weight": unit_weight,
        "fines_content": fines_content,
        "friction_angle": float(interpolated_params.get("friction_angle", 30.0)),
        "depth_mid_m": target_depth,
        "depth_from_m": float(interpolated_params.get("depth_from_m", target_depth - 0.75)),
        "depth_to_m": float(interpolated_params.get("depth_to_m", target_depth + 0.75)),
        "groundwater_depth_m": groundwater_depth_m,
        "moisture_content": float(interpolated_params.get("moisture_content", fines_content)),
        "plasticity_index": float(interpolated_params.get("plasticity_index", 0.0)),
        "mean_particle_size_d50": float(interpolated_params.get("mean_particle_size_d50", 0.1)),
        "relative_density_percent": float(
            interpolated_params.get(
                "relative_density_percent",
                min(100.0, max(0.0, (spt_n60 / 60.0) * 100.0))
            )
        ),
        "elastic_modulus_es": float(interpolated_params.get("elastic_modulus_es", 10000.0)),
        "pga_g": float(interpolated_params.get("pga_g", 0.3)),
    }

    # IMPORTANT: only use features that the no-leakage model was trained on
    features = {feature: derived.get(feature, 0.0) for feature in EXPECTED_FEATURES}

    features_df = pd.DataFrame([features], columns=EXPECTED_FEATURES)

    print(f"[DEBUG] Expected features: {EXPECTED_FEATURES}")
    print(f"[DEBUG] Input features: {features_df.columns.tolist()}")

    return features_df


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    if DEPENDENCIES_AVAILABLE:
        try:
            load_model()
            print("[OK] Models loaded successfully on startup")
        except Exception as e:
            print(f"[ERROR] Failed to load models on startup: {e}")


def run_pipeline_background():
    """Run pipeline in background"""
    global _pipeline_status
    _pipeline_status = {"status": "running", "message": "Pipeline started",
                        "timestamp": datetime.now().isoformat()}
    print("[INFO] Starting pipeline...")

    try:
        if not PIPELINE_AVAILABLE:
            raise Exception("Pipeline module not available")

        pipeline = GeotechnicalPipeline()
        success = pipeline.run()

        if success:
            _pipeline_status = {
                "status": "completed",
                "message": "Pipeline completed successfully",
                "timestamp": datetime.now().isoformat()
            }
            print("[OK] Pipeline completed successfully")
        else:
            _pipeline_status = {
                "status": "failed",
                "message": "Pipeline execution failed",
                "timestamp": datetime.now().isoformat()
            }
            print("[ERROR] Pipeline failed")
    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        _pipeline_status = {
            "status": "failed",
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        print(f"[ERROR] {error_msg}")


def run_training_background():
    """Run training in background"""
    global _training_status
    _training_status = {"status": "running", "message": "Training started",
                        "timestamp": datetime.now().isoformat()}
    print("[INFO] Starting model training...")

    try:
        if not TRAINING_AVAILABLE:
            raise Exception("Training module not available")

        trainer = MultiOutputANNTraining()
        success = trainer.run()

        if success:
            # Reload models after training
            global _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata
            _scaler = None
            _multi_model = None
            _liq_model = None
            _settle_model = None
            _bearing_model = None
            _model_metadata = None

            try:
                load_model()
                print("[OK] Models reloaded after training")
            except Exception as e:
                print(f"[WARNING] Failed to reload models: {e}")

            _training_status = {
                "status": "completed",
                "message": "Training completed successfully",
                "timestamp": datetime.now().isoformat()
            }
            print("[OK] Training completed successfully")
        else:
            _training_status = {
                "status": "failed",
                "message": "Training execution failed",
                "timestamp": datetime.now().isoformat()
            }
            print("[ERROR] Training failed")
    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        _training_status = {
            "status": "failed",
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        print(f"[ERROR] {error_msg}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Geotechnical Prediction API - Spatial Interpolation",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": _multi_model is not None or _liq_model is not None,
        "features": "Multi-borehole IDW interpolation for location-specific predictions"
    }


@app.get("/health")
@app.head("/health")  # Add this line
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": _multi_model is not None or _liq_model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/pipeline/start")
async def pipeline_start(background_tasks: BackgroundTasks, _: None = Depends(verify_api_key)):
    """Start geotechnical data processing pipeline"""
    if not PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Pipeline module not available")

    if _pipeline_status["status"] == "running":
        raise HTTPException(
            status_code=409, detail="Pipeline is already running")

    background_tasks.add_task(run_pipeline_background)

    return {
        "message": "Pipeline started",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/pipeline/status")
async def pipeline_status(_: None = Depends(verify_api_key)):
    """Get pipeline execution status"""
    return _pipeline_status


@app.post("/train/start")
async def train_start(background_tasks: BackgroundTasks, _: None = Depends(verify_api_key)):
    """Start model training"""
    if not TRAINING_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Training module not available")

    if _training_status["status"] == "running":
        raise HTTPException(
            status_code=409, detail="Training is already running")

    background_tasks.add_task(run_training_background)

    return {
        "message": "Training started",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/train/status")
async def train_status(_: None = Depends(verify_api_key)):
    """Get training execution status"""
    return _training_status


def run_full_workflow_background():
    """Run pipeline then training sequentially"""
    global _pipeline_status, _training_status

    try:
        # Run pipeline
        print("[INFO] Starting full workflow: pipeline -> training")
        run_pipeline_background()

        # Check if pipeline succeeded
        if _pipeline_status["status"] != "completed":
            print("[ERROR] Pipeline failed, skipping training")
            _training_status = {
                "status": "skipped",
                "message": "Skipped due to pipeline failure",
                "timestamp": datetime.now().isoformat()
            }
            return

        print("[INFO] Pipeline completed, starting training...")

        # Run training
        run_training_background()

    except Exception as e:
        error_msg = f"Workflow error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        if _pipeline_status["status"] != "completed":
            _pipeline_status = {
                "status": "failed",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        else:
            _training_status = {
                "status": "failed",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }


@app.post("/pipeline-and-train/start")
async def pipeline_and_train_start(background_tasks: BackgroundTasks, _: None = Depends(verify_api_key)):
    """Start pipeline followed by training (complete workflow)"""
    if not PIPELINE_AVAILABLE or not TRAINING_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Pipeline or training module not available")
    if _pipeline_status["status"] == "running" or _training_status["status"] == "running":
        raise HTTPException(
            status_code=409, detail="Pipeline or training is already running")

    background_tasks.add_task(run_full_workflow_background)

    return {
        "message": "Pipeline and training workflow started",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "note": "Check /pipeline/status and /train/status for progress"
    }


@app.post("/predict", response_model=PredictionResponse)
@app.get("/predict-by-location", response_model=PredictionResponse)
async def predict(
    request: Optional[PredictionRequest] = None,
    latitude: Optional[float] = Query(None, ge=-90, le=90),
    longitude: Optional[float] = Query(None, ge=-180, le=180),
    depth: Optional[float] = Query(None, ge=0, le=100),
    municipality: Optional[str] = Query(None),
    n_boreholes: Optional[int] = Query(
        5, ge=1, le=10, description="Number of nearest boreholes to use for interpolation"),
    q_actual: Optional[float] = Query(
        50.0, ge=0, description="Actual building contact pressure (kPa)"),
    magnitude: Optional[float] = Query(
        6.5, ge=4.0, le=10.0, description="Earthquake moment magnitude (Mw)"),
    _: None = Depends(verify_api_key),
):
    """
    Predict liquefaction using spatial interpolation from multiple boreholes.

    Extra analysis parameters:
    - **q_actual**: Actual building contact pressure in kPa (default 50)
    - **magnitude**: Earthquake moment magnitude Mw (default 6.5)
    """
    if not DEPENDENCIES_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Dependencies not available")

    # Handle both POST and GET
    if request:
        lat       = request.latitude
        lon       = request.longitude
        depth_m   = request.depth_m
        munic     = request.municipality
        q_actual  = request.q_actual  if request.q_actual  is not None else q_actual
        magnitude = request.magnitude if request.magnitude is not None else magnitude
    else:
        lat, lon, depth_m, munic = latitude, longitude, depth, municipality

    # Ensure defaults
    q_actual  = float(q_actual  or 50.0)
    magnitude = float(magnitude or 6.5)

    if lat is None or lon is None:
        raise HTTPException(
            status_code=400, detail="Latitude and longitude required")

    # ── Cache lookup ───────────────────────────────────────────────────────
    ck = _cache_key(lat, lon, q_actual, magnitude)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    try:
        scaler, multi_model, liq_model, settle_model, bearing_model, metadata = load_model()

        # Interpolate soil parameters from multiple boreholes
        interpolated_params, interpolation_info = interpolate_soil_parameters(
            lat, lon, depth_m, n_boreholes=n_boreholes
        )

        interpolated_layers = interpolated_params  # rename: now a list of per-layer dicts

        if not interpolated_layers:
            raise HTTPException(
                status_code=404,
                detail="No borehole data found within 100km radius"
            )

        # ── Magnitude Scaling Factor (Idriss 1999) — needed before per-layer FS ─
        import math as _math
        msf = (10 ** 2.24) / (magnitude ** 2.56)

        # ── Per-layer FS computation ───────────────────────────────────────────
        _RISK_ORD_3 = {'VERY HIGH': 5, 'HIGH': 4, 'MEDIUM': 3, 'LOW': 2, 'VERY LOW': 1}
        for lyr in interpolated_layers:
            crr_l = float(lyr.get('cyclic_strength_ratio') or lyr.get('crr') or 0)
            csr_l = float(lyr.get('csr') or 0)
            crr_l = crr_l if crr_l > 0 else 0.3
            csr_l = csr_l if csr_l > 0 else 0.2
            lyr['fs'] = (crr_l * msf) / (csr_l + 1e-9)

        # ── Critical layer: lowest FS (most liquefiable) ───────────────────────
        critical_layer = min(interpolated_layers, key=lambda l: l['fs'])

        # Use critical layer as representative params for ANN + bearing capacity
        interpolated_params = critical_layer

        # Resolve depth_m from critical layer if not supplied
        if depth_m is None:
            depth_m = float(critical_layer.get('depth_to_m') or 1.5)

        # Engineer features from critical layer (ANN input)
        features_df = engineer_features_from_interpolated(
            interpolated_params, lat, lon, depth_m)

        print(f"[DEBUG] Features shape: {features_df.shape}")
        print(f"[DEBUG] Critical layer: #{critical_layer['layer_number']} "
              f"({critical_layer['depth_from_m']}–{critical_layer['depth_to_m']} m), "
              f"FS={critical_layer['fs']:.3f}")

        # Scale and predict
        features_scaled = scaler.transform(features_df)

        # ANN model predicts foundation design parameters [B (m), D (m), L/B ratio]
        if multi_model is not None:
            predictions = multi_model.predict(features_scaled)[0]

            B_pred = max(1.0, min(10.0, float(predictions[0])))
            D_pred = max(0.5, min(6.0, float(predictions[1])))

            # L/B removed from model
            lb_pred = None
        else:
            N_spt = max(1.0, float(interpolated_params.get('spt_n60', 15) or 15))
            if N_spt >= 6.0:
                B_pred = 2.0
            else:
                B_calc = 50.0 / (N_spt * 8.49)
                B_pred = max(2.5, min(5.0, round(B_calc * 2) / 2))
            D_pred  = 1.5
            lb_pred = 1.0

        # ── Extract critical-layer soil parameters ─────────────────────────────
        spt_n60       = float(interpolated_params.get('spt_n60', 0))
        unit_weight   = float(interpolated_params.get('unit_weight', 18))
        csr_val       = float(interpolated_params.get('csr', 0.2)) or 0.2
        crr_val       = float(interpolated_params.get('crr') or interpolated_params.get('cyclic_strength_ratio') or 0)
        crr_val       = crr_val if crr_val > 0 else 0.3
        gwl           = float(interpolated_params.get('groundwater_depth_m', 5))
        fines_percent = float(interpolated_params.get('fines_content', 10))
        fs_adjusted   = critical_layer['fs']   # critical layer FS (magnitude-adjusted)

        # ── Risk level from critical layer FS ──────────────────────────────────
        if fs_adjusted < 1.0:
            liq_adjusted_prob = min(100.0, (1.0 - fs_adjusted) * 100)
        elif fs_adjusted < 1.5:
            liq_adjusted_prob = 30.0
        else:
            liq_adjusted_prob = 10.0

        db_risk_prob = float(interpolated_params.get('risk_probability', 0))
        if db_risk_prob > 0:
            nearest_dist_km_blend = interpolation_info.get('nearest_distance_km', 10)
            db_weight = max(0.2, 0.7 - (nearest_dist_km_blend / 10.0) * 0.5)
            liquefaction_prob = db_weight * db_risk_prob + (1.0 - db_weight) * liq_adjusted_prob
        else:
            liquefaction_prob = liq_adjusted_prob

        if liquefaction_prob >= 60:
            risk_level, severity = "HIGH", "Severe"
        elif liquefaction_prob >= 30:
            risk_level, severity = "MEDIUM", "Moderate"
        else:
            risk_level, severity = "LOW", "Minor"

        # ── DB override: use stored classification when within 5 km ────────────
        data_source = "ANN Model + IDW Interpolation (per-layer)"
        nearest_dist_km = interpolation_info.get('nearest_distance_km', 999)
        if nearest_dist_km < 5.0:
            bh_uuid  = interpolation_info.get('_nearest_borehole_uuid')
            bh_label = interpolation_info.get('_nearest_borehole_label', 'N/A')
            if bh_uuid:
                db_override = get_nearest_borehole_db_risk(bh_uuid, bh_label)
                if db_override:
                    db_ov_weight = max(0.5, 1.0 - (nearest_dist_km / 5.0) * 0.5)
                    liquefaction_prob = (
                        db_ov_weight * db_override['probability']
                        + (1.0 - db_ov_weight) * liquefaction_prob
                    )
                    if liquefaction_prob >= 60:
                        risk_level, severity = "HIGH", "Severe"
                    elif liquefaction_prob >= 30:
                        risk_level, severity = "MEDIUM", "Moderate"
                    else:
                        risk_level, severity = "LOW", "Minor"
                    data_source = db_override['source']

        # ── Bearing capacity (Meyerhof/Bowles using ANN-predicted B and D) ─────
        B = B_pred; D = D_pred; FS_DESIGN = 3.0; SI_ALLOW = 25.0
        N  = max(1.0, spt_n60)
        phi_deg = min(45.0, max(25.0, 27.1 + 0.3 * N - 0.00054 * N ** 2))
        phi_rad = _math.radians(phi_deg)
        Nq  = _math.exp(_math.pi * _math.tan(phi_rad)) * _math.tan(_math.radians(45) + phi_rad / 2) ** 2
        Ng  = 2.0 * (Nq + 1.0) * _math.tan(phi_rad)
        Fqs = 1.0 + _math.tan(phi_rad)
        Fgs = 0.6
        qu_site = unit_weight * D * Nq * Fqs + 0.5 * unit_weight * B * Ng * Fgs
        qa_site = max(1.0, qu_site / FS_DESIGN)

        if qa_site < q_actual:
            contrib_depth = unit_weight * D * Nq * Fqs
            denom = 0.5 * unit_weight * Ng * Fgs
            if denom > 0:
                B_needed = (q_actual * FS_DESIGN - contrib_depth) / denom
                B_needed = round(B_needed * 2) / 2
                B = min(10.0, max(B, B_needed))
                B_pred = B
                qu_site = unit_weight * D * Nq * Fqs + 0.5 * unit_weight * B * Ng * Fgs
                qa_site = max(1.0, qu_site / FS_DESIGN)

        # Pre-liquefaction settlement (Bowles)
        pre_liq_settlement_cm = (q_actual / qa_site) * SI_ALLOW / 10.0  # cm

        # ── Per-layer Tokimatsu & Seed (1987) settlement ───────────────────────
        _DPWH_TO_3 = {'VERY HIGH': 'HIGH', 'HIGH': 'HIGH',
                      'MEDIUM': 'MEDIUM', 'LOW': 'LOW', 'VERY LOW': 'LOW'}
        total_settlement_cm = 0.0
        liquefiable_layers  = []

        for lyr in interpolated_layers:
            fs_lyr    = lyr['fs']
            risk_3lyr = _DPWH_TO_3.get(lyr.get('liquefaction_risk_level', 'LOW'), 'LOW')

            # Cap FS based on stored risk level to honour DB-blended classification
            if risk_3lyr == 'HIGH':
                fs_settle = min(fs_lyr, 0.7)
            elif risk_3lyr == 'MEDIUM':
                fs_settle = min(fs_lyr, 0.9)
            else:
                fs_settle = fs_lyr

            if fs_settle < 1.0:
                N_lyr    = max(1.0, float(lyr.get('spt_n_value', 20)))
                csr_lyr  = float(lyr.get('csr', 0.2)) or 0.2
                thickness = float(lyr.get('layer_thickness', 1.5))

                if N_lyr < 5:    ev_max = 4.0
                elif N_lyr < 10: ev_max = 3.0
                elif N_lyr < 15: ev_max = 2.0
                elif N_lyr < 20: ev_max = 1.0
                else:            ev_max = 0.5

                ev  = ev_max * min(csr_lyr / 0.3, 1.0) * (1.0 - fs_settle)
                ev  = max(0.0, min(ev, ev_max))
                lyr_settle = round((ev / 100.0) * thickness * 100.0, 2)
                total_settlement_cm += lyr_settle

                liquefiable_layers.append({
                    'layer':        lyr['layer_number'],
                    'depth':        f"{lyr['depth_from_m']}–{lyr['depth_to_m']} m",
                    'spt_n':        round(N_lyr, 1),
                    'fs':           round(fs_lyr, 3),
                    'settlement_cm': lyr_settle,
                })
                print(f"[INFO] Layer {lyr['layer_number']} "
                      f"({lyr['depth_from_m']}–{lyr['depth_to_m']} m): "
                      f"FS={fs_lyr:.3f}, ev={ev:.2f}%, settle={lyr_settle:.2f} cm")

        # Apply minimum floors for HIGH/MEDIUM site risk
        if risk_level == "HIGH":
            total_settlement_cm = max(total_settlement_cm, 5.0)
        elif risk_level == "MEDIUM":
            total_settlement_cm = max(total_settlement_cm, 2.0)

        settlement_cm = round(total_settlement_cm, 2) if liquefiable_layers else pre_liq_settlement_cm

        # Pre / post liquefaction bearing capacity
        pre_bearing = qa_site
        if risk_level == "HIGH":
            post_bearing = max(0.0, pre_bearing * 0.2)
        elif risk_level == "MEDIUM":
            post_bearing = max(0.0, pre_bearing * 0.6)
        else:
            post_bearing = pre_bearing
        capacity_reduction = ((pre_bearing - post_bearing) /
                              pre_bearing) * 100 if pre_bearing > 0 else 0

        # Settlement factor of safety against q_actual
        settlement_fs = qa_site / q_actual if q_actual > 0 else float('inf')

        csr  = csr_val
        crr  = crr_val
        settlement_severity = "High" if settlement_cm > 10 else "Moderate" if settlement_cm > 5 else "Low"

        # Recommendations based on confidence
        recommendations = []
        confidence = interpolation_info.get('confidence', 'Low')

        if confidence in ['Very Low', 'Low']:
            recommendations.append(
                f"⚠️ {confidence} confidence prediction - Site-specific investigation REQUIRED"
            )

        if risk_level == "HIGH":
            recommendations.extend([
                "Consider deep foundation systems (piles or caissons)",
                "Implement ground improvement techniques (vibro-compaction, stone columns)",
                "Design for significant post-liquefaction settlement",
                "Conduct detailed site-specific investigation"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Perform detailed site investigation",
                "Consider shallow foundation with ground improvement",
                "Design structures to accommodate moderate settlement",
                "Monitor groundwater levels"
            ])
        else:
            recommendations.extend([
                "Standard foundation design practices may be acceptable",
                "Monitor soil conditions during construction",
                "Consider minor ground improvement for critical structures"
            ])

        # Add distance-based warnings
        nearest_dist = interpolation_info.get('nearest_distance_km', 0)
        if nearest_dist > 20:
            recommendations.insert(0,
                                   f"⚠️ Nearest data is {nearest_dist:.1f} km away - High uncertainty in predictions"
                                   )
        elif nearest_dist > 10:
            recommendations.insert(0,
                                   f"⚠️ Nearest data is {nearest_dist:.1f} km away - Moderate uncertainty in predictions"
                                   )

        nearest_dist_km = interpolation_info.get('nearest_distance_km', 0)

        result = PredictionResponse(
            location={
                "latitude": lat,
                "longitude": lon,
                "nearest_borehole_distance_km": round(nearest_dist_km, 2),
                "municipality": munic or "Unknown"
            },
            risk_assessment={
                "risk_level": risk_level,
                "probability": round(liquefaction_prob, 1),
                "severity": severity,
                "factor_of_safety": round(fs_adjusted, 3),
                "confidence": confidence,
                "data_source": data_source,
            },
            soil_parameters={
                "spt_n60": round(spt_n60, 1),
                "unit_weight": round(unit_weight, 2),
                "csr": round(csr, 4),
                "crr": round(crr, 4),
                "gwl": round(gwl, 2),
                "fines_percent": round(fines_percent, 1),
                "source": "Spatial Interpolation"
            },
            settlement={
                "predicted_cm": round(settlement_cm, 2),           # W/ Liq  (post-liquefaction, model output)
                "pre_liquefaction_cm": round(pre_liq_settlement_cm, 2),  # W/o Liq (Bowles q_actual/Qa×25mm)
                "severity": settlement_severity
            },
            bearing_capacity={
                "pre_liquefaction_kpa": round(pre_bearing, 0),
                "post_liquefaction_kpa": round(post_bearing, 0),
                "capacity_reduction_percent": round(capacity_reduction, 1)
            },
            foundation_recommendation={
                "base_m": round(B_pred, 2),
                "depth_m": round(D_pred, 2),
            },
            recommendations=recommendations,
            analysis_parameters={
                "q_actual_kpa": q_actual,
                "magnitude_mw": magnitude,
                "msf": round(msf, 4),
                "fs_design": FS_DESIGN,
                "settlement_fs": round(settlement_fs, 2)
            },
            interpolation_info=interpolation_info
        )
        _cache_set(ck, result)
        return result
    except Exception as e:
        import traceback
        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/boreholes")
async def get_boreholes(
    municipality: Optional[str] = Query(None, description="Filter by municipality name"),
    _: None = Depends(verify_api_key),
):
    """
    Return all boreholes with geolocations and liquefaction risk classification.

    Each borehole includes a **marker_color** for map pins:
    - `red`    → LIQUEFIABLE    (HIGH / VERY HIGH risk)
    - `orange` → MARGINAL       (MEDIUM risk)
    - `green`  → NON-LIQUEFIABLE (LOW / VERY LOW risk)
    - `gray`   → NO DATA
    """
    try:
        client = get_supabase_client()

        # ── 1. Fetch all boreholes (with municipality name) ──────────────────
        bh_query = client.table('boreholes').select(
            'id, borehole_id, latitude, longitude, elevation, depth_total_m, municipalities(name)'
        )
        bh_result = bh_query.execute()

        if not bh_result.data:
            return {"boreholes": [], "total": 0, "legend": {}}

        boreholes = bh_result.data

        # Optional municipality filter (client-side after join)
        if municipality:
            boreholes = [
                b for b in boreholes
                if isinstance(b.get('municipalities'), dict)
                and b['municipalities'].get('name', '').lower() == municipality.lower()
            ]

        # ── 2. Fetch soil layer risk data for all boreholes ──────────────────
        bh_ids = [b['id'] for b in boreholes]
        soil_result = client.table('soil_layers').select(
            'borehole_id, liquefaction_risk_level, liquefaction, csr, cyclic_strength_ratio, spt_n_value'
        ).in_('borehole_id', bh_ids).execute()

        # ── 3. Aggregate worst-case risk per borehole ────────────────────────
        RISK_ORDER = {'VERY HIGH': 5, 'HIGH': 4, 'MEDIUM': 3, 'LOW': 2, 'VERY LOW': 1}
        COLOR_MAP  = {'VERY HIGH': 'red',    'HIGH': 'red',
                      'MEDIUM':    'orange',
                      'LOW':       'green',  'VERY LOW': 'green'}
        STATUS_MAP = {'VERY HIGH': 'LIQUEFIABLE',     'HIGH': 'LIQUEFIABLE',
                      'MEDIUM':    'MARGINAL',
                      'LOW':       'NON-LIQUEFIABLE',  'VERY LOW': 'NON-LIQUEFIABLE'}

        from collections import defaultdict
        layers_by_bh: dict = defaultdict(list)
        for layer in (soil_result.data or []):
            layers_by_bh[layer['borehole_id']].append(layer)

        def _avg(values):
            valid = [v for v in values if v is not None]
            return round(sum(valid) / len(valid), 4) if valid else None

        borehole_risk = {}
        for bh_id, layers in layers_by_bh.items():
            worst = max(
                (l.get('liquefaction_risk_level') for l in layers),
                key=lambda r: RISK_ORDER.get(r, 0),
                default=None,
            )
            borehole_risk[bh_id] = {
                'risk_level':        worst or 'UNKNOWN',
                'marker_color':      COLOR_MAP.get(worst, 'gray'),
                'liquefaction_status': STATUS_MAP.get(worst, 'UNKNOWN'),
                'layer_count':       len(layers),
                'liquefiable_layers': sum(1 for l in layers if l.get('liquefaction')),
                'avg_csr':  _avg([l.get('csr') for l in layers]),
                'avg_crr':  _avg([l.get('cyclic_strength_ratio') for l in layers]),
                'avg_spt_n': _avg([l.get('spt_n_value') for l in layers]),
            }

        # ── 4. Build response features ───────────────────────────────────────
        NO_DATA = {
            'risk_level': 'NO DATA', 'marker_color': 'gray',
            'liquefaction_status': 'NO DATA', 'layer_count': 0,
            'liquefiable_layers': 0, 'avg_csr': None, 'avg_crr': None, 'avg_spt_n': None,
        }

        features = []
        for bh in boreholes:
            risk = borehole_risk.get(bh['id'], NO_DATA)
            muni = bh.get('municipalities')
            features.append({
                'borehole_id':         bh.get('borehole_id'),
                'latitude':            bh.get('latitude'),
                'longitude':           bh.get('longitude'),
                'elevation':           bh.get('elevation'),
                'municipality':        muni.get('name') if isinstance(muni, dict) else None,
                'risk_level':          risk['risk_level'],
                'liquefaction_status': risk['liquefaction_status'],
                'marker_color':        risk['marker_color'],
                'layer_count':         risk['layer_count'],
                'liquefiable_layers':  risk['liquefiable_layers'],
                'avg_csr':             risk['avg_csr'],
                'avg_crr':             risk['avg_crr'],
                'avg_spt_n':           risk['avg_spt_n'],
            })

        # ── 5. Legend + counts ───────────────────────────────────────────────
        color_counts: dict = defaultdict(int)
        for f in features:
            color_counts[f['marker_color']] += 1

        return {
            "boreholes": features,
            "total": len(features),
            "legend": {
                "red":    {"label": "LIQUEFIABLE",      "risk_levels": ["HIGH", "VERY HIGH"], "count": color_counts['red']},
                "orange": {"label": "MARGINAL",          "risk_levels": ["MEDIUM"],            "count": color_counts['orange']},
                "green":  {"label": "NON-LIQUEFIABLE",   "risk_levels": ["LOW", "VERY LOW"],   "count": color_counts['green']},
                "gray":   {"label": "NO DATA",           "risk_levels": [],                    "count": color_counts['gray']},
            },
        }

    except Exception as e:
        import traceback
        print(f"[ERROR] /boreholes — {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
async def features_info(_: None = Depends(verify_api_key)):
    """Get feature information"""
    return {
        "feature_count": len(EXPECTED_FEATURES),
        "feature_names": EXPECTED_FEATURES,
        "description": "Features expected by the trained model"
    }


@app.get("/model-info")
async def model_info(_: None = Depends(verify_api_key)):
    """Get model metadata"""
    if _model_metadata is None:
        try:
            _, _, _, _, _, metadata = load_model()
            return metadata or {"message": "Metadata not available"}
        except:
            raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_metadata


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
