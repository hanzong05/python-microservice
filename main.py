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

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
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
except ImportError:
    PIPELINE_AVAILABLE = False
    print("[WARNING] Pipeline module not available")

try:
    from train_ann import MultiOutputANNTraining
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("[WARNING] Training module not available")

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

EXPECTED_FEATURES = [
    'spt_n_value', 'spt_n60', 'unit_weight', 'fines_content',
    'groundwater_depth_m', 'pga_g', 'csr', 'depth_from_m', 'depth_to_m',
    'layer_number', 'elastic_modulus_es', 'feature_11', 'feature_12',
    'feature_13', 'feature_14', 'feature_15', 'feature_16'
]


class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    depth_m: Optional[float] = None
    municipality: Optional[str] = None


class PredictionResponse(BaseModel):
    location: Dict
    risk_assessment: Dict
    soil_parameters: Dict
    settlement: Dict
    bearing_capacity: Dict
    recommendations: List[str]
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
    """Load models from Supabase Storage"""
    global _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata

    if _scaler is not None:
        return _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata

    print("[INFO] Loading models from Supabase Storage...")

    client = get_supabase_client()
    bucket_name = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')

    try:
        scaler_data = client.storage.from_(
            bucket_name).download('models/scaler.pkl')
        _scaler = joblib.load(io.BytesIO(scaler_data))
        print(
            f"  [OK] Scaler loaded (expects {_scaler.n_features_in_} features)")

        try:
            multi_data = client.storage.from_(
                bucket_name).download('models/ann_multi_output.pkl')
            _multi_model = joblib.load(io.BytesIO(multi_data))
            print("  [OK] Multi-output model loaded")
        except:
            _multi_model = None

        try:
            liq_data = client.storage.from_(bucket_name).download(
                'models/ann_liquefaction_classifier.pkl')
            _liq_model = joblib.load(io.BytesIO(liq_data))
            print("  [OK] Liquefaction classifier loaded")
        except:
            _liq_model = None

        try:
            settle_data = client.storage.from_(bucket_name).download(
                'models/ann_settlement_regressor.pkl')
            _settle_model = joblib.load(io.BytesIO(settle_data))
            print("  [OK] Settlement regressor loaded")
        except:
            _settle_model = None

        try:
            bearing_data = client.storage.from_(bucket_name).download(
                'models/ann_bearing_capacity_regressor.pkl')
            _bearing_model = joblib.load(io.BytesIO(bearing_data))
            print("  [OK] Bearing capacity regressor loaded")
        except:
            _bearing_model = None

        try:
            metadata_data = client.storage.from_(
                bucket_name).download('models/ann_metadata.json')
            _model_metadata = json.loads(metadata_data.decode('utf-8'))
            print("  [OK] Metadata loaded")
        except:
            _model_metadata = {}

        return _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata
    except Exception as e:
        print(f"  [ERROR] Failed to load models: {e}")
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


def interpolate_soil_parameters(
    latitude: float,
    longitude: float,
    depth_m: Optional[float] = None,
    n_boreholes: int = 5
) -> Tuple[Dict, Dict]:
    """
    Interpolate soil parameters using Inverse Distance Weighting (IDW)
    Returns: (interpolated_params, interpolation_info)
    """
    # Find nearest boreholes
    nearest_boreholes = find_nearest_boreholes(
        latitude, longitude, n=n_boreholes)

    if not nearest_boreholes:
        print("[WARNING] No boreholes found for interpolation")
        return None, None

    # Collect soil data from each borehole
    borehole_data = []
    for borehole, distance_km in nearest_boreholes:
        borehole_id = borehole.get('borehole_id')
        soil_data = get_borehole_soil_data(borehole_id, depth_m)

        if soil_data:
            # Average parameters for this borehole
            avg_params = {
                'spt_n_value': np.mean([r.get('spt_n_value', 20) or 20 for r in soil_data]),
                'spt_n60': np.mean([r.get('spt_n60', 20) or 20 for r in soil_data]),
                'unit_weight': np.mean([r.get('unit_weight', 18) or 18 for r in soil_data]),
                'fines_content': np.mean([r.get('fines_content', 10) or 10 for r in soil_data]),
                'groundwater_depth_m': np.mean([r.get('groundwater_depth_m', 5) or 5 for r in soil_data]),
                'pga_g': np.mean([r.get('pga_g', 0.3) or 0.3 for r in soil_data]),
                'csr': np.mean([r.get('csr', 0.2) or 0.2 for r in soil_data]),
                'elastic_modulus_es': np.mean([r.get('elastic_modulus_es', 10000) or 10000 for r in soil_data]),
                'crr': np.mean([r.get('crr', 0) or r.get('cyclic_strength_ratio', 0) or 0 for r in soil_data]),
            }

            borehole_data.append({
                'borehole_id': borehole_id,
                'distance_km': distance_km,
                'params': avg_params,
                'latitude': borehole.get('latitude'),
                'longitude': borehole.get('longitude')
            })

    if not borehole_data:
        print("[WARNING] No soil data found in any nearby boreholes")
        return None, None

    # Inverse Distance Weighting (IDW)
    # Weight = 1 / (distance^power)
    # Use power=2 for standard IDW
    power = 2

    # Calculate weights
    weights = []
    for bd in borehole_data:
        # Add small epsilon to avoid division by zero for exact matches
        distance = max(bd['distance_km'], 0.001)
        weight = 1.0 / (distance ** power)
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Interpolate each parameter
    interpolated = {}
    for param_name in borehole_data[0]['params'].keys():
        weighted_sum = sum(
            bd['params'][param_name] * weight
            for bd, weight in zip(borehole_data, normalized_weights)
        )
        interpolated[param_name] = weighted_sum

    # Add location-based adjustments
    # Adjust based on distance to nearest data point
    nearest_distance = borehole_data[0]['distance_km']

    # Uncertainty increases with distance
    # Increase PGA slightly for distant locations (conservative)
    if nearest_distance > 10:
        distance_factor = min(1.2, 1.0 + (nearest_distance - 10) / 100)
        interpolated['pga_g'] *= distance_factor
        interpolated['csr'] *= distance_factor

    # Create interpolation info
    interpolation_info = {
        'method': 'Inverse Distance Weighting (IDW)',
        'power': power,
        'boreholes_used': len(borehole_data),
        'nearest_distance_km': round(nearest_distance, 2),
        'farthest_distance_km': round(borehole_data[-1]['distance_km'], 2),
        'borehole_contributions': [
            {
                'id': bd['borehole_id'],
                'distance_km': round(bd['distance_km'], 2),
                'weight': round(w, 3)
            }
            for bd, w in zip(borehole_data, normalized_weights)
        ],
        'confidence': calculate_interpolation_confidence(nearest_distance, len(borehole_data))
    }

    print(f"[INFO] Interpolated from {len(borehole_data)} boreholes")
    print(
        f"[INFO] Nearest: {borehole_data[0]['borehole_id']} ({nearest_distance:.2f} km)")
    print(f"[INFO] Confidence: {interpolation_info['confidence']}")

    return interpolated, interpolation_info


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


def engineer_features_from_interpolated(
    interpolated_params: Dict,
    latitude: float,
    longitude: float,
    depth_m: Optional[float] = None
) -> pd.DataFrame:
    """Create feature vector from interpolated parameters"""

    target_depth = depth_m if depth_m else 0.0

    features = {
        'spt_n_value': float(interpolated_params.get('spt_n_value', 20)),
        'spt_n60': float(interpolated_params.get('spt_n60', 20)),
        'unit_weight': float(interpolated_params.get('unit_weight', 18)),
        'fines_content': float(interpolated_params.get('fines_content', 10)),
        'groundwater_depth_m': float(interpolated_params.get('groundwater_depth_m', 5)),
        'pga_g': float(interpolated_params.get('pga_g', 0.3)),
        'csr': float(interpolated_params.get('csr', 0.2)),
        'depth_from_m': target_depth,
        'depth_to_m': target_depth + 1.5,
        'layer_number': 1,
        'elastic_modulus_es': float(interpolated_params.get('elastic_modulus_es', 10000)),
        'feature_11': 0.0,
        'feature_12': 0.0,
        'feature_13': 0.0,
        'feature_14': 0.0,
        'feature_15': 0.0,
        'feature_16': 0.0
    }

    return pd.DataFrame([features], columns=EXPECTED_FEATURES)


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
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": _multi_model is not None or _liq_model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/pipeline/start")
async def pipeline_start(background_tasks: BackgroundTasks):
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
async def pipeline_status():
    """Get pipeline execution status"""
    return _pipeline_status


@app.post("/train/start")
async def train_start(background_tasks: BackgroundTasks):
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
async def train_status():
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
async def pipeline_and_train_start(background_tasks: BackgroundTasks):
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
        5, ge=1, le=10, description="Number of nearest boreholes to use for interpolation")
):
    """
    Predict liquefaction using spatial interpolation from multiple boreholes
    """
    if not DEPENDENCIES_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Dependencies not available")

    # Handle both POST and GET
    if request:
        lat, lon, depth_m, munic = request.latitude, request.longitude, request.depth_m, request.municipality
    else:
        lat, lon, depth_m, munic = latitude, longitude, depth, municipality

    if lat is None or lon is None:
        raise HTTPException(
            status_code=400, detail="Latitude and longitude required")

    try:
        scaler, multi_model, liq_model, settle_model, bearing_model, metadata = load_model()

        # Interpolate soil parameters from multiple boreholes
        interpolated_params, interpolation_info = interpolate_soil_parameters(
            lat, lon, depth_m, n_boreholes=n_boreholes
        )

        if not interpolated_params:
            raise HTTPException(
                status_code=404,
                detail="No borehole data found within 100km radius"
            )

        # Engineer features from interpolated parameters
        features_df = engineer_features_from_interpolated(
            interpolated_params, lat, lon, depth_m)

        print(f"[DEBUG] Features shape: {features_df.shape}")

        # Scale and predict
        features_scaled = scaler.transform(features_df)

        if multi_model is not None:
            predictions = multi_model.predict(features_scaled)[0]
            liquefaction_prob_raw = float(predictions[0])
            settlement_cm = float(predictions[1])
            bearing_capacity_kpa = float(predictions[2])

            liquefaction_prob = max(0, min(100, liquefaction_prob_raw * 100))
        else:
            if liq_model is not None:
                prediction_proba = liq_model.predict_proba(features_scaled)[0]
                liquefaction_prob = float(prediction_proba[1] * 100)
            else:
                csr = float(interpolated_params.get('csr', 0.2))
                liquefaction_prob = min(100, csr * 250)

            settlement_cm = float(settle_model.predict(
                features_scaled)[0]) if settle_model else 0.0
            bearing_capacity_kpa = float(bearing_model.predict(
                features_scaled)[0]) if bearing_model else 200.0

        # Risk assessment
        if liquefaction_prob >= 60:
            risk_level, severity = "HIGH", "Severe"
        elif liquefaction_prob >= 30:
            risk_level, severity = "MEDIUM", "Moderate"
        else:
            risk_level, severity = "LOW", "Minor"

        # Extract parameters from interpolation
        spt_n60 = float(interpolated_params.get('spt_n60', 0))
        unit_weight = float(interpolated_params.get('unit_weight', 0))
        csr = float(interpolated_params.get('csr', 0))
        crr = float(interpolated_params.get('crr', 0))
        gwl = float(interpolated_params.get('groundwater_depth_m', 0))
        fines_percent = float(interpolated_params.get('fines_content', 0))

        settlement_severity = "High" if settlement_cm > 10 else "Moderate" if settlement_cm > 5 else "Low"

        # Bearing capacity
        if bearing_capacity_kpa < 50:
            pre_bearing = 200 + (spt_n60 * 10)
        else:
            pre_bearing = bearing_capacity_kpa * (1 + liquefaction_prob / 100)

        post_bearing = max(0, bearing_capacity_kpa)
        capacity_reduction = ((pre_bearing - post_bearing) /
                              pre_bearing) * 100 if pre_bearing > 0 else 0

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

        return PredictionResponse(
            location={
                "latitude": lat,
                "longitude": lon,
                "municipality": munic or "Unknown"
            },
            risk_assessment={
                "risk_level": risk_level,
                "probability": round(liquefaction_prob, 1),
                "severity": severity,
                "confidence": confidence
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
                "predicted_cm": round(settlement_cm, 2),
                "severity": settlement_severity
            },
            bearing_capacity={
                "pre_liquefaction_kpa": round(pre_bearing, 0),
                "post_liquefaction_kpa": round(post_bearing, 0),
                "capacity_reduction_percent": round(capacity_reduction, 1)
            },
            recommendations=recommendations,
            interpolation_info=interpolation_info
        )
    except Exception as e:
        import traceback
        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
async def features_info():
    """Get feature information"""
    return {
        "feature_count": len(EXPECTED_FEATURES),
        "feature_names": EXPECTED_FEATURES,
        "description": "Features expected by the trained model"
    }


@app.get("/model-info")
async def model_info():
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
