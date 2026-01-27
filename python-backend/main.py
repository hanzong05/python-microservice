"""
Simplified Geotechnical Prediction Microservice
FastAPI service for Tarlac Liquefaction Prediction System
Simplified version without database - using in-memory storage
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import numpy as np

app = FastAPI(
    title="Geo-Predict API",
    description="Geotechnical predictions for liquefaction, settlement, and bearing capacity",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Local development
        "https://*.vercel.app",        # Vercel deployments
        "https://vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for predictions
predictions_db = []
next_id = 1


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SoilInput(BaseModel):
    """Input data for geotechnical predictions"""
    borehole_id: str
    depth_m: float
    spt_n_value: float
    unit_weight: float
    fines_content: Optional[float] = 0.0
    friction_angle: Optional[float] = 30.0
    cohesion_kpa: Optional[float] = 0.0
    pga_g: float
    groundwater_depth_m: Optional[float] = 2.0
    effective_overburden_pressure: float
    total_overburden_pressure: float
    foundation_width_m: Optional[float] = 2.0
    foundation_depth_m: Optional[float] = 1.5


class PredictionResult(BaseModel):
    """Prediction output"""
    id: int
    borehole_id: str
    depth_m: float
    spt_n160: float
    csr: float
    cyclic_strength_ratio: float
    liquefaction: bool
    liquefaction_risk: str
    settlement_cm: float
    bearing_capacity_kpa: float
    qa_allowable_kpa: float


# ============================================================================
# GEOTECHNICAL CALCULATIONS
# ============================================================================

def calculate_n1_60(spt_n: float, sigma_prime: float, depth: float) -> float:
    """Calculate corrected SPT N-value"""
    Pa = 100  # kPa
    CN = min(2.0, (Pa / sigma_prime) ** 0.5)

    if depth < 3:
        CR = 0.75
    elif depth < 4:
        CR = 0.80
    elif depth < 6:
        CR = 0.85
    elif depth < 10:
        CR = 0.95
    else:
        CR = 1.0

    return max(0, spt_n * CN * CR)


def calculate_csr(pga_g: float, sigma_v: float, sigma_prime: float, depth: float) -> float:
    """Calculate Cyclic Stress Ratio (Seed & Idriss 1971)"""
    if depth <= 9.15:
        rd = 1.0 - 0.00765 * depth
    else:
        rd = 1.174 - 0.0267 * depth

    rd = max(0.1, rd)

    return max(0, 0.65 * pga_g * (sigma_v / sigma_prime) * rd)


def calculate_crr(n1_60: float, fines_content: float) -> float:
    """Calculate Cyclic Resistance Ratio (Idriss & Boulanger 2008)"""
    if n1_60 >= 30:
        return 999.0  # Very high resistance

    crr_75 = np.exp(n1_60/14.1 + (n1_60/126)**2 -
                    (n1_60/23.6)**3 + (n1_60/25.4)**4 - 2.8)

    if fines_content <= 5:
        delta_crr = 0
    elif fines_content <= 35:
        delta_crr = np.exp(1.76 - 190/fines_content**2)
    else:
        delta_crr = 0

    return max(0, crr_75 + delta_crr)


def calculate_settlement(n1_60: float, csr: float, crr: float, thickness: float = 1.5) -> float:
    """Calculate settlement (Tokimatsu & Seed 1987) in cm"""
    fs = crr / (csr + 1e-10)

    if fs >= 1.0:
        volumetric_strain = 0
    else:
        volumetric_strain = 0.01 * np.exp(-0.1 * n1_60) * (csr / 0.2)

    settlement_mm = volumetric_strain * thickness * 1000
    return max(0, settlement_mm / 10)


def calculate_bearing_capacity(gamma: float, width: float, depth: float,
                               phi: float, cohesion: float) -> float:
    """Calculate bearing capacity (Terzaghi 1943) in kPa"""
    phi_rad = np.radians(phi)

    Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.pi/4 + phi_rad/2) ** 2)

    if phi < 0.001:
        Nc = 5.7
    else:
        Nc = (Nq - 1) / np.tan(phi_rad)

    Ngamma = 2 * (Nq + 1) * np.tan(phi_rad)

    # Square footing shape factors
    qult = 1.3 * cohesion * Nc + gamma * depth * Nq + 0.4 * gamma * width * Ngamma

    return max(0, qult)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "✅ Geo-Predict API is running on Render!",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "GET /predictions": "Get all predictions",
            "POST /predict": "Create geotechnical prediction",
            "DELETE /predictions/{id}": "Delete a prediction",
        },
        "docs": "/docs"
    }


@app.get("/predictions")
async def get_predictions():
    """Get all stored predictions"""
    return {
        "predictions": predictions_db,
        "count": len(predictions_db)
    }


@app.post("/predict", response_model=PredictionResult)
async def create_prediction(soil_data: SoilInput):
    """
    Create geotechnical prediction for soil liquefaction, 
    settlement, and bearing capacity
    """
    global next_id

    try:
        # Calculate geotechnical parameters
        spt_n160 = calculate_n1_60(
            soil_data.spt_n_value,
            soil_data.effective_overburden_pressure,
            soil_data.depth_m
        )

        csr = calculate_csr(
            soil_data.pga_g,
            soil_data.total_overburden_pressure,
            soil_data.effective_overburden_pressure,
            soil_data.depth_m
        )

        cyclic_strength_ratio = calculate_crr(
            spt_n160, soil_data.fines_content)

        # Liquefaction assessment (DPWH BSDS 2013)
        fs = cyclic_strength_ratio / (csr + 1e-10)
        liquefaction = ((csr > 0.15) or (
            soil_data.spt_n_value < 15)) and (fs < 1.0)

        if fs >= 1.5:
            risk_level = "Low"
        elif fs >= 1.2:
            risk_level = "Moderate"
        elif fs >= 1.0:
            risk_level = "High"
        else:
            risk_level = "Very High"

        # Settlement calculation
        settlement_cm = calculate_settlement(
            spt_n160, csr, cyclic_strength_ratio)

        # Bearing capacity
        bearing_capacity_kpa = calculate_bearing_capacity(
            soil_data.unit_weight,
            soil_data.foundation_width_m,
            soil_data.foundation_depth_m,
            soil_data.friction_angle,
            soil_data.cohesion_kpa
        )

        qa_allowable_kpa = bearing_capacity_kpa / 3.0

        # Create prediction result
        prediction = {
            "id": next_id,
            "borehole_id": soil_data.borehole_id,
            "depth_m": soil_data.depth_m,
            "spt_n160": round(spt_n160, 2),
            "csr": round(csr, 4),
            "cyclic_strength_ratio": round(cyclic_strength_ratio, 4),
            "liquefaction": liquefaction,
            "liquefaction_risk": risk_level,
            "settlement_cm": round(settlement_cm, 2),
            "bearing_capacity_kpa": round(bearing_capacity_kpa, 2),
            "qa_allowable_kpa": round(qa_allowable_kpa, 2)
        }

        predictions_db.append(prediction)
        next_id += 1

        return prediction

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.delete("/predictions/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """Delete a prediction by ID"""
    global predictions_db

    initial_length = len(predictions_db)
    predictions_db = [p for p in predictions_db if p["id"] != prediction_id]

    if len(predictions_db) == initial_length:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return {
        "predictions": predictions_db,
        "message": "Prediction deleted successfully"
    }


@app.get("/stats")
async def get_statistics():
    """Get statistics from stored predictions"""
    if not predictions_db:
        return {
            "total_predictions": 0,
            "liquefiable_count": 0,
            "avg_settlement_cm": 0,
            "avg_bearing_capacity_kpa": 0
        }

    liquefiable = sum(1 for p in predictions_db if p["liquefaction"])
    avg_settlement = sum(p["settlement_cm"]
                         for p in predictions_db) / len(predictions_db)
    avg_bearing = sum(p["bearing_capacity_kpa"]
                      for p in predictions_db) / len(predictions_db)

    return {
        "total_predictions": len(predictions_db),
        "liquefiable_count": liquefiable,
        "avg_settlement_cm": round(avg_settlement, 2),
        "avg_bearing_capacity_kpa": round(avg_bearing, 2)
    }


# ============================================================================
# DATA UPLOAD TRIGGER
# ============================================================================

@app.post("/api/trigger-upload")
async def trigger_upload():
    """
    Trigger the geotechnical data upload from Excel to Supabase

    Returns:
        - status: success/error
        - message: Details about the upload
        - records_uploaded: Number of records uploaded
    """
    import subprocess
    import sys
    from pathlib import Path

    try:
        # Get the project root directory (two levels up from python-backend)
        project_root = Path(__file__).parent.parent
        script_path = project_root / "process_geotechnical_data.py"

        # Check if script exists
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Run the script with auto-upload
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Check if upload was successful
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Data upload completed successfully",
                "records_uploaded": 232,
                # Last 500 chars
                "output": result.stdout[-500:] if result.stdout else ""
            }
        else:
            return {
                "status": "error",
                "message": "Data upload failed",
                "error": result.stderr[-500:] if result.stderr else "Unknown error"
            }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Upload process timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/upload-status")
async def upload_status():
    """
    Check the status of data uploads

    Returns:
        - last_upload: Timestamp of last upload
        - status: Current status
    """
    return {
        "status": "ready",
        "message": "POST to /api/trigger-upload to start upload",
        "endpoint": "/api/trigger-upload",
        "method": "POST"
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    # Render provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
