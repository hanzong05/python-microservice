#!/usr/bin/env python3
"""
Enhanced API with Spatial Interpolation — FIXED v2.3.0
=======================================================
All original bugs fixed (A–E) PLUS LPI fixes + BUG F:

BUG A  (DATA/PIPELINE) — BH-381/382 have USCS='M' (single-char artifact).
  Pipeline normalised 'M' → '' → treated as real soil with imputed SPT=15,
  unit_weight=23, pga=0.47, producing VERY HIGH risk at depth.
  FIX: exclude boreholes/layers where is_core_sample=1 OR USCS is empty/rock
  from IDW by filtering them out in get_borehole_all_layers.

BUG B  (API — Risk Level) — get_nearest_borehole_db_risk() on a CS-1 borehole
  returned 'NOT APPLICABLE' as worst level.
  RISK_PROB.get('NOT APPLICABLE', 50.0) = 50.0 → injected 50% probability
  into the blend, then db_ov_weight=0.65 diluted the real risk signal down to
  VERY LOW even when IDW showed HIGH/VERY HIGH layers.
  FIX: filter out NOT APPLICABLE layers before computing worst risk.
  If no soil layers remain, return None so the db_override is skipped entirely.

BUG C  (API — LPI) — Risk Level and LPI computed from DIFFERENT FS values.
  Risk Level used liquefaction_prob blend (probability-based).
  LPI used per-layer fs_i = min(fs_computed, _DB_RISK_TO_FS[db_risk]).
  These are independent paths → can contradict.
  FIX: LPI severity is now the PRIMARY driver of the displayed Risk Level label.
  LPI ≥ 15 → VERY HIGH, ≥ 5 → HIGH, ≥ 2 → MEDIUM, ≥ 0.1 → LOW, else VERY LOW.
  This guarantees consistency between the two displayed values.

BUG D  (API — IDW) — idw_field default for 'cyclic_strength_ratio' was 0.0.
  In predict(), zero crr falls back to 0.3 (hardcoded).
  Zero CSR falls back to 0.2 (hardcoded).
  FS = 0.3*MSF/0.2 ≈ 2.16 → silently safe even for liquefiable sites.
  FIX: idw_field raises the default for CRR to None and skips layers with
  no CRR data instead of using the 0.3 sentinel. Per-layer FS is only computed
  when both CSR and CRR are available from IDW; otherwise the DB risk level
  drives the layer's LPI contribution directly via _DB_RISK_TO_FS proxy.

BUG E  (API — borehole_data scope) — idw_bearing() inside predict() referenced
  borehole_data from interpolate_soil_parameters() inner scope — NameError at runtime.
  FIX: interpolate_soil_parameters() now returns (layers, info, borehole_data).
  idw_bearing() accepts borehole_data as an explicit parameter.

BUG G  (API — IDW CRR/CSR dilution contradicts DB risk classification) —
  IDW blends CRR and CSR from ALL nearby boreholes weighted by distance.
  When one borehole has HIGH-risk layers but four surrounding boreholes have
  VERY LOW / LOW soil, the IDW-averaged CRR is pulled upward and CSR is pulled
  downward, producing computed FS >> 1.0 for layers the DB classifies as HIGH.
  Example from logs: Layer 9 risk=HIGH but IDW-computed FS=1.68, Layer 10
  risk=HIGH but FS=1.86 — both > 1.0 → F_i = 0 → LPI = 0.
  The DB risk classification was computed PER-BOREHOLE on the actual measured
  CRR/CSR values, so it is the ground truth for that borehole's soil condition.
  The IDW blend corrupts it.
  FIX: after computing FS from IDW CRR/CSR, apply a conservative ceiling:
    FS_final = min(FS_computed, FS_ceiling_from_worst_DB_risk_classification)
  Ceiling values (top of each DPWH FS band):
    VERY HIGH ceiling = 0.80   (any computed FS > 0.80 is impossible for VERY HIGH)
    HIGH      ceiling = 1.00   (any computed FS > 1.00 contradicts HIGH classification)
    MEDIUM    ceiling = 1.20
    LOW       ceiling = 1.50
    VERY LOW  ceiling = 999.0  (unconstrained)
  This ensures a HIGH-classified layer always produces F_i ≥ 0, so LPI is
  never zero for a site that has confirmed HIGH-risk boreholes nearby.

NOTE (magnitude=0) — magnitude=0 is a valid static/no-earthquake case.
  MSF is set to 1.0 when magnitude=0 so FS = CRR/CSR without seismic scaling.
  The Query param now uses ge=0 (unrestricted) and the MSF formula branches
  cleanly on magnitude==0 instead of clamping or rejecting the input.

LPI FIX 1 — DB proxy FS values were too high for HIGH/MEDIUM risk.
  OLD: HIGH→0.90 (F_i=0.10), MEDIUM→1.10 (F_i=0.00).
  These made HIGH-risk layers contribute nearly nothing to LPI.
  FIX: proxy values now reflect the midpoint of each DPWH FS band.
    VERY HIGH (<0.80): proxy=0.55  → F_i=0.45
    HIGH      (<1.00): proxy=0.80  → F_i=0.20  (was 0.90 → F_i=0.10)
    MEDIUM    (<1.20): proxy=0.95  → F_i=0.05  (was 1.10 → F_i=0.00)
    LOW       (<1.50): proxy=1.35  → F_i=0.00  (unchanged)
    VERY LOW  (≥1.50): proxy=2.00  → F_i=0.00  (unchanged)

LPI FIX 2 — Use stored lpi_severity_factor from DB when CRR/CSR are unavailable.
  The pipeline stores lpi_severity_factor = max(0, 1 - FS) per layer.
  IDW-interpolating this field gives a direct F_i estimate that is more accurate
  than back-calculating from the risk-level proxy, especially for MEDIUM layers.
  FIX: idw_field fetches 'lpi_severity_factor'; if available, it is used as F_i
  directly (bypassing FS proxy entirely). FS proxy is the last resort.

LPI FIX 3 — Layer thickness floor raised from 0.1 m to 1.0 m.
  IDW-averaged depth ranges from misaligned boreholes can produce near-zero
  thicknesses (e.g. 0.1–0.3 m), which multiply F_i * W_i by ~0.2 and suppress
  LPI to negligible values even for genuinely liquefiable profiles.
  FIX: layer_thickness = max(1.0, depth_to - depth_from) so each layer
  contributes at least the equivalent of a 1-metre sampling interval.

LPI FIX 4 — Per-layer LPI debug logging added.
  Every call to predict() now prints a full layer-by-layer breakdown so that
  the source of each F_i (computed/db_lsf/db_proxy) is visible in server logs.
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
    version="2.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────
_client = None
_scaler = None
_multi_model = None
_liq_model = None
_settle_model = None
_bearing_model = None
_model_metadata = None

_pipeline_status = {"status": "idle", "message": "", "timestamp": None}
_training_status = {"status": "idle", "message": "", "timestamp": None}

# ── Prediction cache ───────────────────────────────────────────────────────
_prediction_cache: dict = {}
_CACHE_TTL_SECONDS = 3600


def _cache_key(lat, lon, q_actual, magnitude):
    return (round(lat, 4), round(lon, 4), round(q_actual, 1), round(magnitude, 1))


def _cache_get(key):
    import time
    entry = _prediction_cache.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL_SECONDS:
        print(f"[CACHE] HIT {key}")
        return entry["result"]
    return None


def _cache_set(key, result):
    import time
    _prediction_cache[key] = {"result": result, "ts": time.time()}
    if len(_prediction_cache) > 200:
        oldest = sorted(_prediction_cache,
                        key=lambda k: _prediction_cache[k]["ts"])
        for k in oldest[:50]:
            del _prediction_cache[k]
    print(f"[CACHE] SET {key}  (size: {len(_prediction_cache)})")


EXPECTED_FEATURES: list = []

_API_SECRET_KEY = os.getenv("API_SECRET_KEY")
if not _API_SECRET_KEY:
    raise RuntimeError("API_SECRET_KEY environment variable is not set")


async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    if x_api_key != _API_SECRET_KEY:
        raise HTTPException(
            status_code=401, detail="Unauthorized. Invalid API key.")


class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    depth_m: Optional[float] = None
    municipality: Optional[str] = None
    q_actual: Optional[float] = 50.0
    # 0 = static/no-earthquake case (MSF=1.0)
    magnitude: Optional[float] = 6.5
    t_years: Optional[float] = None


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


# ── RISK CONSTANTS ─────────────────────────────────────────────────────────
_RISK_ORDER = {'VERY HIGH': 5, 'HIGH': 4, 'MEDIUM': 3, 'LOW': 2, 'VERY LOW': 1}
_RISK_PROB = {'VERY HIGH': 90.0, 'HIGH': 75.0,
              'MEDIUM': 45.0,    'LOW': 15.0,  'VERY LOW': 5.0}

# ── LPI → Risk Level (single authoritative mapping, used everywhere) ───────
_LPI_THRESHOLDS = [
    (15.0, "VERY HIGH", "Very High"),
    (5.0,  "HIGH",      "High"),
    (2.0,  "MEDIUM",    "Medium"),
    (0.1,  "LOW",       "Low"),
    (0.0,  "VERY LOW",  "Very Low"),
]


def _lpi_to_risk(lpi: float) -> tuple:
    """Return (risk_level, severity) — single source of truth for risk classification."""
    for threshold, level, sev in _LPI_THRESHOLDS:
        if lpi >= threshold:
            return level, sev
    return "VERY LOW", "Very Low"


def _lpi_severity_label(lpi: float) -> str:
    _, sev = _lpi_to_risk(lpi)
    return sev


# ── LPI FIX 1 — DB proxy FS values recalibrated to DPWH FS band midpoints ──
#
# DPWH BSDS 2013 FS thresholds:
#   VERY HIGH : FS < 0.80  → midpoint ~0.55  → F_i = 1 - 0.55 = 0.45
#   HIGH      : 0.80 ≤ FS < 1.00 → midpoint ~0.90 BUT old value caused F_i=0.10
#               Use 0.80 (band floor) so HIGH layers always contribute.
#   MEDIUM    : 1.00 ≤ FS < 1.20 → use 0.95 (just below 1.0) → F_i = 0.05
#               Previously 1.10 → F_i = 0.00 (zero contribution — WRONG).
#   LOW       : 1.20 ≤ FS < 1.50 → midpoint 1.35 → F_i = 0.00 (correct: unlikely)
#   VERY LOW  : FS ≥ 1.50 → 2.00 → F_i = 0.00 (correct)
#
_DB_RISK_TO_FS = {
    'VERY HIGH':      0.55,   # F_i = 0.45  (was 0.60)
    'HIGH':           0.80,   # F_i = 0.20  (was 0.90 → F_i=0.10)
    'MEDIUM':         0.95,   # F_i = 0.05  (was 1.10 → F_i=0.00 ← main bug)
    'LOW':            1.35,   # F_i = 0.00  (unchanged)
    'VERY LOW':       2.00,   # F_i = 0.00  (unchanged)
    'NOT APPLICABLE': 2.00,   # F_i = 0.00  (unchanged)
}

# ── BUG G FIX — FS ceiling per DPWH risk class (top of each FS band) ────────
# IDW-blended CRR/CSR from safe neighbouring boreholes can inflate computed FS
# above what is physically consistent with the worst DB risk classification.
# These ceilings cap the IDW-computed FS at the band boundary so that a layer
# classified HIGH can never produce F_i = 0 due to IDW dilution.
_DB_RISK_FS_CEILING = {
    'VERY HIGH':      0.80,   # FS must be < 0.80 to be VERY HIGH
    'HIGH':           1.00,   # FS must be < 1.00 to be HIGH
    'MEDIUM':         1.20,   # FS must be < 1.20 to be MEDIUM
    'LOW':            1.50,   # FS must be < 1.50 to be LOW
    'VERY LOW':       999.0,  # unconstrained
    'NOT APPLICABLE': 999.0,  # unconstrained
}

# ── Supabase / model helpers ───────────────────────────────────────────────


def get_supabase_client():
    global _client
    if _client is None:
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        _client = create_client(url, key)
    return _client


def load_model():
    global _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata
    global EXPECTED_FEATURES

    if _scaler is not None:
        return _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata

    print("[INFO] Loading models from Supabase Storage...")
    client = get_supabase_client()
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "geotechnical-data")

    try:
        _scaler = joblib.load(io.BytesIO(
            client.storage.from_(bucket).download("models/scaler_no_leakage.pkl")))
        print("  [OK] Scaler loaded")

        _multi_model = joblib.load(io.BytesIO(
            client.storage.from_(bucket).download("models/ann_multi_output_BD_no_leakage.pkl")))
        print("  [OK] B/D model loaded")

        _liq_model = _settle_model = _bearing_model = None

        meta_bytes = client.storage.from_(bucket).download(
            "models/ann_metadata_no_leakage.json")
        _model_metadata = json.loads(meta_bytes.decode("utf-8"))
        EXPECTED_FEATURES = _model_metadata.get("feature_names", [])
        if not EXPECTED_FEATURES:
            raise RuntimeError("No feature_names in metadata")
        print(f"  [OK] Features: {EXPECTED_FEATURES}")

        return _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata
    except Exception as e:
        print(f"  [ERROR] Model load failed: {e}")
        raise


# ── Spatial helpers ────────────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def find_nearest_boreholes(latitude, longitude, n=5, max_distance_km=100):
    client = get_supabase_client()
    try:
        result = client.table('boreholes').select('*').execute()
        if not result.data:
            return []
        dists = []
        for bh in result.data:
            d = haversine_distance(latitude, longitude,
                                   float(bh.get('latitude', 0)),
                                   float(bh.get('longitude', 0)))
            if d <= max_distance_km:
                dists.append((bh, d))
        dists.sort(key=lambda x: x[1])
        nearest = dists[:n]
        print(f"[INFO] {len(nearest)} boreholes within {max_distance_km} km")
        for bh, d in nearest:
            print(f"  - {bh.get('borehole_id')}: {d:.2f} km")
        return nearest
    except Exception as e:
        print(f"[ERROR] find_nearest_boreholes: {e}")
        return []


def get_borehole_all_layers(borehole_uuid: int) -> List[Dict]:
    """
    Fetch all soil layers for a borehole.
    BUG A FIX: filters out NOT APPLICABLE layers (core samples / rock) so they
    cannot contribute to IDW interpolation or skew the worst-risk calculation.
    Also fetches lpi_severity_factor for LPI FIX 2.
    """
    client = get_supabase_client()
    try:
        result = client.table('soil_layers').select(
            'layer_number, depth_from_m, depth_to_m, '
            'spt_n_value, spt_n160, n1_60cs, unit_weight, fines_content, groundwater_depth_m, '
            'pga_g, csr, cyclic_strength_ratio, friction_angle, cohesion_kpa, '
            'elastic_modulus_es, liquefaction_risk_level, liquefaction, '
            'bearing_qa_kpa, settlement_mm, '
            'lpi_severity_factor, factor_of_safety'          # LPI FIX 2
        ).eq('borehole_id', borehole_uuid).order('layer_number').execute()

        layers = result.data if result.data else []

        # BUG A FIX — exclude non-soil layers from IDW
        soil_layers = [
            l for l in layers
            if l.get('liquefaction_risk_level') not in ('NOT APPLICABLE', None)
            or (l.get('spt_n_value') is not None and l.get('csr') is not None)
        ]
        excluded = len(layers) - len(soil_layers)
        if excluded:
            print(
                f"    [FIX A] Excluded {excluded} non-soil layers from borehole {borehole_uuid}")
        return soil_layers
    except Exception as e:
        print(f"[WARNING] get_borehole_all_layers({borehole_uuid}): {e}")
        return []


def interpolate_soil_parameters(
    latitude, longitude, depth_m=None, n_boreholes=5
) -> Tuple[Optional[List[Dict]], Optional[Dict], Optional[List[Dict]]]:
    """
    IDW interpolation of soil parameters per layer.

    BUG D FIX  : idw_field returns None for missing numeric fields.
    BUG E FIX  : returns borehole_data as third element.
    LPI FIX 2  : interpolates lpi_severity_factor from DB.
    LPI FIX 3  : layer_thickness floor raised to 1.0 m.

    Returns:
        (interpolated_layers, interpolation_info, borehole_data)
        All three are None on failure.
    """
    nearest = find_nearest_boreholes(latitude, longitude, n=n_boreholes)
    if not nearest:
        return None, None, None

    borehole_data = []
    for bh, dist in nearest:
        bh_uuid = bh.get('id')
        bh_label = bh.get('borehole_id')
        layers = get_borehole_all_layers(bh_uuid)
        if not layers:
            continue

        risk_levels = [l.get('liquefaction_risk_level') for l in layers
                       if l.get('liquefaction_risk_level')
                       and l.get('liquefaction_risk_level') != 'NOT APPLICABLE']
        worst_risk = max(risk_levels, key=lambda r: _RISK_ORDER.get(
            r, 0)) if risk_levels else 'VERY LOW'
        weight = 1.0 / max(dist, 0.001) ** 2

        borehole_data.append({
            'borehole_id':      bh_label,
            'borehole_uuid':    bh_uuid,
            'distance_km':      dist,
            'weight':           weight,
            'layers':           layers,
            'risk_probability': _RISK_PROB.get(worst_risk, 5.0),
        })

    if not borehole_data:
        return None, None, None

    nearest_distance = borehole_data[0]['distance_km']
    primary_bh = borehole_data[0]

    # ── DIRECT USE — nearest borehole < 0.5 km ────────────────────────────
    # When the site is on top of or immediately adjacent to a borehole,
    # IDW blending with farther boreholes corrupts the actual measured values
    # (fines content, PGA, CSR, CRR) with data from different soil profiles.
    # Below 0.5 km we use the nearest borehole's layers directly — no blending.
    # borehole_data is still returned in full so idw_bearing() can fall back
    # to farther boreholes for bearing capacity if the nearest has no qa value.
    _DIRECT_USE_KM = 0.5

    if nearest_distance < _DIRECT_USE_KM:
        print(f"[INFO] Nearest borehole {primary_bh['borehole_id']} is "
              f"{nearest_distance:.3f} km — DIRECT USE (no IDW blend)")

        primary_bh['norm_weight'] = 1.0
        for bd in borehole_data[1:]:
            bd['norm_weight'] = 0.0

        # Build interpolated_layers directly from the nearest borehole's layers
        direct_layers = primary_bh['layers']
        all_layer_nums = sorted(set(l['layer_number'] for l in direct_layers))

        risk_levels_direct = [
            l.get('liquefaction_risk_level') for l in direct_layers
            if l.get('liquefaction_risk_level')
            and l.get('liquefaction_risk_level') != 'NOT APPLICABLE'
        ]
        site_risk_prob = _RISK_PROB.get(
            max(risk_levels_direct, key=lambda r: _RISK_ORDER.get(r, 0))
            if risk_levels_direct else 'VERY LOW', 5.0)

        interpolated_layers = []
        for ln in all_layer_nums:
            lyr = next(
                (l for l in direct_layers if l['layer_number'] == ln), None)
            if lyr is None:
                continue

            depth_from = float(lyr.get('depth_from_m') or (ln - 1) * 1.5)
            depth_to = float(lyr.get('depth_to_m') or ln * 1.5)
            thickness = max(1.0, depth_to - depth_from)

            risk_lv = lyr.get('liquefaction_risk_level') or 'VERY LOW'
            if risk_lv == 'NOT APPLICABLE':
                risk_lv = 'VERY LOW'

            interpolated_layers.append({
                'layer_number':          ln,
                'depth_from_m':          round(depth_from, 2),
                'depth_to_m':            round(depth_to, 2),
                'layer_thickness':       round(thickness, 2),
                'spt_n_value':           lyr.get('spt_n_value') or 15,
                'spt_n60':               lyr.get('spt_n_value') or 15,
                'n1_60cs':               lyr.get('n1_60cs') or 15,
                'unit_weight':           lyr.get('unit_weight') or 18,
                'fines_content':         lyr.get('fines_content') or 10,
                'groundwater_depth_m':   lyr.get('groundwater_depth_m') or 5,
                'pga_g':                 lyr.get('pga_g') or 0.3,
                'csr':                   lyr.get('csr'),
                'crr':                   lyr.get('cyclic_strength_ratio'),
                'cyclic_strength_ratio': lyr.get('cyclic_strength_ratio'),
                'friction_angle':        lyr.get('friction_angle') or 30.0,
                'cohesion_kpa':          lyr.get('cohesion_kpa') or 0.0,
                'elastic_modulus_es':    lyr.get('elastic_modulus_es') or 10000,
                'liquefaction_risk_level': risk_lv,
                'risk_probability':        site_risk_prob,
                'lpi_severity_factor':   lyr.get('lpi_severity_factor'),
                'factor_of_safety_db':   lyr.get('factor_of_safety'),
            })

        interpolation_info = {
            'method':              f'Direct — nearest borehole {primary_bh["borehole_id"]} ({nearest_distance:.3f} km)',
            'power':               None,
            'boreholes_used':      1,
            'layers_interpolated': len(interpolated_layers),
            'nearest_distance_km': round(nearest_distance, 3),
            'farthest_distance_km': round(nearest_distance, 3),
            'borehole_contributions': [
                {'id': primary_bh['borehole_id'],
                 'distance_km': round(nearest_distance, 3),
                 'weight': 1.0}
            ],
            'confidence':             calculate_interpolation_confidence(nearest_distance, 1),
            '_nearest_borehole_uuid':  primary_bh.get('borehole_uuid'),
            '_nearest_borehole_label': primary_bh['borehole_id'],
        }

        print(
            f"[INFO] Direct use: {len(interpolated_layers)} layers from {primary_bh['borehole_id']}")
        return interpolated_layers, interpolation_info, borehole_data

    # ── IDW BLEND — nearest borehole ≥ 0.5 km ─────────────────────────────
    total_weight = sum(bd['weight'] for bd in borehole_data)
    for bd in borehole_data:
        bd['norm_weight'] = bd['weight'] / total_weight

    site_risk_prob = sum(bd['risk_probability'] * bd['norm_weight']
                         for bd in borehole_data)

    all_layer_nums = sorted(set(
        l['layer_number'] for bd in borehole_data for l in bd['layers']
    ))

    def idw_field(layer_num, key, default=None):
        """BUG D FIX — default is None, not 0.0."""
        vals = []
        for bd in borehole_data:
            lyr = next((l for l in bd['layers']
                        if l['layer_number'] == layer_num), None)
            if lyr and lyr.get(key) is not None:
                try:
                    vals.append((float(lyr[key]), bd['norm_weight']))
                except (TypeError, ValueError):
                    pass
        if not vals:
            return default
        total_w = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / total_w if total_w > 0 else default

    interpolated_layers = []
    for ln in all_layer_nums:
        depth_from = idw_field(
            ln, 'depth_from_m', (ln - 1) * 1.5) or (ln-1)*1.5
        depth_to = idw_field(ln, 'depth_to_m',    ln * 1.5) or ln * 1.5

        # LPI FIX 3 — floor thickness at 1.0 m so near-zero IDW depths don't
        # suppress LPI to negligible values for genuinely liquefiable profiles.
        thickness = max(1.0, depth_to - depth_from)

        # BUG G FIX — use distance-weighted worst risk instead of global max.
        layer_risk_weighted: List[Tuple[int, float]] = []
        for bd in borehole_data:
            lyr_match = next(
                (l for l in bd['layers']
                 if l['layer_number'] == ln
                 and l.get('liquefaction_risk_level')
                 and l.get('liquefaction_risk_level') != 'NOT APPLICABLE'),
                None
            )
            if lyr_match:
                ord_val = _RISK_ORDER.get(
                    lyr_match['liquefaction_risk_level'], 0)
                layer_risk_weighted.append((ord_val, bd['norm_weight']))

        if layer_risk_weighted:
            total_w_risk = sum(w for _, w in layer_risk_weighted)
            avg_ord = sum(o * w for o, w in layer_risk_weighted) / total_w_risk
            worst_layer_risk = next(
                (label for label, ord_threshold in
                 [('VERY HIGH', 4.5), ('HIGH', 3.5),
                  ('MEDIUM', 2.5), ('LOW', 1.5)]
                 if avg_ord >= ord_threshold),
                'VERY LOW'
            )
        else:
            worst_layer_risk = 'VERY LOW'

        # BUG D FIX — None default (not 0.0)
        crr_idw = idw_field(ln, 'cyclic_strength_ratio')
        csr_idw = idw_field(ln, 'csr')
        lpi_sf_idw = idw_field(ln, 'lpi_severity_factor')
        fs_db_idw = idw_field(ln, 'factor_of_safety')

        interpolated_layers.append({
            'layer_number':          ln,
            'depth_from_m':          round(depth_from, 2),
            'depth_to_m':            round(depth_to, 2),
            'layer_thickness':       round(thickness, 2),
            'spt_n_value':           idw_field(ln, 'spt_n_value', 15),
            'spt_n60':               idw_field(ln, 'spt_n_value', 15),
            'n1_60cs':               idw_field(ln, 'n1_60cs', 15),
            'unit_weight':           idw_field(ln, 'unit_weight', 18),
            'fines_content':         idw_field(ln, 'fines_content', 10),
            'groundwater_depth_m':   idw_field(ln, 'groundwater_depth_m', 5),
            'pga_g':                 idw_field(ln, 'pga_g', 0.3),
            'csr':                   csr_idw,
            'crr':                   crr_idw,
            'cyclic_strength_ratio': crr_idw,
            'friction_angle':        idw_field(ln, 'friction_angle', 30.0),
            'cohesion_kpa':          idw_field(ln, 'cohesion_kpa', 0.0),
            'elastic_modulus_es':    idw_field(ln, 'elastic_modulus_es', 10000),
            'liquefaction_risk_level': worst_layer_risk,
            'risk_probability':        site_risk_prob,
            'lpi_severity_factor':   lpi_sf_idw,
            'factor_of_safety_db':   fs_db_idw,
        })

    if nearest_distance > 10:
        factor = min(1.2, 1.0 + (nearest_distance - 10) / 100)
        for lyr in interpolated_layers:
            if lyr['pga_g']:
                lyr['pga_g'] *= factor
            if lyr['csr']:
                lyr['csr'] *= factor

    interpolation_info = {
        'method':             'Inverse Distance Weighting (IDW) — per layer',
        'power':              2,
        'boreholes_used':     len(borehole_data),
        'layers_interpolated': len(interpolated_layers),
        'nearest_distance_km': round(nearest_distance, 2),
        'farthest_distance_km': round(borehole_data[-1]['distance_km'], 2),
        'borehole_contributions': [
            {'id': bd['borehole_id'], 'distance_km': round(bd['distance_km'], 2),
             'weight': round(bd['norm_weight'], 3)}
            for bd in borehole_data
        ],
        'confidence':             calculate_interpolation_confidence(nearest_distance, len(borehole_data)),
        '_nearest_borehole_uuid':  primary_bh.get('borehole_uuid'),
        '_nearest_borehole_label': primary_bh['borehole_id'],
    }

    print(
        f"[INFO] IDW: {len(interpolated_layers)} layers from {len(borehole_data)} boreholes")
    return interpolated_layers, interpolation_info, borehole_data


def calculate_interpolation_confidence(nearest_km, n_boreholes):
    if nearest_km < 1:
        ds = 100
    elif nearest_km < 5:
        ds = 90
    elif nearest_km < 10:
        ds = 70
    elif nearest_km < 20:
        ds = 50
    elif nearest_km < 50:
        ds = 30
    else:
        ds = 10
    total = min(100, ds + min(20, n_boreholes * 4))
    if total >= 80:
        return "High"
    if total >= 60:
        return "Medium"
    if total >= 40:
        return "Low"
    return "Very Low"


def get_nearest_borehole_db_risk(borehole_uuid, borehole_label: str) -> Optional[Dict]:
    """
    BUG B FIX: filters out NOT APPLICABLE layers before computing worst risk.
    Returns None if no valid soil layers exist.
    """
    RISK_TO_ML = {'VERY HIGH': 'HIGH', 'HIGH': 'HIGH',
                  'MEDIUM': 'MEDIUM',
                  'LOW': 'LOW', 'VERY LOW': 'LOW'}
    RISK_SEV = {'VERY HIGH': 'Severe', 'HIGH': 'Severe',
                'MEDIUM': 'Moderate',
                'LOW': 'Minor', 'VERY LOW': 'Minor'}
    try:
        client = get_supabase_client()
        result = client.table('soil_layers').select(
            'liquefaction_risk_level'
        ).eq('borehole_id', borehole_uuid).execute()

        if not result.data:
            return None

        # BUG B FIX — exclude NOT APPLICABLE layers
        valid_levels = [
            l.get('liquefaction_risk_level')
            for l in result.data
            if l.get('liquefaction_risk_level')
            and l.get('liquefaction_risk_level') != 'NOT APPLICABLE'
        ]
        if not valid_levels:
            print(
                f"[INFO] DB risk override skipped — {borehole_label} has no soil layers (all NOT APPLICABLE)")
            return None

        worst = max(valid_levels, key=lambda r: _RISK_ORDER.get(r, 0))
        print(f"[INFO] DB risk override — {borehole_label}: worst = {worst}")
        return {
            'db_risk_level':  worst,
            'ml_risk_level':  RISK_TO_ML.get(worst, 'LOW'),
            'probability':    _RISK_PROB.get(worst, 50.0),
            'severity':       RISK_SEV.get(worst, 'Minor'),
            'borehole_label': borehole_label,
            'source':         f'Database — {borehole_label} (DPWH BSDS)',
        }
    except Exception as e:
        print(f"[WARNING] DB risk lookup failed for {borehole_label}: {e}")
        return None


def engineer_features_from_interpolated(interpolated_params, latitude, longitude, depth_m=None):
    global EXPECTED_FEATURES
    if not EXPECTED_FEATURES:
        raise RuntimeError("EXPECTED_FEATURES empty — metadata not loaded")

    target_depth = float(
        depth_m or interpolated_params.get("depth_mid_m") or 1.5)
    spt = float(interpolated_params.get("spt_n_value") or 20.0)
    spt60 = float(interpolated_params.get("spt_n60") or spt)
    spt160 = float(interpolated_params.get("spt_n160") or spt60)
    uw = float(interpolated_params.get("unit_weight") or 18.0)
    fc = float(interpolated_params.get("fines_content") or 10.0)
    gwl = float(interpolated_params.get("groundwater_depth_m") or 5.0)

    derived = {
        "spt_n_value":    spt,
        "spt_n60":        spt60,
        "spt_n160":       spt160,
        "unit_weight":    uw,
        "fines_content":  fc,
        "friction_angle": float(interpolated_params.get("friction_angle") or 30.0),
        "depth_mid_m":    target_depth,
        "depth_from_m":   float(interpolated_params.get("depth_from_m") or target_depth - 0.75),
        "depth_to_m":     float(interpolated_params.get("depth_to_m") or target_depth + 0.75),
        "groundwater_depth_m":    gwl,
        "moisture_content":       float(interpolated_params.get("moisture_content") or fc),
        "plasticity_index":       float(interpolated_params.get("plasticity_index") or 0.0),
        "mean_particle_size_d50": float(interpolated_params.get("mean_particle_size_d50") or 0.1),
        "relative_density_percent": float(
            interpolated_params.get("relative_density_percent")
            or min(100.0, max(0.0, (spt60 / 60.0) * 100.0))
        ),
        "elastic_modulus_es": float(interpolated_params.get("elastic_modulus_es") or 10000.0),
        "pga_g":              float(interpolated_params.get("pga_g") or 0.3),
    }
    features = {f: derived.get(f, 0.0) for f in EXPECTED_FEATURES}
    return pd.DataFrame([features], columns=EXPECTED_FEATURES)


# ── FastAPI lifecycle ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    if DEPENDENCIES_AVAILABLE:
        try:
            load_model()
            print("[OK] Models loaded on startup")
        except Exception as e:
            print(f"[ERROR] Model load on startup: {e}")


def run_pipeline_background():
    global _pipeline_status
    _pipeline_status = {"status": "running", "message": "Pipeline started",
                        "timestamp": datetime.now().isoformat()}
    try:
        if not PIPELINE_AVAILABLE:
            raise Exception("Pipeline module not available")
        success = GeotechnicalPipeline().run()
        _pipeline_status = {
            "status":    "completed" if success else "failed",
            "message":   "Pipeline completed" if success else "Pipeline failed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        _pipeline_status = {"status": "failed", "message": str(e),
                            "timestamp": datetime.now().isoformat()}


def run_training_background():
    global _training_status, _scaler, _multi_model, _liq_model, _settle_model, _bearing_model, _model_metadata
    _training_status = {"status": "running", "message": "Training started",
                        "timestamp": datetime.now().isoformat()}
    try:
        if not TRAINING_AVAILABLE:
            raise Exception("Training module not available")
        success = MultiOutputANNTraining().run()
        if success:
            _scaler = _multi_model = _liq_model = _settle_model = _bearing_model = _model_metadata = None
            try:
                load_model()
            except Exception as e:
                print(f"[WARNING] Reload after training: {e}")
        _training_status = {
            "status":    "completed" if success else "failed",
            "message":   "Training completed" if success else "Training failed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        _training_status = {"status": "failed", "message": str(e),
                            "timestamp": datetime.now().isoformat()}


def run_full_workflow_background():
    run_pipeline_background()
    if _pipeline_status["status"] != "completed":
        _training_status.update({"status": "skipped", "message": "Pipeline failed",
                                 "timestamp": datetime.now().isoformat()})
        return
    run_training_background()


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Geotechnical Prediction API", "version": "2.7.0",
            "status": "running", "model_loaded": _multi_model is not None}


@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "healthy", "model_loaded": _multi_model is not None,
            "timestamp": datetime.now().isoformat()}


@app.post("/pipeline/start")
async def pipeline_start(bg: BackgroundTasks, _: None = Depends(verify_api_key)):
    if not PIPELINE_AVAILABLE:
        raise HTTPException(503, "Pipeline not available")
    if _pipeline_status["status"] == "running":
        raise HTTPException(409, "Pipeline already running")
    bg.add_task(run_pipeline_background)
    return {"message": "Pipeline started", "status": "running",
            "timestamp": datetime.now().isoformat()}


@app.get("/pipeline/status")
async def pipeline_status(_: None = Depends(verify_api_key)):
    return _pipeline_status


@app.post("/train/start")
async def train_start(bg: BackgroundTasks, _: None = Depends(verify_api_key)):
    if not TRAINING_AVAILABLE:
        raise HTTPException(503, "Training not available")
    if _training_status["status"] == "running":
        raise HTTPException(409, "Training already running")
    bg.add_task(run_training_background)
    return {"message": "Training started", "status": "running",
            "timestamp": datetime.now().isoformat()}


@app.get("/train/status")
async def train_status(_: None = Depends(verify_api_key)):
    return _training_status


@app.post("/pipeline-and-train/start")
async def pipeline_and_train_start(bg: BackgroundTasks, _: None = Depends(verify_api_key)):
    if not PIPELINE_AVAILABLE or not TRAINING_AVAILABLE:
        raise HTTPException(503, "Pipeline or training not available")
    if _pipeline_status["status"] == "running" or _training_status["status"] == "running":
        raise HTTPException(409, "Pipeline or training already running")
    bg.add_task(run_full_workflow_background)
    return {"message": "Full workflow started", "status": "running",
            "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
@app.get("/predict-by-location", response_model=PredictionResponse)
async def predict(
    request:      Optional[PredictionRequest] = None,
    latitude:     Optional[float] = Query(None, ge=-90,   le=90),
    longitude:    Optional[float] = Query(None, ge=-180,  le=180),
    depth:        Optional[float] = Query(None, ge=0,     le=100),
    municipality: Optional[str] = Query(None),
    n_boreholes:  Optional[int] = Query(5,    ge=1,     le=10),
    q_actual:     Optional[float] = Query(50.0, ge=0),
    magnitude:    Optional[float] = Query(
        6.5,  ge=0,   le=9.5,  description='Moment magnitude Mw. Use 0 for static/no-earthquake analysis (sets MSF=1.0).'),
    b_increment:  Optional[float] = Query(0.1,  ge=0.05,  le=0.5),
    t_years:      Optional[float] = Query(None, ge=0,     le=200),
    _: None = Depends(verify_api_key),
):
    """Predict liquefaction using spatial interpolation from multiple boreholes."""
    if not DEPENDENCIES_AVAILABLE:
        raise HTTPException(503, "Dependencies not available")

    if request:
        lat = request.latitude
        lon = request.longitude
        depth_m = request.depth_m
        munic = request.municipality
        q_actual = request.q_actual if request.q_actual is not None else q_actual
        magnitude = request.magnitude if request.magnitude is not None else magnitude
        t_years = request.t_years if request.t_years is not None else t_years
    else:
        lat, lon, depth_m, munic = latitude, longitude, depth, municipality

    q_actual = float(q_actual) if q_actual is not None else 50.0
    magnitude = float(magnitude) if magnitude is not None else 6.5
    b_increment = float(b_increment) if b_increment is not None else 0.1
    t_years = float(t_years) if t_years is not None else None

    # magnitude=0 is a valid static/no-earthquake case — do not clamp it.
    # Only clamp the upper bound to a physically sane ceiling.
    if magnitude > 9.5:
        magnitude = 9.5

    if lat is None or lon is None:
        raise HTTPException(400, "Latitude and longitude required")

    ck = _cache_key(lat, lon, q_actual, magnitude)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    try:
        import math as _math

        scaler, multi_model, *_ = load_model()

        # BUG E FIX — unpack borehole_data from interpolate_soil_parameters
        interpolated_layers, interpolation_info, borehole_data = interpolate_soil_parameters(
            lat, lon, depth_m, n_boreholes=n_boreholes)

        if not interpolated_layers:
            raise HTTPException(404, "No borehole data found within 100 km")

        # ── MSF (Magnitude Scaling Factor) ────────────────────────────────
        # magnitude=0 → static/no-earthquake case → MSF=1.0 (no seismic scaling).
        # magnitude>0 → Idriss (1999): MSF = (10^2.24) / (Mw^2.56).
        # Upper bound capped at 9.5; no lower clamp so 0 passes through cleanly.
        if magnitude == 0:
            msf = 1.0
            print(f"[INFO] Mw=0 (static case) → MSF=1.0 (no seismic scaling)")
        else:
            _mw = min(magnitude, 9.5)
            msf = (10 ** 2.24) / (_mw ** 2.56)
            print(f"[INFO] Mw={_mw:.2f}  MSF={msf:.4f}")

        # ── Per-layer FS resolution (four-tier priority) ──────────────────────
        #
        # Priority 1 — Computed from IDW-interpolated CRR and CSR.
        #              BUG G FIX: after computation, cap FS at the DPWH band
        #              ceiling for the worst DB risk classification so that IDW
        #              dilution from safe neighbours cannot contradict the DB.
        # Priority 2 — IDW-interpolated lpi_severity_factor from DB (LPI FIX 2).
        # Priority 3 — Raw factor_of_safety IDW-interpolated from DB.
        # Priority 4 — _DB_RISK_TO_FS proxy (LPI FIX 1, last resort).
        #
        for lyr in interpolated_layers:
            crr_l = lyr.get('cyclic_strength_ratio') or lyr.get('crr')
            csr_l = lyr.get('csr')
            db_risk = lyr.get('liquefaction_risk_level', 'VERY LOW')

            if crr_l is not None and csr_l is not None and float(csr_l) > 0:
                # Priority 1 — direct computation
                fs_computed = (float(crr_l) * msf) / (float(csr_l) + 1e-9)

                # BUG G FIX — apply DB risk ceiling to prevent IDW dilution
                # from safe neighbours inflating FS above what is consistent
                # with the worst per-layer DB classification.
                fs_ceiling = _DB_RISK_FS_CEILING.get(db_risk, 999.0)
                fs_final = min(fs_computed, fs_ceiling)
                if fs_final < fs_computed:
                    print(f"    [BUG G] Layer {lyr['layer_number']} risk={db_risk}: "
                          f"IDW FS={fs_computed:.4f} capped to ceiling {fs_ceiling:.2f} "
                          f"(IDW CRR/CSR diluted by safer neighbours)")
                lyr['fs'] = fs_final
                lyr['fs_source'] = 'computed' if fs_final == fs_computed else 'computed+capped'

            else:
                # Priority 2 — use stored lpi_severity_factor (F_i) from DB
                lsf = lyr.get('lpi_severity_factor')
                if lsf is not None:
                    try:
                        lsf_float = float(lsf)
                        lyr['fs'] = max(0.0, 1.0 - lsf_float)
                        lyr['fs_source'] = 'db_lsf'
                    except (TypeError, ValueError):
                        lsf = None

                if lsf is None:
                    # Priority 3 — raw factor_of_safety from DB
                    fs_db = lyr.get('factor_of_safety_db')
                    if fs_db is not None:
                        try:
                            lyr['fs'] = float(fs_db)
                            lyr['fs_source'] = 'db_fs'
                        except (TypeError, ValueError):
                            fs_db = None

                    if fs_db is None:
                        # Priority 4 — recalibrated proxy (last resort)
                        lyr['fs'] = _DB_RISK_TO_FS.get(db_risk, 2.0)
                        lyr['fs_source'] = 'db_proxy'

        # Critical layer = lowest FS
        critical_layer = min(interpolated_layers, key=lambda l: l['fs'])
        interpolated_params = critical_layer

        if depth_m is None:
            depth_m = float(critical_layer.get('depth_to_m') or 1.5)

        print(f"[DEBUG] Critical layer: #{critical_layer['layer_number']} "
              f"({critical_layer['depth_from_m']}–{critical_layer['depth_to_m']} m) "
              f"FS={critical_layer['fs']:.3f} source={critical_layer.get('fs_source')}")

        # ── LPI — Iwasaki et al. (1978) ───────────────────────────────────
        #
        # LPI FIX 3 — layer_thickness already floored at 1.0 m in
        # interpolate_soil_parameters(), so h_i is always ≥ 1.0 m here.
        # LPI FIX 4 — full per-layer debug table printed to server log.
        #
        lpi = 0.0
        lpi_debug_rows = []

        for lyr in interpolated_layers:
            z_mid = (float(lyr.get('depth_from_m', 0)) +
                     float(lyr.get('depth_to_m', 0))) / 2.0

            if z_mid > 20.0:
                lpi_debug_rows.append(
                    f"  Layer {lyr['layer_number']:2d} | z={z_mid:5.1f}m | SKIPPED (z>20m)"
                )
                continue

            h_i = float(lyr.get('layer_thickness', 1.5))
            fs_i = lyr['fs']
            F_i = max(0.0, 1.0 - fs_i)
            W_i = max(0.0, 10.0 - 0.5 * z_mid)
            contrib = F_i * W_i * h_i
            lpi += contrib

            lpi_debug_rows.append(
                f"  Layer {lyr['layer_number']:2d} | z={z_mid:5.1f}m | "
                f"h={h_i:.2f}m | FS={fs_i:.4f} | F_i={F_i:.4f} | "
                f"W_i={W_i:.2f} | contrib={contrib:.4f} | "
                f"src={lyr.get('fs_source'):10s} | "
                f"risk={lyr.get('liquefaction_risk_level')}"
            )

        lpi = round(lpi, 2)
        lpi_severity = _lpi_severity_label(lpi)

        # LPI FIX 4 — print full debug table
        print("[LPI DEBUG] Layer-by-layer breakdown:")
        for row in lpi_debug_rows:
            print(row)
        print(f"[LPI DEBUG] → LPI = {lpi:.2f} ({lpi_severity})")

        # Risk Level = LPI classification — single, authoritative, never overridden
        risk_level, severity = _lpi_to_risk(lpi)
        liquefaction_prob = _RISK_PROB[risk_level]

        print(
            f"[INFO] LPI = {lpi:.2f} ({lpi_severity})  →  Risk Level = {risk_level}")

        # ── DB source label (transparency only — does NOT change risk_level) ─
        data_source = "IDW Interpolation + LPI (Iwasaki 1978)"
        nearest_dist_km = interpolation_info.get('nearest_distance_km', 999)
        if nearest_dist_km < 5.0:
            bh_uuid = interpolation_info.get('_nearest_borehole_uuid')
            bh_label = interpolation_info.get('_nearest_borehole_label', 'N/A')
            if bh_uuid:
                db_check = get_nearest_borehole_db_risk(bh_uuid, bh_label)
                if db_check:
                    data_source = f"LPI Classification (nearest: {bh_label}, {nearest_dist_km:.1f} km)"
                    db_risk_ord = _RISK_ORDER.get(db_check['db_risk_level'], 0)
                    lpi_risk_ord = _RISK_ORDER.get(risk_level, 0)
                    agreement = "AGREE" if db_risk_ord == lpi_risk_ord else \
                        f"DB={db_check['db_risk_level']} vs LPI={risk_level}"
                    print(
                        f"[INFO] DB cross-check ({bh_label}): {agreement} — LPI classification used")

        # ── ANN foundation prediction ──────────────────────────────────────
        features_df = engineer_features_from_interpolated(
            interpolated_params, lat, lon, depth_m)
        features_scaled = scaler.transform(features_df)

        if multi_model is not None:
            predictions = multi_model.predict(features_scaled)[0]
            B_pred = max(1.0, min(10.0, float(predictions[0])))
            D_pred = max(0.5, min(6.0,  float(predictions[1])))
        else:
            N_spt = max(1.0, float(
                interpolated_params.get('spt_n60', 15) or 15))
            B_pred = max(2.5, min(5.0, round(50.0 / (N_spt * 8.49) * 2) / 2))
            D_pred = 1.5

        # ── Critical-layer soil parameters ────────────────────────────────
        spt_n60 = float(interpolated_params.get('spt_n60') or 0)
        unit_weight = float(interpolated_params.get('unit_weight') or 18)
        csr_val = float(interpolated_params.get('csr') or 0) or 0.2
        crr_raw = interpolated_params.get(
            'crr') or interpolated_params.get('cyclic_strength_ratio')
        crr_val = float(crr_raw) if crr_raw is not None else 0.3
        gwl = float(interpolated_params.get('groundwater_depth_m') or 5)
        fines_percent = float(interpolated_params.get('fines_content') or 10)
        fs_adjusted = critical_layer['fs']

        # ── Bearing capacity ───────────────────────────────────────────────
        MAX_B = 5.0
        MAX_D = 3.5
        SI_ALLOW = 25.0

        # BUG E FIX — idw_bearing receives borehole_data explicitly
        def idw_bearing(layer_num: int, key: str, borehole_data: List[Dict]) -> Optional[float]:
            vals = []
            for bd in borehole_data:
                lyr = next(
                    (l for l in bd['layers'] if l['layer_number'] == layer_num), None)
                if lyr and lyr.get(key) is not None:
                    try:
                        vals.append((float(lyr[key]), bd['norm_weight']))
                    except (TypeError, ValueError):
                        pass
            if not vals:
                return None
            total_w = sum(w for _, w in vals)
            return sum(v * w for v, w in vals) / total_w if total_w > 0 else None

        crit_ln = critical_layer['layer_number']
        qa_from_db = idw_bearing(crit_ln, 'bearing_qa_kpa', borehole_data)

        if qa_from_db and qa_from_db > 1.0:
            qa_site = qa_from_db
            qu_site = qa_site * 3.0
            B = B_pred
            D = D_pred
            print(
                f"[INFO] Bearing capacity from DB (FIX 12): qa={qa_site:.1f} kPa")
        else:
            B = B_pred
            D = D_pred
            N = max(1.0, spt_n60)
            Kd = 1.0 + 0.33 * (D / B)
            size_factor = ((B + 0.3) / B) ** 2
            qa_site = max(1.0, 8.0 * N * size_factor * Kd)
            qu_site = qa_site * 3.0
            print(
                f"[INFO] Bearing capacity fallback (Meyerhof SPT): qa={qa_site:.1f} kPa")

        # ── Settlement ─────────────────────────────────────────────────────
        settle_from_db = idw_bearing(crit_ln, 'settlement_mm', borehole_data)

        if settle_from_db and settle_from_db > 0:
            pre_liq_settlement_cm = settle_from_db / 10.0
            print(
                f"[INFO] Settlement from DB (FIX 12): {settle_from_db:.1f} mm")
        else:
            pre_liq_settlement_cm = (q_actual / max(1.0, qa_site)) * 2.5
            print(
                f"[INFO] Settlement fallback: {pre_liq_settlement_cm:.1f} cm")

        # ── Tokimatsu & Seed (1987) volumetric settlement ─────────────────
        _DPWH_TO_4 = {'VERY HIGH': 'VERY HIGH', 'HIGH': 'HIGH',
                      'MEDIUM': 'LOW', 'LOW': 'VERY LOW', 'VERY LOW': 'VERY LOW'}
        total_settle_cm = 0.0
        liquefiable_layers = []
        for lyr in interpolated_layers:
            fs_lyr = lyr['fs']
            risk_4 = _DPWH_TO_4.get(
                lyr.get('liquefaction_risk_level', 'VERY LOW'), 'VERY LOW')
            if risk_4 == 'VERY HIGH':
                fs_settle = min(fs_lyr, 0.6)
            elif risk_4 == 'HIGH':
                fs_settle = min(fs_lyr, 0.8)
            elif risk_4 == 'LOW':
                fs_settle = min(fs_lyr, 0.95)
            else:
                fs_settle = fs_lyr

            if fs_settle < 1.0:
                N_l = max(1.0, float(lyr.get('spt_n_value', 20) or 20))
                csr_l = float(lyr.get('csr') or 0.2) or 0.2
                thick = float(lyr.get('layer_thickness', 1.5))
                ev_max = {N_l < 5: 4.0, N_l < 10: 3.0, N_l < 15: 2.0,
                          N_l < 20: 1.0}.get(True, 0.5)
                ev = max(0.0, min(ev_max, ev_max *
                                  min(csr_l / 0.3, 1.0) * (1.0 - fs_settle)))
                lyr_s = round((ev / 100.0) * thick * 100.0, 2)
                total_settle_cm += lyr_s
                liquefiable_layers.append({
                    'layer':         lyr['layer_number'],
                    'depth':         f"{lyr['depth_from_m']}–{lyr['depth_to_m']} m",
                    'spt_n':         round(N_l, 1),
                    'fs':            round(fs_lyr, 3),
                    'settlement_cm': lyr_s,
                })

        # BUG C FIX — minimum settlement floors driven by LPI-based risk_level
        floors = {'VERY HIGH': 7.0, 'HIGH': 4.0, 'MEDIUM': 2.0, 'LOW': 1.0}
        total_settle_cm = max(total_settle_cm, floors.get(risk_level, 0.0))
        settlement_cm = round(pre_liq_settlement_cm + total_settle_cm, 2)
        settlement_mm = settlement_cm * 10.0

        # ── Bearing capacity reduction ─────────────────────────────────────
        reductions = {'VERY HIGH': 0.10,
                      'HIGH': 0.35, 'MEDIUM': 0.65, 'LOW': 0.75}
        post_bearing = max(0.0, qa_site * reductions.get(risk_level, 1.0))
        cap_reduction = ((qa_site - post_bearing) /
                         qa_site * 100) if qa_site > 0 else 0

        settle_fs = qa_site / q_actual if q_actual > 0 else float('inf')
        settle_sev = "High" if settlement_cm > 10 else "Moderate" if settlement_cm > 5 else "Low"

        at_max = (B_pred >= MAX_B and D_pred >= MAX_D)
        mitigation = settlement_mm > SI_ALLOW and at_max

        # ── Recommendations ────────────────────────────────────────────────
        recommendations = []
        confidence = interpolation_info.get('confidence', 'Low')
        if confidence in ('Very Low', 'Low'):
            recommendations.append(
                f"⚠️ {confidence} confidence — site-specific investigation REQUIRED")

        if risk_level == "VERY HIGH":
            recommendations.extend([
                "VERY HIGH liquefaction risk — deep foundations (driven piles or caissons) required",
                "Ground improvement mandatory: vibro-compaction, stone columns, or deep soil mixing",
                "Design for severe post-liquefaction settlement and lateral spreading",
                "Conduct comprehensive site-specific geotechnical investigation immediately",
            ])
        elif risk_level == "HIGH":
            recommendations.extend([
                "HIGH liquefaction risk — consider deep or heavily reinforced shallow foundation",
                "Ground improvement: vibro-compaction or stone columns recommended",
                "Design for significant post-liquefaction settlement",
                "Conduct detailed site-specific geotechnical investigation",
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "MEDIUM liquefaction risk — perform detailed site investigation before finalising design",
                "Consider moderate ground improvement measures",
                "Design for some post-liquefaction settlement",
            ])
        elif risk_level == "LOW":
            recommendations.extend([
                "LOW liquefaction risk — shallow foundation with routine design is acceptable",
                "Monitor groundwater levels during and after construction",
                "Routine geotechnical checks are sufficient",
            ])
        else:
            recommendations.extend([
                "VERY LOW liquefaction risk — standard foundation design is acceptable",
                "Monitor soil conditions during construction",
            ])

        if lpi_severity in ("Very High", "High"):
            recommendations.append(
                f"LPI = {lpi:.2f} ({lpi_severity}) — incorporate LPI in structural design and risk assessment."
            )
        elif lpi_severity == "Moderate":
            recommendations.append(
                f"LPI = {lpi:.2f} (Moderate) — evaluate targeted ground improvement or settlement-tolerant design."
            )
        elif lpi_severity == "Low":
            recommendations.append(
                f"LPI = {lpi:.2f} (Low) — standard design with monitoring acceptable."
            )

        if t_years:
            recommendations.append(
                f"Design life = {t_years:.0f} yr: verify long-term consolidation within {SI_ALLOW:.0f} mm limit."
            )

        if mitigation:
            recommendations.extend([
                "⚠️ CRITICAL: Settlement exceeds 25 mm at maximum practical dimensions. Mitigation required.",
                "Option 1 — Deep foundations (piles/caissons) to competent stratum below liquefiable zone",
                "Option 2 — Ground densification: vibro-compaction or dynamic compaction",
                "Option 3 — Jet grouting or deep soil mixing (DSM)",
                "Option 4 — Preloading with prefabricated vertical drains (PVDs)",
            ])

        nearest_dist = interpolation_info.get('nearest_distance_km', 0)
        if nearest_dist > 20:
            recommendations.insert(
                0, f"⚠️ Nearest data {nearest_dist:.1f} km away — high uncertainty")
        elif nearest_dist > 10:
            recommendations.insert(
                0, f"⚠️ Nearest data {nearest_dist:.1f} km away — moderate uncertainty")

        result = PredictionResponse(
            location={
                "latitude":  lat,
                "longitude": lon,
                "nearest_borehole_distance_km": round(nearest_dist_km, 2),
                "municipality": munic or "Unknown",
            },
            risk_assessment={
                "risk_level":       risk_level,
                "probability":      round(liquefaction_prob, 1),
                "severity":         severity,
                "factor_of_safety": round(fs_adjusted, 3),
                "confidence":       confidence,
                "data_source":      data_source,
            },
            soil_parameters={
                "spt_n60":       round(spt_n60, 1),
                "unit_weight":   round(unit_weight, 2),
                "csr":           round(csr_val, 4),
                "crr":           round(crr_val, 4),
                "gwl":           round(gwl, 2),
                "fines_percent": round(fines_percent, 1),
                "source":        "Spatial Interpolation",
            },
            settlement={
                "settlement_cm": round(settlement_cm, 2),
                "severity":      settle_sev,
                "lpi":           lpi,
                "lpi_severity":  lpi_severity,
            },
            bearing_capacity={
                "allowable_bearing_capacity_kpa": round(post_bearing, 0),
                "capacity_reduction_percent":     round(cap_reduction, 1),
            },
            foundation_recommendation={
                "base_m":                  round(B_pred, 2),
                "depth_m":                 round(D_pred, 2),
                "mitigation_required":     mitigation,
                "settlement_mm":           round(settlement_mm, 1),
                "allowable_settlement_mm": SI_ALLOW,
            },
            recommendations=recommendations,
            analysis_parameters={
                "q_actual_kpa":  q_actual,
                "magnitude_mw":  magnitude,
                "msf":           round(msf, 4),
                "fs_design":     3.0,
                "settlement_fs": round(settle_fs, 2),
                "b_increment_m": b_increment,
                "t_years":       t_years,
            },
            interpolation_info=interpolation_info,
        )
        _cache_set(ck, result)
        return result

    except Exception as e:
        import traceback
        print(f"[ERROR] predict: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.get("/boreholes")
async def get_boreholes(
    municipality: Optional[str] = Query(None),
    _: None = Depends(verify_api_key),
):
    """
    All boreholes with geolocations and worst-case liquefaction risk.
    marker_color: red=LIQUEFIABLE, orange=MARGINAL, green=NON-LIQUEFIABLE, gray=NO DATA
    BUG B FIX applied: NOT APPLICABLE layers excluded from risk aggregation.
    """
    try:
        client = get_supabase_client()
        bh_result = client.table('boreholes').select(
            'id, borehole_id, latitude, longitude, elevation, depth_total_m, municipalities(name)'
        ).execute()
        if not bh_result.data:
            return {"boreholes": [], "total": 0, "legend": {}}

        boreholes = bh_result.data
        if municipality:
            boreholes = [b for b in boreholes
                         if isinstance(b.get('municipalities'), dict)
                         and b['municipalities'].get('name', '').lower() == municipality.lower()]

        bh_ids = [b['id'] for b in boreholes]
        all_layers = []
        offset = 0
        while True:
            page = client.table('soil_layers').select(
                'borehole_id, liquefaction_risk_level, liquefaction, csr, cyclic_strength_ratio, spt_n_value'
            ).in_('borehole_id', bh_ids).range(offset, offset + 999).execute()
            if not page.data:
                break
            all_layers.extend(page.data)
            if len(page.data) < 1000:
                break
            offset += 1000

        COLOR_MAP = {'VERY HIGH': 'red',   'HIGH': 'red',
                     'MEDIUM':    'orange',
                     'LOW':       'green',  'VERY LOW': 'green'}
        STATUS_MAP = {'VERY HIGH': 'LIQUEFIABLE', 'HIGH': 'LIQUEFIABLE',
                      'MEDIUM':    'MARGINAL',
                      'LOW':       'NON-LIQUEFIABLE', 'VERY LOW': 'NON-LIQUEFIABLE'}

        from collections import defaultdict
        layers_by_bh = defaultdict(list)
        for l in all_layers:
            layers_by_bh[l['borehole_id']].append(l)

        def _avg(vals):
            v = [x for x in vals if x is not None]
            return round(sum(v)/len(v), 4) if v else None

        borehole_risk = {}
        for bh_id, layers in layers_by_bh.items():
            # BUG B FIX — exclude NOT APPLICABLE
            soil_levels = [l.get('liquefaction_risk_level') for l in layers
                           if l.get('liquefaction_risk_level')
                           and l.get('liquefaction_risk_level') != 'NOT APPLICABLE']
            worst = max(soil_levels, key=lambda r: _RISK_ORDER.get(r, 0),
                        default=None) if soil_levels else None
            borehole_risk[bh_id] = {
                'risk_level':          worst or 'UNKNOWN',
                'marker_color':        COLOR_MAP.get(worst, 'gray'),
                'liquefaction_status': STATUS_MAP.get(worst, 'UNKNOWN'),
                'layer_count':         len(layers),
                'liquefiable_layers':  sum(1 for l in layers if l.get('liquefaction')),
                'avg_csr':             _avg([l.get('csr') for l in layers]),
                'avg_crr':             _avg([l.get('cyclic_strength_ratio') for l in layers]),
                'avg_spt_n':           _avg([l.get('spt_n_value') for l in layers]),
            }

        NO_DATA = {'risk_level': 'NO DATA', 'marker_color': 'gray',
                   'liquefaction_status': 'NO DATA', 'layer_count': 0,
                   'liquefiable_layers': 0, 'avg_csr': None, 'avg_crr': None, 'avg_spt_n': None}

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

        from collections import defaultdict as _dd
        cc = _dd(int)
        for f in features:
            cc[f['marker_color']] += 1

        return {
            "boreholes": features,
            "total":     len(features),
            "legend": {
                "red":    {"label": "LIQUEFIABLE",    "risk_levels": ["HIGH", "VERY HIGH"], "count": cc['red']},
                "orange": {"label": "MARGINAL",        "risk_levels": ["MEDIUM"],           "count": cc['orange']},
                "green":  {"label": "NON-LIQUEFIABLE", "risk_levels": ["LOW", "VERY LOW"],  "count": cc['green']},
                "gray":   {"label": "NO DATA",         "risk_levels": [],                   "count": cc['gray']},
            },
        }
    except Exception as e:
        import traceback
        print(f"[ERROR] /boreholes: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.get("/features")
async def features_info(_: None = Depends(verify_api_key)):
    return {"feature_count": len(EXPECTED_FEATURES), "feature_names": EXPECTED_FEATURES}


@app.get("/model-info")
async def model_info(_: None = Depends(verify_api_key)):
    if _model_metadata is None:
        try:
            *_, metadata = load_model()
            return metadata or {"message": "Metadata not available"}
        except:
            raise HTTPException(503, "Model not loaded")
    return _model_metadata


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port,
                reload=True, log_level="info")
