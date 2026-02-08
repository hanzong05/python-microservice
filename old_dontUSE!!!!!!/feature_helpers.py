import re
import io
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


def parse_pga_value(pga_str) -> Optional[float]:
    if pd.isna(pga_str) or pga_str == '' or pga_str is None:
        return None
    try:
        pga_str = str(pga_str).lower().strip()
        matches = re.findall(r'(\d+\.?\d*)g', pga_str)
        if matches:
            values = [float(m) for m in matches]
            return float(np.mean(values))
        return None
    except Exception:
        return None


def parse_relative_density(value) -> Optional[float]:
    if pd.isna(value) or value == '' or value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass

    density_map = {
        'very loose': 15.0,
        'loose': 35.0,
        'loose to medium dense': 50.0,
        'medium': 50.0,
        'medium dense': 65.0,
        'dense': 80.0,
        'very dense': 95.0,
        'hard': 90.0,
    }
    value_str = str(value).lower().strip()
    return density_map.get(value_str, None)


def extract_depth_range(depth_layer: str) -> Tuple[float, float]:
    try:
        parts = str(depth_layer).replace('m', '').split('-')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return 0.0, 1.5


def upload_bytes_to_supabase_storage(file_bytes, bucket_name, storage_path, client, content_type='text/csv'):
    """Upload file bytes directly to Supabase Storage (no local files)
    Requires an active `client` returned by `get_supabase_client()`.
    """
    try:
        if client is None:
            return False
        client.storage.from_(bucket_name).upload(
            storage_path,
            file_bytes,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )
        return True
    except Exception:
        return False


def safe_float(value, default=0.0):
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def load_medians_from_csv_or_bytes(local_path=None, bytes_obj=None):
    medians = {}
    try:
        if local_path is not None:
            df = pd.read_csv(local_path)
        elif bytes_obj is not None:
            df = pd.read_csv(io.BytesIO(bytes_obj))
        else:
            return medians

        med = df.median(numeric_only=True)
        medians = med.to_dict()
    except Exception:
        pass
    return medians


def compute_borehole_aggregates(bh_df: pd.DataFrame, spt_col='spt_n_value', qa_col='qa_allowable_kpa', defaults=None):
    if defaults is None:
        defaults = {}
    spt_default = defaults.get('spt', 15.0)
    qa_default = defaults.get('qa', 1000.0)

    if spt_col not in bh_df.columns or bh_df[spt_col].isna().all():
        bh_avg_spt = spt_default
        bh_min_spt = spt_default
        bh_max_spt = spt_default
        bh_std_spt = 0.0
    else:
        bh_avg_spt = float(bh_df[spt_col].mean())
        bh_min_spt = float(bh_df[spt_col].min())
        bh_max_spt = float(bh_df[spt_col].max())
        bh_std_spt = float(bh_df[spt_col].std()) if not np.isnan(
            bh_df[spt_col].std()) else 0.0

    if qa_col not in bh_df.columns or bh_df[qa_col].isna().all():
        bh_avg_qa = qa_default
        bh_min_qa = qa_default
        bh_max_qa = qa_default
    else:
        bh_avg_qa = float(bh_df[qa_col].mean())
        bh_min_qa = float(bh_df[qa_col].min())
        bh_max_qa = float(bh_df[qa_col].max())

    return {
        'bh_avg_spt': bh_avg_spt,
        'bh_min_spt': bh_min_spt,
        'bh_max_spt': bh_max_spt,
        'bh_std_spt': bh_std_spt,
        'bh_avg_qa': bh_avg_qa,
        'bh_min_qa': bh_min_qa,
        'bh_max_qa': bh_max_qa
    }


def compute_layer_aggregates(df_all: pd.DataFrame, layer_number, spt_col='spt_n_value'):
    layer_defaults = {
        'layer_avg_spt': None,
        'layer_std_spt': 0.0,
        'layer_avg_unit_weight': None,
        'layer_avg_fines': None,
        'layer_avg_qa': None
    }
    try:
        if layer_number is None:
            return layer_defaults
        layer_df = df_all[df_all.get('layer_number') == layer_number]
        if layer_df is None or len(layer_df) == 0:
            return layer_defaults

        if spt_col in layer_df.columns and not layer_df[spt_col].isna().all():
            layer_defaults['layer_avg_spt'] = float(layer_df[spt_col].mean())
            layer_defaults['layer_std_spt'] = float(
                layer_df[spt_col].std()) if not np.isnan(layer_df[spt_col].std()) else 0.0
        if 'unit_weight' in layer_df.columns and not layer_df['unit_weight'].isna().all():
            layer_defaults['layer_avg_unit_weight'] = float(
                layer_df['unit_weight'].mean())
        if 'fines_content' in layer_df.columns and not layer_df['fines_content'].isna().all():
            layer_defaults['layer_avg_fines'] = float(
                layer_df['fines_content'].mean())
        if 'qa_allowable_kpa' in layer_df.columns and not layer_df['qa_allowable_kpa'].isna().all():
            layer_defaults['layer_avg_qa'] = float(
                layer_df['qa_allowable_kpa'].mean())

        return layer_defaults
    except Exception:
        return layer_defaults


def fetch_muni_stats(client, municipality_name: str):
    defaults = {
        'muni_avg_spt_n': None,
        'avg_unit_weight': None,
        'avg_bearing_capacity_kpa': None,
        'borehole_count': None,
        'total_samples': None
    }
    try:
        if client is None or municipality_name is None:
            return defaults
        res = client.table('v_municipality_statistics').select(
            '*').eq('municipality', municipality_name).limit(1).execute()
        if res and getattr(res, 'data', None):
            row = res.data[0]
            defaults['muni_avg_spt_n'] = row.get(
                'avg_spt_n') if 'avg_spt_n' in row else row.get('muni_avg_spt_n')
            defaults['avg_unit_weight'] = row.get(
                'avg_unit_weight') if 'avg_unit_weight' in row else row.get('unit_weight')
            defaults['avg_bearing_capacity_kpa'] = row.get(
                'avg_bearing_capacity_kpa') if 'avg_bearing_capacity_kpa' in row else row.get('avg_bc_kpa')
            defaults['borehole_count'] = row.get('borehole_count')
            defaults['total_samples'] = row.get('total_samples')
    except Exception:
        pass
    return defaults


def calculate_liquefaction_probability(csr, crr, spt_n160, fines_pct=None):
    """
    Calculate liquefaction probability (0-100%) using Factor of Safety.

    Args:
        csr: Cyclic Stress Ratio
        crr: Cyclic Resistance Ratio (computed from SPT N1(60))
        spt_n160: SPT N1(60) value
        fines_pct: Fines content percentage (optional, used for adjustment)

    Returns:
        Liquefaction probability (0-100%)
    """
    try:
        if crr <= 0 or csr <= 0:
            return 0.0

        # Factor of Safety against liquefaction
        fs = crr / (csr + 1e-6)

        # Convert FS to probability using empirical relationship
        # FS > 1.5: very low risk (0-5%)
        # FS 1.0-1.5: low risk (5-25%)
        # FS 0.5-1.0: moderate-high risk (25-75%)
        # FS < 0.5: very high risk (75-100%)

        if fs >= 1.5:
            prob = 0.0
        elif fs >= 1.0:
            prob = (1.5 - fs) / 0.5 * 25  # Linear interpolation from 0-25%
        elif fs >= 0.5:
            prob = 25 + (1.0 - fs) / 0.5 * 50  # Linear from 25-75%
        else:
            prob = 75 + min(0.5, 0.5 - fs) / 0.5 * 25  # Linear from 75-100%

        # Adjust for fines content (clean sand more susceptible)
        if fines_pct is not None:
            fines_pct = safe_float(fines_pct, 15.0)
            if fines_pct < 5:
                # Clean sand: increase susceptibility
                prob = prob * 1.15
            elif fines_pct > 35:
                # Fine-grained: decrease susceptibility
                prob = prob * 0.7

        # Clamp to 0-100
        return float(np.clip(prob, 0, 100))
    except Exception:
        return 0.0


def calculate_settlement_cm(spt_n160, depth_mid_m, effective_stress_kpa,
                            qa_allowable_kpa, fines_pct=None,
                            liquefaction_prob=None, foundation_width_m=1.0):
    """
    Estimate settlement (cm) based on SPT N1(60), depth, stress, and bearing capacity.

    Uses correlations from Meyerhof, Schmertmann, and load-bearing methods.

    Args:
        spt_n160: SPT N1(60) value
        depth_mid_m: Depth to mid-layer (m)
        effective_stress_kpa: Effective overburden pressure (kPa)
        qa_allowable_kpa: Allowable bearing capacity (kPa)
        fines_pct: Fines content (optional)
        liquefaction_prob: Liquefaction probability (optional; increases settlement)
        foundation_width_m: Foundation width (default 1.0 m)

    Returns:
        Settlement estimate in cm
    """
    try:
        spt_n160 = safe_float(spt_n160, 15.0)
        depth_mid_m = safe_float(depth_mid_m, 0.75)
        effective_stress_kpa = safe_float(effective_stress_kpa, 15.0)
        qa_allowable_kpa = safe_float(qa_allowable_kpa, 1000.0)
        fines_pct = safe_float(
            fines_pct, 15.0) if fines_pct is not None else 15.0
        foundation_width_m = safe_float(foundation_width_m, 1.0)

        # Base settlement from Meyerhof correlation (SPT-based)
        # Settlement ~ 1 / (SPT N-value) * stress factor
        # Higher SPT N = lower settlement
        if spt_n160 > 0:
            settlement_cm = max(0.5, (100 / spt_n160) *
                                (effective_stress_kpa / 100.0))
        else:
            settlement_cm = 2.0

        # Adjust for depth (deeper = slightly less settlement due to confinement)
        depth_factor = np.clip(1.0 - (depth_mid_m / 20.0), 0.5, 1.0)
        settlement_cm *= depth_factor

        # Adjust for fines content (fines increase compressibility)
        if fines_pct > 35:
            settlement_cm *= 1.3  # Fine-grained soils settle more
        elif fines_pct < 5:
            settlement_cm *= 0.8  # Clean sand settles less

        # Width factor (Schmertmann: wider foundations settle more)
        width_factor = 1.0 + (foundation_width_m - 1.0) * 0.1
        settlement_cm *= width_factor

        # Amplify settlement if liquefaction is likely
        if liquefaction_prob is not None:
            liquefaction_prob = safe_float(liquefaction_prob, 0.0)
            if liquefaction_prob > 50:
                # Post-liquefaction settlement amplification
                settlement_cm *= (1.0 + liquefaction_prob / 100.0)

        # Clamp to reasonable range (0-15 cm for typical geotechnical predictions)
        return float(np.clip(settlement_cm, 0, 15))
    except Exception:
        return 0.5  # Default to minimal settlement on error
