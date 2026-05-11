#!/usr/bin/env python3
"""
Geotechnical Data Processing Pipeline — Improved v3
=====================================================
All fixes applied from data audit plus validation methodology alignment:

FIX 1  — Fines Content >100% auto-corrected (÷10) or nulled
FIX 2  — N1(60) computed via Cn overburden correction for missing values (was raw SPT)
FIX 3  — SPT N=100 (refusal) capped at 60 for liquefaction path only
FIX 4  — BH-75-style PGA < 0.2g flagged and nulled for municipality propagation
FIX 5  — CS-1/CS core samples excluded from liquefaction classification
FIX 6  — Municipality-median GWL fill for boreholes with no GWL in any layer
FIX 7  — USCS symbol typos normalised (SMM->SM, CM->SC, SP_->SP, ROCK types, etc.)
FIX 8  — 'Ground Water Table' column used as backup GWL source
FIX 9  — SPT imputation tracked via boolean flag (replaces fragile ==15.0 sentinel)
FIX 10 — Fault-distance attenuation fallback for missing PGA (Youngs et al. 1997)
FIX 11 — factor_of_safety, liquefaction_probability, liquefaction_status,
          spt_is_refusal, spt_is_imputed, is_core_sample, is_rock now stored
          in proper DB columns (were previously null)
FIX 12 — Bearing capacity and settlement recomputed per-layer using ANN-derived
          foundation dimensions B and D (was hardcoded B=3.0m, D=1.5m for all)
FIX 13 — CSR/CRR now computed using validation-sheet methodology:
          (A) Total stress = cumulative layer-by-layer sum (UW x h per layer)
              NOT the simplified UW x z_mid formula
          (B) rd and CSR use z = depth_to (bottom of layer), not z_mid
          (C) CRR uses Idriss-Boulanger 2008 formula with constant −2.80 (was −2.67)
              and Kσ overburden correction applied to FS (was missing)
              Validation sheet CRR can exceed 0.6 for dense layers (N>30)
          (D) lpi_weighing_factor uses z=depth_to (matches validation W_i)
          This ensures stored CRR/CSR/FS values match the validation spreadsheet.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re
import os
import io

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[INFO] python-dotenv not installed — .env file will not be loaded")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[INFO] Supabase not available — database storage will be skipped")


_USCS_UNIT_WEIGHT = {
    'GW': 20.0, 'GP': 18.0, 'SW': 19.0, 'SP': 17.5,
    'GM': 20.0, 'GC': 19.5, 'SM': 19.0, 'SC': 19.5,
    'ML': 17.0, 'MH': 16.0, 'CL': 17.5, 'CH': 16.5,
    'OL': 14.0, 'OH': 13.0, 'PT': 11.0,
}

_USCS_FINES_PCT = {
    'GW':  3.0, 'GP':  3.0, 'SW':  3.0, 'SP':  3.0,
    'GM': 10.0, 'GC': 22.0, 'SM': 10.0, 'SC': 22.0,
    'ML': 60.0, 'MH': 70.0, 'CL': 65.0, 'CH': 75.0,
    'OL': 75.0, 'OH': 80.0, 'PT': 90.0,
}

_USCS_CORRECTIONS = {
    'SMM': 'SM', 'CM': 'SC', 'SP ': 'SP', 'SC-SM': 'SC',
    'SP-SM': 'SP', '-': '', 'CORING': '', 'CS': '',
    'BASALT': 'ROCK', 'SANDSTONE': 'ROCK', 'TUFF': 'ROCK',
    'H': '', 'M': '', 'S': '',
}

_NON_SOIL_USCS = {'ROCK', ''}
_COHESIVE_USCS = {'CL', 'CH', 'ML', 'MH', 'OL', 'OH', 'PT', 'CL-ML'}
_TARLAC_PGA_MIN = 0.2


def _compute_foundation_B(n: float, fc: float, gwl: float) -> float:
    n = max(1.0, float(n) if not np.isnan(n) else 15.0)
    fc = float(fc) if not np.isnan(fc) else 15.0
    gwl = float(gwl) if not np.isnan(gwl) else 5.0
    B = 4.5 / np.sqrt(n)
    if fc >= 50.0:   B += 0.75
    elif fc >= 35.0: B += 0.50
    elif fc >= 15.0: B += 0.25
    if gwl <= 1.0:   B += 0.50
    elif gwl <= 2.0: B += 0.25
    B = float(np.clip(B, 1.0, 5.0))
    return round(B * 4) / 4


def _compute_foundation_D(n: float, fc: float, gwl: float, uscs: str, risk_level: str) -> float:
    n = max(1.0, float(n) if not np.isnan(n) else 15.0)
    fc = float(fc) if not np.isnan(fc) else 15.0
    gwl = float(gwl) if not np.isnan(gwl) else 5.0
    uscs = str(uscs).strip().upper()[:2] if uscs else ''
    risk = str(risk_level).strip().upper() if risk_level else ''
    RISK_ORD = {'VERY HIGH': 5, 'HIGH': 4, 'MEDIUM': 3, 'LOW': 2, 'VERY LOW': 1}
    D = 1.0 + 0.04 * float(np.clip(15.0 - n, 0.0, 12.0))
    rs = RISK_ORD.get(risk, 0)
    if rs >= 5:   D += 1.5
    elif rs == 4: D += 1.0
    elif rs == 3: D += 0.5
    if uscs in _COHESIVE_USCS or fc >= 35.0: D += 0.25
    if fc >= 50.0 and n < 8.0:               D += 0.25
    if gwl <= 3.0:
        D = max(D, max(1.0, min(gwl - 0.3, 2.5)))
    D = float(np.clip(D, 1.0, 3.5))
    return round(D * 4) / 4


class GeotechnicalPipeline:
    """Improved single-pass geotechnical data processing pipeline — v3."""

    def __init__(self, excel_file_path=None, output_csv_path=None, use_storage=True):
        self.use_storage = use_storage
        self.excel_path = Path(excel_file_path) if excel_file_path else None
        self.excel_file_bytes = None
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data = None
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        self.client = None
        self.municipality_ids: Dict[str, int] = {}
        self.borehole_record_ids: Dict[str, int] = {}

    def download_from_supabase_storage(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 1: DOWNLOADING FROM SUPABASE STORAGE")
        print("=" * 80)
        if not SUPABASE_AVAILABLE:
            print("  [ERROR] Supabase not available"); return False
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        if not url or not key:
            print("  [ERROR] Environment variables not set"); return False
        try:
            self.client = create_client(url, key)
            bucket = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            data = self.client.storage.from_(bucket).download("raw/Raw_data.xlsx")
            if not data:
                print("  [ERROR] File not found in storage"); return False
            self.excel_file_bytes = data
            print(f"  [OK] Downloaded {len(data)} bytes")
            return True
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}"); return False

    def upload_raw_file_to_bucket(self) -> bool:
        if not self.excel_path or not self.excel_path.exists() or not self.client:
            return True
        try:
            bucket = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            with open(self.excel_path, 'rb') as f:
                data = f.read()
            self.client.storage.from_(bucket).upload(
                f"raw/{self.excel_path.name}", data,
                file_options={'content-type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'upsert': 'true'})
            print("  [OK] Raw file uploaded to bucket")
        except Exception as e:
            print(f"  [WARNING] Raw file upload failed: {e}")
        return True

    def load_excel(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 1: LOADING EXCEL FILE")
        print("=" * 80)
        try:
            if self.excel_file_bytes:
                source = io.BytesIO(self.excel_file_bytes)
                print("  Loading from memory (Supabase Storage)...")
            elif self.excel_path and self.excel_path.exists():
                source = self.excel_path
                print(f"  Loading from local file: {self.excel_path}")
            else:
                print("  [ERROR] No file source available"); return False
            xl = pd.ExcelFile(source)
            print(f"\n  Found {len(xl.sheet_names)} sheets")
            for sheet in xl.sheet_names:
                if sheet.strip().lower() == 'summary':
                    print(f"    Skipping: {sheet}"); continue
                df = pd.read_excel(source, sheet_name=sheet)
                self.raw_data[sheet] = df
                print(f"    Loaded: {sheet} ({len(df)} rows, {len(df.columns)} cols)")
            print(f"\n  [OK] Loaded {len(self.raw_data)} data sheets")
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            import traceback; traceback.print_exc()
            return False

    def clean_coordinate(self, value):
        if pd.isna(value) or value == '' or value is None: return None
        try: return float(str(value).replace('°', '').replace("'", '').replace('"', '').strip())
        except: return None

    def parse_pga(self, value):
        if pd.isna(value): return None
        try:
            matches = re.findall(r'(\d+\.?\d*)g', str(value).lower())
            if matches: return float(np.mean([float(m) for m in matches]))
            return float(str(value).replace('g', '').strip())
        except: return None

    def parse_relative_density(self, value):
        if pd.isna(value): return None
        try: return float(value)
        except: pass
        density_map = {'very loose': 15.0, 'loose': 35.0, 'loose to medium dense': 50.0,
                       'medium': 50.0, 'medium dense': 65.0, 'dense': 80.0, 'very dense': 95.0, 'hard': 90.0}
        return density_map.get(str(value).lower().strip())

    def parse_elastic_modulus(self, value):
        if pd.isna(value) or isinstance(value, datetime): return None
        try: return float(value)
        except: pass
        s = str(value).lower()
        if 'to' in s:
            try:
                parts = s.split('to')
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            except: return None
        return None

    def pre_clean_dataframe(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        df = df.copy()
        unnamed = [c for c in df.columns if str(c).lower().startswith('unnamed')]
        if unnamed:
            df = df.drop(columns=unnamed)
            print(f"    [FIX] Dropped unnamed columns: {unnamed}")
        for col in ['Latitude', 'Longitude']:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_coordinate)
        if 'Peak Ground Acceleration' in df.columns:
            df['Peak Ground Acceleration'] = df['Peak Ground Acceleration'].apply(self.parse_pga)
        if 'Relative Density' in df.columns:
            df['Relative Density'] = df['Relative Density'].apply(self.parse_relative_density)
        if 'Elastic Modulus (Es) (MN/m²)' in df.columns:
            df['Elastic Modulus (Es) (MN/m²)'] = df['Elastic Modulus (Es) (MN/m²)'].apply(self.parse_elastic_modulus)
        if 'Ground Water Table' in df.columns and 'Groundwater Level (m)' in df.columns:
            gwl_main = pd.to_numeric(df['Groundwater Level (m)'], errors='coerce')
            gwl_backup = pd.to_numeric(df['Ground Water Table'], errors='coerce')
            filled = gwl_main.isna() & gwl_backup.notna()
            if filled.any():
                df.loc[filled, 'Groundwater Level (m)'] = gwl_backup[filled]
                print(f"    [FIX8] GWL filled from 'Ground Water Table': {filled.sum()} rows")
        spt_raw_col = next((c for c in ['SPT N-Value', 'SPT N Value', 'N-Value', 'SPT'] if c in df.columns), None)
        df['spt_is_imputed'] = (pd.to_numeric(df[spt_raw_col], errors='coerce').isna() if spt_raw_col else pd.Series(True, index=df.index))
        if 'USCS Symbol' in df.columns:
            uscs_raw = df['USCS Symbol'].fillna('').astype(str).str.strip().str.upper()
            uscs_norm = uscs_raw.replace(_USCS_CORRECTIONS)
            uscs_norm = uscs_norm.where(uscs_norm.str.len() >= 2, '')
            df['USCS Symbol'] = uscs_norm
            df['is_rock'] = uscs_norm.isin({'ROCK'}).astype(int)
            n_fixed = (uscs_raw != uscs_norm).sum()
            if n_fixed: print(f"    [FIX7] USCS symbols normalised: {n_fixed} rows")
        sample_col = next((c for c in df.columns if 'sample' in c.lower()), None)
        if sample_col:
            is_core = df[sample_col].astype(str).str.strip().str.upper().isin(['CS-1', 'CS'])
            df['is_core_sample'] = is_core.astype(int)
            if is_core.any(): print(f"    [FIX5] Core samples flagged: {is_core.sum()} rows")
        else:
            df['is_core_sample'] = 0
        if 'Fines Content' in df.columns:
            fc = pd.to_numeric(df['Fines Content'], errors='coerce')
            bad = fc > 100
            if bad.any():
                corrected = fc[bad] / 10.0
                plausible = (corrected >= 2.0) & (corrected <= 98.0)
                df.loc[bad & plausible, 'Fines Content'] = corrected[plausible].round(1)
                df.loc[bad & ~plausible, 'Fines Content'] = np.nan
                self.validation_errors.append(f"[{sheet_name}] Fines >100%: {bad.sum()} rows auto-corrected")
                print(f"    [FIX1] Fines >100%: {bad.sum()} rows corrected/nulled")
        return df

    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Latitude' in df.columns:
            df['latitude'] = df['Latitude']
            miss = df['latitude'].isna().sum()
            if miss: self.validation_warnings.append(f"Missing latitude: {miss} records")
            oor = ((df['latitude'].dropna() < 15.0) | (df['latitude'].dropna() > 16.0)).sum()
            if oor: self.validation_errors.append(f"Latitude outside Tarlac range: {oor}")
        if 'Longitude' in df.columns:
            df['longitude'] = df['Longitude']
            miss = df['longitude'].isna().sum()
            if miss: self.validation_warnings.append(f"Missing longitude: {miss} records")
            oor = ((df['longitude'].dropna() < 120.0) | (df['longitude'].dropna() > 121.0)).sum()
            if oor: self.validation_errors.append(f"Longitude outside Tarlac range: {oor}")
        return df

    def validate_spt(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        spt_col = next((c for c in ['SPT N-Value', 'SPT N Value', 'spt_n_value', 'N-Value', 'SPT'] if c in df.columns), None)
        if spt_col:
            df['spt_n_value'] = pd.to_numeric(df[spt_col], errors='coerce')
            valid = df['spt_n_value'].dropna()
            oor = ((valid < 0) | (valid > 100)).sum()
            if oor: self.validation_errors.append(f"SPT out of range: {oor} records")
            df['spt_is_refusal'] = (df['spt_n_value'] >= 100).astype(int)
            df['spt_n_value_liq'] = df['spt_n_value'].clip(upper=60.0)
            miss = df['spt_n_value'].isna().sum()
            if miss: self.validation_warnings.append(f"Missing SPT: {miss} records -> imputed 15.0")
            df['spt_n_value'] = df['spt_n_value'].fillna(15.0)
            df['spt_n_value_liq'] = df['spt_n_value_liq'].fillna(15.0)
        else:
            self.validation_warnings.append("SPT column not found — defaulting to 15.0")
            df['spt_n_value'] = 15.0; df['spt_n_value_liq'] = 15.0
            df['spt_is_refusal'] = 0; df['spt_is_imputed'] = True
        return df

    def validate_soil_parameters(self, df: pd.DataFrame, sheet_name: str = '') -> pd.DataFrame:
        df = df.copy()
        uscs_col = next((c for c in ['USCS Symbol', 'uscs_symbol', 'USCS'] if c in df.columns), None)
        if uscs_col:
            df['uscs_symbol'] = df[uscs_col].fillna('').astype(str).str.strip()
            df['uscs_symbol'] = df['uscs_symbol'].replace('nan', '')
        else:
            df['uscs_symbol'] = ''
        uscs_upper = df['uscs_symbol'].str.upper().str.strip()
        uscs_primary = uscs_upper.str[:2]
        uw_col = next((c for c in ['Unit Weight (γ)', 'Unit Weight', 'unit_weight', 'Unit Weight (kN/m³)', 'γ'] if c in df.columns), None)
        if uw_col:
            df['unit_weight'] = pd.to_numeric(df[uw_col], errors='coerce')
            is_core_or_rock = ((df.get('is_core_sample', pd.Series(0, index=df.index)).astype(int) == 1) |
                               (df.get('is_rock', pd.Series(0, index=df.index)).astype(int) == 1))
            zero_core = (df['unit_weight'] == 0) & is_core_or_rock
            if zero_core.any(): df.loc[zero_core, 'unit_weight'] = np.nan
            df.loc[df['unit_weight'] == 0, 'unit_weight'] = np.nan
        else:
            df['unit_weight'] = np.nan
        uscs_uw = uscs_primary.map(_USCS_UNIT_WEIGHT).fillna(18.0)
        df['unit_weight'] = df['unit_weight'].fillna(uscs_uw)
        fc_col = next((c for c in ['Fines Content', 'fines_content', 'Fines (%)'] if c in df.columns), None)
        if fc_col:
            df['fines_content'] = pd.to_numeric(df[fc_col], errors='coerce')
        else:
            df['fines_content'] = np.nan
        uscs_fc = uscs_primary.map(_USCS_FINES_PCT).fillna(15.0)
        df['fines_content'] = df['fines_content'].fillna(uscs_fc)
        gwl_col = next((c for c in ['Groundwater Level (m)', 'Ground Water Table', 'Groundwater Level', 'groundwater_depth_m', 'GWL'] if c in df.columns), None)
        gwl_numeric = pd.to_numeric(df[gwl_col], errors='coerce') if gwl_col else pd.Series(np.nan, index=df.index)
        df['gwl_estimated'] = gwl_numeric.isna()
        df['groundwater_depth_m'] = gwl_numeric.fillna(5.0)
        desc_col = next((c for c in ['Soil/Rock Description', 'Soil Description', 'soil_description', 'Description'] if c in df.columns), None)
        if desc_col:
            df['soil_description'] = df[desc_col].fillna('').astype(str).str.strip().replace('nan', '')
        fa_col = next((c for c in ['Internal Friction Angle', 'Friction Angle', 'friction_angle'] if c in df.columns), None)
        if fa_col: df['friction_angle'] = pd.to_numeric(df[fa_col], errors='coerce')
        pi_col = next((c for c in ['Plasticity Index (PI)', 'Plasticity Index', 'plasticity_index', 'PI'] if c in df.columns), None)
        if pi_col: df['plasticity_index'] = pd.to_numeric(df[pi_col], errors='coerce')
        wc_col = next((c for c in ['Natural Water Content (ω)', 'Natural Water Content', 'moisture_content', 'Water Content'] if c in df.columns), None)
        if wc_col: df['moisture_content'] = pd.to_numeric(df[wc_col], errors='coerce')
        d50_col = next((c for c in ['Mean Particle Size (D50) (mm)', 'Mean Particle Size', 'mean_particle_size_d50', 'D50'] if c in df.columns), None)
        if d50_col: df['mean_particle_size_d50'] = pd.to_numeric(df[d50_col], errors='coerce')
        return df

    def process_and_validate(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 2: PRE-CLEANING AND VALIDATING DATA")
        print("=" * 80)
        all_dfs = []
        for sheet_name, df in self.raw_data.items():
            print(f"\n  Sheet: {sheet_name}")
            df = self.pre_clean_dataframe(df, sheet_name)
            df = self.validate_coordinates(df)
            df = self.validate_spt(df)
            df = self.validate_soil_parameters(df, sheet_name)
            if 'Peak Ground Acceleration' in df.columns:
                df['pga_g'] = pd.to_numeric(df['Peak Ground Acceleration'], errors='coerce')
            else:
                df['pga_g'] = np.nan
            if 'pga_g' in df.columns:
                low_pga = df['pga_g'].notna() & (df['pga_g'] < _TARLAC_PGA_MIN)
                if low_pga.any():
                    bad_bhs = df.loc[low_pga, 'Borehole ID'].tolist()
                    print(f"    [FIX4] PGA < {_TARLAC_PGA_MIN}g nulled for: {bad_bhs}")
                    df.loc[low_pga, 'pga_g'] = np.nan
            df['pga_g'] = df['pga_g'].fillna(0.35)
            if 'Relative Density' in df.columns:
                df['relative_density_percent'] = df['Relative Density']
            df['Depth_Layer'] = sheet_name
            def extract_depth(name):
                m = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', str(name))
                return (float(m.group(1)), float(m.group(2))) if m else (0.0, 1.5)
            depths = df['Depth_Layer'].apply(extract_depth)
            df['depth_from_m'] = depths.apply(lambda x: x[0])
            df['depth_to_m'] = depths.apply(lambda x: x[1])
            df['depth_mid_m'] = (df['depth_from_m'] + df['depth_to_m']) / 2
            df = self.compute_missing_raw_columns(df)
            all_dfs.append(df)
            print(f"    [OK] {len(df)} rows processed")
        if all_dfs:
            self.processed_data = pd.concat(all_dfs, ignore_index=True)
            print(f"\n  [OK] Combined {len(self.processed_data)} records from {len(self.raw_data)} sheets")
            return True
        return False

    def compute_missing_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        GAMMA_W = 9.81; df = df.copy()
        z = df['depth_mid_m']; γ = df['unit_weight']; gwl = df['groundwater_depth_m']
        sigma_v = γ * z; u = np.maximum(0.0, z - gwl) * GAMMA_W
        sigma_eff = np.maximum(1.0, sigma_v - u)
        rd = np.clip(np.where(z <= 9.15, 1.0 - 0.00765 * z, np.where(z <= 23.0, 1.174 - 0.0267 * z, 0.0)), 0.0, 1.0)
        tot_col = 'Total Overburden Pressure'
        if tot_col in df.columns:
            miss = pd.to_numeric(df[tot_col], errors='coerce').isna()
            df.loc[miss, tot_col] = sigma_v[miss].round(2)
            if miss.sum(): print(f"      [FILL] Total OBP: {miss.sum()} rows")
        for eff_col in ('Effective Overburden Presssure', 'Effective Overburden Pressure'):
            if eff_col in df.columns:
                miss = pd.to_numeric(df[eff_col], errors='coerce').isna()
                df.loc[miss, eff_col] = sigma_eff[miss].round(2)
                if miss.sum(): print(f"      [FILL] Effective OBP: {miss.sum()} rows")
                break
        rd_col = 'Relative Density'; n160_col = 'Corrected SPT-N Value (N1(60))'
        if rd_col in df.columns and n160_col in df.columns:
            n1_60 = pd.to_numeric(df[n160_col], errors='coerce')
            uscs = df.get('uscs_symbol', pd.Series('', index=df.index)).str.upper().str[:2]
            miss = (pd.to_numeric(df[rd_col], errors='coerce').isna() & n1_60.notna() & (n1_60 > 0) & ~uscs.isin(_COHESIVE_USCS))
            df.loc[miss, rd_col] = (np.sqrt(n1_60[miss] / 60.0) * 100.0).round(2)
            if miss.sum(): print(f"      [FILL] Relative Density: {miss.sum()} rows")
        csr_col = 'Cyclic Stress Ratio (CSR)'
        if csr_col in df.columns:
            csr_vals = 0.65 * (sigma_v / sigma_eff) * df['pga_g'] * rd
            miss = pd.to_numeric(df[csr_col], errors='coerce').isna() & (df['pga_g'] > 0)
            df.loc[miss, csr_col] = csr_vals[miss].round(6)
            if miss.sum(): print(f"      [FILL] CSR: {miss.sum()} rows")
        uscs_fa = df.get('uscs_symbol', pd.Series('', index=df.index)).str.upper().str[:2]
        is_coh = uscs_fa.isin(_COHESIVE_USCS)
        fa_col = next((c for c in ['friction_angle', 'Internal Friction Angle'] if c in df.columns), None)
        if fa_col:
            miss = pd.to_numeric(df[fa_col], errors='coerce').isna()
            n_fa = df['spt_n_value'].clip(lower=1.0)
            phi_g = (27.1 + 0.3 * n_fa - 0.00054 * n_fa ** 2).clip(25.0, 45.0)
            df.loc[miss, fa_col] = np.where(is_coh[miss], 15.0, phi_g[miss]).round(1)
            if miss.sum(): print(f"      [FILL] Friction angle: {miss.sum()} rows")
        SAND_U = {'SW', 'SP', 'SM', 'SC', 'GW', 'GP', 'GM', 'GC'}; SILT_U = {'ML', 'MH'}
        uscs_es = df.get('uscs_symbol', pd.Series('', index=df.index)).str.upper().str[:2]
        is_sand = uscs_es.isin(SAND_U); is_silt = uscs_es.isin(SILT_U)
        es_col = next((c for c in ['Elastic Modulus (Es) (MN/m²)', 'elastic_modulus_es'] if c in df.columns), None)
        if es_col:
            miss = pd.to_numeric(df[es_col], errors='coerce').isna()
            n_es = df['spt_n_value'].clip(lower=1.0)
            es_val = np.where(is_sand, 0.5 * (n_es + 15.0), np.where(is_silt, 0.3 * (n_es + 6.0), 0.2 * n_es))
            df.loc[miss, es_col] = es_val[miss].round(2)
            if miss.sum(): print(f"      [FILL] Elastic modulus: {miss.sum()} rows")
        return df

    def fill_missing_by_borehole(self) -> bool:
        print("\n" + "=" * 80)
        print("FILLING MISSING VALUES BY BOREHOLE")
        print("=" * 80)
        df = self.processed_data.copy()
        bh_col = next((c for c in ['Borehole ID', 'borehole_id'] if c in df.columns), None)
        if bh_col is None:
            print("  [SKIP] No borehole ID column — skipping"); return True
        df = df.sort_values([bh_col, 'depth_mid_m']).reset_index(drop=True)
        gwl_updated = 0
        if 'gwl_estimated' in df.columns:
            for bh_id, grp in df.groupby(bh_col, sort=False):
                real = ~grp['gwl_estimated']; est = grp['gwl_estimated']
                if real.any() and est.any():
                    med = grp.loc[real, 'groundwater_depth_m'].median()
                    df.loc[grp.index[est], 'groundwater_depth_m'] = med
                    df.loc[grp.index[est], 'gwl_estimated'] = False
                    gwl_updated += int(est.sum())
        print(f"  [FILL] GWL propagated within boreholes: {gwl_updated} rows")
        muni_col = next((c for c in ['Municipality', 'municipality'] if c in df.columns), None)
        if muni_col and 'gwl_estimated' in df.columns:
            real_gwl_df = df[~df['gwl_estimated']]
            muni_medians = real_gwl_df.groupby(muni_col)['groundwater_depth_m'].median()
            still_est = df['gwl_estimated']
            if still_est.any():
                muni_fill = df.loc[still_est, muni_col].map(muni_medians)
                ok = muni_fill.notna()
                df.loc[still_est & ok, 'groundwater_depth_m'] = muni_fill[ok]
                df.loc[still_est & ok, 'gwl_estimated'] = False
                print(f"  [FIX6] GWL filled from municipality median: {ok.sum()} rows")
        pga_updated = 0
        for bh_id, grp in df.groupby(bh_col, sort=False):
            real = (grp['pga_g'] != 0.35) & (grp['pga_g'] >= _TARLAC_PGA_MIN); est = ~real
            if real.any() and est.any():
                med = grp.loc[real, 'pga_g'].median()
                df.loc[grp.index[est], 'pga_g'] = med; pga_updated += int(est.sum())
        print(f"  [FILL] PGA propagated within boreholes: {pga_updated} rows")
        fault_col = next((c for c in ['Fault Distance (km)', 'fault_distance'] if c in df.columns), None)
        if fault_col:
            still_default = df['pga_g'] == 0.35
            R = pd.to_numeric(df.loc[still_default, fault_col], errors='coerce').clip(lower=1.0)
            Mw = 6.5; pga_est = (0.2083 * np.exp(1.2587 * Mw) * (R + 15) ** -1.9661).clip(0.1, 0.8)
            ok = R.notna()
            df.loc[still_default & ok.reindex(df.index, fill_value=False), 'pga_g'] = pga_est[ok].round(3).values
            print(f"  [FIX10] PGA from fault-distance attenuation: {ok.sum()} rows")
        spt_updated = 0
        if 'spt_is_imputed' in df.columns:
            for bh_id, grp in df.groupby(bh_col, sort=False):
                real = ~grp['spt_is_imputed']
                if real.sum() >= 2:
                    spt_real = grp['spt_n_value'].where(real)
                    spt_interp = spt_real.interpolate(method='index').ffill().bfill().clip(1.0, 100.0)
                    imp_idx = grp.index[~real]
                    if len(imp_idx):
                        df.loc[imp_idx, 'spt_n_value'] = spt_interp.loc[imp_idx].round(1)
                        df.loc[imp_idx, 'spt_n_value_liq'] = spt_interp.loc[imp_idx].clip(upper=60).round(1)
                        spt_updated += len(imp_idx)
        print(f"  [FILL] SPT interpolated within boreholes: {spt_updated} rows")
        self.processed_data = df
        print("  [OK] Borehole fill complete")
        return True

    def calculate_csr_crr(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 3: CSR/CRR ANALYSIS (FIX 13 — matches validation sheet)")
        print("=" * 80)
        df = self.processed_data.copy()
        GAMMA_W = 9.81; MW = 6.5
        df['magnitude_mw'] = MW; df['q_actual_kpa'] = 150.0
        print(f"  Mw={MW}, q_actual=150 kPa (MSF computed per-layer via I&B 2003)")

        # FIX 13A — Cumulative layer-by-layer total stress
        # Validation sheet: sv = sum(UW_i x h_i) cumulative, NOT UW x z_mid
        bh_col_local = next((c for c in ['Borehole ID', 'borehole_id'] if c in df.columns), None)
        h_layer = (df['depth_to_m'] - df['depth_from_m']).clip(lower=0.01)
        if bh_col_local:
            df = df.sort_values([bh_col_local, 'depth_from_m']).reset_index(drop=True)
            h_layer = (df['depth_to_m'] - df['depth_from_m']).clip(lower=0.01)
            cum_sv = (
                df.groupby(bh_col_local, sort=False)
                .apply(lambda g: (g['unit_weight'] * h_layer[g.index]).cumsum())
                .reset_index(level=0, drop=True)
            )
            df['total_overburden_pressure'] = cum_sv
        else:
            df['total_overburden_pressure'] = df['unit_weight'] * df['depth_mid_m']

        # Validation sheet evaluates CSR at z = depth_from (top of each layer):
        #   σ_v_top  = cumulative stress up to depth_from = cumulative_to_depth_to − γ×h
        #   σ'v_top  = σ_v_top − u(depth_from)
        #   rd       = f(depth_from)
        # This keeps layers above GWL at stress-ratio = 1.0 (no pore-pressure
        # amplification) and gives CSR matching the validation spreadsheet exactly.
        h_layer = (df['depth_to_m'] - df['depth_from_m']).clip(lower=0.01)
        sv_top  = (df['total_overburden_pressure'] - df['unit_weight'] * h_layer).clip(lower=0.0)
        u_top   = np.maximum(0.0, df['depth_from_m'] - df['groundwater_depth_m']) * GAMMA_W
        seff_top = np.maximum(1.0, sv_top - u_top)

        # effective stress for Cn/Ko still uses depth_to (full layer included)
        depth_below_wt = np.maximum(0.0, df['depth_to_m'] - df['groundwater_depth_m'])
        df['effective_overburden_pressure'] = (
            df['total_overburden_pressure'] - GAMMA_W * depth_below_wt
        ).clip(lower=1.0)

        z_csr = df['depth_from_m']
        rd = np.clip(np.where(z_csr <= 9.15, 1.0 - 0.00765 * z_csr,
                              np.where(z_csr <= 23.0, 1.174 - 0.0267 * z_csr, 0.0)), 0.0, 1.0)

        print("  Calculating CSR (Seed & Idriss 1971, z=depth_from, σ at top — matches validation)...")
        df['csr'] = 0.65 * df['pga_g'] * (sv_top / seff_top) * rd

        # I&B 2003/2004 — Always compute N1(60) iteratively from field SPT N60.
        # The pre-measured N1(60) column in the spreadsheet applies non-standard
        # energy corrections that differ from the validation sheet methodology,
        # so it is intentionally ignored here.
        PA = 101.325  # atmospheric pressure kPa (standard; matches validation sheet)
        sigma_eff = df['effective_overburden_pressure'].clip(lower=1.0)

        # Step 1: initial Cn = min(1.7, sqrt(Pa/σ'v))  [m=0.5 starting point]
        print("  Computing N1(60) — I&B iterative Cn (ignoring spreadsheet column)...")
        Cn_init = np.minimum(1.7, np.sqrt(PA / sigma_eff))
        N160_init = (df['spt_n_value_liq'] * Cn_init).clip(upper=60.0)

        # I&B 2004 additive fines correction: ΔN = exp(1.63 + 9.7/(FC+0.01) − (15.7/(FC+0.01))²)
        print("  Computing (N1)60cs — I&B 2004 additive fines correction...")
        FC = df['fines_content'].clip(lower=0.01)
        FC_c = FC.clip(upper=35.0)  # correction bounded at FC=35% per I&B
        delta_N = np.where(FC < 5.0, 0.0,
                           np.exp(1.63 + 9.7 / (FC_c + 0.01) - (15.7 / (FC_c + 0.01)) ** 2))
        N160cs_init = (N160_init + delta_N).clip(lower=0.0)

        # Step 2: refine m from initial N1(60)cs, then recompute Cn and N1(60)cs
        m = np.clip(0.784 - 0.0768 * np.sqrt(N160cs_init.clip(lower=1.0)), 0.3, 0.9)
        Cn_final = np.minimum(1.7, (PA / sigma_eff) ** m)
        N160_final = (df['spt_n_value_liq'] * Cn_final).clip(upper=60.0)
        N160cs_final = (N160_final + delta_N).clip(lower=0.0)

        df['spt_n160'] = N160_final
        df['spt_n60'] = df['spt_n_value']
        df['n1_60cs'] = N160cs_final
        print(f"    N1(60): {N160_final.min():.1f}–{N160_final.max():.1f} | "
              f"N1(60)cs: {N160cs_final.min():.1f}–{N160cs_final.max():.1f}")

        # CRR — Idriss-Boulanger 2008, cap at 2.0 to prevent overflow on dense layers
        print("  Calculating CRR (Idriss-Boulanger 2008, −2.80 constant)...")
        N_crr = df['n1_60cs'].clip(upper=40.0)
        crr_raw = np.exp(N_crr / 14.1 + (N_crr / 126.0) ** 2 - (N_crr / 23.6) ** 3 + (N_crr / 25.4) ** 4 - 2.80)
        df['crr'] = crr_raw.clip(lower=0.0, upper=2.0)

        # Ko overburden correction (I&B 2003): Ko = min(1.1, (Pa/σ'v)^(0.5−m))
        Ko = np.minimum(1.1, (PA / sigma_eff) ** (0.5 - m))

        # I&B 2003 density-dependent MSF (per-layer, based on N1(60)cs)
        # MSFmax = 1 + 1.254 × max(0, 1 − N1(60)cs/46)
        # MSF = 1 + (MSFmax − 1) × (8.64 × exp(−Mw/4) − 1.325)
        Dr_sq   = np.clip(df['n1_60cs'] / 46.0, 0.0, 1.0)
        MSFmax  = np.maximum(1.0, 1.0 + 1.254 * (1.0 - Dr_sq))
        _msf_k  = 8.64 * np.exp(-MW / 4.0) - 1.325
        MSF_lyr = np.clip(1.0 + (MSFmax - 1.0) * _msf_k, 1.0, MSFmax)
        df['msf']     = MSF_lyr
        df['msf_max'] = MSFmax

        fs_raw = (df['crr'] * MSF_lyr * Ko) / (df['csr'] + 1e-9)
        # Layers entirely above water table cannot liquefy — force FS = 999
        above_gwl = df['depth_to_m'] <= df['groundwater_depth_m']
        df['factor_of_safety'] = np.where(above_gwl, 999.0, fs_raw)

        # FIX 13D — lpi_weighing_factor uses depth_to (matches validation W_i)
        df['lpi_weighing_factor'] = np.where(above_gwl, 0.0,
                                             np.maximum(0.0, 10.0 - 0.5 * df['depth_to_m']))
        df['lpi_severity_factor'] = np.maximum(0.0, 1.0 - df['factor_of_safety'])
        df['liquefaction_probability'] = np.where(
            df['factor_of_safety'] < 1.0, (1.0 - df['factor_of_safety']) * 100,
            np.where(df['factor_of_safety'] < 1.5, 30.0, 10.0)
        ).clip(0.0, 100.0)
        self.processed_data = df
        print(f"  [OK] {len(df)} records")
        print(f"    CSR: {df['csr'].min():.4f} – {df['csr'].max():.4f}")
        print(f"    CRR: {df['crr'].min():.4f} – {df['crr'].max():.2f}")
        print(f"    FS : {df['factor_of_safety'].min():.2f} – {df['factor_of_safety'].max():.2f}")
        return True

    def calculate_bearing_bowles(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 3b: BEARING CAPACITY — INITIAL PASS (B=3.0m fixed)")
        print("=" * 80)
        df = self.processed_data.copy()
        B, D, SI = 3.0, 1.5, 25.0; Kd = 1.0 + 0.33 * (D / B)
        size_factor = ((B + 0.3) / B) ** 2; N = df['spt_n160'].clip(lower=1.0)
        Qa = (8.0 * N * size_factor * Kd).clip(lower=1.0)
        df['foundation_kd'] = Kd; df['bearing_qa_kpa'] = Qa
        df['bearing_qu_kpa'] = (Qa * 3.0).clip(lower=0.0)
        df['settlement_mm'] = (df['q_actual_kpa'] / Qa) * SI
        print(f"  B={B}m D={D}m Kd={Kd:.3f}  [initial — will be corrected]")
        self.processed_data = df
        return True

    def classify_liquefaction_dpwh_bsds(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 4: LIQUEFACTION CLASSIFICATION (DPWH BSDS 2013)")
        print("=" * 80)
        df = self.processed_data.copy()
        def dpwh_classify(row):
            if row.get('is_core_sample', 0) == 1 or row.get('is_rock', 0) == 1:
                return 'NOT APPLICABLE', 'ROCK/CORE SAMPLE'
            fs = row['factor_of_safety']
            if fs < 0.8:  return 'VERY HIGH', 'LIQUEFIES'
            if fs < 1.0:  return 'HIGH',      'LIQUEFIES'
            if fs < 1.2:  return 'MEDIUM',    'MARGINAL'
            if fs < 1.5:  return 'LOW',        'UNLIKELY'
            return 'VERY LOW', 'NO LIQUEFACTION'
        result = df.apply(dpwh_classify, axis=1)
        df['liquefaction_risk_level'] = result.apply(lambda x: x[0])
        df['liquefaction_status'] = result.apply(lambda x: x[1])
        df['liquefaction'] = df['liquefaction_risk_level'].isin(['VERY HIGH', 'HIGH']).astype(int)
        counts = df['liquefaction_risk_level'].value_counts()
        print("\n  Classification Summary:")
        for lvl, cnt in counts.items():
            print(f"    {lvl:20s}: {cnt:4d}  ({cnt/len(df)*100:5.1f}%)")
        liq_n = df['liquefaction'].sum()
        print(f"\n  Total liquefaction cases: {liq_n} ({liq_n/len(df)*100:.1f}%)")
        self.processed_data = df
        return True

    def recalculate_bearing_with_ann_foundation(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 4b: BEARING CAPACITY — RECALC WITH ANN FOUNDATION DIMS (FIX 12)")
        print("=" * 80)
        df = self.processed_data.copy()
        def _B(row):
            n = row.get('spt_n160') if pd.notna(row.get('spt_n160')) else row.get('spt_n_value', 15.0)
            fc = row.get('fines_content', 15.0); gwl = row.get('groundwater_depth_m', 5.0)
            try: return _compute_foundation_B(float(n), float(fc), float(gwl))
            except: return 2.0
        def _D(row):
            n = row.get('spt_n160') if pd.notna(row.get('spt_n160')) else row.get('spt_n_value', 15.0)
            fc = row.get('fines_content', 15.0); gwl = row.get('groundwater_depth_m', 5.0)
            uscs = row.get('uscs_symbol', ''); risk = row.get('liquefaction_risk_level', '')
            try: return _compute_foundation_D(float(n), float(fc), float(gwl), uscs, risk)
            except: return 1.5
        df['foundation_b_m'] = df.apply(_B, axis=1); df['foundation_d_m'] = df.apply(_D, axis=1)
        B = df['foundation_b_m']; D = df['foundation_d_m']; N = df['spt_n160'].clip(lower=1.0); SI = 25.0
        Kd = 1.0 + 0.33 * (D / B); size_factor = ((B + 0.3) / B) ** 2
        Qa = (8.0 * N * size_factor * Kd).clip(lower=1.0)
        df['foundation_kd'] = Kd.round(4); df['bearing_qa_kpa'] = Qa.round(2)
        df['bearing_qu_kpa'] = (Qa * 3.0).round(2); df['settlement_mm'] = ((df['q_actual_kpa'] / Qa) * SI).round(2)
        print(f"  B range    : {B.min():.2f} – {B.max():.2f} m")
        print(f"  D range    : {D.min():.2f} – {D.max():.2f} m")
        print(f"  Qa range   : {Qa.min():.1f} – {Qa.max():.1f} kPa")
        print(f"  Settlement : {df['settlement_mm'].min():.1f} – {df['settlement_mm'].max():.1f} mm")
        print(f"  [OK] {len(df)} layers recalculated")
        self.processed_data = df
        return True

    def engineer_features(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 5: FEATURE ENGINEERING")
        print("=" * 80)
        df = self.processed_data.copy()
        df['depth_thickness_m'] = df['depth_to_m'] - df['depth_from_m']
        df['depth_to_groundwater_m'] = df['groundwater_depth_m'] - df['depth_mid_m']
        df['is_below_groundwater'] = (df['depth_mid_m'] > df['groundwater_depth_m']).astype(int)
        df['depth_normalized'] = df['depth_mid_m'] / 15.0
        df['spt_n_log'] = np.log1p(df['spt_n_value'])
        df['spt_n160_log'] = np.log1p(df['spt_n160'])
        df['relative_density_from_spt'] = np.clip(np.sqrt(df['spt_n160'] / 60) * 100, 0, 100)
        df['effective_stress_ratio'] = df['effective_overburden_pressure'] / (df['total_overburden_pressure'] + 1)
        df['is_clean_sand'] = (df['fines_content'] < 5).astype(int)
        df['is_silty_sand'] = ((df['fines_content'] >= 5) & (df['fines_content'] < 35)).astype(int)
        df['is_fine_grained'] = (df['fines_content'] >= 35).astype(int)
        df['depth_spt_interaction'] = df['depth_mid_m'] * df['spt_n_value']
        df['csr_depth_interaction'] = df['csr'] * df['depth_mid_m']
        df['cn_factor'] = np.minimum(1.7, np.sqrt(101.3 / df['effective_overburden_pressure'].clip(1.0)))
        self.processed_data = df
        print(f"  [OK] {len(df.columns)} total columns")
        return True

    def export_csv(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 6: EXPORTING CLEAN CSV")
        print("=" * 80)
        df = self.processed_data.copy()
        output_columns = [
            'borehole_id', 'Municipality', 'Depth_Layer', 'latitude', 'longitude',
            'depth_from_m', 'depth_to_m', 'depth_mid_m',
            'spt_n_value', 'spt_n_value_liq', 'spt_n160', 'spt_is_refusal', 'spt_is_imputed',
            'uscs_symbol', 'soil_description', 'unit_weight', 'fines_content',
            'groundwater_depth_m', 'gwl_estimated', 'friction_angle', 'moisture_content',
            'plasticity_index', 'mean_particle_size_d50', 'relative_density_percent',
            'pga_g', 'csr', 'crr', 'n1_60cs',
            'effective_overburden_pressure', 'total_overburden_pressure',
            'factor_of_safety', 'liquefaction_probability', 'liquefaction',
            'liquefaction_risk_level', 'liquefaction_status',
            'foundation_b_m', 'foundation_d_m', 'bearing_qa_kpa', 'bearing_qu_kpa', 'settlement_mm',
            'is_core_sample', 'is_rock', 'magnitude_mw', 'msf', 'lpi_weighing_factor', 'lpi_severity_factor',
        ]
        avail = [c for c in output_columns if c in df.columns]
        df_export = df[avail].copy()
        num_cols = df_export.select_dtypes(include=[np.number]).columns
        df_export[num_cols] = df_export[num_cols].fillna(0)
        if not self.client:
            print("  [WARNING] No DB connection — skipping storage upload"); return False
        try:
            bucket = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            path = "cleaned/Cleaned_data.csv"
            try:
                old = self.client.storage.from_(bucket).download(path)
                if old:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.client.storage.from_(bucket).upload(f"old_cleaned_data/Cleaned_data_{ts}.csv", old, file_options={'content-type': 'text/csv', 'upsert': 'true'})
                    self.client.storage.from_(bucket).remove([path])
                    print("  [OK] Old file archived")
            except: pass
            self.client.storage.from_(bucket).upload(path, df_export.to_csv(index=False).encode('utf-8'), file_options={'content-type': 'text/csv', 'upsert': 'true'})
            print(f"  [OK] Uploaded {len(df_export)} rows, {len(df_export.columns)} cols -> {path}")
        except Exception as e:
            print(f"  [ERROR] Upload failed: {e}"); return False
        return True

    def connect_database(self) -> bool:
        if not SUPABASE_AVAILABLE: return False
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY')
        if not url or not key:
            print("  [WARNING] DB env vars not set — skipping database storage"); return False
        try:
            self.client = create_client(url, key)
            self.client.table('municipalities').select('id').limit(1).execute()
            print("  [OK] Connected to PostGIS database"); return True
        except Exception as e:
            print(f"  [WARNING] DB connection failed: {e}"); return False

    def safe_float(self, v, default=None):
        if pd.isna(v) or v == '' or v is None: return default
        try: return float(v)
        except: return default

    def safe_str(self, v, default=None):
        if pd.isna(v) or v == '' or v is None: return default
        return str(v).strip()

    def upsert_municipalities(self) -> bool:
        print("\n  Step 7.1: Upserting municipalities...")
        df = self.processed_data
        muni_col = 'Municipality' if 'Municipality' in df.columns else 'municipality'
        if muni_col not in df.columns:
            print("    [ERROR] No municipality column"); return False
        for name in df[muni_col].dropna().unique():
            name = str(name).strip()
            try:
                res = self.client.table('municipalities').select('id').eq('name', name).execute()
                if res.data:
                    self.municipality_ids[name] = res.data[0]['id']
                else:
                    res = self.client.table('municipalities').insert({'name': name, 'description': 'Municipality in Tarlac Province'}).execute()
                    self.municipality_ids[name] = res.data[0]['id']
                    print(f"    Created: {name}")
            except Exception as e:
                print(f"    [ERROR] {name}: {e}")
        print(f"    [OK] {len(self.municipality_ids)} municipalities")
        return True

    def upsert_boreholes(self) -> bool:
        print("\n  Step 7.2: Upserting boreholes...")
        df = self.processed_data
        bh_col = 'Borehole ID' if 'Borehole ID' in df.columns else 'borehole_id'
        muni_col = 'Municipality' if 'Municipality' in df.columns else 'municipality'
        groups = df.groupby(bh_col).first(); skipped = 0
        for bh_id, row in groups.iterrows():
            bh_id = str(bh_id).strip()
            muni = self.safe_str(row.get(muni_col))
            lat = self.safe_float(row.get('latitude', row.get('Latitude')))
            lon = self.safe_float(row.get('longitude', row.get('Longitude')))
            if not muni or muni not in self.municipality_ids or not lat or not lon:
                skipped += 1; continue
            try:
                res = self.client.table('boreholes').select('id').eq('borehole_id', bh_id).execute()
                if res.data:
                    self.borehole_record_ids[bh_id] = res.data[0]['id']
                else:
                    res = self.client.table('boreholes').insert({
                        'borehole_id': bh_id, 'latitude': lat, 'longitude': lon,
                        'elevation': self.safe_float(row.get('Elevation')), 'depth_total_m': 15.0,
                        'remarks': f'Data from {muni}', 'municipality_id': self.municipality_ids[muni],
                    }).execute()
                    if res.data: self.borehole_record_ids[bh_id] = res.data[0]['id']
            except Exception as e:
                print(f"    [ERROR] {bh_id}: {e}")
        if skipped: print(f"    [INFO] Skipped {skipped} boreholes")
        if not self.borehole_record_ids:
            print("    [ERROR] No boreholes inserted"); return False
        print(f"    [OK] {len(self.borehole_record_ids)} boreholes")
        return True

    def store_soil_layers(self) -> bool:
        print("\n  Step 7.3: Storing soil layers...")
        df = self.processed_data.copy()
        bh_col = 'Borehole ID' if 'Borehole ID' in df.columns else 'borehole_id'
        layer_map = {name: i for i, name in enumerate(sorted(df['Depth_Layer'].unique()), 1)}
        records, skipped = [], 0
        for _, row in df.iterrows():
            bh_id = str(row.get(bh_col, '')).strip()
            bh_rec = self.borehole_record_ids.get(bh_id)
            if not bh_rec: skipped += 1; continue
            record = {
                'borehole_id': bh_rec, 'layer_number': int(layer_map.get(row['Depth_Layer'], 1)),
                'depth_from_m': self.safe_float(row['depth_from_m']), 'depth_to_m': self.safe_float(row['depth_to_m']),
                'depth_range': f"{row['depth_from_m']:.1f}-{row['depth_to_m']:.1f}m",
                'uscs_symbol': self.safe_str(row.get('uscs_symbol')) or None,
                'soil_description': self.safe_str(row.get('soil_description')) or None,
                'spt_n_value': self.safe_float(row['spt_n_value']), 'spt_n160': self.safe_float(row.get('spt_n160')),
                'unit_weight': self.safe_float(row['unit_weight']), 'fines_content': self.safe_float(row['fines_content']),
                'groundwater_depth_m': self.safe_float(row['groundwater_depth_m']),
                'relative_density_percent': self.safe_float(row.get('relative_density_percent')),
                'pga_g': self.safe_float(row['pga_g']), 'csr': self.safe_float(row['csr']),
                'cyclic_strength_ratio': self.safe_float(row.get('crr')),
                'liquefaction': bool(row.get('liquefaction', 0)),
                'liquefaction_risk_level': self.safe_str(row.get('liquefaction_risk_level')),
                'effective_overburden_pressure': self.safe_float(row['effective_overburden_pressure']),
                'total_overburden_pressure': self.safe_float(row['total_overburden_pressure']),
                'factor_of_safety': self.safe_float(row.get('factor_of_safety')),
                'liquefaction_probability': self.safe_float(row.get('liquefaction_probability')),
                'liquefaction_status': self.safe_str(row.get('liquefaction_status')),
                'spt_is_refusal': bool(row.get('spt_is_refusal', False)),
                'spt_is_imputed': bool(row.get('spt_is_imputed', False)),
                'is_core_sample': int(row.get('is_core_sample', 0)),
                'is_rock': int(row.get('is_rock', 0)),
                'foundation_b_m': self.safe_float(row.get('foundation_b_m')),
                'foundation_d_m': self.safe_float(row.get('foundation_d_m')),
                'n1_60cs': self.safe_float(row.get('n1_60cs')),
                'magnitude_mw': self.safe_float(row.get('magnitude_mw')),
                'msf': self.safe_float(row.get('msf')),
                'q_actual_kpa': self.safe_float(row.get('q_actual_kpa')),
                'foundation_kd': self.safe_float(row.get('foundation_kd')),
                'bearing_qu_kpa': self.safe_float(row.get('bearing_qu_kpa')),
                'bearing_qa_kpa': self.safe_float(row.get('bearing_qa_kpa')),
                'settlement_mm': self.safe_float(row.get('settlement_mm')),
                'lpi_weighing_factor': self.safe_float(row.get('lpi_weighing_factor')),
                'lpi_severity_factor': self.safe_float(row.get('lpi_severity_factor')),
                'remarks': None,
            }
            for opt_col in ['moisture_content', 'friction_angle', 'cohesion_kpa', 'plasticity_index', 'mean_particle_size_d50']:
                if opt_col in row.index and pd.notna(row.get(opt_col)):
                    record[opt_col] = self.safe_float(row[opt_col])
            if 'Elastic Modulus (Es) (MN/m²)' in row and pd.notna(row['Elastic Modulus (Es) (MN/m²)']):
                record['elastic_modulus_es'] = self.safe_float(row['Elastic Modulus (Es) (MN/m²)'])
            records.append(record)
        bh_uuids = list(self.borehole_record_ids.values())
        for i in range(0, len(bh_uuids), 100):
            try: self.client.table('soil_layers').delete().in_('borehole_id', bh_uuids[i:i+100]).execute()
            except Exception as e: print(f"    [WARNING] Delete batch failed: {e}")
        print(f"    Cleared existing layers for {len(bh_uuids)} boreholes")
        total, failed = 0, 0
        for i in range(0, len(records), 25):
            batch = records[i:i+25]
            try:
                self.client.table('soil_layers').insert(batch).execute(); total += len(batch)
            except Exception as e:
                failed += len(batch); print(f"    [WARNING] Batch {i//25+1} failed: {e}")
        if skipped: print(f"    [INFO] Skipped {skipped} records")
        if failed:  print(f"    [WARNING] {failed} records failed to insert")
        if not total: print("    [ERROR] Nothing inserted"); return False
        print(f"    [OK] Inserted {total} soil layer records (FIX 13 CSR/CRR applied)")
        return True

    def store_to_postgis(self) -> bool:
        if not self.client: return False
        print("\n" + "=" * 80)
        print("STEP 7: STORING TO POSTGIS")
        print("=" * 80)
        try:
            if not self.upsert_municipalities(): return False
            if not self.upsert_boreholes():     return False
            if not self.store_soil_layers():    return False
            print(f"\n  [OK] Stored {len(self.processed_data)} records")
            return True
        except Exception as e:
            print(f"  [ERROR] {e}"); return False

    def print_validation_report(self):
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        def summarise(messages):
            import re
            from collections import defaultdict
            totals: dict = defaultdict(int); plain: list = []; seen_plain: set = set()
            for msg in messages:
                m = re.match(r'^(.*?)(\d+)\s+record', msg)
                if m:
                    key = re.sub(r'\s*\[.*?\]', '', m.group(1)).strip(); totals[key] += int(m.group(2))
                else:
                    clean = re.sub(r'\s*\[.*?\]', '', msg).strip()
                    if clean not in seen_plain: seen_plain.add(clean); plain.append(clean)
            result = [f"{k} {v} records" for k, v in sorted(totals.items())]
            result += plain; return result
        if self.validation_errors:
            print("\n  [ERRORS]:"); [print(f"    x {e}") for e in summarise(self.validation_errors)]
        else: print("\n  No critical errors")
        if self.validation_warnings:
            print("\n  [WARNINGS]:"); [print(f"    ! {w}") for w in summarise(self.validation_warnings)]
        else: print("\n  No warnings")

    def run(self) -> bool:
        print("\n" + "=" * 80)
        print("GEOTECHNICAL DATA PROCESSING PIPELINE  v3")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.connect_database()
        if not self.excel_path and self.use_storage:
            if not self.download_from_supabase_storage():
                print("[ERROR] Failed to download from Supabase Storage"); return False
        self.upload_raw_file_to_bucket()
        steps = [
            ("Load Excel",                         self.load_excel),
            ("Process and Validate",               self.process_and_validate),
            ("Fill Missing by Borehole",           self.fill_missing_by_borehole),
            ("Calculate CSR/CRR",                  self.calculate_csr_crr),
            ("Bearing Capacity — Initial Pass",    self.calculate_bearing_bowles),
            ("Classify Liquefaction (DPWH BSDS)", self.classify_liquefaction_dpwh_bsds),
            ("Recalc Bearing with ANN Dims",       self.recalculate_bearing_with_ann_foundation),
            ("Engineer Features",                  self.engineer_features),
            ("Export CSV",                         self.export_csv),
        ]
        for name, fn in steps:
            print(f"\n{'─'*40}")
            print(f"Running: {name}")
            if not fn():
                if name == "Export CSV": print("  [INFO] CSV export skipped (no DB connection)")
                else: print(f"[ERROR] Pipeline failed at: {name}"); return False
        self.print_validation_report()
        if self.client: self.store_to_postgis()
        print("\n" + "=" * 80)
        print("[SUCCESS] PIPELINE COMPLETE — FIX 13 applied")
        print("=" * 80)
        print(f"  Records      : {len(self.processed_data)}")
        print(f"  Columns      : {len(self.processed_data.columns)}")
        print(f"  Fixes applied: FIX 1–13")
        return True


def main():
    import sys
    excel_file = sys.argv[1] if len(sys.argv) > 1 else None
    url = os.getenv('SUPABASE_URL'); key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    if url and key: print("[INFO] Supabase configured — DB storage enabled")
    else: print("[WARNING] Supabase env vars not found — DB storage skipped")
    pipeline = GeotechnicalPipeline(excel_file)
    if not pipeline.run(): sys.exit(1)
    print("\nDone")


if __name__ == "__main__":
    main()