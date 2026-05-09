#!/usr/bin/env python3
"""
Geotechnical Data Processing Pipeline — Improved v2
=====================================================
All fixes applied from data audit:

FIX 1  — Fines Content >100% auto-corrected (÷10) or nulled
FIX 2  — N1(60) computed via Cn overburden correction for missing values (was raw SPT)
FIX 3  — SPT N=100 (refusal) capped at 60 for liquefaction path only
FIX 4  — BH-75-style PGA < 0.2g flagged and nulled for municipality propagation
FIX 5  — CS-1/CS core samples excluded from liquefaction classification
FIX 6  — Municipality-median GWL fill for boreholes with no GWL in any layer
FIX 7  — USCS symbol typos normalised (SMM→SM, CM→SC, SP_→SP, ROCK types, etc.)
FIX 8  — 'Ground Water Table' column used as backup GWL source
FIX 9  — SPT imputation tracked via boolean flag (replaces fragile ==15.0 sentinel)
FIX 10 — Fault-distance attenuation fallback for missing PGA (Youngs et al. 1997)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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


# ---------------------------------------------------------------------------
# USCS lookup tables  (Bowles 1988 + USCS classification)
# ---------------------------------------------------------------------------
_USCS_UNIT_WEIGHT = {
    'GW': 20.0, 'GP': 18.0, 'SW': 19.0, 'SP': 17.5,
    'GM': 20.0, 'GC': 19.5, 'SM': 19.0, 'SC': 19.5,
    'ML': 17.0, 'MH': 16.0, 'CL': 17.5, 'CH': 16.5,
    'OL': 14.0, 'OH': 13.0, 'PT': 11.0,
}  # kN/m³

_USCS_FINES_PCT = {
    'GW':  3.0, 'GP':  3.0, 'SW':  3.0, 'SP':  3.0,
    'GM': 10.0, 'GC': 22.0, 'SM': 10.0, 'SC': 22.0,
    'ML': 60.0, 'MH': 70.0, 'CL': 65.0, 'CH': 75.0,
    'OL': 75.0, 'OH': 80.0, 'PT': 90.0,
}  # percent passing #200 sieve

# FIX 7 — USCS typo / non-soil normalisation map
# Maps raw cell values → canonical 2-letter USCS (or '' for non-soil)
_USCS_CORRECTIONS = {
    'SMM':       'SM',    # typo
    'CM':        'SC',    # likely SC mis-typed
    'SP ':       'SP',    # trailing space
    'SC-SM':     'SC',    # dual — use primary
    'SP-SM':     'SP',    # dual — use primary
    '-':         '',      # no data
    'CORING':    '',      # core run, no soil
    'CS':        '',      # core sample marker
    'BASALT':    'ROCK',
    'SANDSTONE': 'ROCK',
    'TUFF':      'ROCK',
    'H':         '',      # artefact
    'M':         '',
    'S':         '',
}

# Non-soil USCS values → exclude from liquefaction
_NON_SOIL_USCS = {'ROCK', ''}

# Cohesive USCS groups (Dr and liquefaction formulae don't apply)
_COHESIVE_USCS = {'CL', 'CH', 'ML', 'MH', 'OL', 'OH', 'PT', 'CL-ML'}

# Minimum plausible PGA for Tarlac Province (g)
_TARLAC_PGA_MIN = 0.2


# ===========================================================================
# Pipeline class
# ===========================================================================
class GeotechnicalPipeline:
    """Improved single-pass geotechnical data processing pipeline."""

    def __init__(
        self,
        excel_file_path: Optional[str] = None,
        output_csv_path: Optional[str] = None,
        use_storage: bool = True,
    ):
        self.use_storage = use_storage
        self.excel_path = Path(excel_file_path) if excel_file_path else None
        self.excel_file_bytes: Optional[bytes] = None

        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Optional[pd.DataFrame] = None
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

        self.client = None
        self.municipality_ids: Dict[str, int] = {}
        self.borehole_record_ids: Dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Storage helpers
    # -----------------------------------------------------------------------
    def download_from_supabase_storage(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 1: DOWNLOADING FROM SUPABASE STORAGE")
        print("=" * 80)
        if not SUPABASE_AVAILABLE:
            print("  [ERROR] Supabase not available")
            return False
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        if not url or not key:
            print("  [ERROR] Environment variables not set")
            return False
        try:
            self.client = create_client(url, key)
            bucket = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            data = self.client.storage.from_(
                bucket).download("raw/Raw_data.xlsx")
            if not data:
                print("  [ERROR] File not found in storage")
                return False
            self.excel_file_bytes = data
            print(f"  [OK] Downloaded {len(data)} bytes")
            return True
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            return False

    def upload_raw_file_to_bucket(self) -> bool:
        if not self.excel_path or not self.excel_path.exists() or not self.client:
            return True
        try:
            bucket = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            with open(self.excel_path, 'rb') as f:
                data = f.read()
            self.client.storage.from_(bucket).upload(
                f"raw/{self.excel_path.name}", data,
                file_options={
                    'content-type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'upsert': 'true'},
            )
            print(f"  [OK] Raw file uploaded to bucket")
        except Exception as e:
            print(f"  [WARNING] Raw file upload failed: {e}")
        return True

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
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
                print("  [ERROR] No file source available")
                return False

            xl = pd.ExcelFile(source)
            print(f"\n  Found {len(xl.sheet_names)} sheets")
            for sheet in xl.sheet_names:
                if sheet.strip().lower() == 'summary':
                    print(f"    Skipping: {sheet}")
                    continue
                df = pd.read_excel(source, sheet_name=sheet)
                self.raw_data[sheet] = df
                print(
                    f"    Loaded: {sheet} ({len(df)} rows, {len(df.columns)} cols)")
            print(f"\n  [OK] Loaded {len(self.raw_data)} data sheets")
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return False

    # -----------------------------------------------------------------------
    # Value parsers
    # -----------------------------------------------------------------------
    def clean_coordinate(self, value) -> Optional[float]:
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return float(str(value).replace('°', '').replace("'", '').replace('"', '').strip())
        except:
            return None

    def parse_pga(self, value) -> Optional[float]:
        if pd.isna(value):
            return None
        try:
            matches = re.findall(r'(\d+\.?\d*)g', str(value).lower())
            if matches:
                return float(np.mean([float(m) for m in matches]))
            return float(str(value).replace('g', '').strip())
        except:
            return None

    def parse_relative_density(self, value) -> Optional[float]:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except:
            pass
        density_map = {
            'very loose': 15.0, 'loose': 35.0, 'loose to medium dense': 50.0,
            'medium': 50.0, 'medium dense': 65.0, 'dense': 80.0,
            'very dense': 95.0, 'hard': 90.0,
        }
        return density_map.get(str(value).lower().strip())

    def parse_elastic_modulus(self, value) -> Optional[float]:
        if pd.isna(value) or isinstance(value, datetime):
            return None
        try:
            return float(value)
        except:
            pass
        s = str(value).lower()
        if 'to' in s:
            try:
                parts = s.split('to')
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            except:
                return None
        return None

    # -----------------------------------------------------------------------
    # PRE-CLEAN  (FIX 1, 7, 8, 9)
    # -----------------------------------------------------------------------
    def pre_clean_dataframe(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """
        Must run FIRST.
        FIX 1  — fines content >100 corrected / nulled
        FIX 7  — USCS typos normalised
        FIX 8  — 'Ground Water Table' used as backup GWL source
        FIX 9  — SPT imputation tracked via 'spt_is_imputed' flag
        Also: drop unnamed cols, fix StringDtype, clean coords/PGA/Dr/Es.
        """
        df = df.copy()

        # Drop unnamed Excel artefact columns
        unnamed = [c for c in df.columns if str(
            c).lower().startswith('unnamed')]
        if unnamed:
            df = df.drop(columns=unnamed)
            print(f"    [FIX] Dropped unnamed columns: {unnamed}")

        # StringDtype → object (pandas 2.x compatibility)
        str_cols = [c for c in df.columns if str(df[c].dtype) == 'str']
        if str_cols:
            df[str_cols] = df[str_cols].astype(object)

        # Coordinates
        for col in ['Latitude', 'Longitude']:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_coordinate)

        # PGA
        if 'Peak Ground Acceleration' in df.columns:
            df['Peak Ground Acceleration'] = df['Peak Ground Acceleration'].apply(
                self.parse_pga)

        # Relative density
        if 'Relative Density' in df.columns:
            df['Relative Density'] = df['Relative Density'].apply(
                self.parse_relative_density)

        # Elastic modulus
        if 'Elastic Modulus (Es) (MN/m²)' in df.columns:
            df['Elastic Modulus (Es) (MN/m²)'] = df['Elastic Modulus (Es) (MN/m²)'].apply(
                self.parse_elastic_modulus)

        # FIX 8 — populate 'Groundwater Level (m)' from 'Ground Water Table' where missing
        if 'Ground Water Table' in df.columns and 'Groundwater Level (m)' in df.columns:
            gwl_main = pd.to_numeric(
                df['Groundwater Level (m)'], errors='coerce')
            gwl_backup = pd.to_numeric(
                df['Ground Water Table'], errors='coerce')
            filled_from_backup = gwl_main.isna() & gwl_backup.notna()
            if filled_from_backup.any():
                df.loc[filled_from_backup,
                       'Groundwater Level (m)'] = gwl_backup[filled_from_backup]
                print(
                    f"    [FIX8] GWL filled from 'Ground Water Table': {filled_from_backup.sum()} rows")

        # FIX 9 — mark which SPT values are genuinely missing BEFORE filling
        spt_raw_col = next((c for c in ['SPT N-Value', 'SPT N Value', 'N-Value', 'SPT']
                            if c in df.columns), None)
        if spt_raw_col:
            df['spt_is_imputed'] = pd.to_numeric(
                df[spt_raw_col], errors='coerce').isna()
        else:
            df['spt_is_imputed'] = True

        # FIX 7 — normalise USCS symbol
        if 'USCS Symbol' in df.columns:
            uscs_raw = df['USCS Symbol'].fillna(
                '').astype(str).str.strip().str.upper()
            uscs_norm = uscs_raw.replace(_USCS_CORRECTIONS)
            # Handle single-char artefacts not in map
            uscs_norm = uscs_norm.where(uscs_norm.str.len() >= 2, '')
            df['USCS Symbol'] = uscs_norm
            # Flag rock/non-soil
            df['is_rock'] = uscs_norm.isin({'ROCK'}).astype(int)
            n_fixed = (uscs_raw != uscs_norm).sum()
            if n_fixed:
                print(f"    [FIX7] USCS symbols normalised: {n_fixed} rows")

        # FIX 5 — flag core samples (CS-1, CS)
        sample_col = next(
            (c for c in df.columns if 'sample' in c.lower()), None)
        if sample_col:
            is_core = df[sample_col].astype(
                str).str.strip().str.upper().isin(['CS-1', 'CS'])
            df['is_core_sample'] = is_core.astype(int)
            if is_core.any():
                print(f"    [FIX5] Core samples flagged: {is_core.sum()} rows")
        else:
            df['is_core_sample'] = 0

        # FIX 1 — fix fines content > 100%
        if 'Fines Content' in df.columns:
            fc = pd.to_numeric(df['Fines Content'], errors='coerce')
            bad = fc > 100
            if bad.any():
                corrected = fc[bad] / 10.0
                plausible = (corrected >= 2.0) & (corrected <= 98.0)
                df.loc[bad & plausible,
                       'Fines Content'] = corrected[plausible].round(1)
                # let USCS fallback handle
                df.loc[bad & ~plausible, 'Fines Content'] = np.nan
                self.validation_errors.append(
                    f"[{sheet_name}] Fines >100%: {bad.sum()} rows auto-corrected (÷10) or nulled"
                )
                print(
                    f"    [FIX1] Fines >100%: {bad.sum()} rows corrected/nulled")

        return df

    # -----------------------------------------------------------------------
    # Coordinate validation
    # -----------------------------------------------------------------------
    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Latitude' in df.columns:
            df['latitude'] = df['Latitude']
            miss = df['latitude'].isna().sum()
            if miss:
                self.validation_warnings.append(
                    f"Missing latitude: {miss} records")
            oor = ((df['latitude'].dropna() < 15.0) | (
                df['latitude'].dropna() > 16.0)).sum()
            if oor:
                self.validation_errors.append(
                    f"Latitude outside Tarlac range (15–16°): {oor}")
        if 'Longitude' in df.columns:
            df['longitude'] = df['Longitude']
            miss = df['longitude'].isna().sum()
            if miss:
                self.validation_warnings.append(
                    f"Missing longitude: {miss} records")
            oor = ((df['longitude'].dropna() < 120.0) | (
                df['longitude'].dropna() > 121.0)).sum()
            if oor:
                self.validation_errors.append(
                    f"Longitude outside Tarlac range (120–121°): {oor}")
        return df

    # -----------------------------------------------------------------------
    # SPT validation  (FIX 3, 9)
    # -----------------------------------------------------------------------
    def validate_spt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX 3 — N=100 (refusal) flagged; capped at 60 for liquefaction path.
        FIX 9 — imputation tracked via spt_is_imputed (set in pre_clean).
        """
        df = df.copy()
        spt_col = next((c for c in ['SPT N-Value', 'SPT N Value', 'spt_n_value', 'N-Value', 'SPT']
                        if c in df.columns), None)

        if spt_col:
            df['spt_n_value'] = pd.to_numeric(df[spt_col], errors='coerce')

            valid = df['spt_n_value'].dropna()
            oor = ((valid < 0) | (valid > 100)).sum()
            if oor:
                self.validation_errors.append(
                    f"SPT out of range (0–100): {oor} records")

            # FIX 3 — refusal flag
            df['spt_is_refusal'] = (df['spt_n_value'] >= 100).astype(int)
            # Separate column capped at 60 for liquefaction use
            df['spt_n_value_liq'] = df['spt_n_value'].clip(upper=60.0)

            # FIX 9 — fill missing with 15.0 AFTER flagging
            miss = df['spt_n_value'].isna().sum()
            if miss:
                self.validation_warnings.append(
                    f"Missing SPT: {miss} records → imputed 15.0")
            df['spt_n_value'] = df['spt_n_value'].fillna(15.0)
            df['spt_n_value_liq'] = df['spt_n_value_liq'].fillna(15.0)
        else:
            self.validation_warnings.append(
                "SPT column not found — defaulting to 15.0")
            df['spt_n_value'] = 15.0
            df['spt_n_value_liq'] = 15.0
            df['spt_is_refusal'] = 0
            df['spt_is_imputed'] = True

        return df

    # -----------------------------------------------------------------------
    # Soil parameter validation
    # -----------------------------------------------------------------------
    def validate_soil_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # USCS (already normalised in pre_clean)
        uscs_col = next(
            (c for c in ['USCS Symbol', 'uscs_symbol', 'USCS'] if c in df.columns), None)
        if uscs_col:
            df['uscs_symbol'] = df[uscs_col].fillna('').astype(str).str.strip()
            df['uscs_symbol'] = df['uscs_symbol'].replace('nan', '')
        else:
            df['uscs_symbol'] = ''

        uscs_upper = df['uscs_symbol'].str.upper().str.strip()
        uscs_primary = uscs_upper.str[:2]

        # Unit weight
        uw_col = next((c for c in ['Unit Weight (γ)', 'Unit Weight', 'unit_weight',
                                   'Unit Weight (kN/m³)', 'γ'] if c in df.columns), None)
        if uw_col:
            df['unit_weight'] = pd.to_numeric(df[uw_col], errors='coerce')
            oor = ((df['unit_weight'].dropna() < 10) | (
                df['unit_weight'].dropna() > 25)).sum()
            if oor:
                self.validation_warnings.append(
                    f"Unit weight outside 10–25 kN/m³: {oor}")
        else:
            df['unit_weight'] = np.nan
        uscs_uw = uscs_primary.map(_USCS_UNIT_WEIGHT).fillna(18.0)
        df['unit_weight'] = df['unit_weight'].fillna(uscs_uw)

        # Fines content (FIX 1 already applied in pre_clean; fill remaining NaN here)
        fc_col = next((c for c in ['Fines Content', 'fines_content', 'Fines (%)']
                       if c in df.columns), None)
        if fc_col:
            df['fines_content'] = pd.to_numeric(df[fc_col], errors='coerce')
            oor = ((df['fines_content'].dropna() < 0) | (
                df['fines_content'].dropna() > 100)).sum()
            if oor:
                self.validation_errors.append(
                    f"Fines content outside 0–100%: {oor} (post-correction)")
        else:
            df['fines_content'] = np.nan
        uscs_fc = uscs_primary.map(_USCS_FINES_PCT).fillna(15.0)
        df['fines_content'] = df['fines_content'].fillna(uscs_fc)

        # Groundwater level (FIX 8 backup already applied; FIX 9 flag tracks imputation)
        gwl_col = next((c for c in ['Groundwater Level (m)', 'Ground Water Table',
                                    'Groundwater Level', 'groundwater_depth_m', 'GWL']
                        if c in df.columns), None)
        gwl_numeric = pd.to_numeric(
            df[gwl_col], errors='coerce') if gwl_col else pd.Series(np.nan, index=df.index)
        df['gwl_estimated'] = gwl_numeric.isna()
        df['groundwater_depth_m'] = gwl_numeric.fillna(
            5.0)  # initial fallback; replaced in later steps

        # Soil description
        desc_col = next((c for c in ['Soil/Rock Description', 'Soil Description',
                                     'soil_description', 'Description'] if c in df.columns), None)
        if desc_col:
            df['soil_description'] = df[desc_col].fillna(
                '').astype(str).str.strip().replace('nan', '')

        # Friction angle
        fa_col = next((c for c in ['Internal Friction Angle', 'Friction Angle',
                                   'friction_angle'] if c in df.columns), None)
        if fa_col:
            df['friction_angle'] = pd.to_numeric(df[fa_col], errors='coerce')

        # Plasticity index
        pi_col = next((c for c in ['Plasticity Index (PI)', 'Plasticity Index',
                                   'plasticity_index', 'PI'] if c in df.columns), None)
        if pi_col:
            df['plasticity_index'] = pd.to_numeric(df[pi_col], errors='coerce')

        # Moisture content
        wc_col = next((c for c in ['Natural Water Content (ω)', 'Natural Water Content',
                                   'moisture_content', 'Water Content'] if c in df.columns), None)
        if wc_col:
            df['moisture_content'] = pd.to_numeric(df[wc_col], errors='coerce')

        # D50
        d50_col = next((c for c in ['Mean Particle Size (D50) (mm)', 'Mean Particle Size',
                                    'mean_particle_size_d50', 'D50'] if c in df.columns), None)
        if d50_col:
            df['mean_particle_size_d50'] = pd.to_numeric(
                df[d50_col], errors='coerce')

        return df

    # -----------------------------------------------------------------------
    # process_and_validate
    # -----------------------------------------------------------------------
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
            df = self.validate_soil_parameters(df)

            # PGA  (FIX 4 applied below in fill_missing_by_borehole after propagation)
            if 'Peak Ground Acceleration' in df.columns:
                df['pga_g'] = pd.to_numeric(
                    df['Peak Ground Acceleration'], errors='coerce')
            else:
                df['pga_g'] = np.nan
            # Flag implausibly low PGA (FIX 4)
            if 'pga_g' in df.columns:
                low_pga = df['pga_g'].notna() & (df['pga_g'] < _TARLAC_PGA_MIN)
                if low_pga.any():
                    bad_bhs = df.loc[low_pga, 'Borehole ID'].tolist()
                    self.validation_errors.append(
                        f"[{sheet_name}] PGA < {_TARLAC_PGA_MIN}g (likely decimal error): {bad_bhs}"
                    )
                    print(
                        f"    [FIX4] PGA < {_TARLAC_PGA_MIN}g nulled for: {bad_bhs}")
                    # municipality propagation will fix
                    df.loc[low_pga, 'pga_g'] = np.nan
            # sentinel; replaced by propagation
            df['pga_g'] = df['pga_g'].fillna(0.35)

            # Relative density
            if 'Relative Density' in df.columns:
                df['relative_density_percent'] = df['Relative Density']

            # Depth info from sheet name
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
            print(
                f"\n  [OK] Combined {len(self.processed_data)} records from {len(self.raw_data)} sheets")
            return True
        return False

    # -----------------------------------------------------------------------
    # Compute missing raw columns
    # -----------------------------------------------------------------------
    def compute_missing_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NULL raw cells using empirical relationships."""
        GAMMA_W = 9.81
        df = df.copy()
        z = df['depth_mid_m']
        γ = df['unit_weight']
        gwl = df['groundwater_depth_m']

        sigma_v = γ * z
        u = np.maximum(0.0, z - gwl) * GAMMA_W
        sigma_eff = np.maximum(1.0, sigma_v - u)

        rd = np.clip(np.where(z <= 9.15, 1.0 - 0.00765 * z,
                              np.where(z <= 23.0, 1.174 - 0.0267 * z, 0.0)), 0.0, 1.0)

        # Total overburden
        tot_col = 'Total Overburden Pressure'
        if tot_col in df.columns:
            miss = pd.to_numeric(df[tot_col], errors='coerce').isna()
            df.loc[miss, tot_col] = sigma_v[miss].round(2)
            if miss.sum():
                print(f"      [FILL] Total OBP: {miss.sum()} rows")

        # Effective overburden (handles typo 'Presssure')
        for eff_col in ('Effective Overburden Presssure', 'Effective Overburden Pressure'):
            if eff_col in df.columns:
                miss = pd.to_numeric(df[eff_col], errors='coerce').isna()
                df.loc[miss, eff_col] = sigma_eff[miss].round(2)
                if miss.sum():
                    print(f"      [FILL] Effective OBP: {miss.sum()} rows")
                break

        # Relative density (Skempton 1986) — granular only
        rd_col = 'Relative Density'
        n160_col = 'Corrected SPT-N Value (N1(60))'
        if rd_col in df.columns and n160_col in df.columns:
            n1_60 = pd.to_numeric(df[n160_col], errors='coerce')
            uscs = df.get('uscs_symbol', pd.Series(
                '', index=df.index)).str.upper().str[:2]
            miss = (pd.to_numeric(df[rd_col], errors='coerce').isna()
                    & n1_60.notna() & (n1_60 > 0)
                    & ~uscs.isin(_COHESIVE_USCS))
            df.loc[miss, rd_col] = (
                np.sqrt(n1_60[miss] / 60.0) * 100.0).round(2)
            if miss.sum():
                print(
                    f"      [FILL] Relative Density: {miss.sum()} rows (Skempton 1986)")

        # CSR (Seed & Idriss 1971)
        csr_col = 'Cyclic Stress Ratio (CSR)'
        if csr_col in df.columns:
            csr_vals = 0.65 * (sigma_v / sigma_eff) * df['pga_g'] * rd
            miss = (pd.to_numeric(
                df[csr_col], errors='coerce').isna() & (df['pga_g'] > 0))
            df.loc[miss, csr_col] = csr_vals[miss].round(6)
            if miss.sum():
                print(f"      [FILL] CSR: {miss.sum()} rows")

        # Friction angle (Wolff 1989 granular / 15° cohesive floor)
        uscs_fa = df.get('uscs_symbol', pd.Series(
            '', index=df.index)).str.upper().str[:2]
        is_coh = uscs_fa.isin(_COHESIVE_USCS)
        fa_col = next(
            (c for c in ['friction_angle', 'Internal Friction Angle'] if c in df.columns), None)
        if fa_col:
            miss = pd.to_numeric(df[fa_col], errors='coerce').isna()
            n_fa = df['spt_n_value'].clip(lower=1.0)
            phi_g = (27.1 + 0.3 * n_fa - 0.00054 * n_fa ** 2).clip(25.0, 45.0)
            df.loc[miss, fa_col] = np.where(
                is_coh[miss], 15.0, phi_g[miss]).round(1)
            if miss.sum():
                print(f"      [FILL] Friction angle: {miss.sum()} rows")

        # Elastic modulus (Bowles 1988)
        SAND_U = {'SW', 'SP', 'SM', 'SC', 'GW', 'GP', 'GM', 'GC'}
        SILT_U = {'ML', 'MH'}
        uscs_es = df.get('uscs_symbol', pd.Series(
            '', index=df.index)).str.upper().str[:2]
        is_sand = uscs_es.isin(SAND_U)
        is_silt = uscs_es.isin(SILT_U)
        es_col = next((c for c in ['Elastic Modulus (Es) (MN/m²)', 'elastic_modulus_es']
                       if c in df.columns), None)
        if es_col:
            miss = pd.to_numeric(df[es_col], errors='coerce').isna()
            n_es = df['spt_n_value'].clip(lower=1.0)
            es_val = np.where(is_sand, 0.5 * (n_es + 15.0),
                              np.where(is_silt, 0.3 * (n_es + 6.0), 0.2 * n_es))
            df.loc[miss, es_col] = es_val[miss].round(2)
            if miss.sum():
                print(f"      [FILL] Elastic modulus: {miss.sum()} rows")

        return df

    # -----------------------------------------------------------------------
    # Fill missing by borehole  (FIX 4, 6, 9)
    # -----------------------------------------------------------------------
    def fill_missing_by_borehole(self) -> bool:
        """
        FIX 4  — PGA < 0.2g (nulled earlier) replaced by borehole median.
        FIX 6  — GWL still missing after borehole propagation → municipality median.
        FIX 9  — SPT interpolation uses spt_is_imputed flag instead of ==15 sentinel.
        """
        print("\n" + "=" * 80)
        print("FILLING MISSING VALUES BY BOREHOLE")
        print("=" * 80)

        df = self.processed_data.copy()

        bh_col = next(
            (c for c in ['Borehole ID', 'borehole_id'] if c in df.columns), None)
        if bh_col is None:
            print("  [SKIP] No borehole ID column — skipping")
            return True

        df = df.sort_values([bh_col, 'depth_mid_m']).reset_index(drop=True)

        # ── GWL within-borehole propagation ───────────────────────────────
        gwl_updated = 0
        if 'gwl_estimated' in df.columns:
            for bh_id, grp in df.groupby(bh_col, sort=False):
                real = ~grp['gwl_estimated']
                est = grp['gwl_estimated']
                if real.any() and est.any():
                    med = grp.loc[real, 'groundwater_depth_m'].median()
                    df.loc[grp.index[est], 'groundwater_depth_m'] = med
                    df.loc[grp.index[est], 'gwl_estimated'] = False
                    gwl_updated += int(est.sum())
        print(f"  [FILL] GWL propagated within boreholes: {gwl_updated} rows")

        # FIX 6 — municipality-median GWL for boreholes with NO real GWL
        muni_col = next(
            (c for c in ['Municipality', 'municipality'] if c in df.columns), None)
        if muni_col and 'gwl_estimated' in df.columns:
            real_gwl_df = df[~df['gwl_estimated']]
            muni_medians = real_gwl_df.groupby(
                muni_col)['groundwater_depth_m'].median()
            still_est = df['gwl_estimated']
            if still_est.any():
                muni_fill = df.loc[still_est, muni_col].map(muni_medians)
                ok = muni_fill.notna()
                df.loc[still_est & ok, 'groundwater_depth_m'] = muni_fill[ok]
                df.loc[still_est & ok, 'gwl_estimated'] = False
                print(
                    f"  [FIX6] GWL filled from municipality median: {ok.sum()} rows")

        # ── PGA propagation (FIX 4 — also covers borehole-median replacement) ──
        pga_updated = 0
        for bh_id, grp in df.groupby(bh_col, sort=False):
            # Real PGA = not 0.35 sentinel AND not suspiciously low
            real = (grp['pga_g'] != 0.35) & (grp['pga_g'] >= _TARLAC_PGA_MIN)
            est = ~real
            if real.any() and est.any():
                med = grp.loc[real, 'pga_g'].median()
                df.loc[grp.index[est], 'pga_g'] = med
                pga_updated += int(est.sum())
        print(f"  [FILL] PGA propagated within boreholes: {pga_updated} rows")

        # FIX 10 — fault-distance attenuation fallback for remaining missing PGA
        fault_col = next((c for c in ['Fault Distance (km)', 'fault_distance']
                          if c in df.columns), None)
        if fault_col:
            still_default = df['pga_g'] == 0.35
            R = pd.to_numeric(
                df.loc[still_default, fault_col], errors='coerce').clip(lower=1.0)
            Mw = 6.5
            # Simplified Youngs et al. (1997) interface equation
            pga_est = (0.2083 * np.exp(1.2587 * Mw) *
                       (R + 15) ** -1.9661).clip(0.1, 0.8)
            ok = R.notna()
            df.loc[still_default & ok.reindex(df.index, fill_value=False), 'pga_g'] = \
                pga_est[ok].round(3).values
            print(
                f"  [FIX10] PGA from fault-distance attenuation: {ok.sum()} rows")

        # ── SPT interpolation (FIX 9 — use flag, not ==15 sentinel) ──────
        spt_updated = 0
        if 'spt_is_imputed' in df.columns:
            for bh_id, grp in df.groupby(bh_col, sort=False):
                real = ~grp['spt_is_imputed']
                if real.sum() >= 2:
                    spt_real = grp['spt_n_value'].where(real)
                    spt_interp = (spt_real.interpolate(method='index')
                                  .ffill().bfill()
                                  .clip(1.0, 100.0))
                    imp_idx = grp.index[~real]
                    if len(imp_idx):
                        df.loc[imp_idx, 'spt_n_value'] = spt_interp.loc[imp_idx].round(
                            1)
                        df.loc[imp_idx, 'spt_n_value_liq'] = spt_interp.loc[imp_idx].clip(
                            upper=60).round(1)
                        spt_updated += len(imp_idx)
        print(
            f"  [FILL] SPT interpolated within boreholes: {spt_updated} rows")

        self.processed_data = df
        print("  [OK] Borehole fill complete")
        return True

    # -----------------------------------------------------------------------
    # CSR / CRR  (FIX 2, 3)
    # -----------------------------------------------------------------------
    def calculate_csr_crr(self) -> bool:
        """
        FIX 2 — N1(60) computed via Cn for records lacking measured value.
        FIX 3 — spt_n_value_liq (capped at 60) used in N1(60)cs path.
        """
        print("\n" + "=" * 80)
        print("STEP 3: CSR/CRR ANALYSIS")
        print("=" * 80)

        df = self.processed_data.copy()
        GAMMA_W = 9.81
        MW = 6.5
        MSF = (10 ** 2.24) / (MW ** 2.56)

        df['magnitude_mw'] = MW
        df['msf'] = MSF
        df['q_actual_kpa'] = 150.0
        print(f"  Mw={MW}, MSF={MSF:.4f}, q_actual=150 kPa")

        # Overburden pressures
        df['total_overburden_pressure'] = df['unit_weight'] * df['depth_mid_m']
        depth_below_wt = np.maximum(
            0.0, df['depth_mid_m'] - df['groundwater_depth_m'])
        df['effective_overburden_pressure'] = (
            df['total_overburden_pressure'] - GAMMA_W * depth_below_wt
        ).clip(lower=1.0)

        # rd stress reduction coefficient
        z = df['depth_mid_m']
        rd = np.clip(np.where(z <= 9.15, 1.0 - 0.00765 * z,
                              np.where(z <= 23.0, 1.174 - 0.0267 * z, 0.0)), 0.0, 1.0)

        # CSR
        print("  Calculating CSR (Seed & Idriss 1971)...")
        df['csr'] = (0.65 * df['pga_g']
                     * (df['total_overburden_pressure'] / df['effective_overburden_pressure'])
                     * rd)

        # FIX 2 — compute N1(60) via Cn where not measured
        print("  Computing N1(60) with Cn correction (FIX 2)...")
        raw_n160 = pd.to_numeric(
            df.get('Corrected SPT-N Value (N1(60))',
                   pd.Series(np.nan, index=df.index)),
            errors='coerce'
        )
        has_measured_n160 = raw_n160.notna()

        # Overburden correction factor Cn (Liao & Whitman 1986)
        Cn = np.minimum(1.7, np.sqrt(
            101.3 / df['effective_overburden_pressure'].clip(lower=1.0)))
        # FIX 3 — use refusal-capped SPT for liquefaction path
        computed_n160 = (df['spt_n_value_liq'] * Cn).clip(upper=60.0)
        df['spt_n160'] = np.where(has_measured_n160, raw_n160, computed_n160)
        df['spt_n60'] = df['spt_n_value']   # uncorrected, kept for reference

        n160_source_computed = (~has_measured_n160).sum()
        n160_source_measured = has_measured_n160.sum()
        print(
            f"    N1(60) source — measured: {n160_source_measured}, Cn-computed: {n160_source_computed}")

        # NCEER fines correction → (N1)60cs
        print("  Computing (N1)60cs with fines correction (NCEER)...")
        FC = df['fines_content'].clip(lower=0.1)
        alpha = np.where(FC < 5.0,  0.0,
                         np.where(FC <= 35.0, np.exp(1.76 - 190.0 / FC ** 2), 5.0))
        beta = np.where(FC < 5.0,  1.0,
                        np.where(FC <= 35.0, 0.99 + FC ** 1.5 / 1000.0, 1.2))
        df['n1_60cs'] = (alpha + beta * df['spt_n160']).clip(upper=60.0)

        # CRR (Robertson & Wride 1998)
        print("  Calculating CRR (Robertson-Wride)...")
        N = df['n1_60cs'].clip(upper=30.0)
        crr_raw = np.exp(N / 14.1 + (N / 126.0) ** 2
                         - (N / 23.6) ** 3 + (N / 25.4) ** 4 - 2.67)
        df['crr'] = np.where(df['n1_60cs'] >= 30.0, 0.6,
                             crr_raw.clip(0.0, 0.6))

        # Factor of safety
        df['factor_of_safety'] = (df['crr'] * MSF) / (df['csr'] + 1e-9)

        # LPI components
        df['lpi_weighing_factor'] = np.maximum(
            0.0, 10.0 - 0.5 * df['depth_mid_m'])
        df['lpi_severity_factor'] = np.maximum(
            0.0, 1.0 - df['factor_of_safety'])

        # Liquefaction probability
        df['liquefaction_probability'] = np.where(
            df['factor_of_safety'] < 1.0,
            (1.0 - df['factor_of_safety']) * 100,
            np.where(df['factor_of_safety'] < 1.5, 30.0, 10.0)
        ).clip(0.0, 100.0)

        self.processed_data = df
        print(f"  [OK] {len(df)} records")
        print(f"    CSR  : {df['csr'].min():.4f} – {df['csr'].max():.4f}")
        print(f"    CRR  : {df['crr'].min():.4f} – {df['crr'].max():.4f}")
        print(
            f"    FS   : {df['factor_of_safety'].min():.2f} – {df['factor_of_safety'].max():.2f}")
        return True

    # -----------------------------------------------------------------------
    # Bearing capacity
    # -----------------------------------------------------------------------
    def calculate_bearing_bowles(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 3b: BEARING CAPACITY & SETTLEMENT (Bowles 1988)")
        print("=" * 80)
        df = self.processed_data.copy()
        B, D, SI = 3.0, 1.5, 25.0
        Kd = 1.0 + 0.33 * (D / B)
        size_factor = ((B + 0.3) / B) ** 2
        N = df['spt_n160'].clip(lower=1.0)
        Qa = (8.0 * N * size_factor * Kd).clip(lower=1.0)
        df['foundation_kd'] = Kd
        df['bearing_qa_kpa'] = Qa
        df['bearing_qu_kpa'] = (Qa * 3.0).clip(lower=0.0)
        df['settlement_mm'] = (df['q_actual_kpa'] / Qa) * SI
        print(f"  B={B}m D={D}m Kd={Kd:.3f}")
        print(f"  Qa: {Qa.min():.1f} – {Qa.max():.1f} kPa")
        print(
            f"  Settlement: {df['settlement_mm'].min():.2f} – {df['settlement_mm'].max():.2f} mm")
        self.processed_data = df
        return True

    # -----------------------------------------------------------------------
    # Liquefaction classification  (FIX 5)
    # -----------------------------------------------------------------------
    def classify_liquefaction_dpwh_bsds(self) -> bool:
        """FIX 5 — core samples and rock excluded from classification."""
        print("\n" + "=" * 80)
        print("STEP 4: LIQUEFACTION CLASSIFICATION (DPWH BSDS 2013)")
        print("=" * 80)
        df = self.processed_data.copy()

        def dpwh_classify(row):
            # FIX 5 — exclude non-soil records
            if row.get('is_core_sample', 0) == 1 or row.get('is_rock', 0) == 1:
                return 'NOT APPLICABLE', 'ROCK/CORE SAMPLE'
            fs = row['factor_of_safety']
            if fs < 0.8:
                return 'VERY HIGH', 'LIQUEFIES'
            if fs < 1.0:
                return 'HIGH',      'LIQUEFIES'
            if fs < 1.2:
                return 'MEDIUM',    'MARGINAL'
            if fs < 1.5:
                return 'LOW',        'UNLIKELY'
            return 'VERY LOW',    'NO LIQUEFACTION'

        result = df.apply(dpwh_classify, axis=1)
        df['liquefaction_risk_level'] = result.apply(lambda x: x[0])
        df['liquefaction_status'] = result.apply(lambda x: x[1])
        df['liquefaction'] = df['liquefaction_risk_level'].isin(
            ['VERY HIGH', 'HIGH']).astype(int)

        counts = df['liquefaction_risk_level'].value_counts()
        print("\n  Classification Summary:")
        for lvl, cnt in counts.items():
            print(f"    {lvl:20s}: {cnt:4d}  ({cnt/len(df)*100:5.1f}%)")
        liq_n = df['liquefaction'].sum()
        print(
            f"\n  Total liquefaction cases: {liq_n} ({liq_n/len(df)*100:.1f}%)")

        self.processed_data = df
        return True

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------
    def engineer_features(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 5: FEATURE ENGINEERING")
        print("=" * 80)
        df = self.processed_data.copy()

        df['depth_thickness_m'] = df['depth_to_m'] - df['depth_from_m']
        df['depth_to_groundwater_m'] = df['groundwater_depth_m'] - df['depth_mid_m']
        df['is_below_groundwater'] = (
            df['depth_mid_m'] > df['groundwater_depth_m']).astype(int)
        df['depth_normalized'] = df['depth_mid_m'] / 15.0

        df['spt_n_log'] = np.log1p(df['spt_n_value'])
        df['spt_n160_log'] = np.log1p(df['spt_n160'])
        df['relative_density_from_spt'] = np.clip(
            np.sqrt(df['spt_n160'] / 60) * 100, 0, 100)

        df['effective_stress_ratio'] = (df['effective_overburden_pressure'] /
                                        (df['total_overburden_pressure'] + 1))

        df['is_clean_sand'] = (df['fines_content'] < 5).astype(int)
        df['is_silty_sand'] = ((df['fines_content'] >= 5) & (
            df['fines_content'] < 35)).astype(int)
        df['is_fine_grained'] = (df['fines_content'] >= 35).astype(int)

        df['depth_spt_interaction'] = df['depth_mid_m'] * df['spt_n_value']
        df['csr_depth_interaction'] = df['csr'] * df['depth_mid_m']
        df['cn_factor'] = np.minimum(1.7, np.sqrt(
            101.3 / df['effective_overburden_pressure'].clip(1.0)))

        self.processed_data = df
        print(f"  [OK] {len(df.columns)} total columns")
        return True

    # -----------------------------------------------------------------------
    # Export CSV
    # -----------------------------------------------------------------------
    def export_csv(self) -> bool:
        print("\n" + "=" * 80)
        print("STEP 6: EXPORTING CLEAN CSV")
        print("=" * 80)
        df = self.processed_data.copy()

        output_columns = [
            'borehole_id', 'Municipality', 'Depth_Layer',
            'latitude', 'longitude',
            'depth_from_m', 'depth_to_m', 'depth_mid_m',
            'spt_n_value', 'spt_n_value_liq', 'spt_n160', 'spt_is_refusal', 'spt_is_imputed',
            'uscs_symbol', 'soil_description',
            'unit_weight', 'fines_content', 'groundwater_depth_m', 'gwl_estimated',
            'friction_angle', 'moisture_content', 'plasticity_index', 'mean_particle_size_d50',
            'relative_density_percent',
            'pga_g', 'csr', 'crr', 'n1_60cs',
            'effective_overburden_pressure', 'total_overburden_pressure',
            'factor_of_safety', 'liquefaction_probability', 'liquefaction',
            'liquefaction_risk_level', 'liquefaction_status',
            'bearing_qa_kpa', 'bearing_qu_kpa', 'settlement_mm',
            'is_core_sample', 'is_rock',
            'magnitude_mw', 'msf', 'lpi_weighing_factor', 'lpi_severity_factor',
        ]
        avail = [c for c in output_columns if c in df.columns]
        df_export = df[avail].copy()
        num_cols = df_export.select_dtypes(include=[np.number]).columns
        df_export[num_cols] = df_export[num_cols].fillna(0)

        if not self.client:
            print("  [WARNING] No DB connection — skipping storage upload")
            return False

        try:
            bucket = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            path = "cleaned/Cleaned_data.csv"
            try:
                old = self.client.storage.from_(bucket).download(path)
                if old:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.client.storage.from_(bucket).upload(
                        f"old_cleaned_data/Cleaned_data_{ts}.csv", old,
                        file_options={'content-type': 'text/csv', 'upsert': 'true'})
                    self.client.storage.from_(bucket).remove([path])
                    print(f"  [OK] Old file archived")
            except:
                pass
            self.client.storage.from_(bucket).upload(
                path, df_export.to_csv(index=False).encode('utf-8'),
                file_options={'content-type': 'text/csv', 'upsert': 'true'})
            print(
                f"  [OK] Uploaded {len(df_export)} rows, {len(df_export.columns)} cols → {path}")
        except Exception as e:
            print(f"  [ERROR] Upload failed: {e}")
            return False
        return True

    # -----------------------------------------------------------------------
    # Database helpers
    # -----------------------------------------------------------------------
    def connect_database(self) -> bool:
        if not SUPABASE_AVAILABLE:
            return False
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv(
            'SUPABASE_KEY')
        if not url or not key:
            print("  [WARNING] DB env vars not set — skipping database storage")
            return False
        try:
            self.client = create_client(url, key)
            self.client.table('municipalities').select('id').limit(1).execute()
            print("  [OK] Connected to PostGIS database")
            return True
        except Exception as e:
            print(f"  [WARNING] DB connection failed: {e}")
            return False

    def safe_float(self, v, default=None):
        if pd.isna(v) or v == '' or v is None:
            return default
        try:
            return float(v)
        except:
            return default

    def safe_str(self, v, default=None):
        if pd.isna(v) or v == '' or v is None:
            return default
        return str(v).strip()

    def upsert_municipalities(self) -> bool:
        print("\n  Step 7.1: Upserting municipalities...")
        df = self.processed_data
        muni_col = 'Municipality' if 'Municipality' in df.columns else 'municipality'
        if muni_col not in df.columns:
            print("    [ERROR] No municipality column")
            return False
        for name in df[muni_col].dropna().unique():
            name = str(name).strip()
            try:
                res = self.client.table('municipalities').select(
                    'id').eq('name', name).execute()
                if res.data:
                    self.municipality_ids[name] = res.data[0]['id']
                else:
                    res = self.client.table('municipalities').insert(
                        {'name': name, 'description': 'Municipality in Tarlac Province'}
                    ).execute()
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
        groups = df.groupby(bh_col).first()
        skipped = 0
        for bh_id, row in groups.iterrows():
            bh_id = str(bh_id).strip()
            muni = self.safe_str(row.get(muni_col))
            lat = self.safe_float(row.get('latitude', row.get('Latitude')))
            lon = self.safe_float(row.get('longitude', row.get('Longitude')))
            if not muni or muni not in self.municipality_ids or not lat or not lon:
                skipped += 1
                continue
            try:
                res = self.client.table('boreholes').select(
                    'id').eq('borehole_id', bh_id).execute()
                if res.data:
                    self.borehole_record_ids[bh_id] = res.data[0]['id']
                else:
                    res = self.client.table('boreholes').insert({
                        'borehole_id': bh_id, 'latitude': lat, 'longitude': lon,
                        'elevation': self.safe_float(row.get('Elevation')),
                        'depth_total_m': 15.0,
                        'remarks': f'Data from {muni}',
                        'municipality_id': self.municipality_ids[muni],
                    }).execute()
                    if res.data:
                        self.borehole_record_ids[bh_id] = res.data[0]['id']
            except Exception as e:
                print(f"    [ERROR] {bh_id}: {e}")
        if skipped:
            print(f"    [INFO] Skipped {skipped} boreholes")
        if not self.borehole_record_ids:
            print("    [ERROR] No boreholes inserted")
            return False
        print(f"    [OK] {len(self.borehole_record_ids)} boreholes")
        return True

    def store_soil_layers(self) -> bool:
        print("\n  Step 7.3: Storing soil layers...")
        df = self.processed_data.copy()
        bh_col = 'Borehole ID' if 'Borehole ID' in df.columns else 'borehole_id'

        layer_map = {name: i for i, name in enumerate(
            sorted(df['Depth_Layer'].unique()), 1)}
        records, skipped = [], 0

        for _, row in df.iterrows():
            bh_id = str(row.get(bh_col, '')).strip()
            bh_rec = self.borehole_record_ids.get(bh_id)
            if not bh_rec:
                skipped += 1
                continue

            records.append({
                'borehole_id':                   bh_rec,
                'layer_number':                  int(layer_map.get(row['Depth_Layer'], 1)),
                'depth_from_m':                  self.safe_float(row['depth_from_m']),
                'depth_to_m':                    self.safe_float(row['depth_to_m']),
                'depth_range':                   f"{row['depth_from_m']:.1f}-{row['depth_to_m']:.1f}m",
                'spt_n_value':                   self.safe_float(row['spt_n_value']),
                'spt_n160':                      self.safe_float(row.get('spt_n160')),
                'spt_is_refusal':                bool(row.get('spt_is_refusal', 0)),
                'spt_is_imputed':                bool(row.get('spt_is_imputed', False)),
                'uscs_symbol':                   self.safe_str(row.get('uscs_symbol')) or None,
                'soil_description':              self.safe_str(row.get('soil_description')) or None,
                'unit_weight':                   self.safe_float(row['unit_weight']),
                'fines_content':                 self.safe_float(row['fines_content']),
                'groundwater_depth_m':           self.safe_float(row['groundwater_depth_m']),
                'pga_g':                         self.safe_float(row['pga_g']),
                'csr':                           self.safe_float(row['csr']),
                'cyclic_strength_ratio':         self.safe_float(row.get('crr')),
                'n1_60cs':                       self.safe_float(row.get('n1_60cs')),
                'effective_overburden_pressure': self.safe_float(row['effective_overburden_pressure']),
                'total_overburden_pressure':     self.safe_float(row['total_overburden_pressure']),
                'factor_of_safety':              self.safe_float(row['factor_of_safety']),
                'liquefaction_probability':      self.safe_float(row.get('liquefaction_probability')),
                'liquefaction':                  bool(row.get('liquefaction', 0)),
                'liquefaction_risk_level':       self.safe_str(row.get('liquefaction_risk_level')),
                'liquefaction_status':           self.safe_str(row.get('liquefaction_status')),
                'relative_density_percent':      self.safe_float(row.get('relative_density_percent')),
                'is_core_sample':                int(row.get('is_core_sample', 0)),
                'is_rock':                       int(row.get('is_rock', 0)),
                'magnitude_mw':                  self.safe_float(row.get('magnitude_mw')),
                'msf':                           self.safe_float(row.get('msf')),
                'q_actual_kpa':                  self.safe_float(row.get('q_actual_kpa')),
                'foundation_kd':                 self.safe_float(row.get('foundation_kd')),
                'bearing_qu_kpa':                self.safe_float(row.get('bearing_qu_kpa')),
                'bearing_qa_kpa':                self.safe_float(row.get('bearing_qa_kpa')),
                'settlement_mm':                 self.safe_float(row.get('settlement_mm')),
                'lpi_weighing_factor':           self.safe_float(row.get('lpi_weighing_factor')),
                'lpi_severity_factor':           self.safe_float(row.get('lpi_severity_factor')),
            })
            for opt in ['moisture_content', 'friction_angle', 'cohesion_kpa',
                        'plasticity_index', 'mean_particle_size_d50']:
                if opt in row.index and pd.notna(row.get(opt)):
                    records[-1][opt] = self.safe_float(row[opt])
            if 'Elastic Modulus (Es) (MN/m²)' in row and pd.notna(row['Elastic Modulus (Es) (MN/m²)']):
                records[-1]['elastic_modulus_es'] = self.safe_float(
                    row['Elastic Modulus (Es) (MN/m²)'])

        EXTRA_COLS = {'spt_is_refusal', 'spt_is_imputed', 'is_core_sample', 'is_rock',
                      'magnitude_mw', 'msf', 'n1_60cs', 'q_actual_kpa', 'foundation_kd',
                      'bearing_qu_kpa', 'bearing_qa_kpa', 'settlement_mm',
                      'lpi_weighing_factor', 'lpi_severity_factor',
                      'liquefaction_probability', 'liquefaction_status'}

        def strip_extra(batch):
            return [{k: v for k, v in r.items() if k not in EXTRA_COLS} for r in batch]

        # Clear old layers
        bh_uuids = list(self.borehole_record_ids.values())
        for i in range(0, len(bh_uuids), 100):
            try:
                self.client.table('soil_layers').delete().in_(
                    'borehole_id', bh_uuids[i:i+100]).execute()
            except Exception as e:
                print(f"    [WARNING] Delete batch failed: {e}")
        print(f"    Cleared existing layers")

        total, fallback = 0, False
        for i in range(0, len(records), 25):
            batch = records[i:i+25]
            try:
                self.client.table('soil_layers').insert(
                    strip_extra(batch) if fallback else batch).execute()
                total += len(batch)
            except Exception as e:
                if 'PGRST204' in str(e) or 'schema cache' in str(e):
                    if not fallback:
                        print(
                            "    [WARNING] Extended columns missing — falling back to base schema")
                        fallback = True
                    try:
                        self.client.table('soil_layers').insert(
                            strip_extra(batch)).execute()
                        total += len(batch)
                    except Exception as e2:
                        print(
                            f"    [WARNING] Batch {i//25+1} failed on fallback: {e2}")
                else:
                    print(f"    [WARNING] Batch {i//25+1} failed: {e}")

        if skipped:
            print(f"    [INFO] Skipped {skipped} records (borehole not found)")
        if not total:
            print("    [ERROR] Nothing inserted")
            return False
        print(f"    [OK] Inserted {total} soil layer records")
        return True

    def store_to_postgis(self) -> bool:
        if not self.client:
            return False
        print("\n" + "=" * 80)
        print("STEP 7: STORING TO POSTGIS")
        print("=" * 80)
        try:
            if not self.upsert_municipalities():
                return False
            if not self.upsert_boreholes():
                return False
            if not self.store_soil_layers():
                return False
            print(f"\n  [OK] Stored {len(self.processed_data)} records")
            return True
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False

    # -----------------------------------------------------------------------
    # Validation report
    # -----------------------------------------------------------------------
    def print_validation_report(self):
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        if self.validation_errors:
            print("\n  [ERRORS]")
            for e in self.validation_errors:
                print(f"    ✗ {e}")
        else:
            print("\n  ✓ No critical errors")
        if self.validation_warnings:
            print("\n  [WARNINGS]")
            for w in self.validation_warnings:
                print(f"    ⚠ {w}")
        else:
            print("\n  ✓ No warnings")

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    def run(self) -> bool:
        print("\n" + "=" * 80)
        print("GEOTECHNICAL DATA PROCESSING PIPELINE  v2")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.connect_database()

        if not self.excel_path and self.use_storage:
            if not self.download_from_supabase_storage():
                print("[ERROR] Failed to download from Supabase Storage")
                return False

        self.upload_raw_file_to_bucket()

        steps = [
            ("Load Excel",                          self.load_excel),
            ("Process and Validate",                self.process_and_validate),
            ("Fill Missing by Borehole",            self.fill_missing_by_borehole),
            ("Calculate CSR/CRR",                   self.calculate_csr_crr),
            ("Bearing Capacity & Settlement",
             self.calculate_bearing_bowles),
            ("Classify Liquefaction (DPWH BSDS)",
             self.classify_liquefaction_dpwh_bsds),
            ("Engineer Features",                   self.engineer_features),
            ("Export CSV",                          self.export_csv),
        ]

        for name, fn in steps:
            print(f"\n{'─'*40}")
            print(f"Running: {name}")
            if not fn():
                # export_csv fails without DB — non-fatal
                if name == "Export CSV":
                    print(f"  [INFO] CSV export skipped (no DB connection)")
                else:
                    print(f"[ERROR] Pipeline failed at: {name}")
                    return False

        self.print_validation_report()

        if self.client:
            self.store_to_postgis()

        print("\n" + "=" * 80)
        print("[SUCCESS] PIPELINE COMPLETE")
        print("=" * 80)
        print(f"  Records  : {len(self.processed_data)}")
        print(f"  Columns  : {len(self.processed_data.columns)}")
        print(f"  Fixes applied: FIX1–FIX10")
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import sys
    excel_file = sys.argv[1] if len(sys.argv) > 1 else None

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    if url and key:
        print(f"[INFO] Supabase configured — DB storage enabled")
    else:
        print("[WARNING] Supabase env vars not found — DB storage skipped")

    pipeline = GeotechnicalPipeline(excel_file)
    if not pipeline.run():
        sys.exit(1)
    print("\n✓ Done")


if __name__ == "__main__":
    main()
