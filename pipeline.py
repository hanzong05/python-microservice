#!/usr/bin/env python3
"""
Geotechnical Data Processing Pipeline
Single-pass processing with validation, CSR/CRR calculation, and DPWH BSDS classification

Features:
✅ Single-pass data processing
✅ Built-in validation (coordinates, SPT, soil parameters)
✅ Automatic CSR/CRR calculation (Seed & Idriss 1971)
✅ Liquefaction classification per DPWH BSDS (2013)
✅ Exports clean CSV ready for database ingestion
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[INFO] python-dotenv not installed - .env file will not be loaded")
    print("Install with: pip install python-dotenv")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[INFO] Supabase not available - database storage will be skipped")


class GeotechnicalPipeline:
    """
    Single-pass geotechnical data processing pipeline
    """

    def __init__(self, excel_file_path: Optional[str] = None, output_csv_path: Optional[str] = None, use_storage: bool = True):
        self.use_storage = use_storage
        self.excel_path = None
        self.excel_file_bytes = None

        if excel_file_path:
            self.excel_path = Path(excel_file_path)
            self.output_path = None
        else:
            self.output_path = None

        self.raw_data = {}
        self.processed_data = None
        self.validation_errors = []
        self.validation_warnings = []

        self.client = None
        self.municipality_ids = {}
        self.borehole_record_ids = {}

    def download_from_supabase_storage(self) -> bool:
        """Download Excel file from Supabase Storage into memory"""
        print("\n" + "="*80)
        print("STEP 1: DOWNLOADING FROM SUPABASE STORAGE")
        print("="*80)

        if not SUPABASE_AVAILABLE:
            print("  [ERROR] Supabase not available")
            return False

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

        if not supabase_url or not supabase_key:
            print("  [ERROR] Environment variables not set")
            return False

        try:
            self.client = create_client(supabase_url, supabase_key)

            storage_path = "raw/Raw_data.xlsx"
            bucket_name = os.getenv(
                'SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            print(f"  Downloading: {storage_path} from bucket: {bucket_name}")

            file_data = self.client.storage.from_(
                bucket_name).download(storage_path)

            if not file_data:
                print(f"  [ERROR] File not found in storage: {storage_path}")
                return False

            self.excel_file_bytes = file_data
            print(f"  [OK] Downloaded {len(file_data)} bytes into memory")
            return True
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def upload_raw_file_to_bucket(self) -> bool:
        """Upload the local raw Excel file to Supabase Storage (raw/ folder)"""
        if not self.excel_path or not self.excel_path.exists():
            return True

        if not self.client:
            print("  [INFO] No database connection - skipping raw file upload")
            return True

        try:
            bucket_name = os.getenv(
                'SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            dest_path = f"raw/{self.excel_path.name}"

            with open(self.excel_path, 'rb') as f:
                file_bytes = f.read()

            self.client.storage.from_(bucket_name).upload(
                dest_path,
                file_bytes,
                file_options={
                    'content-type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'upsert': 'true'
                }
            )
            print(f"  [OK] Raw file uploaded to bucket: {dest_path}")
            return True
        except Exception as e:
            print(f"  [WARNING] Raw file upload failed: {e}")
            return True

    def load_excel(self) -> bool:
        """Load Excel file from local path or memory bytes"""
        print("\n" + "="*80)
        print("STEP 1: LOADING EXCEL FILE")
        print("="*80)

        try:
            if self.excel_file_bytes:
                print("  Loading from memory (Supabase Storage)...")
                xl_file = pd.ExcelFile(io.BytesIO(self.excel_file_bytes))
            elif self.excel_path and self.excel_path.exists():
                print(f"  Loading from local file: {self.excel_path}")
                xl_file = pd.ExcelFile(self.excel_path)
            else:
                print(f"  [ERROR] No file source available")
                if self.excel_path:
                    print(f"  File not found: {self.excel_path}")
                return False

            print(f"\n  Found {len(xl_file.sheet_names)} sheets")

            for sheet_name in xl_file.sheet_names:
                if sheet_name == 'Summary':
                    print(f"    Skipping: {sheet_name}")
                    continue

                if self.excel_file_bytes:
                    df = pd.read_excel(io.BytesIO(
                        self.excel_file_bytes), sheet_name=sheet_name)
                else:
                    df = pd.read_excel(self.excel_path, sheet_name=sheet_name)

                self.raw_data[sheet_name] = df
                print(
                    f"    Loaded: {sheet_name} ({len(df)} rows, {len(df.columns)} columns)")

            print(f"\n  [OK] Loaded {len(self.raw_data)} data sheets")
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return False

    def clean_coordinate(self, value) -> Optional[float]:
        """Clean coordinate value (remove degree symbols, etc.)"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            clean = str(value).replace('°', '').replace(
                "'", '').replace('"', '').strip()
            return float(clean)
        except:
            return None

    def parse_pga(self, value) -> Optional[float]:
        """Parse Peak Ground Acceleration from various formats"""
        if pd.isna(value):
            return None
        try:
            matches = re.findall(r'(\d+\.?\d*)g', str(value).lower())
            if matches:
                return np.mean([float(m) for m in matches])
            return float(str(value).replace('g', '').strip())
        except:
            return None

    def parse_relative_density(self, value) -> Optional[float]:
        """Convert relative density text to numeric percentage"""
        if pd.isna(value):
            return None

        try:
            return float(value)
        except:
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

    def parse_elastic_modulus(self, value) -> Optional[float]:
        """Parse elastic modulus from various formats"""
        if pd.isna(value):
            return None

        if isinstance(value, datetime):
            return None

        try:
            return float(value)
        except:
            pass

        value_str = str(value).lower()
        if 'to' in value_str:
            try:
                parts = value_str.split('to')
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2
            except:
                return None

        return None

    def pre_clean_dataframe(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """
        PRE-CLEAN dataframe — must run FIRST before any numeric operations.

        FIX 1: Drop stray unnamed columns (Excel formatting artifacts that cause
                StringDtype crashes when the pipeline tries to write float values).
        FIX 2: Convert pandas StringDtype → object dtype.
                pandas 2.x reads Excel string columns as StringDtype which is strict
                and rejects float assignments (df.loc[mask, col] = float_series).
                Converting to object dtype restores pandas 1.x behaviour.
        """
        df_clean = df.copy()

        # ── FIX 1: Drop unnamed columns ───────────────────────────────────────
        unnamed_cols = [c for c in df_clean.columns if str(
            c).lower().startswith('unnamed')]
        if unnamed_cols:
            df_clean = df_clean.drop(columns=unnamed_cols)
            print(f"    [FIX] Dropped unnamed columns: {unnamed_cols}")

        # ── FIX 2: StringDtype → object ───────────────────────────────────────
        str_cols = [c for c in df_clean.columns if str(
            df_clean[c].dtype) == 'str']
        if str_cols:
            df_clean[str_cols] = df_clean[str_cols].astype(object)

        # 1. Clean Latitude (remove degree symbols)
        if 'Latitude' in df_clean.columns:
            df_clean['Latitude'] = df_clean['Latitude'].apply(
                self.clean_coordinate)

        # 2. Clean Longitude (remove degree symbols)
        if 'Longitude' in df_clean.columns:
            df_clean['Longitude'] = df_clean['Longitude'].apply(
                self.clean_coordinate)

        # 3. Clean Peak Ground Acceleration (parse complex strings)
        if 'Peak Ground Acceleration' in df_clean.columns:
            df_clean['Peak Ground Acceleration'] = df_clean['Peak Ground Acceleration'].apply(
                self.parse_pga)

        # 4. Clean Relative Density (convert text to numeric)
        if 'Relative Density' in df_clean.columns:
            df_clean['Relative Density'] = df_clean['Relative Density'].apply(
                self.parse_relative_density)

        # 5. Clean Elastic Modulus (parse ranges and dates)
        if 'Elastic Modulus (Es) (MN/m²)' in df_clean.columns:
            df_clean['Elastic Modulus (Es) (MN/m²)'] = df_clean['Elastic Modulus (Es) (MN/m²)'].apply(
                self.parse_elastic_modulus)

        return df_clean

    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate coordinates after pre-cleaning"""
        df_clean = df.copy()

        if 'Latitude' in df_clean.columns:
            df_clean['latitude'] = df_clean['Latitude']
            invalid = df_clean['latitude'].isna().sum()
            if invalid > 0:
                self.validation_warnings.append(
                    f"Missing/invalid latitude: {invalid} records")

            valid_lat = df_clean['latitude'].dropna()
            if len(valid_lat) > 0:
                out_of_range = ((valid_lat < 15.0) | (valid_lat > 16.0)).sum()
                if out_of_range > 0:
                    self.validation_errors.append(
                        f"Latitude out of Tarlac range (15.0-16.0): {out_of_range} records")

        if 'Longitude' in df_clean.columns:
            df_clean['longitude'] = df_clean['Longitude']
            invalid = df_clean['longitude'].isna().sum()
            if invalid > 0:
                self.validation_warnings.append(
                    f"Missing/invalid longitude: {invalid} records")

            valid_lon = df_clean['longitude'].dropna()
            if len(valid_lon) > 0:
                out_of_range = ((valid_lon < 120.0) |
                                (valid_lon > 121.0)).sum()
                if out_of_range > 0:
                    self.validation_errors.append(
                        f"Longitude out of Tarlac range (120.0-121.0): {out_of_range} records")

        return df_clean

    def validate_spt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate SPT values"""
        df_clean = df.copy()

        spt_col = None
        for col in ['SPT N-Value', 'SPT N Value', 'spt_n_value', 'N-Value', 'N Value', 'SPT']:
            if col in df_clean.columns:
                spt_col = col
                break

        if spt_col:
            df_clean['spt_n_value'] = pd.to_numeric(
                df_clean[spt_col], errors='coerce')

            valid_spt = df_clean['spt_n_value'].dropna()
            if len(valid_spt) > 0:
                invalid = ((valid_spt < 0) | (valid_spt > 100)).sum()
                if invalid > 0:
                    self.validation_errors.append(
                        f"SPT values out of valid range (0-100): {invalid} records")

            missing = df_clean['spt_n_value'].isna().sum()
            if missing > 0:
                self.validation_warnings.append(
                    f"Missing SPT values: {missing} records")
                df_clean['spt_n_value'] = df_clean['spt_n_value'].fillna(15.0)
        else:
            self.validation_warnings.append(
                "SPT column not found, using default value 15.0")
            df_clean['spt_n_value'] = 15.0

        return df_clean

    def validate_soil_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate soil parameters"""
        df_clean = df.copy()

        # Unit weight
        unit_weight_col = None
        for col in ['Unit Weight (γ)', 'Unit Weight', 'unit_weight', 'Unit Weight (kN/m³)', 'γ']:
            if col in df_clean.columns:
                unit_weight_col = col
                break

        if unit_weight_col:
            df_clean['unit_weight'] = pd.to_numeric(
                df_clean[unit_weight_col], errors='coerce')
            valid_uw = df_clean['unit_weight'].dropna()
            if len(valid_uw) > 0:
                invalid = ((valid_uw < 10) | (valid_uw > 25)).sum()
                if invalid > 0:
                    self.validation_warnings.append(
                        f"Unit weight out of typical range (10-25 kN/m³): {invalid} records")
            df_clean['unit_weight'] = df_clean['unit_weight'].fillna(18.0)
        else:
            df_clean['unit_weight'] = 18.0

        # Fines content
        fines_col = None
        for col in ['Fines Content', 'fines_content', 'Fines (%)', 'Fines']:
            if col in df_clean.columns:
                fines_col = col
                break

        if fines_col:
            df_clean['fines_content'] = pd.to_numeric(
                df_clean[fines_col], errors='coerce')
            valid_fines = df_clean['fines_content'].dropna()
            if len(valid_fines) > 0:
                invalid = ((valid_fines < 0) | (valid_fines > 100)).sum()
                if invalid > 0:
                    self.validation_errors.append(
                        f"Fines content out of valid range (0-100%): {invalid} records")
            df_clean['fines_content'] = df_clean['fines_content'].fillna(15.0)
        else:
            df_clean['fines_content'] = 15.0

        # Groundwater depth
        gwl_col = None
        for col in ['Groundwater Level (m)', 'Ground Water Table', 'Groundwater Level',
                    'Groundwater Depth', 'groundwater_depth_m', 'GWL', 'Water Table']:
            if col in df_clean.columns:
                gwl_col = col
                break

        if gwl_col:
            df_clean['groundwater_depth_m'] = pd.to_numeric(
                df_clean[gwl_col], errors='coerce')
            df_clean['groundwater_depth_m'] = df_clean['groundwater_depth_m'].fillna(
                5.0)
        else:
            df_clean['groundwater_depth_m'] = 5.0

        # Soil type
        soil_type_col = None
        for col in ['Soil Type', 'soil_type', 'Soil Classification', 'Classification']:
            if col in df_clean.columns:
                soil_type_col = col
                break

        if soil_type_col:
            df_clean['soil_type'] = (df_clean[soil_type_col]
                                     .fillna('')
                                     .astype(str)
                                     .str.strip())
            df_clean['soil_type'] = df_clean['soil_type'].replace('nan', '')

        # USCS symbol
        uscs_col = None
        for col in ['USCS Symbol', 'uscs_symbol', 'USCS Classification', 'USCS']:
            if col in df_clean.columns:
                uscs_col = col
                break

        if uscs_col:
            df_clean['uscs_symbol'] = (df_clean[uscs_col]
                                       .fillna('')
                                       .astype(str)
                                       .str.strip())
            df_clean['uscs_symbol'] = df_clean['uscs_symbol'].replace(
                'nan', '')

        # Soil description
        desc_col = None
        for col in ['Soil/Rock Description', 'Soil Description', 'soil_description', 'Description', 'Soil Name']:
            if col in df_clean.columns:
                desc_col = col
                break

        if desc_col:
            df_clean['soil_description'] = (df_clean[desc_col]
                                            .fillna('')
                                            .astype(str)
                                            .str.strip())
            df_clean['soil_description'] = df_clean['soil_description'].replace(
                'nan', '')

        # Internal friction angle
        friction_col = None
        for col in ['Internal Friction Angle', 'Friction Angle', 'friction_angle', 'φ']:
            if col in df_clean.columns:
                friction_col = col
                break

        if friction_col:
            df_clean['friction_angle'] = pd.to_numeric(
                df_clean[friction_col], errors='coerce')

        # Plasticity Index
        pi_col = None
        for col in ['Plasticity Index (PI)', 'Plasticity Index', 'plasticity_index', 'PI']:
            if col in df_clean.columns:
                pi_col = col
                break

        if pi_col:
            df_clean['plasticity_index'] = pd.to_numeric(
                df_clean[pi_col], errors='coerce')

        # Natural water content
        wc_col = None
        for col in ['Natural Water Content (ω)', 'Natural Water Content', 'moisture_content',
                    'Water Content', 'Moisture Content']:
            if col in df_clean.columns:
                wc_col = col
                break

        if wc_col:
            df_clean['moisture_content'] = pd.to_numeric(
                df_clean[wc_col], errors='coerce')

        # Mean particle size D50
        d50_col = None
        for col in ['Mean Particle Size (D50) (mm)', 'Mean Particle Size', 'mean_particle_size_d50',
                    'D50', 'Particle Size (D50)']:
            if col in df_clean.columns:
                d50_col = col
                break

        if d50_col:
            df_clean['mean_particle_size_d50'] = pd.to_numeric(
                df_clean[d50_col], errors='coerce')

        return df_clean

    def process_and_validate(self) -> bool:
        """Single-pass processing with validation"""
        print("\n" + "="*80)
        print("STEP 2: PRE-CLEANING AND VALIDATING DATA")
        print("="*80)

        all_dfs = []

        for sheet_name, df in self.raw_data.items():
            print(f"\n  Pre-cleaning sheet: {sheet_name}")

            # STEP 1: PRE-CLEAN — must run FIRST (drops unnamed cols + fixes StringDtype)
            df = self.pre_clean_dataframe(df, sheet_name)

            # STEP 2: Validate coordinates
            df = self.validate_coordinates(df)

            # STEP 3: Validate SPT
            df = self.validate_spt(df)

            # STEP 4: Validate soil parameters
            df = self.validate_soil_parameters(df)

            # STEP 5: Map PGA to pga_g
            if 'Peak Ground Acceleration' in df.columns:
                df['pga_g'] = df['Peak Ground Acceleration']
            else:
                df['pga_g'] = 0.35

            df['pga_g'] = pd.to_numeric(
                df['pga_g'], errors='coerce').fillna(0.35)

            # STEP 6: Map Relative Density
            if 'Relative Density' in df.columns:
                df['relative_density_percent'] = df['Relative Density']

            # STEP 7: Add Depth_Layer column
            df['Depth_Layer'] = sheet_name

            # STEP 8: Extract depth from sheet name
            def extract_depth(layer_name):
                match = re.search(
                    r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', str(layer_name))
                if match:
                    return float(match.group(1)), float(match.group(2))
                return 0.0, 1.5

            depths = df['Depth_Layer'].apply(extract_depth)
            df['depth_from_m'] = depths.apply(lambda x: x[0])
            df['depth_to_m'] = depths.apply(lambda x: x[1])
            df['depth_mid_m'] = (df['depth_from_m'] + df['depth_to_m']) / 2

            # STEP 9: Compute missing raw columns from empirical formulas
            df = self.compute_missing_raw_columns(df)

            all_dfs.append(df)
            print(f"    [OK] Processed {len(df)} rows")

        if all_dfs:
            self.processed_data = pd.concat(all_dfs, ignore_index=True)
            print(
                f"\n  [OK] Combined {len(self.processed_data)} total records")
            print(f"  Sheets combined: {len(self.raw_data)}")

            if 'Depth_Layer' in self.processed_data.columns:
                print(f"\n  Depth Layer Distribution:")
                depth_counts = self.processed_data['Depth_Layer'].value_counts(
                ).sort_index()
                for depth, count in depth_counts.items():
                    print(f"    {depth}: {count} rows")

            return True
        return False

    def compute_missing_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in NULL cells in the raw data columns during data cleaning.

        Computes (only for rows where the value is currently missing):
          - Total Overburden Pressure  (σ_v  = γ × z_mid)
          - Effective Overburden Pressure (σ'_v = σ_v − u, min 1 kPa)
          - Relative Density (Dr % = √(N1_60/60) × 100, Skempton 1986)
          - Cyclic Stress Ratio (CSR = 0.65 × σ_v/σ'_v × PGA × rd)
        """
        GAMMA_WATER = 9.81
        COHESIVE_USCS = {"CL", "CH", "ML", "MH", "OL", "OH", "Pt", "CL-ML"}

        df = df.copy()
        z = df['depth_mid_m']
        γ = df['unit_weight']
        gwl = df['groundwater_depth_m']

        sigma_v = γ * z
        u = np.maximum(0.0, z - gwl) * GAMMA_WATER
        sigma_eff = np.maximum(1.0, sigma_v - u)

        rd = np.where(z <= 9.15,
                      1.0 - 0.00765 * z,
                      np.where(z <= 23.0,
                               1.174 - 0.0267 * z,
                               0.0))
        rd = np.clip(rd, 0.0, 1.0)

        # 1. Total Overburden Pressure
        tot_col = 'Total Overburden Pressure'
        if tot_col in df.columns:
            missing = pd.to_numeric(df[tot_col], errors='coerce').isna()
            df.loc[missing, tot_col] = sigma_v[missing].round(2)
            filled = missing.sum()
            if filled:
                print(
                    f"      [FILL] Total Overburden Pressure: {filled} rows computed")

        # 2. Effective Overburden Pressure (typo "Presssure" in raw data)
        for eff_col in ('Effective Overburden Presssure', 'Effective Overburden Pressure'):
            if eff_col in df.columns:
                missing = pd.to_numeric(df[eff_col], errors='coerce').isna()
                df.loc[missing, eff_col] = sigma_eff[missing].round(2)
                filled = missing.sum()
                if filled:
                    print(
                        f"      [FILL] Effective Overburden Pressure: {filled} rows computed")
                break

        # 3. Relative Density (Dr %)
        rd_col = 'Relative Density'
        n160_col = 'Corrected SPT-N Value (N1(60))'
        uscs_col = 'USCS Symbol'
        if rd_col in df.columns and n160_col in df.columns:
            n1_60 = pd.to_numeric(df[n160_col], errors='coerce')
            uscs = (df[uscs_col].fillna('').astype(str).str.strip().str.upper()
                    if uscs_col in df.columns
                    else pd.Series([''] * len(df), index=df.index))
            missing = (
                pd.to_numeric(df[rd_col], errors='coerce').isna()
                & n1_60.notna()
                & (n1_60 > 0)
                & ~uscs.isin(COHESIVE_USCS)
            )
            df.loc[missing, rd_col] = (
                np.sqrt(n1_60[missing] / 60.0) * 100.0
            ).round(2)
            filled = missing.sum()
            if filled:
                print(
                    f"      [FILL] Relative Density: {filled} rows computed (Skempton 1986)")

        # 4. Cyclic Stress Ratio (CSR)
        csr_col = 'Cyclic Stress Ratio (CSR)'
        if csr_col in df.columns:
            csr_vals = 0.65 * (sigma_v / sigma_eff) * df['pga_g'] * rd
            missing = (
                pd.to_numeric(df[csr_col], errors='coerce').isna()
                & (df['pga_g'] > 0)
            )
            df.loc[missing, csr_col] = csr_vals[missing].round(6)
            filled = missing.sum()
            if filled:
                print(
                    f"      [FILL] CSR: {filled} rows computed (Seed & Idriss 1971)")

        return df

    def calculate_csr_crr(self) -> bool:
        """
        Calculate CSR, CRR, MSF, FS, and LPI components per thesis methodology.
        """
        print("\n" + "="*80)
        print("STEP 3: CSR/CRR ANALYSIS (Thesis Methodology)")
        print("="*80)

        df = self.processed_data.copy()
        gamma_water = 9.81

        MAGNITUDE_MW = 6.5
        df['magnitude_mw'] = MAGNITUDE_MW
        print(f"  Magnitude (Mw): {MAGNITUDE_MW} (Tarlac seismic zone)")

        MSF = (10 ** 2.24) / (MAGNITUDE_MW ** 2.56)
        df['msf'] = MSF
        print(f"  MSF: {MSF:.6f}")

        Q_ACTUAL = 50.0
        df['q_actual_kpa'] = Q_ACTUAL
        print(
            f"  q_actual: {Q_ACTUAL} kPa (default building contact pressure)")

        df['spt_n60'] = df['spt_n_value'] * 1.0
        n1_60 = pd.to_numeric(df.get('Corrected SPT-N Value (N1(60))', df['spt_n_value']),
                              errors='coerce').fillna(df['spt_n_value'])
        df['spt_n160'] = n1_60

        df['total_overburden_pressure'] = df['unit_weight'] * df['depth_mid_m']
        depth_below_wt = np.maximum(
            0, df['depth_mid_m'] - df['groundwater_depth_m'])
        df['effective_overburden_pressure'] = (
            df['total_overburden_pressure'] - gamma_water * depth_below_wt
        ).clip(lower=1.0)

        z = df['depth_mid_m']
        rd = np.where(z <= 9.15,
                      1.0 - 0.00765 * z,
                      np.where(z <= 23.0,
                               1.174 - 0.0267 * z,
                               0.0))
        rd = np.clip(rd, 0.0, 1.0)

        print("  Calculating CSR (Seed & Idriss 1971)...")
        df['csr'] = (0.65
                     * df['pga_g']
                     * (df['total_overburden_pressure'] / df['effective_overburden_pressure'])
                     * rd)

        print("  Computing (N1)60cs with fines correction (NCEER)...")
        FC = df['fines_content'].clip(lower=0.1)
        alpha = np.where(FC < 5.0,  0.0,
                         np.where(FC <= 35.0, np.exp(1.76 - 190.0 / FC**2), 5.0))
        beta = np.where(FC < 5.0,  1.0,
                        np.where(FC <= 35.0, 0.99 + FC**1.5 / 1000.0, 1.2))
        df['n1_60cs'] = alpha + beta * df['spt_n160']

        print("  Calculating CRR(7.5) (Robertson-Wride formula)...")
        N = df['n1_60cs'].clip(upper=30.0)
        crr_raw = np.exp(
            N / 14.1
            + (N / 126.0) ** 2
            - (N / 23.6) ** 3
            + (N / 25.4) ** 4
            - 2.67
        )
        df['crr'] = np.where(df['n1_60cs'] >= 30.0, 0.6,
                             crr_raw.clip(0.0, 0.6))

        df['factor_of_safety'] = (df['crr'] * MSF) / (df['csr'] + 1e-9)

        df['lpi_weighing_factor'] = np.maximum(
            0.0, 10.0 - 0.5 * df['depth_mid_m'])
        df['lpi_severity_factor'] = np.maximum(
            0.0, 1.0 - df['factor_of_safety'])

        df['liquefaction_probability'] = np.where(
            df['factor_of_safety'] < 1.0,
            (1.0 - df['factor_of_safety']) * 100,
            np.where(df['factor_of_safety'] < 1.5, 30.0, 10.0)
        ).clip(0.0, 100.0)

        self.processed_data = df
        print(f"  [OK] Calculated for {len(df)} records")
        print(
            f"    CSR  range : {df['csr'].min():.4f} – {df['csr'].max():.4f}")
        print(
            f"    CRR  range : {df['crr'].min():.4f} – {df['crr'].max():.4f}")
        print(
            f"    FS   range : {df['factor_of_safety'].min():.2f} – {df['factor_of_safety'].max():.2f}")
        print(
            f"    (N1)60cs   : {df['n1_60cs'].min():.2f} – {df['n1_60cs'].max():.2f}")
        return True

    def calculate_bearing_bowles(self) -> bool:
        """
        Bearing capacity and settlement per Bowles (1988) / Meyerhof (1956) SPT method.
        """
        print("\n" + "="*80)
        print("STEP 3b: BEARING CAPACITY & SETTLEMENT (Bowles 1988)")
        print("="*80)

        df = self.processed_data.copy()

        B = 3.0       # footing width (m) — matches thesis validation standard
        D = 1.5
        SI_ALLOW = 25.0

        # Depth correction factor (Meyerhof 1956)
        Kd = 1.0 + 0.33 * (D / B)   # = 1.165 for D=1.5 m, B=3.0 m
        df['foundation_kd'] = Kd

        N = df['spt_n160'].clip(lower=1.0)

        # Meyerhof (1956) SPT-based allowable bearing capacity
        # For B > 1.2 m: qa = 8·N·((B+0.3)/B)²·Kd  [kPa, Si = 25 mm]
        # Reference: Bowles (1988), Table 4-4
        size_factor = ((B + 0.3) / B) ** 2          # = 1.21 for B = 3.0 m
        Qa = (8.0 * N * size_factor * Kd).clip(lower=1.0)

        df['bearing_qa_kpa'] = Qa
        df['bearing_qu_kpa'] = (Qa * 3.0).clip(lower=0.0)   # back-computed Qu (FS = 3)

        df['settlement_mm'] = (df['q_actual_kpa'] /
                               df['bearing_qa_kpa']) * SI_ALLOW

        print(
            f"  Foundation: B={B}m, D={D}m, Kd={Kd:.4f}, size_factor={size_factor:.4f}")
        print(
            f"  Qu  range : {df['bearing_qu_kpa'].min():.1f} – {df['bearing_qu_kpa'].max():.1f} kPa")
        print(
            f"  Qa  range : {df['bearing_qa_kpa'].min():.1f} – {df['bearing_qa_kpa'].max():.1f} kPa")
        print(
            f"  Settlement: {df['settlement_mm'].min():.2f} – {df['settlement_mm'].max():.2f} mm")
        print(f"  [OK] Bearing capacity computed for {len(df)} records")

        self.processed_data = df
        return True

    def classify_liquefaction_dpwh_bsds(self) -> bool:
        """
        Classify liquefaction potential per DPWH BSDS (2013)
        """
        print("\n" + "="*80)
        print("STEP 4: LIQUEFACTION CLASSIFICATION (DPWH BSDS 2013)")
        print("="*80)

        df = self.processed_data.copy()

        def dpwh_classify(row):
            fs = row['factor_of_safety']
            if fs < 0.8:
                return 'VERY HIGH', 'LIQUEFIES'
            if fs < 1.0:
                return 'HIGH', 'LIQUEFIES'
            if fs < 1.2:
                return 'MEDIUM', 'MARGINAL'
            if fs < 1.5:
                return 'LOW', 'UNLIKELY'
            return 'VERY LOW', 'NO LIQUEFACTION'

        classification = df.apply(dpwh_classify, axis=1)
        df['liquefaction_risk_level'] = classification.apply(lambda x: x[0])
        df['liquefaction_status'] = classification.apply(lambda x: x[1])
        df['liquefaction'] = df['liquefaction_risk_level'].isin(
            ['VERY HIGH', 'HIGH']).astype(int)

        risk_counts = df['liquefaction_risk_level'].value_counts()
        print("\n  Classification Summary:")
        for risk_level, count in risk_counts.items():
            pct = (count / len(df)) * 100
            print(f"    {risk_level:12s}: {count:4d} records ({pct:5.1f}%)")

        liquefies = df['liquefaction'].sum()
        print(
            f"\n  Total liquefaction cases: {liquefies} ({liquefies/len(df)*100:.1f}%)")

        self.processed_data = df
        return True

    def engineer_features(self) -> bool:
        """Engineer additional features"""
        print("\n" + "="*80)
        print("STEP 5: FEATURE ENGINEERING")
        print("="*80)

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

        df['effective_stress_ratio'] = df['effective_overburden_pressure'] / \
            (df['total_overburden_pressure'] + 1)

        df['is_clean_sand'] = (df['fines_content'] < 5).astype(int)
        df['is_silty_sand'] = ((df['fines_content'] >= 5) & (
            df['fines_content'] < 35)).astype(int)
        df['is_fine_grained'] = (df['fines_content'] >= 35).astype(int)

        df['depth_spt_interaction'] = df['depth_mid_m'] * df['spt_n_value']
        df['csr_depth_interaction'] = df['csr'] * df['depth_mid_m']

        self.processed_data = df
        print(f"  [OK] Created {len(df.columns)} total features")
        return True

    def export_csv(self) -> bool:
        """Export clean CSV ready for database ingestion"""
        print("\n" + "="*80)
        print("STEP 6: EXPORTING CLEAN CSV")
        print("="*80)

        df = self.processed_data.copy()

        output_columns = [
            'borehole_id', 'municipality', 'Depth_Layer',
            'latitude', 'longitude',
            'depth_from_m', 'depth_to_m', 'depth_mid_m',
            'spt_n_value', 'spt_n160',
            'uscs_symbol', 'soil_type', 'soil_description',
            'unit_weight', 'fines_content', 'groundwater_depth_m',
            'friction_angle', 'moisture_content', 'plasticity_index', 'mean_particle_size_d50',
            'relative_density_percent', 'elastic_modulus_es',
            'pga_g', 'csr', 'cyclic_strength_ratio',
            'effective_overburden_pressure', 'total_overburden_pressure',
            'factor_of_safety', 'liquefaction_probability', 'liquefaction',
            'liquefaction_risk_level', 'liquefaction_status',
        ]

        available_cols = [col for col in output_columns if col in df.columns]
        df_export = df[available_cols].copy()

        numeric_cols = df_export.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_export[col] = df_export[col].fillna(0)

        if not self.client:
            print("  [WARNING] No database connection - cannot upload to storage")
            return False

        try:
            bucket_name = os.getenv(
                'SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            storage_path = "cleaned/Cleaned_data.csv"

            try:
                existing_file = self.client.storage.from_(
                    bucket_name).download(storage_path)
                if existing_file:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    old_path_with_timestamp = f"old_cleaned_data/Cleaned_data_{timestamp}.csv"
                    self.client.storage.from_(bucket_name).upload(
                        old_path_with_timestamp,
                        existing_file,
                        file_options={
                            'content-type': 'text/csv', 'upsert': 'true'}
                    )
                    self.client.storage.from_(
                        bucket_name).remove([storage_path])
                    print(
                        f"  [OK] Archived old file to: {old_path_with_timestamp}")
            except:
                pass

            csv_bytes = df_export.to_csv(index=False).encode('utf-8')
            self.client.storage.from_(bucket_name).upload(
                storage_path,
                csv_bytes,
                file_options={'content-type': 'text/csv', 'upsert': 'true'}
            )

            print(f"  [OK] Uploaded to Supabase Storage: {storage_path}")
            print(f"  Records: {len(df_export)}")
            print(f"  Columns: {len(df_export.columns)}")
        except Exception as e:
            print(f"  [ERROR] Failed to upload to storage: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    def connect_database(self) -> bool:
        """Connect to Supabase/PostgreSQL database"""
        if not SUPABASE_AVAILABLE:
            return False

        print("\n" + "="*80)
        print("CONNECTING TO POSTGIS DATABASE")
        print("="*80)

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv(
            'SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY')

        if not supabase_url or not supabase_key:
            print("  [WARNING] Database environment variables not set")
            print("\n  To enable database storage, set environment variables:")
            print("\n  PowerShell:")
            print("    $env:SUPABASE_URL='your_supabase_url'")
            print("    $env:SUPABASE_SERVICE_ROLE_KEY='your_service_role_key'")
            print("\n  Bash/Linux:")
            print("    export SUPABASE_URL='your_supabase_url'")
            print("    export SUPABASE_SERVICE_ROLE_KEY='your_service_role_key'")
            print("\n  Database storage will be skipped")
            return False

        try:
            self.client = create_client(supabase_url, supabase_key)
            self.client.table('municipalities').select('id').limit(1).execute()
            print("  [OK] Connected to PostGIS database")
            return True
        except Exception as e:
            print(f"  [WARNING] Database connection failed: {e}")
            print("  Database storage will be skipped")
            return False

    def safe_float(self, value, default=None):
        """Safely convert to float"""
        if pd.isna(value) or value == '' or value is None:
            return default
        try:
            return float(value)
        except:
            return default

    def safe_str(self, value, default=None):
        """Safely convert to string"""
        if pd.isna(value) or value == '' or value is None:
            return default
        return str(value).strip()

    def upsert_municipalities(self) -> bool:
        """Upsert municipalities following schema"""
        print("\n  Step 7.1: Upserting municipalities...")

        df = self.processed_data
        muni_col = df.get('Municipality', df.get('municipality', pd.Series()))

        if len(muni_col) == 0:
            print("    [ERROR] No municipality column found")
            return False

        for muni_name in muni_col.unique():
            if pd.isna(muni_name):
                continue

            muni_name = str(muni_name).strip()
            try:
                result = self.client.table('municipalities').select(
                    'id, name').eq('name', muni_name).execute()

                if result.data and len(result.data) > 0:
                    self.municipality_ids[muni_name] = result.data[0]['id']
                else:
                    insert_data = {
                        'name': muni_name,
                        'description': f'Municipality in Tarlac Province'
                    }
                    result = self.client.table(
                        'municipalities').insert(insert_data).execute()
                    self.municipality_ids[muni_name] = result.data[0]['id']
                    print(
                        f"    Created: {muni_name} (ID: {self.municipality_ids[muni_name]})")
            except Exception as e:
                print(f"    [ERROR] Failed for {muni_name}: {e}")

        print(
            f"    [OK] Processed {len(self.municipality_ids)} municipalities")
        return True

    def upsert_boreholes(self) -> bool:
        """Upsert boreholes following schema"""
        print("\n  Step 7.2: Upserting boreholes...")

        df = self.processed_data
        borehole_col = df.get(
            'Borehole ID', df.get('borehole_id', pd.Series()))

        if len(borehole_col) == 0:
            print("    [ERROR] No borehole ID column found")
            return False

        borehole_groups = df.groupby(borehole_col.name).first()

        skipped_count = 0
        for borehole_id_str, row in borehole_groups.iterrows():
            borehole_id_str = str(borehole_id_str).strip()
            muni_name = self.safe_str(
                row.get('Municipality', row.get('municipality')))

            if not muni_name or muni_name not in self.municipality_ids:
                skipped_count += 1
                if skipped_count <= 3:
                    print(
                        f"    [SKIP] Borehole {borehole_id_str}: Municipality '{muni_name}' not found")
                continue

            municipality_id = self.municipality_ids[muni_name]

            lat = self.safe_float(row.get('Latitude', row.get('latitude')))
            lon = self.safe_float(row.get('Longitude', row.get('longitude')))

            if not lat or not lon:
                skipped_count += 1
                if skipped_count <= 3:
                    print(
                        f"    [SKIP] Borehole {borehole_id_str}: Missing coordinates (lat={lat}, lon={lon})")
                continue

            try:
                result = self.client.table('boreholes').select(
                    'id').eq('borehole_id', borehole_id_str).execute()

                if result.data and len(result.data) > 0:
                    self.borehole_record_ids[borehole_id_str] = result.data[0]['id']
                    print(
                        f"    Found existing: {borehole_id_str} (ID: {result.data[0]['id']})")
                else:
                    insert_data = {
                        'borehole_id': borehole_id_str,
                        'latitude': lat,
                        'longitude': lon,
                        'elevation': self.safe_float(row.get('Elevation', row.get('elevation'))),
                        'depth_total_m': 15.0,
                        'remarks': f'Data from {muni_name}',
                        'municipality_id': municipality_id
                    }
                    result = self.client.table(
                        'boreholes').insert(insert_data).execute()
                    if result.data and len(result.data) > 0:
                        self.borehole_record_ids[borehole_id_str] = result.data[0]['id']
                        print(
                            f"    Created: {borehole_id_str} (ID: {result.data[0]['id']})")
                    else:
                        print(
                            f"    [WARNING] Insert returned no data for {borehole_id_str}")
            except Exception as e:
                print(f"    [ERROR] Failed for {borehole_id_str}: {e}")
                import traceback
                traceback.print_exc()

        if skipped_count > 0:
            print(
                f"    [INFO] Skipped {skipped_count} boreholes (missing municipality or coordinates)")

        if len(self.borehole_record_ids) == 0:
            print(f"    [ERROR] No boreholes were inserted!")
            print(
                f"    Municipalities available: {list(self.municipality_ids.keys())}")
            return False

        print(f"    [OK] Processed {len(self.borehole_record_ids)} boreholes")
        return True

    def store_soil_layers(self) -> bool:
        """Store soil layers to database following schema"""
        print("\n  Step 7.3: Storing soil layers...")

        df = self.processed_data.copy()

        depth_layer_map = {}
        unique_layers = df['Depth_Layer'].unique()
        for i, layer_name in enumerate(sorted(unique_layers), 1):
            depth_layer_map[layer_name] = i

        records = []
        skipped_count = 0
        for idx, row in df.iterrows():
            borehole_id_str = str(
                row.get('Borehole ID', row.get('borehole_id', ''))).strip()
            borehole_record_id = self.borehole_record_ids.get(borehole_id_str)

            if not borehole_record_id:
                skipped_count += 1
                if skipped_count <= 3:
                    print(
                        f"    [SKIP] Soil layer: Borehole '{borehole_id_str}' not found in boreholes table")
                continue

            layer_number = depth_layer_map.get(row['Depth_Layer'], 1)
            depth_range = f"{row['depth_from_m']:.1f}-{row['depth_to_m']:.1f}m"

            record = {
                'borehole_id': borehole_record_id,
                'layer_number': int(layer_number),
                'depth_from_m': self.safe_float(row['depth_from_m']),
                'depth_to_m': self.safe_float(row['depth_to_m']),
                'depth_range': depth_range,
                'spt_n_value': self.safe_float(row['spt_n_value']),
                'spt_n160': self.safe_float(row.get('spt_n160')),
                'soil_type': self.safe_str(row.get('soil_type')) if row.get('soil_type') not in (None, 'nan', '') else None,
                'uscs_symbol': self.safe_str(row.get('uscs_symbol')) if row.get('uscs_symbol') not in (None, 'nan', '') else None,
                'soil_description': self.safe_str(row.get('soil_description')) if row.get('soil_description') not in (None, 'nan', '') else None,
                'unit_weight': self.safe_float(row['unit_weight']),
                'fines_content': self.safe_float(row['fines_content']),
                'groundwater_depth_m': self.safe_float(row['groundwater_depth_m']),
                'pga_g': self.safe_float(row['pga_g']),
                'csr': self.safe_float(row['csr']),
                'cyclic_strength_ratio': self.safe_float(row.get('crr')),
                'liquefaction': bool(row.get('liquefaction', 0)) if pd.notna(row.get('liquefaction')) else False,
                'liquefaction_risk_level': self.safe_str(row.get('liquefaction_risk_level')),
                'effective_overburden_pressure': self.safe_float(row['effective_overburden_pressure']),
                'total_overburden_pressure': self.safe_float(row['total_overburden_pressure']),
                'relative_density_percent': self.safe_float(row.get('relative_density_percent')),
                'magnitude_mw': self.safe_float(row.get('magnitude_mw')),
                'msf': self.safe_float(row.get('msf')),
                'n1_60cs': self.safe_float(row.get('n1_60cs')),
                'q_actual_kpa': self.safe_float(row.get('q_actual_kpa')),
                'foundation_kd': self.safe_float(row.get('foundation_kd')),
                'bearing_qu_kpa': self.safe_float(row.get('bearing_qu_kpa')),
                'bearing_qa_kpa': self.safe_float(row.get('bearing_qa_kpa')),
                'settlement_mm': self.safe_float(row.get('settlement_mm')),
                'lpi_weighing_factor': self.safe_float(row.get('lpi_weighing_factor')),
                'lpi_severity_factor': self.safe_float(row.get('lpi_severity_factor')),
            }

            if 'moisture_content' in row.index and pd.notna(row.get('moisture_content')):
                record['moisture_content'] = self.safe_float(
                    row['moisture_content'])

            if 'friction_angle' in row.index and pd.notna(row.get('friction_angle')):
                record['friction_angle'] = self.safe_float(
                    row['friction_angle'])

            if 'cohesion_kpa' in row.index and pd.notna(row.get('cohesion_kpa')):
                record['cohesion_kpa'] = self.safe_float(row['cohesion_kpa'])

            if 'plasticity_index' in row.index and pd.notna(row.get('plasticity_index')):
                record['plasticity_index'] = self.safe_float(
                    row['plasticity_index'])

            if 'mean_particle_size_d50' in row.index and pd.notna(row.get('mean_particle_size_d50')):
                record['mean_particle_size_d50'] = self.safe_float(
                    row['mean_particle_size_d50'])

            if 'Elastic Modulus (Es) (MN/m²)' in row and pd.notna(row['Elastic Modulus (Es) (MN/m²)']):
                record['elastic_modulus_es'] = self.safe_float(
                    row['Elastic Modulus (Es) (MN/m²)'])

            records.append(record)

        THESIS_COLUMNS = {
            'magnitude_mw', 'msf', 'n1_60cs', 'q_actual_kpa', 'foundation_kd',
            'bearing_qu_kpa', 'bearing_qa_kpa', 'settlement_mm',
            'lpi_weighing_factor', 'lpi_severity_factor',
        }

        def strip_thesis_columns(batch):
            return [{k: v for k, v in r.items() if k not in THESIS_COLUMNS} for r in batch]

        batch_size = 25
        total_inserted = 0
        schema_fallback = False

        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            batch_num = i // batch_size + 1
            try:
                insert_batch = strip_thesis_columns(
                    batch) if schema_fallback else batch
                self.client.table('soil_layers').insert(insert_batch).execute()
                total_inserted += len(batch)
                if not schema_fallback:
                    print(
                        f"    Inserted batch {batch_num}: {len(batch)} records")
                else:
                    print(
                        f"    Inserted batch {batch_num}: {len(batch)} records (without thesis cols — run SQL migration)")
            except Exception as e:
                err_str = str(e)
                if 'PGRST204' in err_str or 'schema cache' in err_str:
                    if not schema_fallback:
                        print(
                            f"    [WARNING] Thesis columns missing in DB schema — falling back to base columns.")
                        print(
                            f"    [INFO] Run the SQL migration in Supabase to persist thesis analysis fields.")
                        schema_fallback = True
                    try:
                        self.client.table('soil_layers').insert(
                            strip_thesis_columns(batch)).execute()
                        total_inserted += len(batch)
                        print(
                            f"    Inserted batch {batch_num}: {len(batch)} records (fallback)")
                    except Exception as e2:
                        print(
                            f"    [WARNING] Batch {batch_num} failed even on fallback: {e2}")
                else:
                    print(f"    [WARNING] Batch {batch_num} failed: {e}")

        if schema_fallback:
            print("\n" + "="*60)
            print("  ACTION REQUIRED — Run this SQL in Supabase SQL Editor:")
            print("  ALTER TABLE soil_layers")
            print("    ADD COLUMN IF NOT EXISTS magnitude_mw        FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS msf                 FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS n1_60cs             FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS q_actual_kpa        FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS foundation_kd       FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS bearing_qu_kpa      FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS bearing_qa_kpa      FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS settlement_mm       FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS lpi_weighing_factor FLOAT,")
            print("    ADD COLUMN IF NOT EXISTS lpi_severity_factor FLOAT;")
            print("="*60)

        if skipped_count > 0:
            print(
                f"    [INFO] Skipped {skipped_count} soil layer records (borehole not found)")

        if total_inserted == 0:
            print(f"    [WARNING] No soil layers were inserted!")
            print(f"    Total records prepared: {len(records)}")
            print(f"    Boreholes available: {len(self.borehole_record_ids)}")
            return False

        print(f"    [OK] Inserted {total_inserted} soil layer records")
        return True

    def store_to_postgis(self) -> bool:
        """Store processed data to PostGIS database following schema hierarchy"""
        if not self.client:
            return False

        print("\n" + "="*80)
        print("STEP 7: STORING TO POSTGIS DATABASE")
        print("="*80)
        print("  Following schema hierarchy: municipalities → boreholes → soil_layers")

        try:
            if not self.upsert_municipalities():
                return False

            if not self.upsert_boreholes():
                return False

            if not self.store_soil_layers():
                return False

            print(f"\n  [OK] Database storage completed")
            print(f"  Summary:")
            print(f"    Municipalities: {len(self.municipality_ids)}")
            print(f"    Boreholes: {len(self.borehole_record_ids)}")
            print(f"    Soil Layers: {len(self.processed_data)} records")
            print(f"\n  ✓ Spatial indexing enabled (PostGIS)")

            return True
        except Exception as e:
            print(f"  [ERROR] Database storage failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_validation_report(self):
        """Print validation report"""
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)

        if self.validation_errors:
            print("\n  [ERRORS] Critical issues found:")
            for error in self.validation_errors:
                print(f"    ✗ {error}")
        else:
            print("\n  ✓ No critical errors")

        if self.validation_warnings:
            print("\n  [WARNINGS] Non-critical issues:")
            for warning in self.validation_warnings:
                print(f"    ⚠ {warning}")
        else:
            print("\n  ✓ No warnings")

    def run(self) -> bool:
        """Run complete pipeline"""
        print("\n" + "="*80)
        print("GEOTECHNICAL DATA PROCESSING PIPELINE")
        print("="*80)

        if self.excel_path:
            print(f"Input file: {self.excel_path}")
        else:
            print(f"Input: Supabase Storage (raw/Raw_data.xlsx)")

        print(f"Output: Supabase Storage (cleaned/ folder)")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if not self.client:
            self.connect_database()

        if not self.excel_path and self.use_storage:
            if not self.download_from_supabase_storage():
                print("\n[ERROR] Failed to download from Supabase Storage")
                return False

        self.upload_raw_file_to_bucket()

        steps = [
            ("Load Excel", self.load_excel),
            ("Process and Validate", self.process_and_validate),
            ("Calculate CSR/CRR + Magnitude", self.calculate_csr_crr),
            ("Bearing Capacity & Settlement (Bowles 1988)",
             self.calculate_bearing_bowles),
            ("Classify Liquefaction (DPWH BSDS)",
             self.classify_liquefaction_dpwh_bsds),
            ("Engineer Features", self.engineer_features),
            ("Export CSV", self.export_csv),
        ]

        for step_name, step_func in steps:
            if not step_func():
                print(f"\n[ERROR] Pipeline failed at: {step_name}")
                return False

        self.print_validation_report()

        db_connected = self.connect_database()
        if db_connected:
            if self.store_to_postgis():
                print(f"\n  ✓ Data stored to PostGIS database with spatial indexing")
            else:
                print(f"\n  ⚠ Database storage failed, but CSV export succeeded")

        print("\n" + "="*80)
        print("[SUCCESS] PIPELINE COMPLETED")
        print("="*80)
        print(f"\n  Processed: {len(self.processed_data)} records")
        print(f"  Features: {len(self.processed_data.columns)}")
        if db_connected and self.client:
            print(f"  Database: Stored to PostGIS")
            print(f"  CSV: Uploaded to Supabase Storage")
        print(f"\n  ✓ Data ready for use")

        return True


def main():
    """Main execution"""
    import sys

    if len(sys.argv) < 2:
        excel_file = None
        output_csv = None
        print(
            f"[INFO] No file specified, will download from Supabase Storage (raw/Raw_data.xlsx)")
    else:
        excel_file = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if supabase_url and supabase_key:
        print(f"\n[INFO] Environment variables loaded from .env file")
        print(f"  SUPABASE_URL: {supabase_url[:40]}...")
        print(f"  Database storage enabled")
    else:
        print(f"\n[WARNING] Environment variables not found in .env file")
        print(f"  Pipeline will run but skip database storage")
        print(f"  CSV export will still work")
        print(f"\n  Create .env file with:")
        print(f"    SUPABASE_URL=your_url")
        print(f"    SUPABASE_SERVICE_ROLE_KEY=your_key")

    pipeline = GeotechnicalPipeline(excel_file, output_csv)
    success = pipeline.run()

    if not success:
        sys.exit(1)

    print("\n✓ Pipeline completed successfully!")
    if pipeline.client:
        print("✓ Data stored in PostGIS database with spatial indexing enabled")


if __name__ == "__main__":
    main()
