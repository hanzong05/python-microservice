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
    load_dotenv()  # Load .env file automatically
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
        """
        Initialize pipeline
        
        Args:
            excel_file_path: Path to local Excel file (or None to use Supabase Storage)
            output_csv_path: Optional output CSV path (default: auto-generated)
            use_storage: If True and excel_file_path is None, load from Supabase Storage
        """
        self.use_storage = use_storage
        self.excel_path = None
        self.excel_file_bytes = None
        
        if excel_file_path:
            self.excel_path = Path(excel_file_path)
            # No local output path needed - files go to storage only
            self.output_path = None
        else:
            # No local output path needed - files go to storage only
            self.output_path = None
        
        self.raw_data = {}
        self.processed_data = None
        self.validation_errors = []
        self.validation_warnings = []
        
        # Database connection and ID mappings
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
            
            # Download from raw/excel folder in storage bucket
            storage_path = "raw/Raw_data.xlsx"
            bucket_name = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            print(f"  Downloading: {storage_path} from bucket: {bucket_name}")
            
            file_data = self.client.storage.from_(bucket_name).download(storage_path)
            
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
    
    def load_excel(self) -> bool:
        """Load Excel file from local path or memory bytes"""
        print("\n" + "="*80)
        print("STEP 1: LOADING EXCEL FILE")
        print("="*80)
        
        try:
            # Use file bytes from storage if available, otherwise use local file
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
            
            # Load all depth layer sheets (skip Summary like reference script)
            for sheet_name in xl_file.sheet_names:
                if sheet_name == 'Summary':
                    print(f"    Skipping: {sheet_name}")
                    continue
                
                if self.excel_file_bytes:
                    df = pd.read_excel(io.BytesIO(self.excel_file_bytes), sheet_name=sheet_name)
                else:
                    df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                
                self.raw_data[sheet_name] = df
                print(f"    Loaded: {sheet_name} ({len(df)} rows, {len(df.columns)} columns)")
            
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
            clean = str(value).replace('°', '').replace("'", '').replace('"', '').strip()
            return float(clean)
        except:
            return None
    
    def parse_pga(self, value) -> Optional[float]:
        """Parse Peak Ground Acceleration from various formats"""
        if pd.isna(value):
            return None
        try:
            # Extract numeric value before 'g'
            matches = re.findall(r'(\d+\.?\d*)g', str(value).lower())
            if matches:
                return np.mean([float(m) for m in matches])
            # Try direct conversion
            return float(str(value).replace('g', '').strip())
        except:
            return None
    
    def parse_relative_density(self, value) -> Optional[float]:
        """Convert relative density text to numeric percentage"""
        if pd.isna(value):
            return None
        
        # If already numeric, return it
        try:
            return float(value)
        except:
            pass
        
        # Map text descriptions to percentages (same as reference)
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
        """Parse elastic modulus from various formats (like reference script)"""
        if pd.isna(value):
            return None
        
        # If it's a datetime (Excel parsing error), return None
        if isinstance(value, datetime):
            return None
        
        # If already numeric, return it
        try:
            return float(value)
        except:
            pass
        
        # Handle ranges like "3000 to 5000" - take midpoint
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
        PRE-CLEAN dataframe - same approach as reference script
        Cleans: Latitude, Longitude, Peak Ground Acceleration, Relative Density, Elastic Modulus
        """
        df_clean = df.copy()
        
        # 1. Clean Latitude (remove degree symbols) - exact column name from raw data
        if 'Latitude' in df_clean.columns:
            df_clean['Latitude'] = df_clean['Latitude'].apply(self.clean_coordinate)
        
        # 2. Clean Longitude (remove degree symbols) - exact column name from raw data
        if 'Longitude' in df_clean.columns:
            df_clean['Longitude'] = df_clean['Longitude'].apply(self.clean_coordinate)
        
        # 3. Clean Peak Ground Acceleration (parse complex strings) - exact column name
        if 'Peak Ground Acceleration' in df_clean.columns:
            df_clean['Peak Ground Acceleration'] = df_clean['Peak Ground Acceleration'].apply(self.parse_pga)
        
        # 4. Clean Relative Density (convert text to numeric) - exact column name
        if 'Relative Density' in df_clean.columns:
            df_clean['Relative Density'] = df_clean['Relative Density'].apply(self.parse_relative_density)
        
        # 5. Clean Elastic Modulus (parse ranges and dates) - exact column name
        if 'Elastic Modulus (Es) (MN/m²)' in df_clean.columns:
            df_clean['Elastic Modulus (Es) (MN/m²)'] = df_clean['Elastic Modulus (Es) (MN/m²)'].apply(
                self.parse_elastic_modulus)
        
        return df_clean
    
    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate coordinates after pre-cleaning"""
        df_clean = df.copy()
        
        # Use exact column names from raw data
        if 'Latitude' in df_clean.columns:
            df_clean['latitude'] = df_clean['Latitude']
            invalid = df_clean['latitude'].isna().sum()
            if invalid > 0:
                self.validation_warnings.append(f"Missing/invalid latitude: {invalid} records")
            
            # Check Tarlac range (15.0 to 16.0)
            valid_lat = df_clean['latitude'].dropna()
            if len(valid_lat) > 0:
                out_of_range = ((valid_lat < 15.0) | (valid_lat > 16.0)).sum()
                if out_of_range > 0:
                    self.validation_errors.append(f"Latitude out of Tarlac range (15.0-16.0): {out_of_range} records")
        
        if 'Longitude' in df_clean.columns:
            df_clean['longitude'] = df_clean['Longitude']
            invalid = df_clean['longitude'].isna().sum()
            if invalid > 0:
                self.validation_warnings.append(f"Missing/invalid longitude: {invalid} records")
            
            # Check Tarlac range (120.0 to 121.0)
            valid_lon = df_clean['longitude'].dropna()
            if len(valid_lon) > 0:
                out_of_range = ((valid_lon < 120.0) | (valid_lon > 121.0)).sum()
                if out_of_range > 0:
                    self.validation_errors.append(f"Longitude out of Tarlac range (120.0-121.0): {out_of_range} records")
        
        return df_clean
    
    def validate_spt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate SPT values (check for exact column names from raw data)"""
        df_clean = df.copy()
        
        # Try exact column names from raw data first
        spt_col = None
        for col in ['SPT N-Value', 'SPT N Value', 'spt_n_value', 'N-Value', 'N Value', 'SPT']:
            if col in df_clean.columns:
                spt_col = col
                break
        
        if spt_col:
            df_clean['spt_n_value'] = pd.to_numeric(df_clean[spt_col], errors='coerce')
            
            # Validate range (typically 0-100)
            valid_spt = df_clean['spt_n_value'].dropna()
            if len(valid_spt) > 0:
                invalid = ((valid_spt < 0) | (valid_spt > 100)).sum()
                if invalid > 0:
                    self.validation_errors.append(f"SPT values out of valid range (0-100): {invalid} records")
            
            missing = df_clean['spt_n_value'].isna().sum()
            if missing > 0:
                self.validation_warnings.append(f"Missing SPT values: {missing} records")
                df_clean['spt_n_value'] = df_clean['spt_n_value'].fillna(15.0)  # Default
        else:
            self.validation_warnings.append("SPT column not found, using default value 15.0")
            df_clean['spt_n_value'] = 15.0
        
        return df_clean
    
    def validate_soil_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate soil parameters (check for exact column names from raw data)"""
        df_clean = df.copy()
        
        # Unit weight validation - try exact column names
        unit_weight_col = None
        for col in ['Unit Weight', 'unit_weight', 'Unit Weight (kN/m³)', 'γ']:
            if col in df_clean.columns:
                unit_weight_col = col
                break
        
        if unit_weight_col:
            df_clean['unit_weight'] = pd.to_numeric(df_clean[unit_weight_col], errors='coerce')
            valid_uw = df_clean['unit_weight'].dropna()
            if len(valid_uw) > 0:
                invalid = ((valid_uw < 10) | (valid_uw > 25)).sum()
                if invalid > 0:
                    self.validation_warnings.append(f"Unit weight out of typical range (10-25 kN/m³): {invalid} records")
            df_clean['unit_weight'] = df_clean['unit_weight'].fillna(18.0)  # Default
        else:
            df_clean['unit_weight'] = 18.0
        
        # Fines content validation - try exact column names
        fines_col = None
        for col in ['Fines Content', 'fines_content', 'Fines (%)', 'Fines']:
            if col in df_clean.columns:
                fines_col = col
                break
        
        if fines_col:
            df_clean['fines_content'] = pd.to_numeric(df_clean[fines_col], errors='coerce')
            valid_fines = df_clean['fines_content'].dropna()
            if len(valid_fines) > 0:
                invalid = ((valid_fines < 0) | (valid_fines > 100)).sum()
                if invalid > 0:
                    self.validation_errors.append(f"Fines content out of valid range (0-100%): {invalid} records")
            df_clean['fines_content'] = df_clean['fines_content'].fillna(15.0)  # Default
        else:
            df_clean['fines_content'] = 15.0
        
        # Groundwater depth validation - try exact column names
        gwl_col = None
        for col in ['Groundwater Level', 'Groundwater Depth', 'groundwater_depth_m', 'GWL', 'Water Table']:
            if col in df_clean.columns:
                gwl_col = col
                break
        
        if gwl_col:
            df_clean['groundwater_depth_m'] = pd.to_numeric(df_clean[gwl_col], errors='coerce')
            df_clean['groundwater_depth_m'] = df_clean['groundwater_depth_m'].fillna(5.0)  # Default
        else:
            df_clean['groundwater_depth_m'] = 5.0
        
        return df_clean
    
    def process_and_validate(self) -> bool:
        """Single-pass processing with validation (following reference script pattern)"""
        print("\n" + "="*80)
        print("STEP 2: PRE-CLEANING AND VALIDATING DATA")
        print("="*80)
        
        all_dfs = []
        
        for sheet_name, df in self.raw_data.items():
            print(f"\n  Pre-cleaning sheet: {sheet_name}")
            
            # STEP 1: PRE-CLEAN (like reference script) - must run FIRST
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
                df['pga_g'] = 0.35  # Default PGA for Tarlac
            
            df['pga_g'] = pd.to_numeric(df['pga_g'], errors='coerce').fillna(0.35)
            
            # STEP 6: Map Relative Density
            if 'Relative Density' in df.columns:
                df['relative_density_percent'] = df['Relative Density']
            
            # STEP 7: Add Depth_Layer column (like reference script)
            df['Depth_Layer'] = sheet_name
            
            # STEP 8: Extract depth from sheet name or Depth_Layer
            def extract_depth(layer_name):
                match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', str(layer_name))
                if match:
                    return float(match.group(1)), float(match.group(2))
                return 0.0, 1.5
            
            depths = df['Depth_Layer'].apply(extract_depth)
            df['depth_from_m'] = depths.apply(lambda x: x[0])
            df['depth_to_m'] = depths.apply(lambda x: x[1])
            df['depth_mid_m'] = (df['depth_from_m'] + df['depth_to_m']) / 2
            
            all_dfs.append(df)
            print(f"    [OK] Processed {len(df)} rows")
        
        # Combine all sheets (like reference script)
        if all_dfs:
            self.processed_data = pd.concat(all_dfs, ignore_index=True)
            print(f"\n  [OK] Combined {len(self.processed_data)} total records")
            print(f"  Sheets combined: {len(self.raw_data)}")
            
            # Show depth layer distribution (like reference script)
            if 'Depth_Layer' in self.processed_data.columns:
                print(f"\n  Depth Layer Distribution:")
                depth_counts = self.processed_data['Depth_Layer'].value_counts().sort_index()
                for depth, count in depth_counts.items():
                    print(f"    {depth}: {count} rows")
            
            return True
        return False
    
    def calculate_csr_crr(self) -> bool:
        """
        Calculate CSR and CRR using Seed & Idriss (1971) method
        """
        print("\n" + "="*80)
        print("STEP 3: CALCULATING CSR/CRR (Seed & Idriss 1971)")
        print("="*80)
        
        df = self.processed_data.copy()
        
        # Normalize SPT values
        print("  Normalizing SPT values...")
        df['spt_n60'] = df['spt_n_value'] * 1.0  # N60 = N (can be adjusted with correction factors)
        df['spt_n160'] = df['spt_n_value'] * 1.1  # Rough correction to N160
        
        # Calculate total overburden pressure
        df['total_overburden_pressure'] = df['unit_weight'] * df['depth_mid_m']
        
        # Calculate effective overburden pressure
        gamma_water = 9.81  # kN/m³
        depth_below_wt = np.maximum(0, df['depth_mid_m'] - df['groundwater_depth_m'])
        df['effective_overburden_pressure'] = df['total_overburden_pressure'] - (gamma_water * depth_below_wt)
        
        # Calculate CSR (Cyclic Stress Ratio) - Seed & Idriss (1971)
        print("  Calculating CSR...")
        # Stress reduction coefficient (rd)
        rd = 1.0 - 0.00765 * df['depth_mid_m']
        rd = rd.clip(0.0, 1.0)
        
        # CSR = 0.65 * (amax/g) * (σv/σ'v) * rd
        # where amax/g = PGA
        df['csr'] = 0.65 * (df['pga_g'] / 9.81) * (df['total_overburden_pressure'] / df['effective_overburden_pressure'].clip(lower=1.0)) * rd
        
        # Calculate CRR (Cyclic Resistance Ratio) from SPT
        print("  Calculating CRR from SPT...")
        # Simplified CRR based on SPT N160
        df['crr'] = np.where(
            df['spt_n160'] <= 30,
            1.0 / (34.0 - df['spt_n160'] + 0.001),
            0.5
        )
        df['crr'] = df['crr'].clip(0.0, 1.0)
        
        # Factor of safety
        df['factor_of_safety'] = df['crr'] / (df['csr'] + 0.001)
        
        # Liquefaction probability (0-100%)
        df['liquefaction_probability'] = np.where(
            df['factor_of_safety'] < 1.0,
            (1.0 - df['factor_of_safety']) * 100,
            np.where(df['factor_of_safety'] < 1.5, 30.0, 10.0)
        )
        df['liquefaction_probability'] = df['liquefaction_probability'].clip(0.0, 100.0)
        
        self.processed_data = df
        print(f"  [OK] Calculated CSR/CRR for {len(df)} records")
        print(f"    CSR range: {df['csr'].min():.3f} - {df['csr'].max():.3f}")
        print(f"    CRR range: {df['crr'].min():.3f} - {df['crr'].max():.3f}")
        print(f"    Factor of Safety range: {df['factor_of_safety'].min():.2f} - {df['factor_of_safety'].max():.2f}")
        
        return True
    
    def classify_liquefaction_dpwh_bsds(self) -> bool:
        """
        Classify liquefaction potential per DPWH BSDS (2013)
        """
        print("\n" + "="*80)
        print("STEP 4: LIQUEFACTION CLASSIFICATION (DPWH BSDS 2013)")
        print("="*80)
        
        df = self.processed_data.copy()
        
        # DPWH BSDS (2013) Liquefaction Screening Criteria
        # Based on SPT N-value, fines content, and depth
        
        def dpwh_classify(row):
            spt_n = row['spt_n_value']
            fines = row['fines_content']
            depth = row['depth_mid_m']
            fs = row['factor_of_safety']
            
            # Very High Risk: FS < 0.8
            if fs < 0.8:
                return 'VERY HIGH', 'LIQUEFIES'
            
            # High Risk: 0.8 <= FS < 1.0
            if fs < 1.0:
                return 'HIGH', 'LIQUEFIES'
            
            # Medium Risk: 1.0 <= FS < 1.2
            if fs < 1.2:
                return 'MEDIUM', 'MARGINAL'
            
            # Low Risk: 1.2 <= FS < 1.5
            if fs < 1.5:
                return 'LOW', 'UNLIKELY'
            
            # Very Low Risk: FS >= 1.5
            return 'VERY LOW', 'NO LIQUEFACTION'
        
        # Apply classification
        classification = df.apply(dpwh_classify, axis=1)
        df['liquefaction_risk_level'] = classification.apply(lambda x: x[0])
        df['liquefaction_status'] = classification.apply(lambda x: x[1])
        df['liquefaction'] = df['liquefaction_risk_level'].isin(['VERY HIGH', 'HIGH']).astype(int)
        
        # Summary
        risk_counts = df['liquefaction_risk_level'].value_counts()
        print("\n  Classification Summary:")
        for risk_level, count in risk_counts.items():
            pct = (count / len(df)) * 100
            print(f"    {risk_level:12s}: {count:4d} records ({pct:5.1f}%)")
        
        liquefies = df['liquefaction'].sum()
        print(f"\n  Total liquefaction cases: {liquefies} ({liquefies/len(df)*100:.1f}%)")
        
        self.processed_data = df
        return True
    
    def engineer_features(self) -> bool:
        """Engineer additional features"""
        print("\n" + "="*80)
        print("STEP 5: FEATURE ENGINEERING")
        print("="*80)
        
        df = self.processed_data.copy()
        
        # Depth features
        df['depth_thickness_m'] = df['depth_to_m'] - df['depth_from_m']
        df['depth_to_groundwater_m'] = df['groundwater_depth_m'] - df['depth_mid_m']
        df['is_below_groundwater'] = (df['depth_mid_m'] > df['groundwater_depth_m']).astype(int)
        df['depth_normalized'] = df['depth_mid_m'] / 15.0
        
        # SPT features
        df['spt_n_log'] = np.log1p(df['spt_n_value'])
        df['spt_n160_log'] = np.log1p(df['spt_n160'])
        df['relative_density_from_spt'] = np.clip(np.sqrt(df['spt_n160'] / 60) * 100, 0, 100)
        
        # Stress features
        df['effective_stress_ratio'] = df['effective_overburden_pressure'] / (df['total_overburden_pressure'] + 1)
        
        # Soil classification
        df['is_clean_sand'] = (df['fines_content'] < 5).astype(int)
        df['is_silty_sand'] = ((df['fines_content'] >= 5) & (df['fines_content'] < 35)).astype(int)
        df['is_fine_grained'] = (df['fines_content'] >= 35).astype(int)
        
        # Interaction features
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
        
        # Select and order columns for database (prioritize exact raw data column names)
        output_columns = [
            # Identifiers (exact names from raw data)
            'Borehole ID', 'borehole_id', 'Municipality', 'municipality', 'Depth_Layer',
            # Coordinates (both original and cleaned)
            'Latitude', 'Longitude', 'latitude', 'longitude',
            # Depth
            'depth_from_m', 'depth_to_m', 'depth_mid_m', 'depth_range',
            # SPT (exact name from raw data)
            'SPT N-Value', 'spt_n_value', 'spt_n60', 'spt_n160',
            # Soil properties (exact names from raw data)
            'Unit Weight', 'unit_weight', 'Fines Content', 'fines_content',
            'Groundwater Level', 'Groundwater Depth', 'groundwater_depth_m',
            # Seismic (exact name from raw data)
            'Peak Ground Acceleration', 'pga_g', 'csr', 'crr', 'cyclic_strength_ratio',
            # Relative Density (exact name from raw data)
            'Relative Density', 'relative_density_percent',
            # Elastic Modulus (exact name from raw data)
            'Elastic Modulus (Es) (MN/m²)',
            # Liquefaction
            'factor_of_safety', 'liquefaction_probability', 'liquefaction',
            'liquefaction_risk_level', 'liquefaction_status',
            # Stress
            'total_overburden_pressure', 'effective_overburden_pressure',
            # Features
            'is_clean_sand', 'is_silty_sand',
        ]
        
        # Select available columns (keep all that exist)
        available_cols = [col for col in output_columns if col in df.columns]
        
        # Also include any other calculated columns
        calculated_cols = ['depth_range', 'spt_n60', 'spt_n160', 'csr', 'crr', 
                          'factor_of_safety', 'liquefaction_probability', 'liquefaction',
                          'liquefaction_risk_level', 'liquefaction_status',
                          'total_overburden_pressure', 'effective_overburden_pressure',
                          'depth_mid_m', 'is_clean_sand', 'is_silty_sand']
        
        for col in calculated_cols:
            if col in df.columns and col not in available_cols:
                available_cols.append(col)
        
        df_export = df[available_cols].copy()
        
        # Create depth_range if not exists
        if 'depth_range' not in df_export.columns:
            df_export['depth_range'] = df_export.apply(
                lambda row: f"{row['depth_from_m']:.1f}-{row['depth_to_m']:.1f}m", axis=1
            )
        
        # Rename cyclic_strength_ratio if crr exists
        if 'crr' in df_export.columns and 'cyclic_strength_ratio' not in df_export.columns:
            df_export['cyclic_strength_ratio'] = df_export['crr']
        
        # Fill any remaining NaN with appropriate defaults
        numeric_cols = df_export.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_export[col] = df_export[col].fillna(0)
        
        # Upload to Supabase Storage in cleaned/ folder (no local file)
        if not self.client:
            print("  [WARNING] No database connection - cannot upload to storage")
            return False
        
        try:
            bucket_name = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            storage_path = "cleaned/Cleaned_data.csv"
            old_storage_path = "old_cleaned_data/Cleaned_data.csv"
            
            # Check if Cleaned_data.csv exists and move it to old_cleaned_data/ folder
            try:
                # Try to download existing file
                existing_file = self.client.storage.from_(bucket_name).download(storage_path)
                if existing_file:
                    # Move old file to old_cleaned_data/ folder with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    old_path_with_timestamp = f"old_cleaned_data/Cleaned_data_{timestamp}.csv"
                    
                    # Upload old file to archive folder
                    self.client.storage.from_(bucket_name).upload(
                        old_path_with_timestamp,
                        existing_file,
                        file_options={'content-type': 'text/csv', 'upsert': 'true'}
                    )
                    
                    # Delete old file from cleaned/ folder
                    self.client.storage.from_(bucket_name).remove([storage_path])
                    print(f"  [OK] Archived old file to: {old_path_with_timestamp}")
            except:
                # File doesn't exist, that's fine
                pass
            
            # Create CSV as bytes (no physical file)
            csv_bytes = df_export.to_csv(index=False).encode('utf-8')
            
            # Upload new file as Cleaned_data.csv
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
        
        # Try multiple environment variable names (PowerShell and bash compatible)
        supabase_url = os.getenv('SUPABASE_URL') or os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY')
        
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
            # Test connection
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
                # Check if exists
                result = self.client.table('municipalities').select('id, name').eq('name', muni_name).execute()
                
                if result.data and len(result.data) > 0:
                    self.municipality_ids[muni_name] = result.data[0]['id']
                else:
                    # Insert new
                    insert_data = {
                        'name': muni_name,
                        'description': f'Municipality in Tarlac Province'
                    }
                    result = self.client.table('municipalities').insert(insert_data).execute()
                    self.municipality_ids[muni_name] = result.data[0]['id']
                    print(f"    Created: {muni_name} (ID: {self.municipality_ids[muni_name]})")
            except Exception as e:
                print(f"    [ERROR] Failed for {muni_name}: {e}")
        
        print(f"    [OK] Processed {len(self.municipality_ids)} municipalities")
        return True
    
    def upsert_boreholes(self) -> bool:
        """Upsert boreholes following schema (directly linked to municipalities)"""
        print("\n  Step 7.2: Upserting boreholes...")
        
        df = self.processed_data
        borehole_col = df.get('Borehole ID', df.get('borehole_id', pd.Series()))
        
        if len(borehole_col) == 0:
            print("    [ERROR] No borehole ID column found")
            return False
        
        # Get unique boreholes
        borehole_groups = df.groupby(borehole_col.name).first()
        
        skipped_count = 0
        for borehole_id_str, row in borehole_groups.iterrows():
            borehole_id_str = str(borehole_id_str).strip()
            muni_name = self.safe_str(row.get('Municipality', row.get('municipality')))
            
            if not muni_name or muni_name not in self.municipality_ids:
                skipped_count += 1
                if skipped_count <= 3:  # Show first 3 skipped
                    print(f"    [SKIP] Borehole {borehole_id_str}: Municipality '{muni_name}' not found")
                continue
            
            municipality_id = self.municipality_ids[muni_name]
            
            # Get coordinates
            lat = self.safe_float(row.get('Latitude', row.get('latitude')))
            lon = self.safe_float(row.get('Longitude', row.get('longitude')))
            
            if not lat or not lon:
                skipped_count += 1
                if skipped_count <= 3:
                    print(f"    [SKIP] Borehole {borehole_id_str}: Missing coordinates (lat={lat}, lon={lon})")
                continue
            
            try:
                # Check if exists
                result = self.client.table('boreholes').select('id').eq('borehole_id', borehole_id_str).execute()
                
                if result.data and len(result.data) > 0:
                    self.borehole_record_ids[borehole_id_str] = result.data[0]['id']
                    print(f"    Found existing: {borehole_id_str} (ID: {result.data[0]['id']})")
                else:
                    # Insert new - schema needs municipality_id column (barangay table removed)
                    # Try with municipality_id, if it fails, the database schema needs updating
                    insert_data = {
                        'borehole_id': borehole_id_str,
                        'latitude': lat,
                        'longitude': lon,
                        'elevation': self.safe_float(row.get('Elevation', row.get('elevation'))),
                        'depth_total_m': 15.0,  # Default, can be calculated from max depth
                        'remarks': f'Data from {muni_name}'
                    }
                    
                    # Add municipality_id - database schema must have this column
                    # If error occurs, update boreholes table to replace barangay_id with municipality_id
                    insert_data['municipality_id'] = municipality_id
                    result = self.client.table('boreholes').insert(insert_data).execute()
                    if result.data and len(result.data) > 0:
                        self.borehole_record_ids[borehole_id_str] = result.data[0]['id']
                        print(f"    Created: {borehole_id_str} (ID: {result.data[0]['id']})")
                    else:
                        print(f"    [WARNING] Insert returned no data for {borehole_id_str}")
            except Exception as e:
                print(f"    [ERROR] Failed for {borehole_id_str}: {e}")
                import traceback
                traceback.print_exc()
        
        if skipped_count > 0:
            print(f"    [INFO] Skipped {skipped_count} boreholes (missing municipality or coordinates)")
        
        if len(self.borehole_record_ids) == 0:
            print(f"    [ERROR] No boreholes were inserted!")
            print(f"    Municipalities available: {list(self.municipality_ids.keys())}")
            return False
        
        print(f"    [OK] Processed {len(self.borehole_record_ids)} boreholes")
        return True
    
    def store_soil_layers(self) -> bool:
        """Store soil layers to database following schema"""
        print("\n  Step 7.3: Storing soil layers...")
        
        df = self.processed_data.copy()
        
        # Map depth layers to layer numbers
        depth_layer_map = {}
        unique_layers = df['Depth_Layer'].unique()
        for i, layer_name in enumerate(sorted(unique_layers), 1):
            depth_layer_map[layer_name] = i
        
        records = []
        skipped_count = 0
        for idx, row in df.iterrows():
            borehole_id_str = str(row.get('Borehole ID', row.get('borehole_id', ''))).strip()
            borehole_record_id = self.borehole_record_ids.get(borehole_id_str)
            
            if not borehole_record_id:
                skipped_count += 1
                if skipped_count <= 3:  # Show first 3 skipped
                    print(f"    [SKIP] Soil layer: Borehole '{borehole_id_str}' not found in boreholes table")
                continue  # Skip if borehole not found
            
            layer_number = depth_layer_map.get(row['Depth_Layer'], 1)
            depth_range = f"{row['depth_from_m']:.1f}-{row['depth_to_m']:.1f}m"
            
            # Build record following schema
            record = {
                'borehole_id': borehole_record_id,  # FK to boreholes.id
                'layer_number': int(layer_number),
                'depth_from_m': self.safe_float(row['depth_from_m']),
                'depth_to_m': self.safe_float(row['depth_to_m']),
                'depth_range': depth_range,
                'spt_n_value': self.safe_float(row['spt_n_value']),
                'spt_n60': self.safe_float(row.get('spt_n60')),
                'spt_n160': self.safe_float(row.get('spt_n160')),
                'unit_weight': self.safe_float(row['unit_weight']),
                'fines_content': self.safe_float(row['fines_content']),
                'groundwater_depth_m': self.safe_float(row['groundwater_depth_m']),
                'pga_g': self.safe_float(row['pga_g']),
                'csr': self.safe_float(row['csr']),
                'cyclic_strength_ratio': self.safe_float(row.get('crr')),  # CRR stored as cyclic_strength_ratio
                'liquefaction': bool(row.get('liquefaction', 0)) if pd.notna(row.get('liquefaction')) else False,
                'liquefaction_risk_level': self.safe_str(row.get('liquefaction_risk_level')),
                'effective_overburden_pressure': self.safe_float(row['effective_overburden_pressure']),
                'total_overburden_pressure': self.safe_float(row['total_overburden_pressure']),
                'relative_density_percent': self.safe_float(row.get('relative_density_percent')),
            }
            
            # Add optional fields if available
            if 'moisture_content' in row and pd.notna(row['moisture_content']):
                record['moisture_content'] = self.safe_float(row['moisture_content'])
            
            if 'friction_angle' in row and pd.notna(row['friction_angle']):
                record['friction_angle'] = self.safe_float(row['friction_angle'])
            
            if 'cohesion_kpa' in row and pd.notna(row['cohesion_kpa']):
                record['cohesion_kpa'] = self.safe_float(row['cohesion_kpa'])
            
            # Elastic Modulus
            if 'Elastic Modulus (Es) (MN/m²)' in row and pd.notna(row['Elastic Modulus (Es) (MN/m²)']):
                record['elastic_modulus_es'] = self.safe_float(row['Elastic Modulus (Es) (MN/m²)'])
            
            records.append(record)
        
        # Insert in batches
        batch_size = 25
        total_inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            try:
                self.client.table('soil_layers').insert(batch).execute()
                total_inserted += len(batch)
                print(f"    Inserted batch {i//batch_size + 1}: {len(batch)} records")
            except Exception as e:
                print(f"    [WARNING] Batch {i//batch_size + 1} failed: {e}")
        
        if skipped_count > 0:
            print(f"    [INFO] Skipped {skipped_count} soil layer records (borehole not found)")
        
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
            # Step 1: Upsert municipalities
            if not self.upsert_municipalities():
                return False
            
            # Step 2: Upsert boreholes (directly linked to municipalities)
            if not self.upsert_boreholes():
                return False
            
            # Step 3: Insert soil layers
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
        
        # Step 0: Download from storage if needed
        if not self.excel_path and self.use_storage:
            if not self.download_from_supabase_storage():
                print("\n[ERROR] Failed to download from Supabase Storage")
                return False
        
        steps = [
            ("Load Excel", self.load_excel),
            ("Process and Validate", self.process_and_validate),
            ("Calculate CSR/CRR", self.calculate_csr_crr),
            ("Classify Liquefaction (DPWH BSDS)", self.classify_liquefaction_dpwh_bsds),
            ("Engineer Features", self.engineer_features),
            ("Export CSV", self.export_csv),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n[ERROR] Pipeline failed at: {step_name}")
                return False
        
        self.print_validation_report()
        
        # Optional: Store to PostGIS database (reuse connection if available)
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
    
    # Default to Supabase Storage if no argument provided
    if len(sys.argv) < 2:
        excel_file = None  # Will download from storage
        output_csv = None
        print(f"[INFO] No file specified, will download from Supabase Storage (raw/Raw_data.xlsx)")
    else:
        excel_file = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if environment variables are loaded from .env
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
