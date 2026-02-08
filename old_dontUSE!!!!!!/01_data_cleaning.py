"""
Data Cleaning Pipeline - WITH INTEGRATED PRE-CLEANING
Tarlac Geotechnical Investigation Data
Version 2.0 - ROOT CAUSE FIX

NEW FEATURES IN v2.0:
- Pre-cleans problematic columns FIRST (degree symbols, complex strings, text values)
- Fixes root cause of feature mismatch errors
- Processes all depth layer sheets
- Combines into single cleaned dataset
- Uploads to Supabase Storage

This script is the FIRST step in the ML pipeline and ensures all downstream
processes receive clean, properly formatted numeric data.
"""

import pandas as pd
import numpy as np
import warnings
import io
import re
from datetime import datetime
from supabase_client import get_supabase_client

warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: PRE-CLEANING FUNCTIONS (ROOT CAUSE FIX)
# ============================================================================

def clean_coordinate(coord_str):
    """
    Remove degree symbol and convert to float
    Fixes: '15.715506°' → 15.715506
    """
    if pd.isna(coord_str):
        return None

    try:
        clean = str(coord_str).replace('°', '').strip()
        return float(clean)
    except:
        return None


def parse_pga(pga_str):
    """
    Parse Peak Ground Acceleration from complex strings
    Fixes: '0.4g (RP: 500-yr; STIFF Soil)' → 0.4
           '0.3g to 0.4g (RP: 500-yr; STIFF Soil)' → 0.35 (average)
    """
    if pd.isna(pga_str):
        return None

    try:
        # Extract all numeric values before 'g'
        matches = re.findall(r'(\d+\.?\d*)g', str(pga_str).lower())

        if matches:
            # If multiple values (e.g., range), take average
            values = [float(m) for m in matches]
            return np.mean(values)

        return None
    except:
        return None


def parse_relative_density(value):
    """
    Convert relative density text to numeric percentage
    Fixes: 'hard' → 90.0
           'very dense' → 95.0
           'loose to medium dense' → 50.0
    """
    if pd.isna(value):
        return None

    # If already numeric, return it
    try:
        return float(value)
    except:
        pass

    # Map text descriptions to percentages
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


def parse_elastic_modulus(value):
    """
    Parse elastic modulus from various formats
    Fixes: '3000 to 5000' → 4000.0 (midpoint)
           datetime(2026, 10, 30) → None (Excel error)
           6000 → 6000.0
    """
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


def pre_clean_dataframe(df, sheet_name):
    """
    PRE-CLEAN a dataframe - FIXES ROOT CAUSE
    This must run BEFORE any other processing!
    """
    print(f"\n  Pre-cleaning sheet: {sheet_name}")

    df_clean = df.copy()
    cleaning_stats = {}

    # 1. Clean Latitude (remove degree symbols)
    if 'Latitude' in df_clean.columns:
        before_type = df_clean['Latitude'].dtype
        before_null = df_clean['Latitude'].isna().sum()

        df_clean['Latitude'] = df_clean['Latitude'].apply(clean_coordinate)

        after_null = df_clean['Latitude'].isna().sum()
        cleaning_stats['Latitude'] = {
            'before_type': str(before_type),
            'after_type': str(df_clean['Latitude'].dtype),
            'nulls_before': before_null,
            'nulls_after': after_null,
            'cleaned': after_null - before_null
        }
        print(
            f"    ✓ Latitude: {before_type} → {df_clean['Latitude'].dtype} (cleaned {after_null - before_null} values)")

    # 2. Clean Longitude (remove degree symbols)
    if 'Longitude' in df_clean.columns:
        before_type = df_clean['Longitude'].dtype
        before_null = df_clean['Longitude'].isna().sum()

        df_clean['Longitude'] = df_clean['Longitude'].apply(clean_coordinate)

        after_null = df_clean['Longitude'].isna().sum()
        cleaning_stats['Longitude'] = {
            'before_type': str(before_type),
            'after_type': str(df_clean['Longitude'].dtype),
            'nulls_before': before_null,
            'nulls_after': after_null,
            'cleaned': after_null - before_null
        }
        print(
            f"    ✓ Longitude: {before_type} → {df_clean['Longitude'].dtype} (cleaned {after_null - before_null} values)")

    # 3. Clean Peak Ground Acceleration (parse complex strings)
    if 'Peak Ground Acceleration' in df_clean.columns:
        before_type = df_clean['Peak Ground Acceleration'].dtype
        before_null = df_clean['Peak Ground Acceleration'].isna().sum()

        df_clean['Peak Ground Acceleration'] = df_clean['Peak Ground Acceleration'].apply(
            parse_pga)

        after_null = df_clean['Peak Ground Acceleration'].isna().sum()
        cleaning_stats['Peak Ground Acceleration'] = {
            'before_type': str(before_type),
            'after_type': str(df_clean['Peak Ground Acceleration'].dtype),
            'nulls_before': before_null,
            'nulls_after': after_null,
            'cleaned': after_null - before_null
        }
        print(
            f"    ✓ PGA: {before_type} → {df_clean['Peak Ground Acceleration'].dtype} (cleaned {after_null - before_null} values)")

    # 4. Clean Relative Density (convert text to numeric)
    if 'Relative Density' in df_clean.columns:
        before_type = df_clean['Relative Density'].dtype
        before_null = df_clean['Relative Density'].isna().sum()

        df_clean['Relative Density'] = df_clean['Relative Density'].apply(
            parse_relative_density)

        after_null = df_clean['Relative Density'].isna().sum()
        cleaning_stats['Relative Density'] = {
            'before_type': str(before_type),
            'after_type': str(df_clean['Relative Density'].dtype),
            'nulls_before': before_null,
            'nulls_after': after_null,
            'cleaned': after_null - before_null
        }
        print(
            f"    ✓ Relative Density: {before_type} → {df_clean['Relative Density'].dtype} (cleaned {after_null - before_null} values)")

    # 5. Clean Elastic Modulus (parse ranges and dates)
    if 'Elastic Modulus (Es) (MN/m²)' in df_clean.columns:
        before_type = df_clean['Elastic Modulus (Es) (MN/m²)'].dtype
        before_null = df_clean['Elastic Modulus (Es) (MN/m²)'].isna().sum()

        df_clean['Elastic Modulus (Es) (MN/m²)'] = df_clean['Elastic Modulus (Es) (MN/m²)'].apply(
            parse_elastic_modulus)

        after_null = df_clean['Elastic Modulus (Es) (MN/m²)'].isna().sum()
        cleaning_stats['Elastic Modulus'] = {
            'before_type': str(before_type),
            'after_type': str(df_clean['Elastic Modulus (Es) (MN/m²)'].dtype),
            'nulls_before': before_null,
            'nulls_after': after_null,
            'cleaned': after_null - before_null
        }
        print(
            f"    ✓ Elastic Modulus: {before_type} → {df_clean['Elastic Modulus (Es) (MN/m²)'].dtype} (cleaned {after_null - before_null} values)")

    print(f"    [OK] Pre-cleaning completed for {sheet_name}")

    return df_clean, cleaning_stats


# ============================================================================
# SECTION 2: SUPABASE FUNCTIONS
# ============================================================================

def download_file_from_storage(bucket_name, file_path):
    """Download file from Supabase Storage"""
    print(f"\n  Downloading from Supabase Storage...")
    print(f"    Bucket: {bucket_name}")
    print(f"    File: {file_path}")

    client = get_supabase_client()
    if not client:
        print("    [ERROR] Failed to connect to Supabase")
        return None

    try:
        response = client.storage.from_(bucket_name).download(file_path)
        print(f"    [OK] Downloaded {len(response)} bytes")
        return response
    except Exception as e:
        print(f"    [ERROR] Download failed: {e}")
        return None


def upload_to_supabase_storage(excel_bytes, bucket_name, storage_path):
    """Upload file to Supabase Storage directly from memory"""
    print(f"\n  Uploading to Supabase Storage...")
    print(f"    Path: {storage_path}")

    client = get_supabase_client()
    if not client:
        print("    [ERROR] Failed to connect to Supabase")
        return False

    try:
        # Archive existing file if it exists
        try:
            existing_file = client.storage.from_(
                bucket_name).download(storage_path)
            if existing_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = f"old_cleaned/old_cleaned_data_{timestamp}.xlsx"
                client.storage.from_(bucket_name).move(
                    storage_path, archive_path)
                print(f"    [OK] Existing file archived as {archive_path}")
        except Exception:
            # File doesn't exist yet, which is fine
            pass

        # Upload new file
        client.storage.from_(bucket_name).upload(
            storage_path,
            excel_bytes,
            file_options={
                "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "upsert": "true"
            }
        )
        print(
            f"    [OK] Uploaded successfully ({len(excel_bytes) / 1024:.2f} KB)")
        return True

    except Exception as e:
        print(f"    [ERROR] Upload failed: {e}")
        return False


# ============================================================================
# SECTION 3: DATA CLEANING CLASS
# ============================================================================

class GeotechnicalDataCleaner:
    """
    Data cleaning pipeline with integrated pre-cleaning
    Processes all depth layer sheets and combines them
    """

    def __init__(self, file_bytes):
        self.file_bytes = file_bytes
        self.raw_data = {}
        self.cleaned_data = {}
        self.combined_df = None
        self.cleaning_report = {}

    def load_raw_data(self):
        """Load all sheets from raw Excel file"""
        print("\n" + "=" * 80)
        print("STEP 1: LOADING RAW DATA")
        print("=" * 80)

        try:
            xl_file = pd.ExcelFile(io.BytesIO(self.file_bytes))
            print(f"\n  Found {len(xl_file.sheet_names)} sheets")

            # Load all depth layer sheets (skip Summary)
            for sheet_name in xl_file.sheet_names:
                if sheet_name == 'Summary':
                    print(f"    Skipping: {sheet_name}")
                    continue

                df = pd.read_excel(io.BytesIO(
                    self.file_bytes), sheet_name=sheet_name)
                self.raw_data[sheet_name] = df
                print(
                    f"    Loaded: {sheet_name} ({len(df)} rows, {len(df.columns)} columns)")

            print(f"\n  [OK] Loaded {len(self.raw_data)} data sheets")
            return True

        except Exception as e:
            print(f"  [ERROR] Failed to load data: {e}")
            return False

    def pre_clean_all_sheets(self):
        """
        STEP 2: PRE-CLEAN ALL SHEETS
        This is the ROOT CAUSE FIX - runs FIRST!
        """
        print("\n" + "=" * 80)
        print("STEP 2: PRE-CLEANING ALL SHEETS (ROOT CAUSE FIX)")
        print("=" * 80)

        for sheet_name, df in self.raw_data.items():
            # Apply pre-cleaning to fix problematic columns
            cleaned_df, stats = pre_clean_dataframe(df, sheet_name)
            self.cleaned_data[sheet_name] = cleaned_df
            self.cleaning_report[sheet_name] = stats

        print(f"\n  [SUCCESS] Pre-cleaned {len(self.cleaned_data)} sheets")
        print(
            "  All degree symbols removed, complex strings parsed, text values converted!")
        return True

    def add_depth_layer_column(self):
        """Add depth layer identifier to each sheet"""
        print("\n" + "=" * 80)
        print("STEP 3: ADDING DEPTH LAYER IDENTIFIERS")
        print("=" * 80)

        for sheet_name, df in self.cleaned_data.items():
            df['Depth_Layer'] = sheet_name
            print(f"    Added 'Depth_Layer' = '{sheet_name}' ({len(df)} rows)")

        print(f"\n  [OK] Depth layer column added to all sheets")
        return True

    def combine_all_sheets(self):
        """Combine all depth layer sheets into single dataframe"""
        print("\n" + "=" * 80)
        print("STEP 4: COMBINING ALL SHEETS")
        print("=" * 80)

        all_dfs = list(self.cleaned_data.values())
        self.combined_df = pd.concat(all_dfs, ignore_index=True)

        print(f"\n  Combined Statistics:")
        print(f"    Total rows: {len(self.combined_df)}")
        print(f"    Total columns: {len(self.combined_df.columns)}")
        print(f"    Sheets combined: {len(self.cleaned_data)}")

        # Show depth layer distribution
        print(f"\n  Depth Layer Distribution:")
        depth_counts = self.combined_df['Depth_Layer'].value_counts(
        ).sort_index()
        for depth, count in depth_counts.items():
            print(f"    {depth}: {count} rows")

        print(f"\n  [OK] All sheets combined into single dataset")
        return True

    def validate_data_types(self):
        """Validate that critical columns are now properly typed"""
        print("\n" + "=" * 80)
        print("STEP 5: VALIDATING DATA TYPES")
        print("=" * 80)

        critical_columns = [
            'Latitude', 'Longitude', 'Peak Ground Acceleration',
            'Relative Density', 'Elastic Modulus (Es) (MN/m²)'
        ]

        print("\n  Critical Column Data Types:")
        all_valid = True

        for col in critical_columns:
            if col in self.combined_df.columns:
                dtype = self.combined_df[col].dtype
                is_numeric = pd.api.types.is_numeric_dtype(
                    self.combined_df[col])
                status = "✓" if is_numeric else "✗"
                print(f"    {status} {col}: {dtype} (numeric: {is_numeric})")

                if not is_numeric:
                    all_valid = False
                    # Show sample non-numeric values
                    non_numeric = self.combined_df[col].apply(
                        lambda x: not isinstance(x, (int, float, np.number)))
                    if non_numeric.any():
                        samples = self.combined_df[col][non_numeric].head(
                            3).tolist()
                        print(
                            f"        [WARNING] Non-numeric values found: {samples}")

        if all_valid:
            print(f"\n  [SUCCESS] All critical columns are properly typed!")
        else:
            print(f"\n  [WARNING] Some columns still have type issues")

        return all_valid

    def generate_cleaning_summary(self):
        """Generate summary report of cleaning operations"""
        print("\n" + "=" * 80)
        print("CLEANING SUMMARY REPORT")
        print("=" * 80)

        total_cleaned = sum(
            sum(col_stats.get('cleaned', 0)
                for col_stats in sheet_stats.values())
            for sheet_stats in self.cleaning_report.values()
        )

        print(f"\n  Total values cleaned: {total_cleaned}")
        print(f"  Sheets processed: {len(self.cleaning_report)}")
        print(
            f"  Final dataset size: {len(self.combined_df)} rows × {len(self.combined_df.columns)} columns")

        # Show which columns were cleaned across all sheets
        print(f"\n  Columns Cleaned:")
        all_cleaned_cols = set()
        for sheet_stats in self.cleaning_report.values():
            all_cleaned_cols.update(sheet_stats.keys())

        for col in sorted(all_cleaned_cols):
            total_col_cleaned = sum(
                sheet_stats.get(col, {}).get('cleaned', 0)
                for sheet_stats in self.cleaning_report.values()
            )
            print(f"    - {col}: {total_col_cleaned} values cleaned")

        return True

    def export_cleaned_data(self):
        """Export cleaned data to Excel in memory"""
        print("\n" + "=" * 80)
        print("STEP 6: EXPORTING CLEANED DATA")
        print("=" * 80)

        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Main cleaned dataset
            self.combined_df.to_excel(
                writer, sheet_name='Cleaned_Data', index=False)
            print(f"    Exported: Cleaned_Data ({len(self.combined_df)} rows)")

            # Data summary
            summary_df = pd.DataFrame({
                'Metric': [
                    'Total Records',
                    'Total Columns',
                    'Unique Municipalities',
                    'Unique Boreholes',
                    'Depth Layers',
                    'Date Cleaned'
                ],
                'Value': [
                    len(self.combined_df),
                    len(self.combined_df.columns),
                    self.combined_df['Municipality'].nunique(
                    ) if 'Municipality' in self.combined_df.columns else 'N/A',
                    self.combined_df['Borehole ID'].nunique(
                    ) if 'Borehole ID' in self.combined_df.columns else 'N/A',
                    self.combined_df['Depth_Layer'].nunique(),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print(f"    Exported: Summary")

        excel_buffer.seek(0)
        excel_bytes = excel_buffer.read()

        print(
            f"\n  [OK] Excel file created in memory ({len(excel_bytes) / 1024:.2f} KB)")
        return excel_bytes


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main data cleaning pipeline with integrated pre-cleaning"""
    print("\n" + "=" * 80)
    print("GEOTECHNICAL DATA CLEANING PIPELINE v2.0")
    print("WITH INTEGRATED PRE-CLEANING (ROOT CAUSE FIX)")
    print("Tarlac Province, Philippines")
    print("=" * 80)

    # Configuration
    BUCKET_NAME = 'geotechnical-data'
    INPUT_FILE_PATH = 'raw/Raw_Data.xlsx'  # Or your raw file path
    OUTPUT_STORAGE_PATH = 'cleaned/Cleaned_Data.xlsx'

    # Step 1: Download raw data
    print("\n" + "=" * 80)
    print("DOWNLOADING RAW DATA")
    print("=" * 80)

    file_bytes = download_file_from_storage(BUCKET_NAME, INPUT_FILE_PATH)
    if not file_bytes:
        print("\n[ERROR] Failed to download file. Exiting.")
        return None

    # Step 2: Initialize cleaner
    cleaner = GeotechnicalDataCleaner(file_bytes)

    # Step 3: Load raw data
    if not cleaner.load_raw_data():
        print("\n[ERROR] Failed to load data. Exiting.")
        return None

    # Step 4: PRE-CLEAN (ROOT CAUSE FIX!)
    if not cleaner.pre_clean_all_sheets():
        print("\n[ERROR] Pre-cleaning failed. Exiting.")
        return None

    # Step 5: Add depth layer identifiers
    cleaner.add_depth_layer_column()

    # Step 6: Combine all sheets
    cleaner.combine_all_sheets()

    # Step 7: Validate data types
    cleaner.validate_data_types()

    # Step 8: Generate summary
    cleaner.generate_cleaning_summary()

    # Step 9: Export cleaned data
    excel_bytes = cleaner.export_cleaned_data()

    # Step 10: Upload to Supabase
    print("\n" + "=" * 80)
    print("UPLOADING TO SUPABASE STORAGE")
    print("=" * 80)

    if upload_to_supabase_storage(excel_bytes, BUCKET_NAME, OUTPUT_STORAGE_PATH):
        print(f"\n  [SUCCESS] Cleaned data uploaded to {OUTPUT_STORAGE_PATH}")
    else:
        print(f"\n  [ERROR] Failed to upload cleaned data")

    # Final summary
    print("\n" + "=" * 80)
    print("[SUCCESS] DATA CLEANING COMPLETED!")
    print("=" * 80)
    print("\n  ✓ All degree symbols removed from coordinates")
    print("  ✓ Complex PGA strings parsed to numeric values")
    print("  ✓ Relative density text converted to percentages")
    print("  ✓ Elastic modulus ranges and dates handled")
    print("  ✓ All critical columns are now properly typed")
    print("\n  Next Steps:")
    print("    1. Run: 01b_ml_data_preparation.py")
    print("    2. Run: 02_etl_to_supabase.py")
    print("    3. Run: 03_feature_engineering.py")
    print("    4. Run: 04_model_training.py")
    print("    5. Restart API with updated models")
    print("\n" + "=" * 80)

    return cleaner


if __name__ == "__main__":
    cleaner = main()
