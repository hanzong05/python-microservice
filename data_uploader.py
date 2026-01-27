"""
Data cleaning and Supabase upload script for geological survey data with PostGIS support
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Handle data cleaning operations for geological survey data"""

    def __init__(self):
        self.cleaning_report = {
            'total_rows': 0,
            'rows_cleaned': 0,
            'null_values': {},
            'conversions': [],
            'errors': []
        }

    def load_excel(self, file_path: str, sheet_name: str = 0) -> pd.DataFrame:
        """Load Excel file and handle basic validation"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            self.cleaning_report['total_rows'] = len(df)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise

    def clean_numeric_column(self, df: pd.DataFrame, column: str,
                             allow_null: bool = True) -> pd.DataFrame:
        """Clean and convert numeric columns"""
        try:
            if column not in df.columns:
                logger.warning(f"Column {column} not found")
                return df

            # Record null values
            null_count = df[column].isna().sum()
            if null_count > 0:
                self.cleaning_report['null_values'][column] = null_count

            # Convert to numeric, coercing errors
            df[column] = pd.to_numeric(df[column], errors='coerce')

            if not allow_null and df[column].isna().any():
                df = df.dropna(subset=[column])
                logger.info(
                    f"Removed {null_count} rows with null values in {column}")

            self.cleaning_report['conversions'].append(f"{column} -> numeric")
            return df
        except Exception as e:
            logger.error(f"Error cleaning column {column}: {e}")
            self.cleaning_report['errors'].append(str(e))
            return df

    def clean_string_column(self, df: pd.DataFrame, column: str,
                            strip: bool = True) -> pd.DataFrame:
        """Clean string columns"""
        try:
            if column not in df.columns:
                return df

            if strip:
                df[column] = df[column].astype(str).str.strip()

            # Remove rows with only whitespace or 'nan'
            df = df[~df[column].isin(['', 'nan', 'NaN', 'None', 'null'])]

            self.cleaning_report['conversions'].append(
                f"{column} -> string (cleaned)")
            return df
        except Exception as e:
            logger.error(f"Error cleaning string column {column}: {e}")
            return df

    def clean_date_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Clean and convert date columns"""
        try:
            if column not in df.columns:
                return df

            df[column] = pd.to_datetime(df[column], errors='coerce')
            self.cleaning_report['conversions'].append(f"{column} -> date")
            return df
        except Exception as e:
            logger.error(f"Error cleaning date column {column}: {e}")
            return df

    def create_geometry_point(self, latitude: float, longitude: float) -> Optional[Dict]:
        """Create PostGIS geometry point from lat/lon"""
        try:
            if pd.isna(latitude) or pd.isna(longitude):
                return None

            # Validate coordinates
            lat = float(latitude)
            lon = float(longitude)

            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None

            # PostGIS Point format: Point(lon lat SRID=4326)
            return {
                'type': 'Point',
                'coordinates': [lon, lat],
                'crs': {'properties': {'name': 'EPSG:4326'}, 'type': 'name'}
            }
        except Exception as e:
            logger.error(f"Error creating geometry point: {e}")
            return None

    def add_geometry_columns(self, df: pd.DataFrame, lat_col: str,
                             lon_col: str, geo_col_name: str = 'location') -> pd.DataFrame:
        """Create geometry column from latitude and longitude"""
        try:
            if lat_col not in df.columns or lon_col not in df.columns:
                logger.warning(f"Latitude or longitude column not found")
                return df

            df[geo_col_name] = df.apply(
                lambda row: self.create_geometry_point(
                    row[lat_col], row[lon_col]),
                axis=1
            )

            logger.info(
                f"Created {geo_col_name} column from {lat_col} and {lon_col}")
            return df
        except Exception as e:
            logger.error(f"Error adding geometry columns: {e}")
            return df

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset)
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop',
                              threshold: float = 0.5) -> pd.DataFrame:
        """Handle missing values in the dataframe"""
        try:
            if strategy == 'drop':
                # Drop rows with all NaN values
                df = df.dropna(how='all')
                # Drop columns with more than threshold missing values
                mask = df.isnull().sum() / len(df) > threshold
                cols_to_drop = df.columns[mask].tolist()
                if cols_to_drop:
                    logger.info(
                        f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
                    df = df.drop(columns=cols_to_drop)

            elif strategy == 'fill_numeric':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(
                    df[numeric_cols].mean())

            return df
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return df

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate cleaned data"""
        issues = []

        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues

        if df.isnull().all().any():
            null_cols = df.columns[df.isnull().all()].tolist()
            issues.append(f"Columns with all null values: {null_cols}")

        return len(issues) == 0, issues

    def print_report(self):
        """Print cleaning report"""
        logger.info("=" * 50)
        logger.info("DATA CLEANING REPORT")
        logger.info("=" * 50)
        logger.info(
            f"Total rows processed: {self.cleaning_report['total_rows']}")
        logger.info(
            f"Null values by column: {self.cleaning_report['null_values']}")
        logger.info(
            f"Conversions applied: {self.cleaning_report['conversions']}")
        if self.cleaning_report['errors']:
            logger.error(
                f"Errors encountered: {self.cleaning_report['errors']}")
        logger.info("=" * 50)


class SupabaseUploader:
    """Handle data upload to Supabase"""

    def __init__(self, url: str, api_key: str):
        """
        Initialize Supabase client

        Args:
            url: Supabase project URL
            api_key: Supabase API key (service role key for uploads)
        """
        try:
            from supabase import create_client
        except ImportError:
            logger.error(
                "supabase-py not installed. Install with: pip install supabase")
            raise

        self.supabase = create_client(url, api_key)
        self.upload_report = {
            'successful': 0,
            'failed': 0,
            'errors': []
        }

    def upload_municipalities(self, df: pd.DataFrame) -> bool:
        """Upload municipalities data"""
        try:
            data = []
            for _, row in df.iterrows():
                record = {
                    'name': row.get('name'),
                    'center_point': row.get('center_point'),
                    'description': row.get('description'),
                }
                # Remove None values
                record = {k: v for k, v in record.items() if v is not None}
                data.append(record)

            response = self.supabase.table(
                'municipalities').insert(data).execute()
            logger.info(f"Uploaded {len(data)} municipalities")
            self.upload_report['successful'] += len(data)
            return True
        except Exception as e:
            logger.error(f"Error uploading municipalities: {e}")
            self.upload_report['errors'].append(str(e))
            return False

    def upload_barangays(self, df: pd.DataFrame) -> bool:
        """Upload barangays data"""
        try:
            data = []
            for _, row in df.iterrows():
                record = {
                    'municipality_id': row.get('municipality_id'),
                    'name': row.get('name'),
                    'location': row.get('location'),
                    'area_sq_km': row.get('area_sq_km'),
                    'description': row.get('description'),
                }
                record = {k: v for k, v in record.items() if v is not None}
                data.append(record)

            response = self.supabase.table('barangays').insert(data).execute()
            logger.info(f"Uploaded {len(data)} barangays")
            self.upload_report['successful'] += len(data)
            return True
        except Exception as e:
            logger.error(f"Error uploading barangays: {e}")
            self.upload_report['errors'].append(str(e))
            return False

    def upload_boreholes(self, df: pd.DataFrame) -> bool:
        """Upload boreholes data with PostGIS geometry"""
        try:
            data = []
            for _, row in df.iterrows():
                record = {
                    'barangay_id': row.get('barangay_id'),
                    'borehole_id': row.get('borehole_id'),
                    'location': row.get('location'),  # PostGIS Point geometry
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude'),
                    'elevation': row.get('elevation'),
                    'depth_total_m': row.get('depth_total_m'),
                    'drilling_date': row.get('drilling_date'),
                    'drilling_contractor': row.get('drilling_contractor'),
                    'remarks': row.get('remarks'),
                }
                record = {k: v for k, v in record.items() if v is not None}
                data.append(record)

            response = self.supabase.table('boreholes').insert(data).execute()
            logger.info(f"Uploaded {len(data)} boreholes")
            self.upload_report['successful'] += len(data)
            return True
        except Exception as e:
            logger.error(f"Error uploading boreholes: {e}")
            self.upload_report['errors'].append(str(e))
            return False

    def upload_soil_layers(self, df: pd.DataFrame) -> bool:
        """Upload soil layers data"""
        try:
            data = []
            numeric_cols = [
                'depth_from_m', 'depth_to_m', 'spt_n_value', 'spt_n60', 'spt_n160',
                'unit_weight', 'unit_weight_saturated', 'moisture_content',
                'plasticity_index', 'liquid_limit', 'plastic_limit', 'fines_content',
                'mean_particle_size_d50', 'groundwater_depth_m', 'friction_angle',
                'cohesion_kpa', 'shear_modulus_mpa', 'youngs_modulus_mpa',
                'poisson_ratio', 'pga_g', 'csr', 'cyclic_strength_ratio',
                'settlement_cm', 'elastic_modulus_es', 'foundation_width_m',
                'foundation_depth_m', 'bearing_capacity_kpa', 'qa_allowable_kpa',
                'effective_overburden_pressure', 'total_overburden_pressure',
                'relative_density_percent'
            ]

            for _, row in df.iterrows():
                record = {
                    'borehole_id': row.get('borehole_id'),
                    'layer_number': row.get('layer_number'),
                    'soil_type': row.get('soil_type'),
                    'uscs_symbol': row.get('uscs_symbol'),
                    'soil_description': row.get('soil_description'),
                    'color': row.get('color'),
                    'water_table_at_depth': row.get('water_table_at_depth', False),
                    'liquefaction': row.get('liquefaction', False),
                    'liquefaction_risk_level': row.get('liquefaction_risk_level'),
                    'test_laboratory': row.get('test_laboratory'),
                    'observer_name': row.get('observer_name'),
                    'remarks': row.get('remarks'),
                }

                # Add numeric columns if present
                for col in numeric_cols:
                    if col in row.index:
                        record[col] = row.get(col)

                record = {k: v for k, v in record.items() if v is not None}
                data.append(record)

            response = self.supabase.table(
                'soil_layers').insert(data).execute()
            logger.info(f"Uploaded {len(data)} soil layer records")
            self.upload_report['successful'] += len(data)
            return True
        except Exception as e:
            logger.error(f"Error uploading soil layers: {e}")
            self.upload_report['errors'].append(str(e))
            return False

    def print_report(self):
        """Print upload report"""
        logger.info("=" * 50)
        logger.info("UPLOAD REPORT")
        logger.info("=" * 50)
        logger.info(
            f"Successfully uploaded: {self.upload_report['successful']}")
        logger.info(f"Failed uploads: {self.upload_report['failed']}")
        if self.upload_report['errors']:
            logger.error(f"Errors: {self.upload_report['errors']}")
        logger.info("=" * 50)


def main():
    """Main execution function"""
    import os

    # Configuration
    EXCEL_FILE = input("Enter Excel file path: ")
    SUPABASE_URL = os.getenv("SUPABASE_URL") or input("Enter Supabase URL: ")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or input(
        "Enter Supabase Service Key: ")

    # Initialize cleaner
    cleaner = DataCleaner()

    # Load data based on sheet name selection
    print("\nAvailable sheets:")
    print("1. Municipalities")
    print("2. Barangays")
    print("3. Boreholes")
    print("4. Soil Layers")

    choice = input("\nSelect data type to process (1-4): ")

    try:
        if choice == '1':
            process_municipalities(EXCEL_FILE, cleaner,
                                   SUPABASE_URL, SUPABASE_KEY)
        elif choice == '2':
            process_barangays(EXCEL_FILE, cleaner, SUPABASE_URL, SUPABASE_KEY)
        elif choice == '3':
            process_boreholes(EXCEL_FILE, cleaner, SUPABASE_URL, SUPABASE_KEY)
        elif choice == '4':
            process_soil_layers(EXCEL_FILE, cleaner,
                                SUPABASE_URL, SUPABASE_KEY)
        else:
            logger.error("Invalid choice")
            return

        cleaner.print_report()
    except Exception as e:
        logger.error(f"Fatal error: {e}")


def process_municipalities(file_path: str, cleaner: DataCleaner, url: str, key: str):
    """Process and upload municipalities data"""
    df = cleaner.load_excel(file_path, sheet_name='municipalities')

    # Clean data
    df = cleaner.clean_string_column(df, 'name')
    df = cleaner.clean_string_column(df, 'description')
    df = cleaner.remove_duplicates(df, subset=['name'])

    # Validate
    is_valid, issues = cleaner.validate_data(df)
    if not is_valid:
        logger.error(f"Validation failed: {issues}")
        return

    # Upload
    uploader = SupabaseUploader(url, key)
    uploader.upload_municipalities(df)
    uploader.print_report()


def process_barangays(file_path: str, cleaner: DataCleaner, url: str, key: str):
    """Process and upload barangays data"""
    df = cleaner.load_excel(file_path, sheet_name='barangays')

    # Clean data
    df = cleaner.clean_numeric_column(df, 'municipality_id')
    df = cleaner.clean_string_column(df, 'name')
    df = cleaner.clean_numeric_column(df, 'area_sq_km')
    df = cleaner.clean_string_column(df, 'description')

    # Create geometry if lat/lon present
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = cleaner.add_geometry_columns(
            df, 'latitude', 'longitude', 'location')

    df = cleaner.remove_duplicates(df)

    # Validate
    is_valid, issues = cleaner.validate_data(df)
    if not is_valid:
        logger.error(f"Validation failed: {issues}")
        return

    # Upload
    uploader = SupabaseUploader(url, key)
    uploader.upload_barangays(df)
    uploader.print_report()


def process_boreholes(file_path: str, cleaner: DataCleaner, url: str, key: str):
    """Process and upload boreholes data"""
    df = cleaner.load_excel(file_path, sheet_name='boreholes')

    # Clean data
    df = cleaner.clean_numeric_column(df, 'barangay_id')
    df = cleaner.clean_string_column(df, 'borehole_id')
    df = cleaner.clean_numeric_column(df, 'latitude')
    df = cleaner.clean_numeric_column(df, 'longitude')
    df = cleaner.clean_numeric_column(df, 'elevation')
    df = cleaner.clean_numeric_column(df, 'depth_total_m')
    df = cleaner.clean_date_column(df, 'drilling_date')
    df = cleaner.clean_string_column(df, 'drilling_contractor')
    df = cleaner.clean_string_column(df, 'remarks')

    # Create geometry
    df = cleaner.add_geometry_columns(df, 'latitude', 'longitude', 'location')

    df = cleaner.remove_duplicates(df, subset=['borehole_id'])

    # Validate
    is_valid, issues = cleaner.validate_data(df)
    if not is_valid:
        logger.error(f"Validation failed: {issues}")
        return

    # Upload
    uploader = SupabaseUploader(url, key)
    uploader.upload_boreholes(df)
    uploader.print_report()


def process_soil_layers(file_path: str, cleaner: DataCleaner, url: str, key: str):
    """Process and upload soil layers data"""
    df = cleaner.load_excel(file_path, sheet_name='soil_layers')

    # Clean data - numeric columns
    numeric_cols = [
        'depth_from_m', 'depth_to_m', 'spt_n_value', 'spt_n60', 'spt_n160',
        'unit_weight', 'unit_weight_saturated', 'moisture_content',
        'plasticity_index', 'liquid_limit', 'plastic_limit', 'fines_content',
        'mean_particle_size_d50', 'groundwater_depth_m', 'friction_angle',
        'cohesion_kpa', 'shear_modulus_mpa', 'youngs_modulus_mpa',
        'poisson_ratio', 'pga_g', 'csr', 'cyclic_strength_ratio',
        'settlement_cm', 'elastic_modulus_es', 'foundation_width_m',
        'foundation_depth_m', 'bearing_capacity_kpa', 'qa_allowable_kpa',
        'effective_overburden_pressure', 'total_overburden_pressure',
        'relative_density_percent'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df = cleaner.clean_numeric_column(df, col, allow_null=True)

    # String columns
    string_cols = ['soil_type', 'uscs_symbol', 'soil_description', 'color',
                   'liquefaction_risk_level', 'test_laboratory', 'observer_name', 'remarks']
    for col in string_cols:
        if col in df.columns:
            df = cleaner.clean_string_column(df, col)

    df = cleaner.handle_missing_values(df, strategy='drop')
    df = cleaner.remove_duplicates(df)

    # Validate
    is_valid, issues = cleaner.validate_data(df)
    if not is_valid:
        logger.error(f"Validation failed: {issues}")
        return

    # Upload
    uploader = SupabaseUploader(url, key)
    uploader.upload_soil_layers(df)
    uploader.print_report()


if __name__ == "__main__":
    main()
