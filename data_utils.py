"""
Utility functions for data cleaning and PostGIS operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GeometryHelper:
    """Helper class for PostGIS geometry operations"""
    
    @staticmethod
    def create_point(lon: float, lat: float, srid: int = 4326) -> Dict:
        """Create a PostGIS Point geometry"""
        return {
            'type': 'Point',
            'coordinates': [lon, lat],
            'crs': {'properties': {'name': f'EPSG:{srid}'}, 'type': 'name'}
        }
    
    @staticmethod
    def create_polygon(coordinates: List[List[Tuple[float, float]]], 
                      srid: int = 4326) -> Dict:
        """Create a PostGIS Polygon geometry"""
        return {
            'type': 'Polygon',
            'coordinates': coordinates,
            'crs': {'properties': {'name': f'EPSG:{srid}'}, 'type': 'name'}
        }
    
    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """Validate latitude and longitude values"""
        try:
            lat = float(lat)
            lon = float(lon)
            return -90 <= lat <= 90 and -180 <= lon <= 180
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        from math import radians, cos, sin, asin, sqrt
        
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r


class DataValidator:
    """Validate data quality and consistency"""
    
    @staticmethod
    def check_column_exists(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
        """Check if required columns exist in dataframe"""
        missing = [col for col in required_cols if col not in df.columns]
        return len(missing) == 0, missing
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[int, List[str]]:
        """Check for duplicate rows"""
        duplicates = df.duplicated(subset=subset)
        dup_indices = df[duplicates].index.tolist()
        return duplicates.sum(), dup_indices
    
    @staticmethod
    def check_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, bool]:
        """Check if columns match expected data types"""
        results = {}
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = df[col].dtype.name
                results[col] = actual_type == expected_type
        return results
    
    @staticmethod
    def get_null_summary(df: pd.DataFrame) -> Dict:
        """Get summary of null/missing values"""
        null_counts = df.isnull().sum()
        null_percent = (null_counts / len(df) * 100).round(2)
        
        summary = {}
        for col in df.columns:
            if null_counts[col] > 0:
                summary[col] = {
                    'count': int(null_counts[col]),
                    'percentage': float(null_percent[col])
                }
        return summary
    
    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, column: str) -> List[int]:
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
        return outliers


class ExcelHandler:
    """Handle Excel file operations"""
    
    @staticmethod
    def get_sheet_names(file_path: str) -> List[str]:
        """Get all sheet names from Excel file"""
        try:
            xls = pd.ExcelFile(file_path)
            return xls.sheet_names
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return []
    
    @staticmethod
    def preview_sheet(file_path: str, sheet_name: str, rows: int = 5) -> pd.DataFrame:
        """Preview first N rows of a sheet"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=rows)
            return df
        except Exception as e:
            logger.error(f"Error reading sheet preview: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_column_info(file_path: str, sheet_name: str) -> Dict:
        """Get information about columns in a sheet"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            info = {
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'columns': []
            }
            
            for col in df.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'null_count': int(df[col].isnull().sum()),
                    'sample_values': df[col].dropna().head(3).tolist()
                }
                info['columns'].append(col_info)
            
            return info
        except Exception as e:
            logger.error(f"Error getting column info: {e}")
            return {}


class BatchUploader:
    """Handle batch uploads to Supabase"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    def chunk_dataframe(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataframe into chunks"""
        chunks = []
        for i in range(0, len(df), self.batch_size):
            chunks.append(df.iloc[i:i + self.batch_size])
        return chunks
    
    @staticmethod
    def prepare_record(row: pd.Series, field_mapping: Dict[str, str]) -> Dict:
        """Prepare a single record for upload"""
        record = {}
        for target_field, source_field in field_mapping.items():
            if source_field in row.index:
                value = row[source_field]
                # Skip None/NaN values
                if pd.notna(value):
                    record[target_field] = value
        return record


class DataQualityReport:
    """Generate data quality reports"""
    
    def __init__(self):
        self.report = {
            'timestamp': pd.Timestamp.now(),
            'checks': [],
            'warnings': [],
            'errors': []
        }
    
    def add_check(self, check_name: str, passed: bool, details: str = ""):
        """Add a quality check result"""
        self.report['checks'].append({
            'name': check_name,
            'passed': passed,
            'details': details
        })
    
    def add_warning(self, warning: str):
        """Add a warning"""
        self.report['warnings'].append(warning)
    
    def add_error(self, error: str):
        """Add an error"""
        self.report['errors'].append(error)
    
    def get_summary(self) -> Dict:
        """Get report summary"""
        total_checks = len(self.report['checks'])
        passed_checks = sum(1 for c in self.report['checks'] if c['passed'])
        
        return {
            'timestamp': self.report['timestamp'].isoformat(),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'warnings': len(self.report['warnings']),
            'errors': len(self.report['errors']),
            'passed': total_checks == passed_checks and len(self.report['errors']) == 0
        }
    
    def print_report(self):
        """Print formatted report"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed_checks']} | Failed: {summary['failed_checks']}")
        print(f"Warnings: {summary['warnings']} | Errors: {summary['errors']}")
        print("-"*60)
        
        if self.report['checks']:
            print("\nQUALITY CHECKS:")
            for check in self.report['checks']:
                status = "✓ PASS" if check['passed'] else "✗ FAIL"
                print(f"  {status}: {check['name']}")
                if check['details']:
                    print(f"         {check['details']}")
        
        if self.report['warnings']:
            print("\nWARNINGS:")
            for warning in self.report['warnings']:
                print(f"  ⚠ {warning}")
        
        if self.report['errors']:
            print("\nERRORS:")
            for error in self.report['errors']:
                print(f"  ✗ {error}")
        
        print("="*60 + "\n")
