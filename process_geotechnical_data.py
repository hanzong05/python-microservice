"""
Script to process the Sample_Geotechnical_Data.xlsx file
Handles soil layer data by depth ranges and uploads to Supabase with PostGIS
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from data_uploader import DataCleaner, SupabaseUploader
from data_utils import DataQualityReport, ExcelHandler
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeotechnicalDataProcessor:
    """Process geotechnical investigation data from Excel"""

    def __init__(self, excel_file: str):
        self.excel_file = excel_file
        self.cleaner = DataCleaner()
        self.sheets = pd.ExcelFile(excel_file).sheet_names
        self.all_data = {}
        self.quality_report = DataQualityReport()

    def load_all_sheets(self):
        """Load all sheet data"""
        logger.info(f"Loading Excel file: {self.excel_file}")
        logger.info(f"Found sheets: {self.sheets}")

        for sheet in self.sheets:
            try:
                df = pd.read_excel(self.excel_file, sheet_name=sheet)
                self.all_data[sheet] = df
                logger.info(
                    f"[OK] Loaded {sheet}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"[ERROR] Error loading {sheet}: {e}")
                self.quality_report.add_error(
                    f"Failed to load sheet {sheet}: {str(e)}")

    def extract_header_info(self) -> dict:
        """Extract project information from Summary sheet"""
        info = {
            'location': None,
            'project_name': None,
            'coordinates': {'latitude': None, 'longitude': None},
            'description': None
        }

        if 'Summary' in self.all_data:
            df_summary = self.all_data['Summary']
            summary_text = df_summary.iloc[:, 0].astype(str).str.cat(sep=' ')

            # Parse information
            if 'Location:' in summary_text:
                try:
                    location = summary_text.split(
                        'Location:')[1].split('\n')[0].strip()
                    info['location'] = location
                except:
                    pass

            if 'Coordinates' in summary_text or 'latitude' in summary_text.lower():
                try:
                    # Extract coordinates if present
                    lines = summary_text.split('\n')
                    for line in lines:
                        if 'latitude' in line.lower():
                            parts = line.split(':')
                            if len(parts) > 1:
                                try:
                                    info['coordinates']['latitude'] = float(
                                        parts[1].strip())
                                except:
                                    pass
                        if 'longitude' in line.lower():
                            parts = line.split(':')
                            if len(parts) > 1:
                                try:
                                    info['coordinates']['longitude'] = float(
                                        parts[1].strip())
                                except:
                                    pass
                except:
                    pass

        return info

    def parse_depth_layers(self) -> dict:
        """Parse soil layer data from depth range sheets"""
        layers_data = []

        depth_sheets = [s for s in self.sheets if s != 'Summary' and 'm' in s]

        for sheet_name in sorted(depth_sheets):
            if sheet_name not in self.all_data:
                continue

            df = self.all_data[sheet_name].copy()

            # Extract depth range from sheet name
            try:
                parts = sheet_name.split('-')
                depth_from = float(parts[0].replace('m', '').strip())
                depth_to = float(parts[1].replace('m', '').strip())
            except:
                logger.warning(
                    f"Could not parse depth range from sheet {sheet_name}")
                depth_from = None
                depth_to = None

            # Find header row (usually contains test names)
            header_row = None
            for idx, row in df.iterrows():
                row_str = str(row.values)
                if any(keyword in row_str.lower() for keyword in ['boring', 'spt', 'depth', 'soil']):
                    header_row = idx
                    break

            if header_row is None:
                header_row = 0

            # Use discovered header row
            actual_df = df.iloc[header_row:].reset_index(drop=True)

            # Get actual column names
            if header_row > 0:
                new_header = df.iloc[header_row]
                actual_df.columns = new_header
                actual_df = actual_df[1:].reset_index(drop=True)

            # Clean column names
            actual_df.columns = [str(col).strip() for col in actual_df.columns]

            # Add depth information
            for idx, row in actual_df.iterrows():
                layer_data = {
                    'sheet_name': sheet_name,
                    'depth_from_m': depth_from,
                    'depth_to_m': depth_to,
                    'row_data': row.to_dict()
                }
                layers_data.append(layer_data)

            logger.info(
                f"[OK] Parsed {len(actual_df)} records from {sheet_name}")

        return {
            'layers': layers_data,
            'total_records': len(layers_data)
        }

    def create_soil_layers_table(self) -> pd.DataFrame:
        """Create structured soil layers dataframe"""
        parsed_layers = self.parse_depth_layers()
        project_info = self.extract_header_info()

        records = []

        for layer in parsed_layers['layers']:
            row_data = layer['row_data']

            # Create record with standardized column names
            record = {
                'depth_from_m': layer['depth_from_m'],
                'depth_to_m': layer['depth_to_m'],
                'borehole_id': str(row_data.get('Boring', 'BH-001')).strip() or 'BH-001',
                'layer_number': 1,  # Will be set later
                'depth_range': f"{layer['depth_from_m']}-{layer['depth_to_m']}m",

                # Soil classification
                'soil_type': row_data.get('Soil Type', row_data.get('Type', '')),
                'uscs_symbol': row_data.get('USCS', row_data.get('USCS Symbol', '')),
                'soil_description': row_data.get('Description', row_data.get('Remarks', '')),
                'color': row_data.get('Color', ''),

                # SPT Values
                'spt_n_value': self._parse_numeric(row_data.get('N', row_data.get('SPT N', ''))),
                'spt_n60': self._parse_numeric(row_data.get('N60', '')),
                'spt_n160': self._parse_numeric(row_data.get('N160', '')),
                'blow_count_int': self._parse_numeric(row_data.get('Blow Count', '')),

                # Physical properties
                'unit_weight': self._parse_numeric(row_data.get('Unit Weight', row_data.get('γ (kN/m³)', ''))),
                'unit_weight_saturated': self._parse_numeric(row_data.get('Saturated Unit Weight', '')),
                'moisture_content': self._parse_numeric(row_data.get('Moisture Content', row_data.get('w (%)', ''))),

                # Plasticity
                'plasticity_index': self._parse_numeric(row_data.get('PI', row_data.get('Plasticity Index', ''))),
                'liquid_limit': self._parse_numeric(row_data.get('LL', row_data.get('Liquid Limit', ''))),
                'plastic_limit': self._parse_numeric(row_data.get('PL', row_data.get('Plastic Limit', ''))),
                'fines_content': self._parse_numeric(row_data.get('Fines', row_data.get('Fines Content', ''))),

                # Particle size
                'mean_particle_size_d50': self._parse_numeric(row_data.get('D50', row_data.get('Mean Particle Size', ''))),

                # Groundwater
                'groundwater_depth_m': self._parse_numeric(row_data.get('Groundwater Depth', '')),
                'water_table_at_depth': layer['depth_from_m'] == self._parse_numeric(row_data.get('Groundwater Depth', '')),

                # Shear strength
                'friction_angle': self._parse_numeric(row_data.get('φ', row_data.get('Friction Angle', ''))),
                'cohesion_kpa': self._parse_numeric(row_data.get('c', row_data.get('Cohesion', ''))),

                # Elastic properties
                'shear_modulus_mpa': self._parse_numeric(row_data.get('G', row_data.get('Shear Modulus', ''))),
                'youngs_modulus_mpa': self._parse_numeric(row_data.get('E', row_data.get('Youngs Modulus', ''))),
                'poisson_ratio': self._parse_numeric(row_data.get('ν', row_data.get('Poisson Ratio', ''))),

                # Seismic
                'pga_g': self._parse_numeric(row_data.get('PGA', row_data.get('Peak Ground Accel', ''))),
                'csr': self._parse_numeric(row_data.get('CSR', row_data.get('Cyclic Stress Ratio', ''))),
                'cyclic_strength_ratio': self._parse_numeric(row_data.get('CRR', row_data.get('Cyclic Resistance', ''))),

                # Liquefaction
                'liquefaction': self._parse_bool(row_data.get('Liquefaction', False)),
                'liquefaction_risk_level': row_data.get('Liquefaction Risk', ''),

                # Settlement
                'settlement_cm': self._parse_numeric(row_data.get('Settlement', '')),

                # Foundation
                'foundation_width_m': self._parse_numeric(row_data.get('B', row_data.get('Foundation Width', ''))),
                'foundation_depth_m': self._parse_numeric(row_data.get('Df', row_data.get('Foundation Depth', ''))),
                'bearing_capacity_kpa': self._parse_numeric(row_data.get('qult', row_data.get('Bearing Capacity', ''))),
                'qa_allowable_kpa': self._parse_numeric(row_data.get('qa', row_data.get('Allowable Capacity', ''))),

                # Pressure
                'effective_overburden_pressure': self._parse_numeric(row_data.get("σ'v", row_data.get('Effective Overburden', ''))),
                'total_overburden_pressure': self._parse_numeric(row_data.get('σv', row_data.get('Total Overburden', ''))),
                'relative_density_percent': self._parse_numeric(row_data.get('Dr', row_data.get('Relative Density', ''))),

                # Lab info
                'test_laboratory': row_data.get('Lab', row_data.get('Laboratory', '')),
                'observer_name': row_data.get('Observer', row_data.get('Technician', '')),
                'remarks': row_data.get('Remarks', row_data.get('Notes', ''))
            }

            # Add project info if available
            if project_info['location']:
                record['location'] = project_info['location']

            records.append(record)

        # Create dataframe
        df_layers = pd.DataFrame(records)

        # Set layer numbers by depth range
        for (depth_from, depth_to), group in df_layers.groupby(['depth_from_m', 'depth_to_m']):
            layer_num = int((depth_from / 1.5) + 1)
            df_layers.loc[group.index, 'layer_number'] = min(layer_num, 10)

        logger.info(
            f"[OK] Created soil layers table with {len(df_layers)} records")
        self.quality_report.add_check("Layer Data Creation", len(
            df_layers) > 0, f"{len(df_layers)} layers")

        return df_layers

    @staticmethod
    def _parse_numeric(value) -> float:
        """Safely parse numeric values"""
        if pd.isna(value) or value == '':
            return None
        try:
            return float(str(value).strip())
        except:
            return None

    @staticmethod
    def _parse_bool(value) -> bool:
        """Safely parse boolean values"""
        if pd.isna(value):
            return False
        str_val = str(value).lower().strip()
        return str_val in ['true', 'yes', '1', 'y', 'ok']

    def print_summary(self):
        """Print processing summary"""
        logger.info("\n" + "="*60)
        logger.info("GEOTECHNICAL DATA PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Excel file: {self.excel_file}")
        logger.info(f"Sheets processed: {len(self.sheets)}")
        logger.info(f"Depth sheets: {[s for s in self.sheets if 'm' in s]}")
        logger.info("="*60)


def main():
    """Main execution"""
    excel_file = r'c:\xampp\htdocs\test-py\Sample_Geotechnical_Data.xlsx'

    print("="*60)
    print("GEOTECHNICAL DATA PROCESSOR")
    print("="*60)
    print(f"Processing: {excel_file}\n")

    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"ERROR: File not found: {excel_file}")
        return

    # Process data
    processor = GeotechnicalDataProcessor(excel_file)
    processor.load_all_sheets()

    # Create soil layers dataframe
    df_layers = processor.create_soil_layers_table()

    # Display summary
    processor.print_summary()

    # Show data preview
    print("\nSample Data:")
    print("-"*60)
    print(df_layers.head().to_string())

    print("\n" + "-"*60)
    print("Column Summary:")
    for col in df_layers.columns:
        null_count = df_layers[col].isna().sum()
        print(
            f"  {col:<35} | Non-null: {len(df_layers) - null_count}/{len(df_layers)}")

    # Save cleaned data
    output_file = r'c:\xampp\htdocs\test-py\cleaned_geotechnical_data.csv'
    df_layers.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nCleaned data saved to: {output_file}")

    processor.quality_report.print_report()

    # Ask for upload
    print("\n" + "="*60)
    try:
        response = input(
            "\nDo you want to upload this data to Supabase? (yes/no): ").strip().lower()

        if response in ['yes', 'y']:
            url = os.getenv("SUPABASE_URL") or input("Enter Supabase URL: ")
            key = os.getenv("SUPABASE_SERVICE_KEY") or input(
                "Enter Service Key: ")

            uploader = SupabaseUploader(url, key)
            success = uploader.upload_soil_layers(df_layers)
            uploader.print_report()
    except (KeyboardInterrupt, EOFError):
        print("\nSkipping upload. Data already saved to CSV.")


if __name__ == "__main__":
    main()
