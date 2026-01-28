"""
ETL Pipeline: Excel to Supabase Database - SUPABASE STORAGE VERSION
Geotechnical Data for Liquefaction Prediction System

- Downloads ML_Ready_Data.xlsx from Supabase Storage
- Processes and uploads data to Supabase database tables
"""

import json
import warnings
import pandas as pd
import numpy as np
import io
from datetime import datetime
from supabase_client import get_supabase_client  # Import the connection function

warnings.filterwarnings('ignore')


# -----------------------------
# Download File from Supabase
# -----------------------------

def download_file_from_storage(bucket_name, file_path):
    """Download file from Supabase Storage"""
    print(f" Downloading file from Supabase Storage...")
    print(f"   Bucket: {bucket_name}")
    print(f"   File: {file_path}")

    client = get_supabase_client()
    if not client:
        print(" Failed to connect to Supabase")
        return None

    try:
        response = client.storage.from_(bucket_name).download(file_path)
        print(f"✓ Successfully downloaded {len(response)} bytes")
        return response
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None


class GeotechnicalETL:
    """ETL Pipeline for Geotechnical Data"""

    def __init__(self, file_bytes=None):
        self.file_bytes = file_bytes
        self.client = None
        self.df = None
        self.municipality_ids = {}
        self.barangay_ids = {}
        self.borehole_ids = {}

    def connect(self):
        """Connect to Supabase using imported client"""
        print("=" * 70)
        print("CONNECTING TO SUPABASE")
        print("=" * 70)

        self.client = get_supabase_client()

        if self.client:
            print("✓ Connected to Supabase successfully!")
            return True
        else:
            print(" Connection failed")
            return False

    def load_data(self):
        """Load data from Excel bytes"""
        print("\n" + "=" * 70)
        print("LOADING DATA FROM EXCEL")
        print("=" * 70)

        try:
            # Read Excel from bytes instead of file
            self.df = pd.read_excel(io.BytesIO(
                self.file_bytes), sheet_name='Full_Dataset')
            print(f"✓ Loaded {len(self.df)} records from Supabase Storage")
            print(f"✓ Columns: {len(self.df.columns)}")
            print(f"\nSample data:")
            print(
                self.df[['Municipality', 'Borehole ID', 'Depth_Layer']].head())
            return True
        except Exception as e:
            print(f" Failed to load data: {e}")
            return False

    def extract_unique_values(self):
        """Extract unique municipalities, barangays, and boreholes"""
        print("\n" + "=" * 70)
        print("ANALYZING DATA STRUCTURE")
        print("=" * 70)

        municipalities = self.df['Municipality'].unique()
        print(f"\n✓ Found {len(municipalities)} municipalities")

        boreholes = self.df['Borehole ID'].unique()
        print(f"✓ Found {len(boreholes)} boreholes")

        depth_layers = self.df['Depth_Layer'].unique()
        print(f"✓ Found {len(depth_layers)} depth layers")

        return municipalities, boreholes, depth_layers

    def upsert_municipalities(self):
        """Insert or update municipalities"""
        print("\n" + "=" * 70)
        print("STEP 1: UPSERTING MUNICIPALITIES")
        print("=" * 70)

        municipalities = self.df['Municipality'].unique()

        for muni_name in municipalities:
            try:
                result = self.client.table('municipalities').select(
                    'id, name').eq('name', muni_name).execute()

                if result.data and len(result.data) > 0:
                    muni_id = result.data[0]['id']
                    self.municipality_ids[muni_name] = muni_id
                    print(f"  ✓ Found existing: {muni_name} (ID: {muni_id})")
                else:
                    insert_data = {
                        'name': muni_name,
                        'description': f'Municipality in Tarlac Province'
                    }
                    result = self.client.table(
                        'municipalities').insert(insert_data).execute()
                    muni_id = result.data[0]['id']
                    self.municipality_ids[muni_name] = muni_id
                    print(f"  ✓ Inserted new: {muni_name} (ID: {muni_id})")

            except Exception as e:
                print(f"   Error with {muni_name}: {e}")

        print(f"\n✓ Total municipalities: {len(self.municipality_ids)}")

    def upsert_barangays(self):
        """Insert or update barangays"""
        print("\n" + "=" * 70)
        print("STEP 2: UPSERTING BARANGAYS")
        print("=" * 70)

        for muni_name, muni_id in self.municipality_ids.items():
            barangay_name = f"{muni_name} - Central"

            try:
                result = self.client.table('barangays').select('id, name').eq(
                    'municipality_id', muni_id).eq('name', barangay_name).execute()

                if result.data and len(result.data) > 0:
                    barangay_id = result.data[0]['id']
                    self.barangay_ids[(muni_name, barangay_name)] = barangay_id
                else:
                    insert_data = {
                        'municipality_id': muni_id,
                        'name': barangay_name,
                        'description': f'Default barangay for {muni_name}'
                    }
                    result = self.client.table(
                        'barangays').insert(insert_data).execute()
                    barangay_id = result.data[0]['id']
                    self.barangay_ids[(muni_name, barangay_name)] = barangay_id

            except Exception as e:
                print(f"   Error with {barangay_name}: {e}")

        print(f"✓ Total barangays: {len(self.barangay_ids)}")

    def upsert_boreholes(self):
        """Insert or update boreholes WITHOUT PostGIS location"""
        print("\n" + "=" * 70)
        print("STEP 3: UPSERTING BOREHOLES")
        print("=" * 70)

        borehole_groups = self.df.groupby('Borehole ID').first().reset_index()

        for idx, row in borehole_groups.iterrows():
            borehole_id = row['Borehole ID']
            municipality = row['Municipality']

            barangay_key = (municipality, f"{municipality} - Central")
            barangay_id = self.barangay_ids.get(barangay_key)

            if not barangay_id:
                continue

            try:
                result = self.client.table('boreholes').select(
                    'id, borehole_id').eq('borehole_id', borehole_id).execute()

                latitude = float(row['Latitude']) if pd.notna(
                    row['Latitude']) else None
                longitude = float(row['Longitude']) if pd.notna(
                    row['Longitude']) else None

                if not latitude or not longitude:
                    print(
                        f"  ⚠ Missing coordinates for {borehole_id}, skipping")
                    continue

                borehole_data = {
                    'borehole_id': borehole_id,
                    'barangay_id': barangay_id,
                    'latitude': latitude,
                    'longitude': longitude,
                    'elevation': float(row['Elevation']) if pd.notna(row['Elevation']) else None,
                    'depth_total_m': 15.0,
                    'remarks': f'Data from {municipality}'
                }

                if result.data and len(result.data) > 0:
                    db_id = result.data[0]['id']
                    borehole_data['updated_at'] = datetime.now().isoformat()
                    self.client.table('boreholes').update(
                        borehole_data).eq('id', db_id).execute()
                    self.borehole_ids[borehole_id] = db_id
                    print(f"  ✓ Updated: {borehole_id} (ID: {db_id})")
                else:
                    result = self.client.table('boreholes').insert(
                        borehole_data).execute()
                    db_id = result.data[0]['id']
                    self.borehole_ids[borehole_id] = db_id
                    print(f"  ✓ Inserted: {borehole_id} (ID: {db_id})")

            except Exception as e:
                print(f"   Error with {borehole_id}: {e}")

        print(f"\n✓ Total boreholes: {len(self.borehole_ids)}")

    def map_depth_layer_to_number(self, depth_layer):
        """Map depth layer string to layer number (1-10)"""
        layer_mapping = {
            '0m-1.5m': 1, '1.5m-3.0m': 2, '3.0m-4.5m': 3, '4.5m-6.0m': 4,
            '6.0m-7.5m': 5, '7.5m-9.0m': 6, '9.0m-10.5m': 7, '10.5m-12.0m': 8,
            '12.0m-13.5m': 9, '13.5m-15.0m': 10
        }
        return layer_mapping.get(depth_layer, 1)

    def extract_depth_range(self, depth_layer):
        """Extract depth_from_m and depth_to_m from depth layer string"""
        parts = depth_layer.replace('m', '').split('-')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        return 0.0, 1.5

    def safe_float(self, value):
        """Safely convert to float, return None if invalid"""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except:
            return None

    def upsert_soil_layers_batch(self, batch_size=25):
        """Insert or update soil layers using BATCH operations for speed"""
        print("\n" + "=" * 70)
        print("STEP 4: UPSERTING SOIL LAYERS (BATCH MODE)")
        print("=" * 70)

        total_records = len(self.df)
        processed = 0
        errors = 0
        batch_data = []

        print(
            f"\nProcessing {total_records} soil layer records in batches of {batch_size}...")

        for idx, row in self.df.iterrows():
            try:
                borehole_id_str = row['Borehole ID']
                borehole_db_id = self.borehole_ids.get(borehole_id_str)

                if not borehole_db_id:
                    errors += 1
                    continue

                depth_layer = row['Depth_Layer']
                layer_number = self.map_depth_layer_to_number(depth_layer)
                depth_from, depth_to = self.extract_depth_range(depth_layer)

                soil_data = {
                    'borehole_id': borehole_db_id,
                    'layer_number': layer_number,
                    'depth_from_m': depth_from,
                    'depth_to_m': depth_to,
                    'depth_range': depth_layer,
                    'soil_type': row.get('Soil/Rock Description'),
                    'uscs_symbol': row.get('USCS Symbol'),
                    'soil_description': row.get('Soil/Rock Description'),
                    'spt_n_value': self.safe_float(row.get('SPT N-Value')),
                    'spt_n60': self.safe_float(row.get('SPT N-Value')),
                    'spt_n160': self.safe_float(row.get('Corrected SPT-N Value (N1(60))')),
                    'unit_weight': self.safe_float(row.get('Unit Weight (γ)')),
                    'moisture_content': self.safe_float(row.get('Natural Water Content (ω)')),
                    'plasticity_index': self.safe_float(row.get('Plasticity Index (PI)')),
                    'fines_content': self.safe_float(row.get('Fines Content')),
                    'mean_particle_size_d50': self.safe_float(row.get('Mean Particle Size (D50) (mm)')),
                    'groundwater_depth_m': self.safe_float(row.get('Groundwater Level (m)')),
                    'friction_angle': self.safe_float(row.get('Internal Friction Angle')),
                    'cohesion_kpa': self.safe_float(row.get('Cohesion_kPa')),
                    'pga_g': self.safe_float(row.get('Peak Ground Acceleration')),
                    'csr': self.safe_float(row.get('Cyclic Stress Ratio (CSR)')),
                    'cyclic_strength_ratio': self.safe_float(row.get('CRR')),
                    'liquefaction': bool(row.get('Liquefaction_Potential', 0) == 1),
                    'liquefaction_risk_level': 'High' if row.get('Liquefaction_Potential', 0) == 1 else 'Low',
                    'settlement_cm': self.safe_float(row.get('Settlement_m')) * 100 if pd.notna(row.get('Settlement_m')) else None,
                    'bearing_capacity_kpa': self.safe_float(row.get('Ultimate_Bearing_Capacity_kPa')),
                    'qa_allowable_kpa': self.safe_float(row.get('Allowable_Bearing_Capacity_kPa')),
                    'effective_overburden_pressure': self.safe_float(row.get('Effective_Overburden_Stress_kPa')),
                    'total_overburden_pressure': self.safe_float(row.get('Total_Overburden_Stress_kPa')),
                    'relative_density_percent': self.safe_float(row.get('Relative Density')),
                    'foundation_width_m': self.safe_float(row.get('Foundation Width (B)')),
                    'foundation_depth_m': self.safe_float(row.get('Foundation Depth (D)')),
                    'elastic_modulus_es': self.safe_float(row.get('Elastic Modulus (Es) (MN/m²)')),
                }

                # Remove None values
                soil_data = {k: v for k, v in soil_data.items()
                             if v is not None}

                # Add to batch
                batch_data.append(soil_data)

                # When batch is full, insert it
                if len(batch_data) >= batch_size:
                    try:
                        self.client.table('soil_layers').insert(
                            batch_data).execute()
                        processed += len(batch_data)
                        print(
                            f"  ✓ Processed {processed}/{total_records} records ({processed/total_records*100:.1f}%)")
                        batch_data = []  # Clear batch
                    except Exception as batch_error:
                        print(
                            f"  ⚠ Batch insert failed, trying individual inserts...")
                        # Fall back to individual inserts for this batch
                        for item in batch_data:
                            try:
                                self.client.table('soil_layers').insert(
                                    item).execute()
                                processed += 1
                            except:
                                errors += 1
                        batch_data = []

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"   Error at row {idx}: {e}")

        # Insert remaining batch
        if batch_data:
            try:
                self.client.table('soil_layers').insert(batch_data).execute()
                processed += len(batch_data)
            except Exception as batch_error:
                for item in batch_data:
                    try:
                        self.client.table('soil_layers').insert(item).execute()
                        processed += 1
                    except:
                        errors += 1

        print(f"\n✓ Completed!")
        print(f"  Successfully processed: {processed}/{total_records}")
        print(f"  Errors: {errors}")

    def verify_data(self):
        """Verify data was loaded correctly"""
        print("\n" + "=" * 70)
        print("VERIFYING DATA IN DATABASE")
        print("=" * 70)

        try:
            municipalities = self.client.table(
                'municipalities').select('id', count='exact').execute()
            barangays = self.client.table('barangays').select(
                'id', count='exact').execute()
            boreholes = self.client.table('boreholes').select(
                'id', count='exact').execute()
            soil_layers = self.client.table('soil_layers').select(
                'id', count='exact').execute()

            print(f"\n✓ Database Summary:")
            print(f"  Municipalities: {municipalities.count}")
            print(f"  Barangays: {barangays.count}")
            print(f"  Boreholes: {boreholes.count}")
            print(f"  Soil Layers: {soil_layers.count}")

            return True

        except Exception as e:
            print(f" Verification failed: {e}")
            return False

    def print_postgis_instructions(self):
        """Print instructions for updating PostGIS locations"""
        print("\n" + "=" * 70)
        print("IMPORTANT: UPDATE POSTGIS LOCATIONS")
        print("=" * 70)
        print("\nYour boreholes were inserted successfully, but the PostGIS 'location'")
        print("field needs to be updated manually.")
        print("\n Please run this SQL in Supabase SQL Editor:")
        print("-" * 70)
        print("""
UPDATE boreholes 
SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
WHERE location IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL;
""")
        print("-" * 70)
        print("\nThis will populate the location field with PostGIS geometry.")
        print("=" * 70)


def main():
    """Main ETL execution"""
    print("\n" + "=" * 70)
    print("ETL PIPELINE: SUPABASE STORAGE TO DATABASE")
    print("Geotechnical Data for Liquefaction Prediction")
    print("=" * 70 + "\n")

    # Configuration
    BUCKET_NAME = 'geotechnical-data'
    INPUT_FILE_PATH = 'ml_ready/ML_Ready_Data.xlsx'  # Path in Supabase Storage

    # Step 1: Download file from Supabase Storage
    file_bytes = download_file_from_storage(BUCKET_NAME, INPUT_FILE_PATH)
    if not file_bytes:
        print(" Failed to download file from storage. Exiting.")
        return None

    # Step 2: Initialize ETL with downloaded bytes
    etl = GeotechnicalETL(file_bytes=file_bytes)

    if not etl.connect():
        return None

    if not etl.load_data():
        return None

    etl.extract_unique_values()

    # Load data to database
    etl.upsert_municipalities()
    etl.upsert_barangays()
    etl.upsert_boreholes()
    etl.upsert_soil_layers_batch()  # Use batch insert

    etl.verify_data()

    # Print PostGIS update instructions
    etl.print_postgis_instructions()

    print("\n" + "=" * 70)
    print(" ETL PIPELINE COMPLETED!")
    print("=" * 70 + "\n")

    return etl


if __name__ == "__main__":
    etl_system = main()
