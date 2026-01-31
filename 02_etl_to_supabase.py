"""
ETL Pipeline: Excel to Supabase Database - IMPROVED VERSION
Geotechnical Data for Liquefaction Prediction System

Key Improvements:
- Better error handling and logging
- Validates data before insertion
- Handles missing/null values properly
- Batch operations with fallback
- Progress tracking
- Data validation checks
"""

import json
import warnings
import pandas as pd
import numpy as np
import io
from datetime import datetime
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings('ignore')


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 25
BUCKET_NAME = 'geotechnical-data'
INPUT_FILE_PATH = 'ml_ready/ML_Ready_Data.xlsx'


# -----------------------------
# Helper Functions
# -----------------------------

def safe_float(value, default=None) -> Optional[float]:
    """Safely convert to float, return default if invalid"""
    if pd.isna(value) or value == '' or value is None:
        return default
    try:
        result = float(value)
        return result if not np.isnan(result) and not np.isinf(result) else default
    except (ValueError, TypeError):
        return default


def safe_int(value, default=None) -> Optional[int]:
    """Safely convert to int, return default if invalid"""
    if pd.isna(value) or value == '' or value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value, default=False) -> bool:
    """Safely convert to bool"""
    if pd.isna(value) or value == '' or value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def safe_str(value, default=None) -> Optional[str]:
    """Safely convert to string"""
    if pd.isna(value) or value == '' or value is None:
        return default
    return str(value).strip()


# -----------------------------
# Download File from Supabase
# -----------------------------

def download_file_from_storage(client, bucket_name: str, file_path: str) -> Optional[bytes]:
    """Download file from Supabase Storage"""
    print(f"📥 Downloading file from Supabase Storage...")
    print(f"   Bucket: {bucket_name}")
    print(f"   File: {file_path}")

    try:
        response = client.storage.from_(bucket_name).download(file_path)
        print(f"✓ Successfully downloaded {len(response)} bytes")
        return response
    except Exception as e:
        print(f"❌ Error downloading file: {e}")
        return None


class GeotechnicalETL:
    """ETL Pipeline for Geotechnical Data"""

    def __init__(self, file_bytes: Optional[bytes] = None):
        self.file_bytes = file_bytes
        self.client = None
        self.df = None
        self.municipality_ids: Dict[str, int] = {}
        self.barangay_ids: Dict[Tuple[str, str], int] = {}
        self.borehole_ids: Dict[str, int] = {}
        self.stats = {
            'municipalities_created': 0,
            'municipalities_updated': 0,
            'barangays_created': 0,
            'barangays_updated': 0,
            'boreholes_created': 0,
            'boreholes_updated': 0,
            'soil_layers_created': 0,
            'soil_layers_failed': 0,
        }

    def connect(self):
        """Connect to Supabase using imported client"""
        print("=" * 70)
        print("CONNECTING TO SUPABASE")
        print("=" * 70)

        try:
            from supabase_client import get_supabase_client
            self.client = get_supabase_client()

            if self.client:
                print("✓ Connected to Supabase successfully!")
                return True
            else:
                print("❌ Connection failed")
                return False
        except ImportError:
            print("❌ Error: supabase_client module not found")
            print(
                "   Please ensure supabase_client.py exists with get_supabase_client() function")
            return False

    def load_data(self) -> bool:
        """Load data from Excel bytes"""
        print("\n" + "=" * 70)
        print("LOADING DATA FROM EXCEL")
        print("=" * 70)

        try:
            self.df = pd.read_excel(
                io.BytesIO(self.file_bytes),
                sheet_name='Full_Dataset'
            )

            print(f"✓ Loaded {len(self.df)} records")
            print(f"✓ Columns: {len(self.df.columns)}")

            # Validate required columns
            required_cols = ['Municipality', 'Borehole ID', 'Depth_Layer',
                             'Latitude', 'Longitude']
            missing_cols = [
                col for col in required_cols if col not in self.df.columns]

            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return False

            print(f"\n✓ Sample data:")
            print(
                self.df[['Municipality', 'Borehole ID', 'Depth_Layer']].head(3))

            return True

        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False

    def extract_unique_values(self):
        """Extract unique municipalities, barangays, and boreholes"""
        print("\n" + "=" * 70)
        print("ANALYZING DATA STRUCTURE")
        print("=" * 70)

        municipalities = self.df['Municipality'].unique()
        print(f"\n✓ Found {len(municipalities)} municipalities")
        print(f"  {', '.join(sorted(municipalities))}")

        boreholes = self.df['Borehole ID'].unique()
        print(f"\n✓ Found {len(boreholes)} boreholes")

        depth_layers = self.df['Depth_Layer'].unique()
        print(f"\n✓ Found {len(depth_layers)} depth layers")
        print(f"  {', '.join(sorted(depth_layers))}")

        return municipalities, boreholes, depth_layers

    def upsert_municipalities(self):
        """Insert or update municipalities"""
        print("\n" + "=" * 70)
        print("STEP 1: UPSERTING MUNICIPALITIES")
        print("=" * 70)

        municipalities = sorted(self.df['Municipality'].unique())

        for muni_name in municipalities:
            try:
                # Check if exists
                result = self.client.table('municipalities').select(
                    'id, name'
                ).eq('name', muni_name).execute()

                if result.data and len(result.data) > 0:
                    muni_id = result.data[0]['id']
                    self.municipality_ids[muni_name] = muni_id
                    self.stats['municipalities_updated'] += 1
                    print(f"  ✓ Found: {muni_name} (ID: {muni_id})")
                else:
                    # Insert new
                    insert_data = {
                        'name': muni_name,
                        'description': f'Municipality in Tarlac Province'
                    }
                    result = self.client.table('municipalities').insert(
                        insert_data
                    ).execute()

                    muni_id = result.data[0]['id']
                    self.municipality_ids[muni_name] = muni_id
                    self.stats['municipalities_created'] += 1
                    print(f"  ✓ Created: {muni_name} (ID: {muni_id})")

            except Exception as e:
                print(f"  ❌ Error with {muni_name}: {e}")

        print(f"\n✓ Municipalities - Created: {self.stats['municipalities_created']}, "
              f"Found: {self.stats['municipalities_updated']}")

    def upsert_barangays(self):
        """Insert or update barangays"""
        print("\n" + "=" * 70)
        print("STEP 2: UPSERTING BARANGAYS")
        print("=" * 70)

        for muni_name, muni_id in self.municipality_ids.items():
            barangay_name = f"{muni_name} - Central"

            try:
                # Check if exists
                result = self.client.table('barangays').select(
                    'id, name'
                ).eq('municipality_id', muni_id).eq('name', barangay_name).execute()

                if result.data and len(result.data) > 0:
                    barangay_id = result.data[0]['id']
                    self.barangay_ids[(muni_name, barangay_name)] = barangay_id
                    self.stats['barangays_updated'] += 1
                else:
                    # Insert new
                    insert_data = {
                        'municipality_id': muni_id,
                        'name': barangay_name,
                        'description': f'Default barangay for {muni_name}'
                    }
                    result = self.client.table('barangays').insert(
                        insert_data
                    ).execute()

                    barangay_id = result.data[0]['id']
                    self.barangay_ids[(muni_name, barangay_name)] = barangay_id
                    self.stats['barangays_created'] += 1

            except Exception as e:
                print(f"  ❌ Error with {barangay_name}: {e}")

        print(f"✓ Barangays - Created: {self.stats['barangays_created']}, "
              f"Found: {self.stats['barangays_updated']}")

    def upsert_boreholes(self):
        """Insert or update boreholes"""
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
                print(f"  ⚠️  No barangay found for {borehole_id}, skipping")
                continue

            try:
                # Validate coordinates
                latitude = safe_float(row['Latitude'])
                longitude = safe_float(row['Longitude'])

                if not latitude or not longitude:
                    print(
                        f"  ⚠️  Missing coordinates for {borehole_id}, skipping")
                    continue

                # Check if exists
                result = self.client.table('boreholes').select(
                    'id, borehole_id'
                ).eq('borehole_id', borehole_id).execute()

                borehole_data = {
                    'borehole_id': borehole_id,
                    'barangay_id': barangay_id,
                    'latitude': latitude,
                    'longitude': longitude,
                    'elevation': safe_float(row['Elevation']),
                    'depth_total_m': 15.0,
                    'remarks': f'Data from {municipality}'
                }

                if result.data and len(result.data) > 0:
                    # Update existing
                    db_id = result.data[0]['id']
                    borehole_data['updated_at'] = datetime.now().isoformat()

                    self.client.table('boreholes').update(
                        borehole_data
                    ).eq('id', db_id).execute()

                    self.borehole_ids[borehole_id] = db_id
                    self.stats['boreholes_updated'] += 1
                    print(f"  ✓ Updated: {borehole_id} (ID: {db_id})")
                else:
                    # Insert new
                    result = self.client.table('boreholes').insert(
                        borehole_data
                    ).execute()

                    db_id = result.data[0]['id']
                    self.borehole_ids[borehole_id] = db_id
                    self.stats['boreholes_created'] += 1
                    print(f"  ✓ Created: {borehole_id} (ID: {db_id})")

            except Exception as e:
                print(f"  ❌ Error with {borehole_id}: {e}")

        print(f"\n✓ Boreholes - Created: {self.stats['boreholes_created']}, "
              f"Updated: {self.stats['boreholes_updated']}")

    def map_depth_layer_to_number(self, depth_layer: str) -> int:
        """Map depth layer string to layer number (1-10)"""
        layer_mapping = {
            '0m-1.5m': 1, '1.5m-3.0m': 2, '3.0m-4.5m': 3, '4.5m-6.0m': 4,
            '6.0m-7.5m': 5, '7.5m-9.0m': 6, '9.0m-10.5m': 7, '10.5m-12.0m': 8,
            '12.0m-13.5m': 9, '13.5m-15.0m': 10
        }
        return layer_mapping.get(depth_layer, 1)

    def extract_depth_range(self, depth_layer: str) -> Tuple[float, float]:
        """Extract depth_from_m and depth_to_m from depth layer string"""
        try:
            parts = depth_layer.replace('m', '').split('-')
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
        except:
            pass
        return 0.0, 1.5

    def prepare_soil_layer_data(self, row) -> Optional[Dict]:
        """Prepare soil layer data from DataFrame row"""
        borehole_id_str = safe_str(row['Borehole ID'])
        borehole_db_id = self.borehole_ids.get(borehole_id_str)

        if not borehole_db_id:
            return None

        depth_layer = safe_str(row['Depth_Layer'], '0m-1.5m')
        layer_number = self.map_depth_layer_to_number(depth_layer)
        depth_from, depth_to = self.extract_depth_range(depth_layer)

        # Handle PGA which might be string or float
        pga_value = row.get('Peak Ground Acceleration')
        if isinstance(pga_value, str):
            pga_value = safe_float(pga_value.replace('g', '').strip())
        else:
            pga_value = safe_float(pga_value)

        soil_data = {
            'borehole_id': borehole_db_id,
            'layer_number': layer_number,
            'depth_from_m': depth_from,
            'depth_to_m': depth_to,
            'depth_range': depth_layer,
            'soil_type': safe_str(row.get('Soil/Rock Description')),
            'uscs_symbol': safe_str(row.get('USCS Symbol')),
            'soil_description': safe_str(row.get('Soil/Rock Description')),
            'spt_n_value': safe_float(row.get('SPT N-Value')),
            'spt_n60': safe_float(row.get('SPT N-Value')),
            'spt_n160': safe_float(row.get('Corrected SPT-N Value (N1(60))')),
            'unit_weight': safe_float(row.get('Unit Weight (γ)')),
            'moisture_content': safe_float(row.get('Natural Water Content (ω)')),
            'plasticity_index': safe_float(row.get('Plasticity Index (PI)')),
            'fines_content': safe_float(row.get('Fines Content')),
            'mean_particle_size_d50': safe_float(row.get('Mean Particle Size (D50) (mm)')),
            'groundwater_depth_m': safe_float(row.get('Groundwater Level (m)')),
            'friction_angle': safe_float(row.get('Internal Friction Angle')),
            'cohesion_kpa': safe_float(row.get('Cohesion_kPa')),
            'pga_g': pga_value,
            'csr': safe_float(row.get('Cyclic Stress Ratio (CSR)')),
            'cyclic_strength_ratio': safe_float(row.get('CRR')),
            'liquefaction': safe_bool(row.get('Liquefaction_Potential', 0) == 1),
            'liquefaction_risk_level': 'High' if row.get('Liquefaction_Potential', 0) == 1 else 'Low',
            'settlement_cm': safe_float(row.get('Settlement_m')) * 100 if pd.notna(row.get('Settlement_m')) else None,
            'bearing_capacity_kpa': safe_float(row.get('Ultimate_Bearing_Capacity_kPa')),
            'qa_allowable_kpa': safe_float(row.get('Allowable_Bearing_Capacity_kPa')),
            'effective_overburden_pressure': safe_float(row.get('Effective_Overburden_Stress_kPa')),
            'total_overburden_pressure': safe_float(row.get('Total_Overburden_Stress_kPa')),
            'relative_density_percent': safe_float(row.get('Relative Density')),
            'foundation_width_m': safe_float(row.get('Foundation Width (B)')),
            'foundation_depth_m': safe_float(row.get('Foundation Depth (D)')),
            'elastic_modulus_es': safe_float(row.get('Elastic Modulus (Es) (MN/m²)')),
        }

        # Remove None values to avoid database errors
        soil_data = {k: v for k, v in soil_data.items() if v is not None}

        return soil_data

    def upsert_soil_layers_batch(self, batch_size: int = BATCH_SIZE):
        """Insert soil layers using BATCH operations"""
        print("\n" + "=" * 70)
        print(f"STEP 4: UPSERTING SOIL LAYERS (BATCH SIZE: {batch_size})")
        print("=" * 70)

        total_records = len(self.df)
        processed = 0
        batch_data = []

        print(f"\nProcessing {total_records} soil layer records...")

        for idx, row in self.df.iterrows():
            try:
                soil_data = self.prepare_soil_layer_data(row)

                if not soil_data:
                    self.stats['soil_layers_failed'] += 1
                    continue

                batch_data.append(soil_data)

                # Insert when batch is full
                if len(batch_data) >= batch_size:
                    success = self._insert_batch(batch_data)
                    if success:
                        processed += len(batch_data)
                        self.stats['soil_layers_created'] += len(batch_data)
                        progress = (processed / total_records) * 100
                        print(
                            f"  ✓ Progress: {processed}/{total_records} ({progress:.1f}%)")
                    batch_data = []

            except Exception as e:
                self.stats['soil_layers_failed'] += 1
                if self.stats['soil_layers_failed'] <= 5:
                    print(f"  ⚠️  Error at row {idx}: {e}")

        # Insert remaining batch
        if batch_data:
            success = self._insert_batch(batch_data)
            if success:
                processed += len(batch_data)
                self.stats['soil_layers_created'] += len(batch_data)

        print(f"\n✓ Completed!")
        print(
            f"  Successfully created: {self.stats['soil_layers_created']}/{total_records}")
        print(f"  Failed: {self.stats['soil_layers_failed']}")

    def _insert_batch(self, batch_data: List[Dict]) -> bool:
        """Insert a batch of soil layer records"""
        try:
            self.client.table('soil_layers').insert(batch_data).execute()
            return True
        except Exception as batch_error:
            print(f"  ⚠️  Batch insert failed, trying individual inserts...")
            success_count = 0
            for item in batch_data:
                try:
                    self.client.table('soil_layers').insert(item).execute()
                    success_count += 1
                except Exception as e:
                    self.stats['soil_layers_failed'] += 1
            self.stats['soil_layers_created'] += success_count
            return success_count > 0

    def verify_data(self):
        """Verify data was loaded correctly"""
        print("\n" + "=" * 70)
        print("VERIFYING DATA IN DATABASE")
        print("=" * 70)

        try:
            municipalities = self.client.table('municipalities').select(
                'id', count='exact'
            ).execute()

            barangays = self.client.table('barangays').select(
                'id', count='exact'
            ).execute()

            boreholes = self.client.table('boreholes').select(
                'id', count='exact'
            ).execute()

            soil_layers = self.client.table('soil_layers').select(
                'id', count='exact'
            ).execute()

            print(f"\n✓ Database Summary:")
            print(f"  Municipalities: {municipalities.count}")
            print(f"  Barangays: {barangays.count}")
            print(f"  Boreholes: {boreholes.count}")
            print(f"  Soil Layers: {soil_layers.count}")

            return True

        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

    def print_postgis_instructions(self):
        """Print instructions for updating PostGIS locations"""
        print("\n" + "=" * 70)
        print("⚠️  IMPORTANT: UPDATE POSTGIS LOCATIONS")
        print("=" * 70)
        print(
            "\nYour boreholes were inserted, but PostGIS 'location' field needs updating.")
        print("\n📝 Run this SQL in Supabase SQL Editor:")
        print("-" * 70)
        print("""
UPDATE boreholes 
SET location = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
WHERE location IS NULL 
  AND latitude IS NOT NULL 
  AND longitude IS NOT NULL;
""")
        print("-" * 70)
        print("\nThis populates the location field with PostGIS geometry.")
        print("=" * 70)

    def print_summary(self):
        """Print final summary statistics"""
        print("\n" + "=" * 70)
        print("📊 ETL PIPELINE SUMMARY")
        print("=" * 70)
        print(f"\nMunicipalities:")
        print(f"  Created: {self.stats['municipalities_created']}")
        print(f"  Found Existing: {self.stats['municipalities_updated']}")
        print(f"\nBarangays:")
        print(f"  Created: {self.stats['barangays_created']}")
        print(f"  Found Existing: {self.stats['barangays_updated']}")
        print(f"\nBoreholes:")
        print(f"  Created: {self.stats['boreholes_created']}")
        print(f"  Updated: {self.stats['boreholes_updated']}")
        print(f"\nSoil Layers:")
        print(f"  Created: {self.stats['soil_layers_created']}")
        print(f"  Failed: {self.stats['soil_layers_failed']}")


def main():
    """Main ETL execution"""
    print("\n" + "=" * 70)
    print("🚀 ETL PIPELINE: SUPABASE STORAGE TO DATABASE")
    print("   Geotechnical Data for Liquefaction Prediction")
    print("=" * 70 + "\n")

    # Initialize ETL
    etl = GeotechnicalETL()

    # Step 1: Connect
    if not etl.connect():
        print("\n❌ Cannot proceed without database connection")
        return None

    # Step 2: Download file
    file_bytes = download_file_from_storage(
        etl.client,
        BUCKET_NAME,
        INPUT_FILE_PATH
    )

    if not file_bytes:
        print("\n❌ Failed to download file from storage")
        return None

    etl.file_bytes = file_bytes

    # Step 3: Load data
    if not etl.load_data():
        print("\n❌ Failed to load data from Excel")
        return None

    # Step 4: Analyze data
    etl.extract_unique_values()

    # Step 5: Load to database
    etl.upsert_municipalities()
    etl.upsert_barangays()
    etl.upsert_boreholes()
    etl.upsert_soil_layers_batch()

    # Step 6: Verify
    etl.verify_data()

    # Step 7: Print instructions
    etl.print_postgis_instructions()

    # Step 8: Summary
    etl.print_summary()

    print("\n" + "=" * 70)
    print("✅ ETL PIPELINE COMPLETED!")
    print("=" * 70 + "\n")

    return etl


if __name__ == "__main__":
    etl_system = main()
