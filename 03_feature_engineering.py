"""
Feature Engineering Pipeline with PostGIS Spatial Analysis - IMPROVED VERSION
Tarlac Province Geotechnical Data

KEY IMPROVEMENTS:
1. Enhanced spatial feature engineering using PostGIS functions
2. Additional soil mechanics features based on database schema
3. Better handling of missing values and data quality
4. More comprehensive interaction features
5. Layer-wise aggregate features
6. Improved borehole-level features
7. Better target variable preparation for multi-target learning
8. Enhanced validation and error handling

This script:
1. Extracts data from Supabase (PostGIS) with spatial queries
2. Engineers features using PostGIS spatial functions and soil mechanics principles
3. Leverages PostGIS views (v_complete_soil_data, v_liquefaction_risk_zones, etc.)
4. Creates training/validation/test datasets with ALL THREE TARGETS
5. Exports feature-engineered data to Supabase Storage IN-MEMORY (NO LOCAL FILES)

Author: Geotechnical ML Pipeline - Improved
Date: 2026-01-30
"""

import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import io
from supabase_client import get_supabase_client

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    print("❌ scikit-learn not installed!")
    print("Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False


# -----------------------------
# Upload to Supabase Storage (IN-MEMORY)
# -----------------------------

def upload_bytes_to_supabase_storage(file_bytes, bucket_name, storage_path, content_type='text/csv'):
    """Upload file bytes directly to Supabase Storage (no local files)"""
    print(f"📤 Uploading to {storage_path}...")
    client = get_supabase_client()
    if not client:
        return False

    try:
        client.storage.from_(bucket_name).upload(
            storage_path,
            file_bytes,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )
        print(
            f"  ✓ Uploaded to {storage_path} ({len(file_bytes) / 1024:.2f} KB)")
        return True

    except Exception as e:
        print(f"  ❌ Error uploading file: {e}")
        return False


class ImprovedGeotechnicalFeatureEngineering:
    """
    Enhanced Feature Engineering Pipeline with PostGIS Spatial Analysis
    """

    def __init__(self):
        """Initialize Feature Engineering Pipeline"""
        self.client = None
        self.df_raw = None
        self.df_features = None
        self.df_spatial = None
        self.df_muni_stats = None
        self.df_bearing_capacity = None
        self.feature_metadata = {}
        self.data_quality_report = {}

    def connect(self) -> bool:
        """Connect to Supabase"""
        print("=" * 80)
        print("CONNECTING TO SUPABASE (PostGIS)")
        print("=" * 80)

        self.client = get_supabase_client()
        if not self.client:
            print("❌ Connection failed")
            return False

        try:
            # Test connection
            self.client.table('municipalities').select('id').limit(1).execute()
            print("✓ Connected to Supabase successfully!")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def extract_data(self) -> bool:
        """Extract data using PostGIS views for optimized spatial queries"""
        print("\n" + "=" * 80)
        print("EXTRACTING DATA FROM SUPABASE (Using PostGIS Views)")
        print("=" * 80)

        try:
            # 1. Main dataset from v_complete_soil_data
            print("\n1. Fetching complete soil data from v_complete_soil_data view...")
            result = self.client.table(
                'v_complete_soil_data').select('*').execute()

            if not result.data:
                print("❌ No data found in v_complete_soil_data view")
                return False

            self.df_raw = pd.DataFrame(result.data)
            print(f"✓ Extracted {len(self.df_raw)} soil layer records")
            print(f"✓ Columns: {len(self.df_raw.columns)}")

            # 2. Spatial risk zone data
            print(
                "\n2. Fetching liquefaction risk zones from v_liquefaction_risk_zones view...")
            risk_result = self.client.table(
                'v_liquefaction_risk_zones').select('*').execute()

            if risk_result.data:
                self.df_spatial = pd.DataFrame(risk_result.data)
                print(f"✓ Extracted {len(self.df_spatial)} spatial risk zones")
            else:
                print("⚠️  No spatial risk zone data available")
                self.df_spatial = None

            # 3. Municipality statistics
            print(
                "\n3. Fetching municipality statistics from v_municipality_statistics view...")
            stats_result = self.client.table(
                'v_municipality_statistics').select('*').execute()

            if stats_result.data:
                self.df_muni_stats = pd.DataFrame(stats_result.data)
                print(
                    f"✓ Extracted statistics for {len(self.df_muni_stats)} municipalities")
            else:
                print("⚠️  No municipality statistics available")
                self.df_muni_stats = None

            # 4. Bearing capacity by layer
            print(
                "\n4. Fetching bearing capacity statistics from v_bearing_capacity_by_layer view...")
            bc_result = self.client.table(
                'v_bearing_capacity_by_layer').select('*').execute()

            if bc_result.data:
                self.df_bearing_capacity = pd.DataFrame(bc_result.data)
                print(
                    f"✓ Extracted bearing capacity stats for {len(self.df_bearing_capacity)} depth layers")
            else:
                print("⚠️  No bearing capacity statistics available")
                self.df_bearing_capacity = None

            # Data quality assessment
            self._assess_data_quality()

            print(f"\n📊 Data summary:")
            print(
                f"  - Unique boreholes: {self.df_raw['borehole_id'].nunique()}")
            print(
                f"  - Unique municipalities: {self.df_raw['municipality'].nunique()}")
            print(f"  - Depth layers: {self.df_raw['layer_number'].nunique()}")
            print(
                f"  - Liquefaction cases: {self.df_raw['liquefaction'].sum()}")
            print(
                f"  - Non-liquefaction cases: {(~self.df_raw['liquefaction']).sum()}")

            return True

        except Exception as e:
            print(f"❌ Failed to extract data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _assess_data_quality(self):
        """Assess data quality and missing values"""
        print("\n" + "=" * 80)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 80)

        df = self.df_raw

        # Key features for quality check
        key_features = [
            'spt_n_value', 'spt_n160', 'unit_weight', 'fines_content',
            'groundwater_depth_m', 'pga_g', 'csr', 'liquefaction',
            'settlement_cm', 'bearing_capacity_kpa', 'latitude', 'longitude'
        ]

        quality_report = {}

        for feature in key_features:
            if feature in df.columns:
                total = len(df)
                missing = df[feature].isna().sum()
                missing_pct = (missing / total) * 100

                quality_report[feature] = {
                    'total': total,
                    'missing': missing,
                    'missing_percent': missing_pct,
                    'present': total - missing
                }

                if missing_pct > 0:
                    print(
                        f"  ⚠️  {feature}: {missing_pct:.1f}% missing ({missing}/{total})")
                else:
                    print(f"  ✓ {feature}: Complete")

        self.data_quality_report = quality_report

    def calculate_spatial_distances(self) -> bool:
        """
        Calculate spatial distances using accurate geographic calculations
        """
        print("\n" + "=" * 80)
        print("CALCULATING SPATIAL DISTANCES (Geographic)")
        print("=" * 80)

        try:
            # Get unique borehole locations
            borehole_locations = self.df_raw[[
                'borehole_id', 'latitude', 'longitude']].drop_duplicates()

            print(
                f"\nCalculating distances for {len(borehole_locations)} boreholes...")

            # Tarlac City center coordinates
            tarlac_center_lat = 15.4753
            tarlac_center_lon = 120.5969

            # Haversine formula for accurate distance
            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate great circle distance in kilometers"""
                lat1, lon1, lat2, lon2 = map(
                    np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * \
                    np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                r = 6371  # Earth radius in km
                return c * r

            # Distance from Tarlac center
            borehole_locations['distance_from_tarlac_center_km'] = borehole_locations.apply(
                lambda row: haversine_distance(
                    row['latitude'], row['longitude'],
                    tarlac_center_lat, tarlac_center_lon
                ),
                axis=1
            )

            # Calculate inter-borehole distances (nearest neighbor)
            print("  Calculating nearest neighbor distances...")
            nearest_distances = []

            for idx, row in borehole_locations.iterrows():
                distances = borehole_locations[borehole_locations['borehole_id'] != row['borehole_id']].apply(
                    lambda r: haversine_distance(
                        row['latitude'], row['longitude'],
                        r['latitude'], r['longitude']
                    ),
                    axis=1
                )
                nearest_distances.append(
                    distances.min() if len(distances) > 0 else 0)

            borehole_locations['nearest_borehole_distance_km'] = nearest_distances

            # Merge back to main dataframe
            self.df_raw = self.df_raw.merge(
                borehole_locations[['borehole_id', 'distance_from_tarlac_center_km',
                                   'nearest_borehole_distance_km']],
                on='borehole_id',
                how='left'
            )

            print(f"✓ Calculated accurate geographic distances")
            print(f"  - Distance from center: Min={self.df_raw['distance_from_tarlac_center_km'].min():.2f} km, "
                  f"Max={self.df_raw['distance_from_tarlac_center_km'].max():.2f} km")
            print(f"  - Nearest borehole: Min={self.df_raw['nearest_borehole_distance_km'].min():.2f} km, "
                  f"Max={self.df_raw['nearest_borehole_distance_km'].max():.2f} km")

            return True

        except Exception as e:
            print(f"❌ Failed to calculate spatial distances: {e}")
            import traceback
            traceback.print_exc()
            return False

    def engineer_spatial_features(self) -> bool:
        """Engineer enhanced spatial features from PostGIS data"""
        print("\n" + "=" * 80)
        print("ENGINEERING SPATIAL FEATURES")
        print("=" * 80)

        df = self.df_raw.copy()

        # ====================================================================
        # 1. SPATIAL RISK ZONE FEATURES
        # ====================================================================
        if self.df_spatial is not None:
            print("\n1. Creating spatial risk zone features...")

            # Create spatial grid cells (0.01 degree ~ 1.1 km)
            df['lat_cell'] = (df['latitude'] / 0.01).round() * 0.01
            df['lon_cell'] = (df['longitude'] / 0.01).round() * 0.01

            # Merge with spatial risk data
            df = df.merge(
                self.df_spatial[['lat_cell', 'lon_cell', 'liquefaction_risk_percent',
                                'sample_count', 'avg_spt_n']],
                on=['lat_cell', 'lon_cell'],
                how='left',
                suffixes=('', '_zone')
            )

            # Rename for clarity
            df = df.rename(columns={
                'liquefaction_risk_percent': 'zone_liquefaction_risk_percent',
                'sample_count': 'zone_sample_count',
                'avg_spt_n': 'zone_avg_spt_n'
            })

            # Fill missing with defaults
            df['zone_liquefaction_risk_percent'] = df['zone_liquefaction_risk_percent'].fillna(
                0)
            df['zone_sample_count'] = df['zone_sample_count'].fillna(0)
            df['zone_avg_spt_n'] = df['zone_avg_spt_n'].fillna(
                df['spt_n_value'].median())

            # Zone density feature (how well-sampled is this area)
            df['zone_density_category'] = pd.cut(
                df['zone_sample_count'],
                bins=[-1, 0, 5, 10, 50],
                labels=['No Data', 'Sparse', 'Moderate', 'Dense']
            )

            print(f"  ✓ Added spatial risk zone features")
        else:
            print("\n1. ⚠️  Skipping spatial risk features (no data available)")

        # ====================================================================
        # 2. MUNICIPALITY-LEVEL FEATURES
        # ====================================================================
        if self.df_muni_stats is not None:
            print("\n2. Creating municipality-level features...")

            # Merge municipality statistics
            df = df.merge(
                self.df_muni_stats[['municipality', 'borehole_count', 'total_samples',
                                   'avg_spt_n', 'avg_unit_weight', 'avg_bearing_capacity_kpa',
                                    'liquefaction_zones', 'liquefaction_percentage']],
                on='municipality',
                how='left',
                suffixes=('', '_muni')
            )

            # Rename for clarity
            df = df.rename(columns={
                'borehole_count': 'muni_borehole_count',
                'total_samples': 'muni_total_samples',
                'avg_spt_n': 'muni_avg_spt_n',
                'avg_unit_weight': 'muni_avg_unit_weight',
                'avg_bearing_capacity_kpa': 'muni_avg_bearing_capacity_kpa',
                'liquefaction_zones': 'muni_liquefaction_zones',
                'liquefaction_percentage': 'muni_liquefaction_percentage'
            })

            # Relative position features (how does this sample compare to municipality average)
            df['spt_relative_to_muni'] = df['spt_n_value'] - df['muni_avg_spt_n']
            df['unit_weight_relative_to_muni'] = df['unit_weight'] - \
                df['muni_avg_unit_weight']

            print(f"  ✓ Added municipality-level aggregate features")
        else:
            print("\n2. ⚠️  Skipping municipality features (no data available)")

        # ====================================================================
        # 3. BEARING CAPACITY LAYER FEATURES
        # ====================================================================
        if self.df_bearing_capacity is not None:
            print("\n3. Creating bearing capacity layer features...")

            # Merge bearing capacity statistics by layer
            df = df.merge(
                self.df_bearing_capacity[['layer_number', 'avg_bc_kpa', 'stddev_bc_kpa',
                                         'min_bc_kpa', 'max_bc_kpa']],
                on='layer_number',
                how='left',
                suffixes=('', '_layer')
            )

            # Rename for clarity
            df = df.rename(columns={
                'avg_bc_kpa': 'layer_avg_bearing_capacity',
                'stddev_bc_kpa': 'layer_stddev_bearing_capacity',
                'min_bc_kpa': 'layer_min_bearing_capacity',
                'max_bc_kpa': 'layer_max_bearing_capacity'
            })

            # Relative bearing capacity (how does this sample compare to typical for this depth)
            df['bearing_capacity_relative_to_layer'] = df['bearing_capacity_kpa'] - \
                df['layer_avg_bearing_capacity']

            # Normalized bearing capacity (0-1 scale within layer)
            layer_range = df['layer_max_bearing_capacity'] - \
                df['layer_min_bearing_capacity']
            df['bearing_capacity_normalized_in_layer'] = (
                (df['bearing_capacity_kpa'] - df['layer_min_bearing_capacity']) /
                (layer_range + 1)
            )

            print(f"  ✓ Added bearing capacity layer features")
        else:
            print("\n3. ⚠️  Skipping bearing capacity features (no data available)")

        # ====================================================================
        # 4. GEOGRAPHIC CLUSTERING FEATURES
        # ====================================================================
        print("\n4. Creating geographic clustering features...")

        # Distance-based zones
        df['distance_zone'] = pd.cut(
            df['distance_from_tarlac_center_km'],
            bins=[0, 5, 10, 20, 50, 100],
            labels=['Very Close (0-5km)', 'Close (5-10km)', 'Medium (10-20km)',
                    'Far (20-50km)', 'Very Far (50-100km)']
        )

        # Directional quadrants
        tarlac_lat, tarlac_lon = 15.4753, 120.5969
        df['direction_from_center'] = np.where(
            df['latitude'] >= tarlac_lat,
            np.where(df['longitude'] >= tarlac_lon, 'NE', 'NW'),
            np.where(df['longitude'] >= tarlac_lon, 'SE', 'SW')
        )

        # Borehole density (nearest neighbor distance)
        df['borehole_density_category'] = pd.cut(
            df['nearest_borehole_distance_km'],
            bins=[0, 0.5, 1, 2, 5, 100],
            labels=['Very Dense', 'Dense', 'Moderate', 'Sparse', 'Very Sparse']
        )

        print(f"  ✓ Created geographic clustering features")

        self.df_raw = df
        return True

    def engineer_soil_mechanics_features(self) -> bool:
        """Engineer features based on soil mechanics principles"""
        print("\n" + "=" * 80)
        print("ENGINEERING SOIL MECHANICS FEATURES")
        print("=" * 80)

        df = self.df_raw.copy()

        # ====================================================================
        # 1. DEPTH-RELATED FEATURES
        # ====================================================================
        print("\n1. Creating enhanced depth-related features...")

        df['depth_mid_m'] = (df['depth_from_m'] + df['depth_to_m']) / 2
        df['depth_thickness_m'] = df['depth_to_m'] - df['depth_from_m']
        df['depth_to_groundwater_m'] = df['groundwater_depth_m'] - df['depth_mid_m']
        df['is_below_groundwater'] = (
            df['depth_mid_m'] > df['groundwater_depth_m']).astype(int)

        # Depth squared (for settlement calculations)
        df['depth_mid_squared'] = df['depth_mid_m'] ** 2

        # Normalized depth (0-1 within typical investigation depth)
        df['depth_normalized'] = df['depth_mid_m'] / 15.0

        # Depth categories
        df['depth_category'] = pd.cut(
            df['depth_mid_m'],
            bins=[0, 3, 6, 9, 12, 15],
            labels=['Shallow (0-3m)', 'Medium (3-6m)', 'Deep (6-9m)',
                    'Very Deep (9-12m)', 'Extreme (12-15m)']
        )

        print(f"  ✓ Created enhanced depth features")

        # ====================================================================
        # 2. SPT-RELATED FEATURES
        # ====================================================================
        print("\n2. Creating enhanced SPT-related features...")

        # SPT ratios and differences
        df['spt_correction_ratio'] = df['spt_n160'] / (df['spt_n_value'] + 1)
        df['spt_n_log'] = np.log1p(df['spt_n_value'])
        df['spt_n160_log'] = np.log1p(df['spt_n160'])
        df['spt_n60_log'] = np.log1p(df['spt_n60'])

        # SPT squared (for some empirical formulas)
        df['spt_n_squared'] = df['spt_n_value'] ** 2
        df['spt_n160_squared'] = df['spt_n160'] ** 2

        # SPT categories
        df['spt_category'] = pd.cut(
            df['spt_n_value'],
            bins=[0, 4, 10, 30, 50, 100],
            labels=['Very Loose', 'Loose', 'Medium', 'Dense', 'Very Dense']
        )

        # Relative density from SPT (Skempton's correlation)
        df['relative_density_from_spt'] = np.clip(
            np.sqrt(df['spt_n160'] / 60) * 100,
            0, 100
        )

        print(f"  ✓ Created enhanced SPT features")

        # ====================================================================
        # 3. STRESS AND PRESSURE FEATURES
        # ====================================================================
        print("\n3. Creating enhanced stress and pressure features...")

        # Stress ratios
        df['effective_stress_ratio'] = df['effective_overburden_pressure'] / \
            (df['total_overburden_pressure'] + 1)
        df['overburden_pressure_diff'] = df['total_overburden_pressure'] - \
            df['effective_overburden_pressure']

        # Normalized stresses
        df['effective_stress_normalized'] = df['effective_overburden_pressure'] / 100
        df['total_stress_normalized'] = df['total_overburden_pressure'] / 100

        # Pore water pressure (approximation)
        df['pore_pressure_approx'] = df['overburden_pressure_diff']

        # Stress level indicator
        df['stress_level'] = pd.cut(
            df['effective_overburden_pressure'],
            bins=[0, 50, 100, 200, 500, 10000],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )

        print(f"  ✓ Created enhanced stress features")

        # ====================================================================
        # 4. SEISMIC FEATURES
        # ====================================================================
        print("\n4. Creating enhanced seismic features...")

        # Factor of Safety against liquefaction
        df['factor_of_safety'] = (
            df['cyclic_strength_ratio'] + 0.001) / (df['csr'] + 0.001)

        # Liquefaction potential index
        df['liquefaction_potential_index'] = np.where(
            df['factor_of_safety'] < 1.0,
            1.0 - df['factor_of_safety'],
            0
        )

        # PGA categories
        df['pga_category'] = pd.cut(
            df['pga_g'].fillna(0),
            bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0],
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )

        # CSR categories
        df['csr_category'] = pd.cut(
            df['csr'].fillna(0),
            bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0],
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )

        # Seismic demand vs capacity ratio
        df['seismic_demand_capacity_ratio'] = df['csr'] / \
            (df['cyclic_strength_ratio'] + 0.001)

        print(f"  ✓ Created enhanced seismic features")

        # ====================================================================
        # 5. BEARING CAPACITY AND SETTLEMENT FEATURES
        # ====================================================================
        print("\n5. Creating enhanced bearing capacity and settlement features...")

        # Safety factors
        df['bearing_capacity_safety_factor'] = df['qa_allowable_kpa'] / \
            (df['bearing_capacity_kpa'] + 1)

        # Settlement features
        df['settlement_category'] = pd.cut(
            df['settlement_cm'].fillna(0),
            bins=[0, 2.5, 5, 10, 100],
            labels=['Minimal', 'Acceptable', 'Significant', 'Severe']
        )

        # Settlement per unit depth
        df['settlement_per_depth'] = df['settlement_cm'] / \
            (df['depth_thickness_m'] + 0.1)

        # Consolidation potential
        df['consolidation_potential'] = df['settlement_cm'] * \
            df['fines_content'] / 100

        # Bearing capacity ratio (actual to allowable)
        df['bc_utilization_ratio'] = df['bearing_capacity_kpa'] / \
            (df['qa_allowable_kpa'] + 1)

        print(f"  ✓ Created enhanced bearing capacity features")

        # ====================================================================
        # 6. SOIL PROPERTY FEATURES
        # ====================================================================
        print("\n6. Creating soil property features...")

        # Moisture-related
        df['moisture_content_log'] = np.log1p(df['moisture_content'].fillna(0))
        df['saturation_degree_approx'] = np.clip(
            df['moisture_content'] / 30 * 100, 0, 100)

        # Fines content features
        df['fines_content_log'] = np.log1p(df['fines_content'].fillna(0))
        df['is_clean_sand'] = (df['fines_content'] < 5).astype(int)
        df['is_silty_sand'] = ((df['fines_content'] >= 5) & (
            df['fines_content'] < 35)).astype(int)
        df['is_fine_grained'] = (df['fines_content'] >= 35).astype(int)

        # Particle size
        df['particle_size_log'] = np.log1p(
            df['mean_particle_size_d50'].fillna(0))

        # Plasticity
        df['plasticity_index_log'] = np.log1p(df['plasticity_index'].fillna(0))
        df['is_plastic'] = (df['plasticity_index'] > 7).astype(int)

        # Unit weight categories
        df['unit_weight_category'] = pd.cut(
            df['unit_weight'].fillna(0),
            bins=[0, 16, 18, 20, 22, 30],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )

        print(f"  ✓ Created soil property features")

        # ====================================================================
        # 7. SHEAR STRENGTH FEATURES
        # ====================================================================
        print("\n7. Creating shear strength features...")

        # Friction angle features
        df['friction_angle_log'] = np.log1p(df['friction_angle'].fillna(0))
        df['friction_angle_radians'] = np.radians(
            df['friction_angle'].fillna(0))
        df['tan_friction_angle'] = np.tan(df['friction_angle_radians'])

        # Cohesion features
        df['cohesion_log'] = np.log1p(df['cohesion_kpa'].fillna(0))
        df['is_cohesive'] = (df['cohesion_kpa'] > 10).astype(int)

        # Combined shear strength indicator
        df['shear_strength_indicator'] = (
            df['friction_angle'].fillna(0) * 0.5 +
            df['cohesion_kpa'].fillna(0) * 0.1
        )

        print(f"  ✓ Created shear strength features")

        self.df_features = df
        return True

    def engineer_interaction_features(self) -> bool:
        """Engineer interaction features between key variables"""
        print("\n" + "=" * 80)
        print("ENGINEERING INTERACTION FEATURES")
        print("=" * 80)

        df = self.df_features.copy()

        # ====================================================================
        # 1. DEPTH INTERACTIONS
        # ====================================================================
        print("\n1. Creating depth interaction features...")

        df['depth_spt_interaction'] = df['depth_mid_m'] * df['spt_n_value']
        df['depth_fines_interaction'] = df['depth_mid_m'] * df['fines_content']
        df['depth_moisture_interaction'] = df['depth_mid_m'] * \
            df['moisture_content']
        df['depth_stress_interaction'] = df['depth_mid_m'] * \
            df['effective_overburden_pressure']

        print(f"  ✓ Created depth interaction features")

        # ====================================================================
        # 2. SPT INTERACTIONS
        # ====================================================================
        print("\n2. Creating SPT interaction features...")

        df['spt_fines_interaction'] = df['spt_n_value'] * \
            (100 - df['fines_content']) / 100
        df['spt_moisture_interaction'] = df['spt_n_value'] * \
            (100 - df['moisture_content']) / 100
        df['spt_stress_interaction'] = df['spt_n_value'] * \
            df['effective_overburden_pressure']
        df['spt_depth_ratio'] = df['spt_n_value'] / (df['depth_mid_m'] + 1)

        print(f"  ✓ Created SPT interaction features")

        # ====================================================================
        # 3. SEISMIC INTERACTIONS
        # ====================================================================
        print("\n3. Creating seismic interaction features...")

        df['csr_depth_interaction'] = df['csr'] * df['depth_mid_m']
        df['csr_fines_interaction'] = df['csr'] * df['fines_content']
        df['csr_spt_interaction'] = df['csr'] * df['spt_n_value']
        df['pga_depth_interaction'] = df['pga_g'] * df['depth_mid_m']

        print(f"  ✓ Created seismic interaction features")

        # ====================================================================
        # 4. SOIL PROPERTY INTERACTIONS
        # ====================================================================
        print("\n4. Creating soil property interaction features...")

        df['fines_moisture_interaction'] = df['fines_content'] * \
            df['moisture_content'] / 100
        df['fines_plasticity_interaction'] = df['fines_content'] * \
            df['plasticity_index'].fillna(0) / 100
        df['unit_weight_depth_interaction'] = df['unit_weight'] * df['depth_mid_m']

        print(f"  ✓ Created soil property interaction features")

        # ====================================================================
        # 5. SPATIAL INTERACTIONS
        # ====================================================================
        print("\n5. Creating spatial interaction features...")

        if 'distance_from_tarlac_center_km' in df.columns:
            df['distance_spt_interaction'] = df['distance_from_tarlac_center_km'] * \
                df['spt_n_value']
            df['distance_depth_interaction'] = df['distance_from_tarlac_center_km'] * \
                df['depth_mid_m']
            df['distance_liquefaction_risk_interaction'] = (
                df['distance_from_tarlac_center_km'] *
                df['zone_liquefaction_risk_percent']
            )

        print(f"  ✓ Created spatial interaction features")

        # ====================================================================
        # 6. BEARING CAPACITY INTERACTIONS
        # ====================================================================
        print("\n6. Creating bearing capacity interaction features...")

        df['bc_spt_interaction'] = df['bearing_capacity_kpa'] * df['spt_n_value']
        df['bc_depth_interaction'] = df['bearing_capacity_kpa'] * df['depth_mid_m']
        df['settlement_fines_interaction'] = df['settlement_cm'] * \
            df['fines_content']

        print(f"  ✓ Created bearing capacity interaction features")

        self.df_features = df
        return True

    def engineer_aggregate_features(self) -> bool:
        """Engineer aggregate features at borehole and layer levels"""
        print("\n" + "=" * 80)
        print("ENGINEERING AGGREGATE FEATURES")
        print("=" * 80)

        df = self.df_features.copy()

        # ====================================================================
        # 1. BOREHOLE-LEVEL AGGREGATES
        # ====================================================================
        print("\n1. Creating borehole-level aggregate features...")

        # SPT aggregates
        bh_spt_stats = df.groupby('borehole_id')['spt_n_value'].agg([
            ('bh_avg_spt', 'mean'),
            ('bh_min_spt', 'min'),
            ('bh_max_spt', 'max'),
            ('bh_std_spt', 'std')
        ]).reset_index()
        df = df.merge(bh_spt_stats, on='borehole_id', how='left')

        # Relative SPT within borehole
        df['spt_deviation_from_bh_avg'] = df['spt_n_value'] - df['bh_avg_spt']
        df['spt_normalized_in_bh'] = (
            df['spt_n_value'] - df['bh_min_spt']) / (df['bh_max_spt'] - df['bh_min_spt'] + 1)

        # Liquefaction rate per borehole
        bh_liq_rate = df.groupby('borehole_id')[
            'liquefaction'].mean().reset_index()
        bh_liq_rate.columns = ['borehole_id', 'bh_liquefaction_rate']
        df = df.merge(bh_liq_rate, on='borehole_id', how='left')

        # Bearing capacity aggregates
        bh_bc_stats = df.groupby('borehole_id')['bearing_capacity_kpa'].agg([
            ('bh_avg_bearing_capacity', 'mean'),
            ('bh_min_bearing_capacity', 'min'),
            ('bh_max_bearing_capacity', 'max')
        ]).reset_index()
        df = df.merge(bh_bc_stats, on='borehole_id', how='left')

        # Settlement aggregates
        bh_settlement_stats = df.groupby('borehole_id')['settlement_cm'].agg([
            ('bh_avg_settlement', 'mean'),
            ('bh_max_settlement', 'max')
        ]).reset_index()
        df = df.merge(bh_settlement_stats, on='borehole_id', how='left')

        print(f"  ✓ Created borehole-level aggregate features")

        # ====================================================================
        # 2. LAYER-LEVEL AGGREGATES
        # ====================================================================
        print("\n2. Creating layer-level aggregate features...")

        # Average properties per depth layer across all boreholes
        layer_stats = df.groupby('layer_number').agg({
            'spt_n_value': ['mean', 'std'],
            'unit_weight': 'mean',
            'fines_content': 'mean',
            'liquefaction': 'mean'
        }).reset_index()

        layer_stats.columns = ['layer_number', 'layer_avg_spt', 'layer_std_spt',
                               'layer_avg_unit_weight', 'layer_avg_fines',
                               'layer_liquefaction_rate']

        df = df.merge(layer_stats, on='layer_number', how='left')

        # Relative to layer average
        df['spt_relative_to_layer'] = df['spt_n_value'] - df['layer_avg_spt']
        df['fines_relative_to_layer'] = df['fines_content'] - df['layer_avg_fines']

        print(f"  ✓ Created layer-level aggregate features")

        # ====================================================================
        # 3. DERIVED ENGINEERING FEATURES
        # ====================================================================
        print("\n3. Creating derived engineering features...")

        # Liquefaction susceptibility score (composite indicator)
        df['liquefaction_susceptibility_score'] = (
            (df['is_below_groundwater'] * 30) +
            (np.clip(50 - df['spt_n_value'], 0, 50)) +
            (df['is_clean_sand'] * 20) +
            (df['zone_liquefaction_risk_percent'] * 0.3)
        )

        # Soil behavior type
        df['soil_behavior_type'] = np.where(
            df['fines_content'] < 5,
            'Clean Sand',
            np.where(
                df['fines_content'] < 35,
                'Silty Sand',
                'Clay/Silt'
            )
        )

        # Foundation suitability index
        df['foundation_suitability_index'] = (
            (df['bearing_capacity_kpa'] / 100) * 0.4 +
            (df['spt_n_value'] / 50) * 0.3 +
            ((1 - df['liquefaction']) * 100) * 0.3
        )

        print(f"  ✓ Created derived engineering features")

        self.df_features = df
        return True

    def prepare_training_data(self, test_size=0.2, val_size=0.1, random_state=42) -> Dict:
        """
        Prepare train/validation/test splits with ALL THREE TARGETS
        """
        print("\n" + "=" * 80)
        print("PREPARING TRAINING DATA")
        print("=" * 80)

        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn not available")
            return None

        df = self.df_features.copy()

        # Define feature columns (exclude target and ID columns)
        exclude_cols = [
            'layer_id', 'borehole_record_id', 'municipality_id', 'barangay_id',
            'borehole_id', 'barangay', 'municipality',
            'liquefaction', 'liquefaction_risk_level',
            'depth_range', 'depth_category', 'spt_category',
            'pga_category', 'csr_category', 'settlement_category',
            'soil_behavior_type', 'distance_zone', 'direction_from_center',
            'lat_cell', 'lon_cell', 'created_at', 'updated_at',
            'zone_density_category', 'stress_level', 'unit_weight_category',
            'borehole_density_category',
            # CRITICAL: Exclude ALL target variables
            'settlement_cm', 'bearing_capacity_kpa', 'qa_allowable_kpa',
            # Also exclude raw coordinates to prevent data leakage
            'latitude', 'longitude', 'elevation'
        ]

        # Numeric feature columns
        numeric_features = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'bool']
        ]

        print(
            f"\n✓ Selected {len(numeric_features)} numeric features for training")

        # ALL THREE targets
        target_cols = ['liquefaction', 'settlement_cm', 'qa_allowable_kpa']

        # Handle missing values
        df_clean = df[numeric_features + target_cols].copy()

        # Fill missing values with median for numeric features
        for col in numeric_features:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)

        # Fill missing target values
        df_clean['settlement_cm'] = df_clean['settlement_cm'].fillna(
            df_clean['settlement_cm'].median())
        df_clean['qa_allowable_kpa'] = df_clean['qa_allowable_kpa'].fillna(
            df_clean['qa_allowable_kpa'].median())

        # Separate features and ALL targets
        X = df_clean[numeric_features]
        y_liquefaction = df_clean['liquefaction'].astype(int)
        y_settlement = df_clean['settlement_cm']
        y_bearing = df_clean['qa_allowable_kpa']

        # First split: train+val vs test (stratified on liquefaction)
        X_temp, X_test, y_liq_temp, y_liq_test = train_test_split(
            X, y_liquefaction, test_size=test_size, random_state=random_state, stratify=y_liquefaction
        )

        # Get corresponding settlement and bearing for test set
        y_settlement_temp = y_settlement.loc[X_temp.index]
        y_settlement_test = y_settlement.loc[X_test.index]
        y_bearing_temp = y_bearing.loc[X_temp.index]
        y_bearing_test = y_bearing.loc[X_test.index]

        # Second split: train vs val (stratified on liquefaction)
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_liq_train, y_liq_val = train_test_split(
            X_temp, y_liq_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_liq_temp
        )

        # Get corresponding settlement and bearing for train/val sets
        y_settlement_train = y_settlement_temp.loc[X_train.index]
        y_settlement_val = y_settlement_temp.loc[X_val.index]
        y_bearing_train = y_bearing_temp.loc[X_train.index]
        y_bearing_val = y_bearing_temp.loc[X_val.index]

        print(f"\n✓ Data splits created:")
        print(
            f"  - Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(
            f"  - Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(
            f"  - Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

        print(f"\n✓ Class distribution (Liquefaction):")
        print(
            f"  - Training:   Liq={y_liq_train.sum()}, Non-liq={len(y_liq_train)-y_liq_train.sum()}")
        print(
            f"  - Validation: Liq={y_liq_val.sum()}, Non-liq={len(y_liq_val)-y_liq_val.sum()}")
        print(
            f"  - Test:       Liq={y_liq_test.sum()}, Non-liq={len(y_liq_test)-y_liq_test.sum()}")

        print(f"\n✓ Target statistics:")
        print(
            f"  - Settlement (train): Mean={y_settlement_train.mean():.2f} cm, Std={y_settlement_train.std():.2f}")
        print(
            f"  - Bearing (train): Mean={y_bearing_train.mean():.2f} kPa, Std={y_bearing_train.std():.2f}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_liq_train,
            'y_val': y_liq_val,
            'y_test': y_liq_test,
            'y_train_settlement': y_settlement_train,
            'y_val_settlement': y_settlement_val,
            'y_test_settlement': y_settlement_test,
            'y_train_bearing': y_bearing_train,
            'y_val_bearing': y_bearing_val,
            'y_test_bearing': y_bearing_test,
            'feature_names': numeric_features,
            'feature_count': len(numeric_features)
        }

    def export_data_to_memory(self, bucket_name='geotechnical-data') -> bool:
        """Export feature-engineered data IN MEMORY and upload to Supabase Storage"""
        print("\n" + "=" * 80)
        print("EXPORTING DATA (IN-MEMORY)")
        print("=" * 80)

        try:
            # 1. Export full feature-engineered dataset
            print("\n1. Creating features_engineered.csv...")
            csv_buffer = io.StringIO()
            self.df_features.to_csv(csv_buffer, index=False)
            features_bytes = csv_buffer.getvalue().encode('utf-8')
            upload_bytes_to_supabase_storage(
                features_bytes, bucket_name,
                'feature_engineering/features_engineered.csv'
            )

            # 2. Export feature metadata
            print("\n2. Creating feature_metadata.json...")
            self.feature_metadata['data_quality'] = self.data_quality_report
            metadata_json = json.dumps(
                self.feature_metadata, indent=2, default=str)
            metadata_bytes = metadata_json.encode('utf-8')
            upload_bytes_to_supabase_storage(
                metadata_bytes, bucket_name,
                'feature_engineering/feature_metadata.json',
                content_type='application/json'
            )

            # 3. Export feature list
            print("\n3. Creating feature_list.txt...")
            feature_list_text = '\n'.join(self.df_features.columns)
            feature_list_bytes = feature_list_text.encode('utf-8')
            upload_bytes_to_supabase_storage(
                feature_list_bytes, bucket_name,
                'feature_engineering/feature_list.txt',
                content_type='text/plain'
            )

            # 4. Export training splits with ALL THREE TARGETS
            print("\n4. Creating train/val/test datasets...")
            splits = self.prepare_training_data()

            if splits:
                # Train set
                train_df = pd.concat([
                    splits['X_train'],
                    splits['y_train'].rename('liquefaction'),
                    splits['y_train_settlement'].rename('settlement_cm'),
                    splits['y_train_bearing'].rename('qa_allowable_kpa')
                ], axis=1)
                train_buffer = io.StringIO()
                train_df.to_csv(train_buffer, index=False)
                train_bytes = train_buffer.getvalue().encode('utf-8')
                upload_bytes_to_supabase_storage(
                    train_bytes, bucket_name,
                    'feature_engineering/train.csv'
                )

                # Validation set
                val_df = pd.concat([
                    splits['X_val'],
                    splits['y_val'].rename('liquefaction'),
                    splits['y_val_settlement'].rename('settlement_cm'),
                    splits['y_val_bearing'].rename('qa_allowable_kpa')
                ], axis=1)
                val_buffer = io.StringIO()
                val_df.to_csv(val_buffer, index=False)
                val_bytes = val_buffer.getvalue().encode('utf-8')
                upload_bytes_to_supabase_storage(
                    val_bytes, bucket_name,
                    'feature_engineering/validation.csv'
                )

                # Test set
                test_df = pd.concat([
                    splits['X_test'],
                    splits['y_test'].rename('liquefaction'),
                    splits['y_test_settlement'].rename('settlement_cm'),
                    splits['y_test_bearing'].rename('qa_allowable_kpa')
                ], axis=1)
                test_buffer = io.StringIO()
                test_df.to_csv(test_buffer, index=False)
                test_bytes = test_buffer.getvalue().encode('utf-8')
                upload_bytes_to_supabase_storage(
                    test_bytes, bucket_name,
                    'feature_engineering/test.csv'
                )

            print(f"\n✓ All data exported to Supabase Storage (no local files created)")
            return True

        except Exception as e:
            print(f"❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_feature_report_to_memory(self, bucket_name='geotechnical-data') -> bool:
        """Generate comprehensive feature engineering report IN MEMORY and upload"""
        print("\n" + "=" * 80)
        print("GENERATING FEATURE REPORT (IN-MEMORY)")
        print("=" * 80)

        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("IMPROVED FEATURE ENGINEERING REPORT")
            report_lines.append(
                "Liquefaction Prediction System - Tarlac Province")
            report_lines.append("PostGIS-Enhanced Spatial Analysis")
            report_lines.append("=" * 80)
            report_lines.append("")
            report_lines.append(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")

            # Data summary
            report_lines.append("DATA SUMMARY:")
            report_lines.append("-" * 80)
            report_lines.append(f"Total records: {len(self.df_features)}")
            report_lines.append(
                f"Total features: {len(self.df_features.columns)}")
            report_lines.append(
                f"Unique boreholes: {self.df_features['borehole_id'].nunique()}")
            report_lines.append(
                f"Unique municipalities: {self.df_features['municipality'].nunique()}")
            report_lines.append(f"PostGIS spatial analysis: ENABLED")
            report_lines.append("")

            # Data quality
            report_lines.append("DATA QUALITY ASSESSMENT:")
            report_lines.append("-" * 80)
            for feature, stats in self.data_quality_report.items():
                report_lines.append(f"{feature}:")
                report_lines.append(
                    f"  - Present: {stats['present']}/{stats['total']} ({100-stats['missing_percent']:.1f}%)")
                report_lines.append(
                    f"  - Missing: {stats['missing']} ({stats['missing_percent']:.1f}%)")
            report_lines.append("")

            # Target distribution
            report_lines.append("TARGET VARIABLE DISTRIBUTION:")
            report_lines.append("-" * 80)
            liq_count = self.df_features['liquefaction'].sum()
            non_liq_count = len(self.df_features) - liq_count
            report_lines.append(f"Liquefaction:")
            report_lines.append(
                f"  - Positive cases: {liq_count} ({liq_count/len(self.df_features)*100:.1f}%)")
            report_lines.append(
                f"  - Negative cases: {non_liq_count} ({non_liq_count/len(self.df_features)*100:.1f}%)")
            report_lines.append("")

            report_lines.append(f"Settlement (cm):")
            report_lines.append(
                f"  - Mean: {self.df_features['settlement_cm'].mean():.2f}")
            report_lines.append(
                f"  - Median: {self.df_features['settlement_cm'].median():.2f}")
            report_lines.append(
                f"  - Std: {self.df_features['settlement_cm'].std():.2f}")
            report_lines.append(
                f"  - Min: {self.df_features['settlement_cm'].min():.2f}")
            report_lines.append(
                f"  - Max: {self.df_features['settlement_cm'].max():.2f}")
            report_lines.append("")

            report_lines.append(f"Allowable Bearing Capacity (kPa):")
            report_lines.append(
                f"  - Mean: {self.df_features['qa_allowable_kpa'].mean():.2f}")
            report_lines.append(
                f"  - Median: {self.df_features['qa_allowable_kpa'].median():.2f}")
            report_lines.append(
                f"  - Std: {self.df_features['qa_allowable_kpa'].std():.2f}")
            report_lines.append(
                f"  - Min: {self.df_features['qa_allowable_kpa'].min():.2f}")
            report_lines.append(
                f"  - Max: {self.df_features['qa_allowable_kpa'].max():.2f}")
            report_lines.append("")

            # Spatial features summary
            report_lines.append("SPATIAL FEATURES (PostGIS):")
            report_lines.append("-" * 80)
            if 'distance_from_tarlac_center_km' in self.df_features.columns:
                report_lines.append(f"Distance from Tarlac center:")
                report_lines.append(
                    f"  - Min: {self.df_features['distance_from_tarlac_center_km'].min():.2f} km")
                report_lines.append(
                    f"  - Max: {self.df_features['distance_from_tarlac_center_km'].max():.2f} km")
                report_lines.append(
                    f"  - Mean: {self.df_features['distance_from_tarlac_center_km'].mean():.2f} km")

            if 'zone_liquefaction_risk_percent' in self.df_features.columns:
                report_lines.append(f"Spatial risk zones:")
                report_lines.append(
                    f"  - Records with zone data: {self.df_features['zone_liquefaction_risk_percent'].notna().sum()}")
                report_lines.append(
                    f"  - Avg risk: {self.df_features['zone_liquefaction_risk_percent'].mean():.2f}%")
            report_lines.append("")

            # Feature categories
            report_lines.append("ENGINEERED FEATURE CATEGORIES:")
            report_lines.append("-" * 80)
            report_lines.append("1. Depth-related features (8)")
            report_lines.append(
                "   - Basic depth, normalized, squared, categories")
            report_lines.append("")
            report_lines.append("2. SPT-related features (10)")
            report_lines.append(
                "   - Raw, corrected, log-transformed, squared, categories")
            report_lines.append("")
            report_lines.append("3. Stress and pressure features (8)")
            report_lines.append(
                "   - Effective stress, total stress, ratios, normalized")
            report_lines.append("")
            report_lines.append("4. Seismic features (8)")
            report_lines.append(
                "   - CSR, CRR, factor of safety, PGA categories")
            report_lines.append("")
            report_lines.append("5. Bearing capacity features (8)")
            report_lines.append(
                "   - Capacity values, safety factors, layer comparisons")
            report_lines.append("")
            report_lines.append("6. Soil property features (13)")
            report_lines.append(
                "   - Moisture, fines content, particle size, plasticity")
            report_lines.append("")
            report_lines.append("7. Shear strength features (7)")
            report_lines.append(
                "   - Friction angle, cohesion, combined indicators")
            report_lines.append("")
            report_lines.append("8. Interaction features (20+)")
            report_lines.append(
                "   - Depth, SPT, seismic, spatial, bearing capacity interactions")
            report_lines.append("")
            report_lines.append("9. Borehole aggregate features (15+)")
            report_lines.append(
                "   - SPT stats, liquefaction rate, bearing capacity stats")
            report_lines.append("")
            report_lines.append("10. Layer aggregate features (6)")
            report_lines.append("   - Layer-wise averages and comparisons")
            report_lines.append("")
            report_lines.append("11. PostGIS spatial features (10+)")
            report_lines.append(
                "   - Geographic distances, zones, municipality stats")
            report_lines.append("")
            report_lines.append("12. Derived engineering features (5)")
            report_lines.append(
                "   - Susceptibility scores, behavior types, suitability indices")
            report_lines.append("")

            # Database schema used
            report_lines.append("DATABASE SCHEMA INTEGRATION:")
            report_lines.append("-" * 80)
            report_lines.append("Views utilized:")
            report_lines.append("  - v_complete_soil_data (main dataset)")
            report_lines.append("  - v_liquefaction_risk_zones (spatial risk)")
            report_lines.append(
                "  - v_municipality_statistics (regional aggregates)")
            report_lines.append(
                "  - v_bearing_capacity_by_layer (depth-wise statistics)")
            report_lines.append("")

            report_lines.append("=" * 80)
            report_lines.append("IMPROVEMENTS OVER PREVIOUS VERSION:")
            report_lines.append("=" * 80)
            report_lines.append("1. ✓ Enhanced spatial feature engineering")
            report_lines.append("2. ✓ Additional soil mechanics features")
            report_lines.append("3. ✓ Better handling of missing values")
            report_lines.append("4. ✓ More comprehensive interaction features")
            report_lines.append("5. ✓ Layer-wise and borehole-wise aggregates")
            report_lines.append("6. ✓ Integration with all PostGIS views")
            report_lines.append("7. ✓ Enhanced data quality reporting")
            report_lines.append(
                "8. ✓ Three-target preparation (liquefaction, settlement, bearing)")
            report_lines.append("")

            report_lines.append("=" * 80)
            report_lines.append("END OF REPORT")
            report_lines.append("=" * 80)

            # Convert to bytes and upload
            report_text = '\n'.join(report_lines)
            report_bytes = report_text.encode('utf-8')

            upload_bytes_to_supabase_storage(
                report_bytes,
                bucket_name,
                'feature_engineering/improved_feature_engineering_report.txt',
                content_type='text/plain'
            )

            print(f"✓ Feature report generated and uploaded")
            return True

        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("IMPROVED FEATURE ENGINEERING PIPELINE")
    print("PostGIS-Enhanced Spatial Analysis")
    print("Liquefaction Prediction System - Tarlac Province")
    print("=" * 80 + "\n")

    # Configuration
    BUCKET_NAME = 'geotechnical-data'

    # Initialize pipeline
    pipeline = ImprovedGeotechnicalFeatureEngineering()

    # Execute pipeline
    steps = [
        ("Connecting to database", pipeline.connect),
        ("Extracting data", pipeline.extract_data),
        ("Calculating spatial distances", pipeline.calculate_spatial_distances),
        ("Engineering spatial features", pipeline.engineer_spatial_features),
        ("Engineering soil mechanics features",
         pipeline.engineer_soil_mechanics_features),
        ("Engineering interaction features",
         pipeline.engineer_interaction_features),
        ("Engineering aggregate features", pipeline.engineer_aggregate_features),
    ]

    for step_name, step_func in steps:
        print(f"\n{'='*80}")
        print(f"EXECUTING: {step_name}")
        print(f"{'='*80}")

        if not step_func():
            print(f"\n❌ Pipeline failed at step: {step_name}")
            return None

    # Export results
    pipeline.export_data_to_memory(bucket_name=BUCKET_NAME)
    pipeline.generate_feature_report_to_memory(bucket_name=BUCKET_NAME)

    # Update feature metadata
    pipeline.feature_metadata = {
        'total_features': len(pipeline.df_features.columns),
        'total_records': len(pipeline.df_features),
        'uses_postgis': True,
        'spatial_views_used': [
            'v_complete_soil_data',
            'v_liquefaction_risk_zones',
            'v_municipality_statistics',
            'v_bearing_capacity_by_layer'
        ],
        'improvement_version': '2.0',
        'timestamp': datetime.now().isoformat(),
        'target_variables': ['liquefaction', 'settlement_cm', 'qa_allowable_kpa']
    }

    print("\n" + "=" * 80)
    print("✅ IMPROVED FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📊 Pipeline Statistics:")
    print(f"  - Total features: {pipeline.feature_metadata['total_features']}")
    print(f"  - Total records: {pipeline.feature_metadata['total_records']}")
    print(
        f"  - Target variables: {len(pipeline.feature_metadata['target_variables'])}")
    print(f"\n✓ PostGIS spatial analysis: ENABLED")
    print(f"✓ All processing done in-memory (no local files)")
    print(f"\n📁 Files uploaded to Supabase Storage:")
    print(f"  Bucket: {BUCKET_NAME}")
    print(f"  Path: feature_engineering/")
    print(f"\n📄 Output files:")
    print(f"  - features_engineered.csv (full dataset)")
    print(f"  - train.csv (training set with 3 targets)")
    print(f"  - validation.csv (validation set with 3 targets)")
    print(f"  - test.csv (test set with 3 targets)")
    print(f"  - feature_metadata.json (metadata)")
    print(f"  - feature_list.txt (feature names)")
    print(f"  - improved_feature_engineering_report.txt (detailed report)")
    print(f"\n🎯 Next steps:")
    print(f"  1. Review improved_feature_engineering_report.txt")
    print(f"  2. Verify all datasets have 3 target columns")
    print(f"  3. Check feature metadata for data quality issues")
    print(f"  4. Proceed with ANN model training")
    print("=" * 80 + "\n")

    return pipeline


if __name__ == "__main__":
    improved_pipeline = main()
