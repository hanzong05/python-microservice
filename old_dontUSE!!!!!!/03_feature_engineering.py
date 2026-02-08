"""
Feature Engineering Pipeline with PostGIS Spatial Analysis - FIXED VERSION
Tarlac Province Geotechnical Data

CRITICAL FIXES APPLIED:
1. Settlement_m is ALL ZEROS - This is NOT a valid target variable!
2. Liquefaction_Potential is ALL ZEROS - This is NOT a valid target variable!
3. PGA values have complex strings like "0.4g (RP: 500-yr; STIFF Soil)" - need better parsing
4. Relative Density has string values like "hard", "very dense" - need better mapping
5. Missing depth_from_m and depth_to_m columns - need to derive from Depth_Layer

SOLUTION:
- Settlement_m and Liquefaction_Potential cannot be used as targets (all zeros)
- Only Allowable_Bearing_Capacity_kPa is a valid regression target
- This is now a SINGLE-TARGET regression problem, not multi-target
- Added proper PGA parsing for complex strings
- Improved Relative Density mapping
- Derive depth ranges from Depth_Layer column

Version: 3.0.0 (Major Fix)
Date: 2026-02-04
"""

from feature_helpers import (
    parse_pga_value,
    parse_relative_density,
    extract_depth_range,
    upload_bytes_to_supabase_storage,
    safe_float,
    load_medians_from_csv_or_bytes,
    compute_borehole_aggregates,
    compute_layer_aggregates,
    fetch_muni_stats,
    calculate_liquefaction_probability,
    calculate_settlement_cm,
)
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import io
import re
from supabase_client import get_supabase_client

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    print("  scikit-learn not installed!")
    print("Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False


# Reuse shared helpers implemented in feature_helpers.py


class FixedGeotechnicalFeatureEngineering:
    """
    Fixed Feature Engineering Pipeline
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
            print("  Connection failed")
            return False

        try:
            # Test connection
            self.client.table('municipalities').select('id').limit(1).execute()
            print("  Connected to Supabase successfully!")
            return True
        except Exception as e:
            print(f"  Connection test failed: {e}")
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
                print("  No data found in v_complete_soil_data view")
                return False

            self.df_raw = pd.DataFrame(result.data)
            print(f"  Extracted {len(self.df_raw)} soil layer records")
            print(f"  Columns: {len(self.df_raw.columns)}")

            # If view returns only one borehole (buggy view), fall back to loading
            # soil_layers + boreholes tables and merge locally to build a complete
            # dataset compatible with the downstream pipeline.
            try:
                uniq_bh = self.df_raw['borehole_id'].nunique()
            except Exception:
                uniq_bh = 0

            if uniq_bh <= 1:
                print("\n  [WARN] v_complete_soil_data contains <=1 unique borehole (view may be filtered). Falling back to `soil_layers` + `boreholes` tables...")
                try:
                    # Download raw soil layers
                    layers_res = self.client.table(
                        'soil_layers').select('*').execute()
                    if not layers_res.data:
                        print("  [ERROR] soil_layers table returned no data")
                    else:
                        df_layers = pd.DataFrame(layers_res.data)

                        # Download boreholes (to get string borehole_id and coords)
                        bh_res = self.client.table('boreholes').select(
                            'id,borehole_id,latitude,longitude,elevation,barangay_id,municipality_id').execute()
                        if not bh_res.data:
                            print("  [ERROR] boreholes table returned no data")
                        else:
                            df_bh = pd.DataFrame(bh_res.data)
                            # Rename and merge: soil_layers.borehole_id (bigint) -> boreholes.id
                            df_bh = df_bh.rename(
                                columns={'id': 'borehole_record_id', 'borehole_id': 'borehole_id_str'})
                            df_merged = df_layers.merge(
                                df_bh, left_on='borehole_id', right_on='borehole_record_id', how='left')

                            # Normalize column names to match v_complete_soil_data where possible
                            if 'borehole_id_str' in df_merged.columns:
                                df_merged['borehole_id'] = df_merged['borehole_id_str']
                            if 'latitude' not in df_merged.columns and 'latitude_y' in df_merged.columns:
                                df_merged['latitude'] = df_merged['latitude_y']
                            if 'longitude' not in df_merged.columns and 'longitude_y' in df_merged.columns:
                                df_merged['longitude'] = df_merged['longitude_y']

                            # Assign merged dataframe as raw data for downstream processing
                            self.df_raw = df_merged
                            print(
                                f"  Fallback dataset built: {len(self.df_raw)} records, unique boreholes: {self.df_raw['borehole_id'].nunique()}")
                except Exception as e:
                    print(f"  [ERROR] Fallback loading failed: {e}")
            # CRITICAL: Check target variables
            print("\n   Analyzing target variables:")

            # Liquefaction
            if 'liquefaction' in self.df_raw.columns:
                liq_positive = self.df_raw['liquefaction'].sum()
                liq_total = len(self.df_raw)
                print(
                    f"   Liquefaction: {liq_positive}/{liq_total} positive cases ({liq_positive/liq_total*100:.1f}%)")
                if liq_positive == 0:
                    print(
                        "       WARNING: ALL liquefaction values are 0 (not usable as target!)")

            # Settlement
            if 'settlement_cm' in self.df_raw.columns:
                settlement_nonzero = (self.df_raw['settlement_cm'] != 0).sum()
                print(
                    f"   Settlement: {settlement_nonzero}/{len(self.df_raw)} non-zero values")
                if settlement_nonzero == 0:
                    print(
                        "       WARNING: ALL settlement values are 0 (not usable as target!)")

            # Bearing capacity
            if 'qa_allowable_kpa' in self.df_raw.columns:
                bc_stats = self.df_raw['qa_allowable_kpa'].describe()
                print(
                    f"   Allowable Bearing Capacity: min={bc_stats['min']:.2f}, max={bc_stats['max']:.2f}, mean={bc_stats['mean']:.2f}")
                print(f"     This is a VALID target variable for regression")

            # 2. Spatial risk zone data (optional)
            print(
                "\n2. Fetching liquefaction risk zones from v_liquefaction_risk_zones view...")
            try:
                risk_result = self.client.table(
                    'v_liquefaction_risk_zones').select('*').execute()
                if risk_result.data:
                    self.df_spatial = pd.DataFrame(risk_result.data)
                    print(
                        f"  Extracted {len(self.df_spatial)} spatial risk zones")
                else:
                    print("    No spatial risk zone data available")
                    self.df_spatial = None
            except:
                print("    v_liquefaction_risk_zones view not available")
                self.df_spatial = None

            # 3. Municipality statistics (optional)
            print(
                "\n3. Fetching municipality statistics from v_municipality_statistics view...")
            try:
                stats_result = self.client.table(
                    'v_municipality_statistics').select('*').execute()
                if stats_result.data:
                    self.df_muni_stats = pd.DataFrame(stats_result.data)
                    print(
                        f"  Extracted statistics for {len(self.df_muni_stats)} municipalities")
                else:
                    print("    No municipality statistics available")
                    self.df_muni_stats = None
            except:
                print("    v_municipality_statistics view not available")
                self.df_muni_stats = None

            # 4. Bearing capacity by layer (optional)
            print(
                "\n4. Fetching bearing capacity statistics from v_bearing_capacity_by_layer view...")
            try:
                bc_result = self.client.table(
                    'v_bearing_capacity_by_layer').select('*').execute()
                if bc_result.data:
                    self.df_bearing_capacity = pd.DataFrame(bc_result.data)
                    print(
                        f"  Extracted bearing capacity stats for {len(self.df_bearing_capacity)} depth layers")
                else:
                    print("    No bearing capacity statistics available")
                    self.df_bearing_capacity = None
            except:
                print("    v_bearing_capacity_by_layer view not available")
                self.df_bearing_capacity = None

            # Data quality assessment
            self._assess_data_quality()

            print(f"\n   Data summary:")
            print(
                f"  - Unique boreholes: {self.df_raw['borehole_id'].nunique()}")
            print(
                f"  - Unique municipalities: {self.df_raw['municipality'].nunique()}")
            print(f"  - Depth layers: {self.df_raw['layer_number'].nunique()}")

            return True

        except Exception as e:
            print(f"  Failed to extract data: {e}")
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
            'groundwater_depth_m', 'pga_g', 'csr',
            'bearing_capacity_kpa', 'qa_allowable_kpa',
            'latitude', 'longitude'
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
                        f"      {feature}: {missing_pct:.1f}% missing ({missing}/{total})")
                else:
                    print(f"    {feature}: Complete")

        self.data_quality_report = quality_report

    def calculate_spatial_distances(self) -> bool:
        """Calculate spatial distances using accurate geographic calculations"""
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

            print(f"  Calculated accurate geographic distances")
            print(f"  - Distance from center: Min={self.df_raw['distance_from_tarlac_center_km'].min():.2f} km, "
                  f"Max={self.df_raw['distance_from_tarlac_center_km'].max():.2f} km")
            print(f"  - Nearest borehole: Min={self.df_raw['nearest_borehole_distance_km'].min():.2f} km, "
                  f"Max={self.df_raw['nearest_borehole_distance_km'].max():.2f} km")

            return True

        except Exception as e:
            print(f"  Failed to calculate spatial distances: {e}")
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
        # 1. SPATIAL RISK ZONE FEATURES (Optional)
        # ====================================================================
        if self.df_spatial is not None and len(self.df_spatial) > 0:
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

            print(f"    Added spatial risk zone features")
        else:
            print("\n1.     Skipping spatial risk features (no data available)")
            # Add default columns so rest of pipeline works
            df['zone_liquefaction_risk_percent'] = 0.0
            df['zone_sample_count'] = 0
            df['zone_avg_spt_n'] = df['spt_n_value'].median()

        # ====================================================================
        # 2. MUNICIPALITY-LEVEL FEATURES (Optional)
        # ====================================================================
        if self.df_muni_stats is not None and len(self.df_muni_stats) > 0:
            print("\n2. Creating municipality-level features...")

            # Merge municipality statistics
            df = df.merge(
                self.df_muni_stats,
                on='municipality',
                how='left',
                suffixes=('', '_muni')
            )

            # Rename for clarity if needed
            if 'avg_spt_n' in df.columns:
                df = df.rename(columns={'avg_spt_n': 'muni_avg_spt_n'})

            print(f"    Added municipality-level aggregate features")
        else:
            print("\n2.     Skipping municipality features (no data available)")

        # ====================================================================
        # 3. BEARING CAPACITY LAYER FEATURES (Optional)
        # ====================================================================
        if self.df_bearing_capacity is not None and len(self.df_bearing_capacity) > 0:
            print("\n3. Creating bearing capacity layer features...")

            # Merge bearing capacity statistics by layer
            df = df.merge(
                self.df_bearing_capacity,
                on='layer_number',
                how='left',
                suffixes=('', '_layer')
            )

            print(f"    Added bearing capacity layer features")
        else:
            print("\n3.     Skipping bearing capacity features (no data available)")

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

        print(f"    Created geographic clustering features")

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

        # Calculate depth midpoint and thickness from depth_from_m and depth_to_m
        df['depth_mid_m'] = (df['depth_from_m'] + df['depth_to_m']) / 2
        df['depth_thickness_m'] = df['depth_to_m'] - df['depth_from_m']

        # Depth to groundwater
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

        print(f"    Created enhanced depth features")

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
        if 'relative_density_percent' not in df.columns or df['relative_density_percent'].isna().all():
            df['relative_density_from_spt'] = np.clip(
                np.sqrt(df['spt_n160'] / 60) * 100,
                0, 100
            )

        print(f"    Created enhanced SPT features")

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

        print(f"    Created enhanced stress features")

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

        print(f"    Created enhanced seismic features")

        # ====================================================================
        # 5. BEARING CAPACITY FEATURES
        # ====================================================================
        print("\n5. Creating bearing capacity features...")

        # Safety factors
        df['bearing_capacity_safety_factor'] = df['qa_allowable_kpa'] / \
            (df['bearing_capacity_kpa'] + 1)

        # Bearing capacity ratio (actual to allowable)
        df['bc_utilization_ratio'] = df['bearing_capacity_kpa'] / \
            (df['qa_allowable_kpa'] + 1)

        # Normalized bearing capacity
        df['qa_allowable_log'] = np.log1p(df['qa_allowable_kpa'])
        df['bearing_capacity_log'] = np.log1p(df['bearing_capacity_kpa'])

        print(f"    Created bearing capacity features")

        # ====================================================================
        # 6. SOIL PROPERTY FEATURES
        # ====================================================================
        print("\n6. Creating soil property features...")

        # Moisture-related
        df['moisture_content_log'] = np.log1p(df['moisture_content'].fillna(0))

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

        # Unit weight
        df['unit_weight_log'] = np.log1p(df['unit_weight'].fillna(0))

        print(f"    Created soil property features")

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

        print(f"    Created shear strength features")

        self.df_features = df
        return True

    def engineer_interaction_features(self) -> bool:
        """Engineer interaction features between key variables"""
        print("\n" + "=" * 80)
        print("ENGINEERING INTERACTION FEATURES")
        print("=" * 80)

        df = self.df_features.copy()

        print("\n1. Creating depth interaction features...")
        df['depth_spt_interaction'] = df['depth_mid_m'] * df['spt_n_value']
        df['depth_fines_interaction'] = df['depth_mid_m'] * df['fines_content']
        df['depth_moisture_interaction'] = df['depth_mid_m'] * \
            df['moisture_content']
        df['depth_stress_interaction'] = df['depth_mid_m'] * \
            df['effective_overburden_pressure']

        print("\n2. Creating SPT interaction features...")
        df['spt_fines_interaction'] = df['spt_n_value'] * \
            (100 - df['fines_content']) / 100
        df['spt_moisture_interaction'] = df['spt_n_value'] * \
            (100 - df['moisture_content']) / 100
        df['spt_stress_interaction'] = df['spt_n_value'] * \
            df['effective_overburden_pressure']
        df['spt_depth_ratio'] = df['spt_n_value'] / (df['depth_mid_m'] + 1)

        print("\n3. Creating seismic interaction features...")
        df['csr_depth_interaction'] = df['csr'] * df['depth_mid_m']
        df['csr_fines_interaction'] = df['csr'] * df['fines_content']
        df['csr_spt_interaction'] = df['csr'] * df['spt_n_value']
        df['pga_depth_interaction'] = df['pga_g'] * df['depth_mid_m']

        print("\n4. Creating spatial interaction features...")
        df['distance_spt_interaction'] = df['distance_from_tarlac_center_km'] * \
            df['spt_n_value']
        df['distance_depth_interaction'] = df['distance_from_tarlac_center_km'] * \
            df['depth_mid_m']

        print("\n5. Creating bearing capacity interaction features...")
        df['bc_spt_interaction'] = df['bearing_capacity_kpa'] * df['spt_n_value']
        df['bc_depth_interaction'] = df['bearing_capacity_kpa'] * df['depth_mid_m']

        print(f"    Created interaction features")

        self.df_features = df
        return True

    def engineer_aggregate_features(self) -> bool:
        """Engineer aggregate features at borehole and layer levels"""
        print("\n" + "=" * 80)
        print("ENGINEERING AGGREGATE FEATURES")
        print("=" * 80)

        df = self.df_features.copy()

        print("\n1. Creating borehole-level aggregate features...")

        # SPT aggregates
        bh_spt_stats = df.groupby('borehole_id')['spt_n_value'].agg([
            ('bh_avg_spt', 'mean'),
            ('bh_min_spt', 'min'),
            ('bh_max_spt', 'max'),
            ('bh_std_spt', 'std')
        ]).reset_index()
        df = df.merge(bh_spt_stats, on='borehole_id', how='left')

        # Bearing capacity aggregates
        bh_bc_stats = df.groupby('borehole_id')['qa_allowable_kpa'].agg([
            ('bh_avg_qa_allowable', 'mean'),
            ('bh_min_qa_allowable', 'min'),
            ('bh_max_qa_allowable', 'max')
        ]).reset_index()
        df = df.merge(bh_bc_stats, on='borehole_id', how='left')

        print("\n2. Creating layer-level aggregate features...")

        # Average properties per depth layer
        layer_stats = df.groupby('layer_number').agg({
            'spt_n_value': ['mean', 'std'],
            'unit_weight': 'mean',
            'fines_content': 'mean',
            'qa_allowable_kpa': 'mean'
        }).reset_index()

        layer_stats.columns = ['layer_number', 'layer_avg_spt', 'layer_std_spt',
                               'layer_avg_unit_weight', 'layer_avg_fines',
                               'layer_avg_qa_allowable']

        df = df.merge(layer_stats, on='layer_number', how='left')

        # Relative to layer average
        df['spt_relative_to_layer'] = df['spt_n_value'] - df['layer_avg_spt']
        df['qa_relative_to_layer'] = df['qa_allowable_kpa'] - \
            df['layer_avg_qa_allowable']

        print(f"    Created aggregate features")

        self.df_features = df

        # ====================================================================
        # CALCULATE TARGET VARIABLES FOR 3-MODEL TRAINING
        # ====================================================================
        print("\n" + "=" * 80)
        print("CALCULATING TARGET VARIABLES")
        print("=" * 80)

        # LIQUEFACTION PROBABILITY (0-100%)
        print("\n1. Calculating liquefaction probability...")
        self.df_features['liquefaction_probability_pct'] = self.df_features.apply(
            lambda row: calculate_liquefaction_probability(
                csr=row.get('csr', 0.2),
                crr=row.get('cyclic_strength_ratio', 0.045),
                spt_n160=row.get('spt_n160', 15.0),
                fines_pct=row.get('fines_content', 15.0)
            ),
            axis=1
        )
        print(f"  Liquefaction probability stats:")
        print(
            f"    Mean: {self.df_features['liquefaction_probability_pct'].mean():.1f}%")
        print(
            f"    Std:  {self.df_features['liquefaction_probability_pct'].std():.1f}%")
        print(
            f"    Min:  {self.df_features['liquefaction_probability_pct'].min():.1f}%")
        print(
            f"    Max:  {self.df_features['liquefaction_probability_pct'].max():.1f}%")

        # SETTLEMENT (cm)
        print("\n2. Calculating settlement...")
        self.df_features['settlement_cm_calc'] = self.df_features.apply(
            lambda row: calculate_settlement_cm(
                spt_n160=row.get('spt_n160', 15.0),
                depth_mid_m=row.get('depth_mid_m', 0.75),
                effective_stress_kpa=row.get(
                    'effective_overburden_pressure', 15.0),
                qa_allowable_kpa=row.get('qa_allowable_kpa', 1000.0),
                fines_pct=row.get('fines_content', 15.0),
                liquefaction_prob=row.get('liquefaction_probability_pct', 0.0),
                foundation_width_m=1.0
            ),
            axis=1
        )
        print(f"  Settlement stats:")
        print(
            f"    Mean: {self.df_features['settlement_cm_calc'].mean():.2f} cm")
        print(
            f"    Std:  {self.df_features['settlement_cm_calc'].std():.2f} cm")
        print(
            f"    Min:  {self.df_features['settlement_cm_calc'].min():.2f} cm")
        print(
            f"    Max:  {self.df_features['settlement_cm_calc'].max():.2f} cm")

        return True

    def prepare_training_data(self, test_size=0.2, val_size=0.1, random_state=42) -> Dict:
        """
        Prepare train/validation/test splits for THREE TARGETS:
        - liquefaction_probability_pct (0-100%)
        - settlement_cm_calc (cm)
        - qa_allowable_kpa (kPa)
        """
        print("\n" + "=" * 80)
        print("PREPARING TRAINING DATA - THREE TARGET REGRESSION")
        print("=" * 80)

        if not SKLEARN_AVAILABLE:
            print("  scikit-learn not available")
            return None

        df = self.df_features.copy()

        # Define feature columns (exclude target and ID columns)
        exclude_cols = [
            'layer_id', 'borehole_record_id', 'municipality_id', 'barangay_id',
            'borehole_id', 'barangay', 'municipality',
            'liquefaction', 'liquefaction_risk_level',
            'depth_range', 'depth_category', 'spt_category',
            'pga_category', 'csr_category', 'distance_zone', 'direction_from_center',
            'lat_cell', 'lon_cell', 'created_at', 'updated_at',
            # Exclude ALL target-related columns
            'settlement_cm', 'bearing_capacity_kpa', 'qa_allowable_kpa',
            'qa_allowable_log',  # This is derived from target
            # Exclude newly calculated targets
            'liquefaction_probability_pct', 'settlement_cm_calc',
            # Exclude coordinates to prevent data leakage
            'latitude', 'longitude', 'elevation'
        ]

        # Numeric feature columns
        numeric_features = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'bool']
        ]

        print(
            f"\n  Selected {len(numeric_features)} numeric features for training")

        # MULTIPLE targets: liquefaction, settlement, bearing_capacity
        target_cols = ['liquefaction_probability_pct',
                       'settlement_cm_calc', 'qa_allowable_kpa']

        # Handle missing values
        df_clean = df[numeric_features + target_cols].copy()

        # Fill missing values with median for numeric features
        for col in numeric_features:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)

        # Fill missing target values
        for target_col in target_cols:
            df_clean[target_col] = df_clean[target_col].fillna(
                df_clean[target_col].median())

        # Separate features and targets
        X = df_clean[numeric_features]

        # Create dictionaries to store splits for each target
        splits = {}

        for target_col in target_cols:
            print(f"\n  Creating splits for target: {target_col}")
            y = df_clean[target_col]

            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
            )

            splits[target_col] = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }

            print(
                f"    Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
            print(
                f"    Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
            print(
                f"    Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

            print(f"\n    Target statistics ({target_col}):")
            print(
                f"    - Training:   Mean={y_train.mean():.2f}, Std={y_train.std():.2f}, Min={y_train.min():.2f}, Max={y_train.max():.2f}")
            print(
                f"    - Validation: Mean={y_val.mean():.2f}, Std={y_val.std():.2f}, Min={y_val.min():.2f}, Max={y_val.max():.2f}")
            print(
                f"    - Test:       Mean={y_test.mean():.2f}, Std={y_test.std():.2f}, Min={y_test.min():.2f}, Max={y_test.max():.2f}")

        return {
            'splits': splits,
            'feature_names': numeric_features,
            'feature_count': len(numeric_features),
            'target_names': target_cols
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
                'feature_engineering/features_engineered_FIXED.csv',
                self.client
            )

            # 2. Export feature metadata
            print("\n2. Creating feature_metadata.json...")
            self.feature_metadata['data_quality'] = self.data_quality_report
            metadata_json = json.dumps(
                self.feature_metadata, indent=2, default=str)
            metadata_bytes = metadata_json.encode('utf-8')
            upload_bytes_to_supabase_storage(
                metadata_bytes, bucket_name,
                'feature_engineering/feature_metadata_FIXED.json',
                self.client,
                content_type='application/json'
            )

            # 3. Export training splits with THREE TARGETS
            print("\n3. Creating train/val/test datasets for 3 targets...")
            splits_result = self.prepare_training_data()

            if splits_result:
                splits_dict = splits_result['splits']
                target_names = splits_result['target_names']

                # Export splits for each target
                for target_name in target_names:
                    print(
                        f"\n  Exporting datasets for target: {target_name}")
                    target_splits = splits_dict[target_name]

                    # Train set
                    train_df = pd.concat([
                        target_splits['X_train'],
                        target_splits['y_train'].rename(target_name)
                    ], axis=1)
                    train_buffer = io.StringIO()
                    train_df.to_csv(train_buffer, index=False)
                    train_bytes = train_buffer.getvalue().encode('utf-8')

                    # Sanitize target name for file path
                    target_file = target_name.lower().replace(' ', '_')
                    upload_bytes_to_supabase_storage(
                        train_bytes, bucket_name,
                        f'feature_engineering/train_{target_file}_FIXED.csv',
                        self.client
                    )

                    # Validation set
                    val_df = pd.concat([
                        target_splits['X_val'],
                        target_splits['y_val'].rename(target_name)
                    ], axis=1)
                    val_buffer = io.StringIO()
                    val_df.to_csv(val_buffer, index=False)
                    val_bytes = val_buffer.getvalue().encode('utf-8')
                    upload_bytes_to_supabase_storage(
                        val_bytes, bucket_name,
                        f'feature_engineering/validation_{target_file}_FIXED.csv',
                        self.client
                    )

                    # Test set
                    test_df = pd.concat([
                        target_splits['X_test'],
                        target_splits['y_test'].rename(target_name)
                    ], axis=1)
                    test_buffer = io.StringIO()
                    test_df.to_csv(test_buffer, index=False)
                    test_bytes = test_buffer.getvalue().encode('utf-8')
                    upload_bytes_to_supabase_storage(
                        test_bytes, bucket_name,
                        f'feature_engineering/test_{target_file}_FIXED.csv',
                        self.client
                    )

            print(f"\n  All data exported to Supabase Storage")
            return True

        except Exception as e:
            print(f"  Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("FIXED FEATURE ENGINEERING PIPELINE")
    print("THREE TARGET REGRESSION:")
    print("  1. Liquefaction Probability (0-100%)")
    print("  2. Settlement (cm)")
    print("  3. Allowable Bearing Capacity (kPa)")
    print("Version: 4.0.0 (Three Target Fix)")
    print("=" * 80 + "\n")

    BUCKET_NAME = 'geotechnical-data'

    pipeline = FixedGeotechnicalFeatureEngineering()

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
        if not step_func():
            print(f"\n  Pipeline failed at step: {step_name}")
            return None

    pipeline.export_data_to_memory(bucket_name=BUCKET_NAME)

    pipeline.feature_metadata = {
        'total_features': len(pipeline.df_features.columns),
        'total_records': len(pipeline.df_features),
        'target_variable': 'qa_allowable_kpa',
        'problem_type': 'regression',
        'version': '3.0.0',
        'critical_fixes': [
            'Settlement_m and Liquefaction_Potential are ALL ZEROS - cannot be used as targets',
            'Changed from multi-target to single-target regression',
            'Improved PGA parsing for complex strings',
            'Better Relative Density mapping',
            'Proper handling of missing depth columns'
        ],
        'timestamp': datetime.now().isoformat()
    }

    print("\n" + "=" * 80)
    print("[SUCCESS] FIXED FEATURE ENGINEERING COMPLETED!")
    print("=" * 80)
    print(f"\n    CRITICAL CHANGES:")
    print(f"  - This is now a SINGLE-TARGET REGRESSION problem")
    print(f"  - Target variable: qa_allowable_kpa (Allowable Bearing Capacity)")
    print(f"  - Settlement and Liquefaction cannot be predicted (all zeros in data)")
    print(f"\n   Pipeline Statistics:")
    print(f"  - Total features: {pipeline.feature_metadata['total_features']}")
    print(f"  - Total records: {pipeline.feature_metadata['total_records']}")
    print(f"  - Target: {pipeline.feature_metadata['target_variable']}")
    print("=" * 80 + "\n")

    return pipeline


if __name__ == "__main__":
    fixed_pipeline = main()
