"""
Feature Engineering Pipeline for Liquefaction Prediction - CORRECTED VERSION
Tarlac Province Geotechnical Data

This script:
1. Extracts data from Supabase (PostGIS)
2. Engineers features for ML model training
3. Creates training/validation/test datasets with ALL THREE TARGETS
4. Exports feature-engineered data to Supabase Storage

CORRECTION: Now properly exports liquefaction, settlement_cm, AND qa_allowable_kpa

Author: Geotechnical ML Pipeline
Date: 2026-01-28
"""

import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import json
from supabase_client import get_supabase_client

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    print("X scikit-learn not installed!")
    print("Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False


# -----------------------------
# Upload File to Supabase Storage
# -----------------------------

def upload_to_supabase_storage(local_file_path, bucket_name, storage_path):
    """Upload file to Supabase Storage"""
    print(
        f" Uploading {os.path.basename(local_file_path)} to Supabase Storage...")
    client = get_supabase_client()
    if not client:
        return False

    try:
        with open(local_file_path, 'rb') as f:
            file_data = f.read()

        # Determine content type
        if local_file_path.endswith('.csv'):
            content_type = 'text/csv'
        elif local_file_path.endswith('.json'):
            content_type = 'application/json'
        elif local_file_path.endswith('.txt'):
            content_type = 'text/plain'
        else:
            content_type = 'application/octet-stream'

        client.storage.from_(bucket_name).upload(
            storage_path,
            file_data,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )
        print(f"  ✓ Uploaded to {storage_path}")
        return True

    except Exception as e:
        print(f"  X Error uploading file: {e}")
        return False


class GeotechnicalFeatureEngineering:
    """
    Feature Engineering Pipeline for Liquefaction Prediction
    """

    def __init__(self):
        """Initialize Feature Engineering Pipeline"""
        self.client = None
        self.df_raw = None
        self.df_features = None
        self.feature_metadata = {}

    def connect(self) -> bool:
        """Connect to Supabase"""
        print("=" * 80)
        print("CONNECTING TO SUPABASE")
        print("=" * 80)

        self.client = get_supabase_client()
        if not self.client:
            print("X Connection failed")
            return False

        try:
            # Test connection
            self.client.table('municipalities').select('id').limit(1).execute()
            print("✓ Connected to Supabase successfully!")
            return True
        except Exception as e:
            print(f"X Connection test failed: {e}")
            return False

    def extract_data(self) -> bool:
        """Extract all data from Supabase with joins"""
        print("\n" + "=" * 80)
        print("EXTRACTING DATA FROM SUPABASE")
        print("=" * 80)

        try:
            # Query soil layers with borehole and location information
            print("\nFetching soil layers with location data...")

            result = self.client.table('soil_layers').select(
                '''
                *,
                boreholes (
                    borehole_id,
                    latitude,
                    longitude,
                    elevation,
                    depth_total_m,
                    barangays (
                        name,
                        municipalities (
                            name
                        )
                    )
                )
                '''
            ).execute()

            if not result.data:
                print("X No data found in database")
                return False

            # Flatten the nested structure
            data_flat = []
            for row in result.data:
                flat_row = {
                    # Soil layer data
                    'soil_layer_id': row.get('id'),
                    'layer_number': row.get('layer_number'),
                    'depth_from_m': row.get('depth_from_m'),
                    'depth_to_m': row.get('depth_to_m'),
                    'depth_range': row.get('depth_range'),

                    # Soil description
                    'soil_type': row.get('soil_type'),
                    'uscs_symbol': row.get('uscs_symbol'),
                    'soil_description': row.get('soil_description'),

                    # SPT values
                    'spt_n_value': row.get('spt_n_value'),
                    'spt_n60': row.get('spt_n60'),
                    'spt_n160': row.get('spt_n160'),

                    # Physical properties
                    'unit_weight': row.get('unit_weight'),
                    'moisture_content': row.get('moisture_content'),
                    'plasticity_index': row.get('plasticity_index'),
                    'fines_content': row.get('fines_content'),
                    'mean_particle_size_d50': row.get('mean_particle_size_d50'),

                    # Groundwater
                    'groundwater_depth_m': row.get('groundwater_depth_m'),

                    # Strength parameters
                    'friction_angle': row.get('friction_angle'),
                    'cohesion_kpa': row.get('cohesion_kpa'),

                    # Seismic parameters
                    'pga_g': row.get('pga_g'),
                    'csr': row.get('csr'),
                    'cyclic_strength_ratio': row.get('cyclic_strength_ratio'),

                    # Target variables (from data preparation step)
                    'liquefaction': row.get('liquefaction'),
                    'liquefaction_risk_level': row.get('liquefaction_risk_level'),

                    # Settlement and bearing capacity targets
                    'settlement_cm': row.get('settlement_cm'),
                    'bearing_capacity_kpa': row.get('bearing_capacity_kpa'),
                    'qa_allowable_kpa': row.get('qa_allowable_kpa'),

                    # Stress parameters
                    'effective_overburden_pressure': row.get('effective_overburden_pressure'),
                    'total_overburden_pressure': row.get('total_overburden_pressure'),
                    'relative_density_percent': row.get('relative_density_percent'),

                    # Foundation parameters
                    'foundation_width_m': row.get('foundation_width_m'),
                    'foundation_depth_m': row.get('foundation_depth_m'),

                    # Elastic properties
                    'elastic_modulus_es': row.get('elastic_modulus_es'),
                }

                # Add borehole data
                if row.get('boreholes'):
                    bh = row['boreholes']
                    flat_row['borehole_id'] = bh.get('borehole_id')
                    flat_row['latitude'] = bh.get('latitude')
                    flat_row['longitude'] = bh.get('longitude')
                    flat_row['elevation'] = bh.get('elevation')
                    flat_row['depth_total_m'] = bh.get('depth_total_m')

                    # Add location data
                    if bh.get('barangays'):
                        bg = bh['barangays']
                        flat_row['barangay'] = bg.get('name')

                        if bg.get('municipalities'):
                            flat_row['municipality'] = bg['municipalities'].get(
                                'name')

                data_flat.append(flat_row)

            self.df_raw = pd.DataFrame(data_flat)

            print(f"✓ Extracted {len(self.df_raw)} soil layer records")
            print(f"✓ Columns: {len(self.df_raw.columns)}")
            print(f"\nData summary:")
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
            print(f"X Failed to extract data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def engineer_features(self) -> bool:
        """Engineer features for ML model"""
        print("\n" + "=" * 80)
        print("ENGINEERING FEATURES")
        print("=" * 80)

        df = self.df_raw.copy()

        # ====================================================================
        # 1. DEPTH-RELATED FEATURES
        # ====================================================================
        print("\n1. Creating depth-related features...")

        df['depth_mid_m'] = (df['depth_from_m'] + df['depth_to_m']) / 2
        df['depth_thickness_m'] = df['depth_to_m'] - df['depth_from_m']
        df['depth_to_groundwater_m'] = df['groundwater_depth_m'] - df['depth_mid_m']
        df['is_below_groundwater'] = (
            df['depth_mid_m'] > df['groundwater_depth_m']).astype(int)

        # Depth categories
        df['depth_category'] = pd.cut(
            df['depth_mid_m'],
            bins=[0, 3, 6, 9, 12, 15],
            labels=['Shallow (0-3m)', 'Medium (3-6m)', 'Deep (6-9m)',
                    'Very Deep (9-12m)', 'Extreme (12-15m)']
        )

        print(f"  ✓ Created depth features")

        # ====================================================================
        # 2. SPT-RELATED FEATURES
        # ====================================================================
        print("\n2. Creating SPT-related features...")

        # SPT ratios and differences
        df['spt_correction_ratio'] = df['spt_n160'] / \
            (df['spt_n_value'] + 1)  # +1 to avoid division by zero
        df['spt_n_log'] = np.log1p(df['spt_n_value'])
        df['spt_n160_log'] = np.log1p(df['spt_n160'])

        # SPT categories (based on soil consistency)
        df['spt_category'] = pd.cut(
            df['spt_n_value'],
            bins=[0, 4, 10, 30, 50, 100],
            labels=['Very Loose', 'Loose', 'Medium', 'Dense', 'Very Dense']
        )

        # Liquefaction resistance indicator (simplified)
        df['liquefaction_resistance'] = df['spt_n160'] / \
            (df['csr'] + 0.01)  # +0.01 to avoid division by zero

        print(f"  ✓ Created SPT features")

        # ====================================================================
        # 3. SOIL CLASSIFICATION FEATURES
        # ====================================================================
        print("\n3. Creating soil classification features...")

        # Encode USCS symbols
        df['uscs_encoded'] = LabelEncoder().fit_transform(
            df['uscs_symbol'].fillna('Unknown'))

        # Soil type categories (simplified)
        df['soil_type_category'] = df['soil_type'].fillna('Unknown')

        # Fine content categories
        df['fines_category'] = pd.cut(
            df['fines_content'].fillna(0),
            bins=[0, 5, 15, 35, 100],
            labels=['Clean Sand', 'Low Fines', 'Medium Fines', 'High Fines']
        )

        # Particle size categories
        df['particle_size_category'] = pd.cut(
            df['mean_particle_size_d50'].fillna(0),
            bins=[0, 0.075, 0.25, 2.0, 100],
            labels=['Silt/Clay', 'Fine Sand',
                    'Medium Sand', 'Coarse Sand/Gravel']
        )

        print(f"  ✓ Created soil classification features")

        # ====================================================================
        # 4. STRESS AND PRESSURE FEATURES
        # ====================================================================
        print("\n4. Creating stress and pressure features...")

        # Stress ratios
        df['effective_stress_ratio'] = df['effective_overburden_pressure'] / \
            (df['total_overburden_pressure'] + 1)
        df['overburden_pressure_diff'] = df['total_overburden_pressure'] - \
            df['effective_overburden_pressure']

        # Normalized stress (normalize by atmospheric pressure)
        df['normalized_effective_stress'] = df['effective_overburden_pressure'] / 100

        print(f"  ✓ Created stress features")

        # ====================================================================
        # 5. SEISMIC FEATURES
        # ====================================================================
        print("\n5. Creating seismic features...")

        # Factor of Safety against liquefaction
        df['factor_of_safety'] = df['cyclic_strength_ratio'] / \
            (df['csr'] + 0.01)
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

        print(f"  ✓ Created seismic features")

        # ====================================================================
        # 6. BEARING CAPACITY AND SETTLEMENT FEATURES
        # ====================================================================
        print("\n6. Creating bearing capacity and settlement features...")

        # Safety factor for bearing capacity
        df['bearing_capacity_safety_factor'] = df['qa_allowable_kpa'] / \
            (df['bearing_capacity_kpa'] + 1)

        # Settlement categories
        df['settlement_category'] = pd.cut(
            df['settlement_cm'].fillna(0),
            bins=[0, 2.5, 5, 10, 100],
            labels=['Minimal', 'Acceptable', 'Significant', 'Severe']
        )

        # Consolidation indicator
        df['consolidation_potential'] = df['settlement_cm'] * \
            df['fines_content'] / 100

        print(f"  ✓ Created bearing capacity features")

        # ====================================================================
        # 7. GEOGRAPHIC FEATURES
        # ====================================================================
        print("\n7. Creating geographic features...")

        # Encode municipality
        df['municipality_encoded'] = LabelEncoder().fit_transform(
            df['municipality'].fillna('Unknown'))

        # Elevation categories
        df['elevation_category'] = pd.cut(
            df['elevation'].fillna(0),
            bins=[0, 20, 30, 40, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        # Distance from reference point (approximate center of Tarlac)
        ref_lat, ref_lon = 15.48, 120.60
        df['distance_from_center_km'] = np.sqrt(
            (df['latitude'] - ref_lat)**2 + (df['longitude'] - ref_lon)**2
        ) * 111  # Rough conversion to km

        print(f"  ✓ Created geographic features")

        # ====================================================================
        # 8. INTERACTION FEATURES
        # ====================================================================
        print("\n8. Creating interaction features...")

        # Depth × SPT interaction
        df['depth_spt_interaction'] = df['depth_mid_m'] * df['spt_n_value']

        # Fines × Moisture interaction
        df['fines_moisture_interaction'] = df['fines_content'] * \
            df['moisture_content'] / 100

        # CSR × Depth interaction
        df['csr_depth_interaction'] = df['csr'] * df['depth_mid_m']

        # SPT × Fines interaction
        df['spt_fines_interaction'] = df['spt_n_value'] * \
            (100 - df['fines_content']) / 100

        print(f"  ✓ Created interaction features")

        # ====================================================================
        # 9. AGGREGATE FEATURES (Borehole-level)
        # ====================================================================
        print("\n9. Creating borehole aggregate features...")

        # Average SPT per borehole
        bh_spt_avg = df.groupby('borehole_id')['spt_n_value'].transform('mean')
        df['borehole_avg_spt'] = bh_spt_avg

        # SPT variation within borehole
        df['spt_deviation_from_borehole_avg'] = df['spt_n_value'] - \
            df['borehole_avg_spt']

        # Average liquefaction risk per borehole
        bh_liq_avg = df.groupby('borehole_id')[
            'liquefaction'].transform('mean')
        df['borehole_liquefaction_rate'] = bh_liq_avg

        print(f"  ✓ Created borehole aggregate features")

        # ====================================================================
        # 10. DERIVED ENGINEERING FEATURES
        # ====================================================================
        print("\n10. Creating derived engineering features...")

        # Relative density estimation (for sands)
        df['relative_density_estimated'] = np.clip(
            (df['spt_n_value'] - 2) / 50 * 100,
            0, 100
        )

        # Liquefaction susceptibility score (heuristic)
        df['liquefaction_susceptibility_score'] = (
            (df['is_below_groundwater'] * 30) +
            (np.clip(50 - df['spt_n_value'], 0, 50)) +
            (df['fines_content'].fillna(0) < 15).astype(int) * 20
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

        print(f"  ✓ Created derived engineering features")

        # ====================================================================
        # SAVE ENGINEERED FEATURES
        # ====================================================================
        self.df_features = df

        # Feature metadata
        self.feature_metadata = {
            'total_features': len(df.columns),
            'engineered_features': len(df.columns) - len(self.df_raw.columns),
            'original_features': len(self.df_raw.columns),
            'records': len(df),
            'timestamp': datetime.now().isoformat()
        }

        print(f"\n" + "=" * 80)
        print(f"✓ FEATURE ENGINEERING COMPLETE")
        print(f"  - Total features: {self.feature_metadata['total_features']}")
        print(
            f"  - Engineered features: {self.feature_metadata['engineered_features']}")
        print(f"  - Total records: {self.feature_metadata['records']}")
        print("=" * 80)

        return True

    def prepare_training_data(self, test_size=0.2, val_size=0.1, random_state=42) -> Dict:
        """
        Prepare train/validation/test splits with ALL THREE TARGETS

        CORRECTED: Now properly splits liquefaction, settlement_cm, AND qa_allowable_kpa

        Returns:
        Dictionary with train, validation, and test DataFrames
        """
        print("\n" + "=" * 80)
        print("PREPARING TRAINING DATA")
        print("=" * 80)

        if not SKLEARN_AVAILABLE:
            print("X scikit-learn not available")
            return None

        df = self.df_features.copy()

        # Define feature columns (exclude target and ID columns)
        exclude_cols = [
            'soil_layer_id', 'borehole_id', 'barangay', 'municipality',
            'liquefaction', 'liquefaction_risk_level', 'soil_type',
            'uscs_symbol', 'soil_description', 'depth_range', 'depth_category',
            'spt_category', 'fines_category', 'particle_size_category',
            'pga_category', 'settlement_category', 'elevation_category',
            'soil_behavior_type', 'soil_type_category',
            # IMPORTANT: Exclude ALL target variables
            'settlement_cm', 'bearing_capacity_kpa', 'qa_allowable_kpa'
        ]

        # Numeric feature columns
        numeric_features = [
            col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

        print(
            f"\n✓ Selected {len(numeric_features)} numeric features for training")

        # CORRECTED: Include ALL THREE targets
        target_cols = ['liquefaction', 'settlement_cm', 'qa_allowable_kpa']

        # Handle missing values
        df_clean = df[numeric_features + target_cols].copy()
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

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

    def export_data(self, output_dir='ml_data', bucket_name='geotechnical-data') -> bool:
        """Export feature-engineered data for model training and upload to Supabase Storage"""
        print("\n" + "=" * 80)
        print("EXPORTING DATA")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Export full feature-engineered dataset
            features_file = os.path.join(output_dir, 'features_engineered.csv')
            self.df_features.to_csv(features_file, index=False)
            print(f"✓ Exported full dataset: {features_file}")

            # Export feature metadata
            metadata_file = os.path.join(output_dir, 'feature_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.feature_metadata, f, indent=2)
            print(f"✓ Exported metadata: {metadata_file}")

            # Export feature list
            feature_list_file = os.path.join(output_dir, 'feature_list.txt')
            with open(feature_list_file, 'w') as f:
                for col in self.df_features.columns:
                    f.write(f"{col}\n")
            print(f"✓ Exported feature list: {feature_list_file}")

            # Export training splits with ALL THREE TARGETS
            splits = self.prepare_training_data()
            if splits:
                # CORRECTED: Save train with ALL targets
                train_df = pd.concat([
                    splits['X_train'],
                    splits['y_train'].rename('liquefaction'),
                    splits['y_train_settlement'].rename('settlement_cm'),
                    splits['y_train_bearing'].rename('qa_allowable_kpa')
                ], axis=1)
                train_file = os.path.join(output_dir, 'train.csv')
                train_df.to_csv(train_file, index=False)
                print(f"✓ Exported training set: {train_file}")

                # CORRECTED: Save validation with ALL targets
                val_df = pd.concat([
                    splits['X_val'],
                    splits['y_val'].rename('liquefaction'),
                    splits['y_val_settlement'].rename('settlement_cm'),
                    splits['y_val_bearing'].rename('qa_allowable_kpa')
                ], axis=1)
                val_file = os.path.join(output_dir, 'validation.csv')
                val_df.to_csv(val_file, index=False)
                print(f"✓ Exported validation set: {val_file}")

                # CORRECTED: Save test with ALL targets
                test_df = pd.concat([
                    splits['X_test'],
                    splits['y_test'].rename('liquefaction'),
                    splits['y_test_settlement'].rename('settlement_cm'),
                    splits['y_test_bearing'].rename('qa_allowable_kpa')
                ], axis=1)
                test_file = os.path.join(output_dir, 'test.csv')
                test_df.to_csv(test_file, index=False)
                print(f"✓ Exported test set: {test_file}")

            print(f"\n✓ All data exported to: {output_dir}/")

            # Upload to Supabase Storage
            print("\n" + "=" * 80)
            print("UPLOADING TO SUPABASE STORAGE")
            print("=" * 80)

            files_to_upload = [
                (features_file, 'feature_engineering/features_engineered.csv'),
                (metadata_file, 'feature_engineering/feature_metadata.json'),
                (feature_list_file, 'feature_engineering/feature_list.txt'),
                (train_file, 'feature_engineering/train.csv'),
                (val_file, 'feature_engineering/validation.csv'),
                (test_file, 'feature_engineering/test.csv'),
            ]

            for local_path, storage_path in files_to_upload:
                upload_to_supabase_storage(
                    local_path, bucket_name, storage_path)

            print(
                f"\n✓ All files uploaded to Supabase Storage bucket: {bucket_name}")
            return True

        except Exception as e:
            print(f" Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_feature_report(self, output_file='feature_engineering_report.txt', bucket_name='geotechnical-data') -> bool:
        """Generate comprehensive feature engineering report and upload to Supabase Storage"""
        print("\n" + "=" * 80)
        print("GENERATING FEATURE REPORT")
        print("=" * 80)

        try:
            with open(output_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("FEATURE ENGINEERING REPORT\n")
                f.write("Liquefaction Prediction - Tarlac Province\n")
                f.write("=" * 80 + "\n\n")

                f.write(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Data summary
                f.write("DATA SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total records: {len(self.df_features)}\n")
                f.write(f"Total features: {len(self.df_features.columns)}\n")
                f.write(
                    f"Unique boreholes: {self.df_features['borehole_id'].nunique()}\n")
                f.write(
                    f"Unique municipalities: {self.df_features['municipality'].nunique()}\n\n")

                # Target distribution
                f.write("TARGET VARIABLE DISTRIBUTION:\n")
                f.write("-" * 80 + "\n")
                liq_count = self.df_features['liquefaction'].sum()
                non_liq_count = len(self.df_features) - liq_count
                f.write(
                    f"Liquefaction cases: {liq_count} ({liq_count/len(self.df_features)*100:.1f}%)\n")
                f.write(
                    f"Non-liquefaction cases: {non_liq_count} ({non_liq_count/len(self.df_features)*100:.1f}%)\n\n")

                # Target statistics
                f.write("TARGET STATISTICS:\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"Settlement (cm): Mean={self.df_features['settlement_cm'].mean():.2f}, Std={self.df_features['settlement_cm'].std():.2f}\n")
                f.write(
                    f"Bearing Capacity (kPa): Mean={self.df_features['qa_allowable_kpa'].mean():.2f}, Std={self.df_features['qa_allowable_kpa'].std():.2f}\n\n")

                # Feature categories
                f.write("ENGINEERED FEATURE CATEGORIES:\n")
                f.write("-" * 80 + "\n")
                f.write("1. Depth-related features (5)\n")
                f.write("2. SPT-related features (6)\n")
                f.write("3. Soil classification features (4)\n")
                f.write("4. Stress and pressure features (3)\n")
                f.write("5. Seismic features (3)\n")
                f.write("6. Bearing capacity and settlement features (3)\n")
                f.write("7. Geographic features (3)\n")
                f.write("8. Interaction features (4)\n")
                f.write("9. Borehole aggregate features (3)\n")
                f.write("10. Derived engineering features (3)\n\n")

                # Numeric features statistics
                f.write("NUMERIC FEATURES STATISTICS:\n")
                f.write("-" * 80 + "\n")
                numeric_cols = self.df_features.select_dtypes(
                    include=[np.number]).columns
                stats = self.df_features[numeric_cols].describe()
                f.write(stats.to_string())
                f.write("\n\n")

                # Missing values
                f.write("MISSING VALUES:\n")
                f.write("-" * 80 + "\n")
                missing = self.df_features.isnull().sum()
                missing = missing[missing > 0].sort_values(ascending=False)
                if len(missing) > 0:
                    for col, count in missing.items():
                        pct = count / len(self.df_features) * 100
                        f.write(f"{col}: {count} ({pct:.1f}%)\n")
                else:
                    f.write("No missing values\n")
                f.write("\n")

                f.write("=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")

            print(f"✓ Feature report generated: {output_file}")

            # Upload report to Supabase Storage
            upload_to_supabase_storage(
                output_file,
                bucket_name,
                'feature_engineering/feature_engineering_report.txt'
            )

            return True

        except Exception as e:
            print(f" Report generation failed: {e}")
            return False


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING PIPELINE - CORRECTED VERSION")
    print("Liquefaction Prediction - Tarlac Province")
    print("=" * 80 + "\n")

    # Configuration
    BUCKET_NAME = 'geotechnical-data'

    # Initialize pipeline
    pipeline = GeotechnicalFeatureEngineering()

    # Execute pipeline
    if not pipeline.connect():
        return None

    if not pipeline.extract_data():
        return None

    if not pipeline.engineer_features():
        return None

    pipeline.export_data(bucket_name=BUCKET_NAME)
    pipeline.generate_feature_report(bucket_name=BUCKET_NAME)

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nFiles uploaded to Supabase Storage:")
    print(f"  Bucket: {BUCKET_NAME}")
    print("  Path: feature_engineering/")
    print("\nNext steps:")
    print("  1. Review feature_engineering_report.txt in Supabase Storage")
    print("  2. Verify train.csv, validation.csv, test.csv have 3 target columns:")
    print("     - liquefaction")
    print("     - settlement_cm")
    print("     - qa_allowable_kpa")
    print("  3. Start ANN model training")
    print("=" * 80 + "\n")

    return pipeline


if __name__ == "__main__":
    feature_pipeline = main()
