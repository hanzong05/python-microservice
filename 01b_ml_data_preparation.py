"""
Data Preparation for Machine Learning - IN-MEMORY VERSION
Liquefaction, Settlement, and Bearing Capacity Prediction
Tarlac Province, Philippines

- Downloads Cleaned_Data.xlsx from Supabase Storage
- Processes data for ML IN MEMORY
- Archives existing ML_Ready_Data.xlsx to old_ml_data/<timestamp>
- Uploads new ML_Ready_Data.xlsx back to Supabase Storage ONLY
- NO LOCAL FILES CREATED
"""

import pandas as pd
import numpy as np
import warnings
from openpyxl import Workbook
import io
from datetime import datetime
from supabase_client import get_supabase_client

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
        print(f" Error downloading file: {e}")
        return None


# -----------------------------
# Upload Directly from Memory
# -----------------------------

def upload_to_supabase_storage(excel_bytes, bucket_name, storage_path):
    """Upload file to Supabase Storage directly from memory and archive existing file"""
    print(f"\n Uploading to Supabase Storage...")
    client = get_supabase_client()
    if not client:
        return False

    try:
        # Check if file already exists and archive it
        try:
            existing_file = client.storage.from_(
                bucket_name).download(storage_path)
            if existing_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = f"old_ml_data/old_ml_ready_data_{timestamp}.xlsx"
                client.storage.from_(bucket_name).move(
                    storage_path, archive_path)
                print(f"✓ Existing file archived as {archive_path}")
        except Exception:
            # File may not exist, which is fine
            pass

        # Upload from memory
        client.storage.from_(bucket_name).upload(
            storage_path,
            excel_bytes,
            file_options={
                "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "upsert": "true"
            }
        )
        print(f"✓ File uploaded successfully to {storage_path}")
        print(f"✓ File size: {len(excel_bytes) / 1024:.2f} KB")
        return True

    except Exception as e:
        print(f" Error uploading file: {e}")
        return False


# -----------------------------
# Data Preparation Class
# -----------------------------

class GeotechnicalDataPrep:
    """Prepare geotechnical data for machine learning"""

    def __init__(self, file_bytes):
        self.file_bytes = file_bytes
        self.df = None

    def load_data(self):
        """Load the cleaned data from bytes"""
        print("=" * 70)
        print("STEP 1: LOADING DATA")
        print("=" * 70)

        try:
            # Read Excel from bytes
            self.df = pd.read_excel(io.BytesIO(
                self.file_bytes), sheet_name='Cleaned_Data')
            print(f"✓ Successfully loaded {len(self.df)} records")
            print(f"✓ Number of columns: {len(self.df.columns)}")
            print(f"\nColumn names:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"  {i:2d}. {col}")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False

    def analyze_data_quality(self):
        """Analyze data completeness and quality"""
        print("\n" + "=" * 70)
        print("STEP 2: DATA QUALITY ANALYSIS")
        print("=" * 70)

        print(
            f"\nDataset shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")

        # Missing values
        print("\n--- Missing Values Analysis ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)

        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })

        missing_with_nulls = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )

        if len(missing_with_nulls) > 0:
            print(
                f"\n⚠ Found {len(missing_with_nulls)} columns with missing values:")
            print(missing_with_nulls.to_string(index=False))
        else:
            print("✓ No missing values detected!")

        # Data types
        print("\n--- Data Types ---")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")

        # Key columns
        print("\n--- Key Column Statistics ---")
        key_cols = ['Municipality', 'Borehole ID', 'Depth_Layer']
        for col in key_cols:
            if col in self.df.columns:
                unique_count = self.df[col].nunique()
                print(f"  {col}: {unique_count} unique values")

        return missing_with_nulls

    def convert_depth_to_numeric(self):
        """Convert depth layer text to numeric midpoint values"""
        print("\n" + "=" * 70)
        print("STEP 3: DEPTH LAYER CONVERSION")
        print("=" * 70)

        if 'Depth_Layer' not in self.df.columns:
            print("⚠ Warning: 'Depth_Layer' column not found")
            return

        depth_mapping = {
            '0m-1.5m': 0.75,
            '1.5m-3.0m': 2.25,
            '3.0m-4.5m': 3.75,
            '4.5m-6.0m': 5.25,
            '6.0m-7.5m': 6.75,
            '7.5m-9.0m': 8.25,
            '9.0m-10.5m': 9.75,
            '10.5m-12.0m': 11.25,
            '12.0m-13.5m': 12.75,
            '13.5m-15.0m': 14.25
        }

        self.df['Depth_Midpoint_m'] = self.df['Depth_Layer'].map(depth_mapping)
        print("✓ Created 'Depth_Midpoint_m' column")
        print(f"\nDepth distribution:")
        print(self.df['Depth_Layer'].value_counts().sort_index())

    def engineer_soil_features(self):
        """Create engineered features from existing soil properties"""
        print("\n" + "=" * 70)
        print("STEP 4: FEATURE ENGINEERING - SOIL PROPERTIES")
        print("=" * 70)

        features_created = []

        # 1. Soil Classification (based on fines content)
        if 'Fines Content' in self.df.columns:
            self.df['Soil_Type_Category'] = pd.cut(
                self.df['Fines Content'],
                bins=[0, 5, 12, 100],
                labels=['Clean_Sand', 'Sandy_Soil', 'Fine_Grained']
            )
            features_created.append('Soil_Type_Category')
            print("✓ Created Soil_Type_Category based on fines content")

        # 2. Relative Density Classification
        if 'Corrected SPT-N Value (N1(60))' in self.df.columns:
            self.df['Relative_Density_Class'] = pd.cut(
                self.df['Corrected SPT-N Value (N1(60))'],
                bins=[0, 4, 10, 30, 50, 100],
                labels=['Very_Loose', 'Loose', 'Medium', 'Dense', 'Very_Dense']
            )
            features_created.append('Relative_Density_Class')
            print("✓ Created Relative_Density_Class from N1(60)")

        # 3. Total Overburden Stress
        if 'Unit Weight (γ)' in self.df.columns and 'Depth_Midpoint_m' in self.df.columns:
            self.df['Total_Overburden_Stress_kPa'] = (
                self.df['Unit Weight (γ)'] * self.df['Depth_Midpoint_m']
            )
            features_created.append('Total_Overburden_Stress_kPa')
            print("✓ Created Total_Overburden_Stress_kPa")

        # 4. Effective Overburden Stress
        if 'Groundwater Level (m)' in self.df.columns and 'Depth_Midpoint_m' in self.df.columns:
            gamma_water = 9.81  # kN/m³
            depth_below_wt = np.maximum(
                0, self.df['Depth_Midpoint_m'] -
                self.df['Groundwater Level (m)']
            )

            if 'Total_Overburden_Stress_kPa' in self.df.columns:
                self.df['Effective_Overburden_Stress_kPa'] = (
                    self.df['Total_Overburden_Stress_kPa'] -
                    (gamma_water * depth_below_wt)
                )
                features_created.append('Effective_Overburden_Stress_kPa')
                print("✓ Created Effective_Overburden_Stress_kPa")

        if features_created:
            print(f"\n✓ Total features created: {len(features_created)}")
        else:
            print("⚠ No features could be created (missing base columns)")

        return features_created

    def calculate_liquefaction_parameters(self):
        """Calculate liquefaction-related parameters"""
        print("\n" + "=" * 70)
        print("STEP 5: LIQUEFACTION PARAMETER CALCULATION")
        print("=" * 70)

        parameters_created = []

        N1_60_col = 'Corrected SPT-N Value (N1(60))'
        CSR_col = 'Cyclic Stress Ratio (CSR)'

        if N1_60_col not in self.df.columns:
            print(
                f"⚠ Cannot calculate liquefaction parameters. Missing: {N1_60_col}")
            return parameters_created

        if CSR_col not in self.df.columns:
            print(
                f"⚠ Cannot calculate liquefaction parameters. Missing: {CSR_col}")
            return parameters_created

        # 1. Calculate CRR (Cyclic Resistance Ratio)
        N160 = self.df[N1_60_col]
        self.df['CRR'] = np.where(N160 <= 30, 1 / (34 - N160 + 0.001), 0.5)

        # Adjust for fines content if available
        if 'Fines Content' in self.df.columns:
            fc = self.df['Fines Content']
            fc_factor = 1 + 0.004 * fc
            self.df['CRR'] = self.df['CRR'] * fc_factor

        parameters_created.append('CRR')
        print("✓ Created CRR (Cyclic Resistance Ratio)")

        # 2. Factor of Safety against Liquefaction
        CSR = self.df[CSR_col]
        self.df['FS_Liquefaction'] = self.df['CRR'] / (CSR + 0.0001)
        parameters_created.append('FS_Liquefaction')
        print("✓ Created FS_Liquefaction (Factor of Safety)")

        # 3. Liquefaction Potential Classification
        self.df['Liquefaction_Potential'] = np.where(
            self.df['FS_Liquefaction'] < 1.0, 1, 0
        )
        parameters_created.append('Liquefaction_Potential')
        print("✓ Created Liquefaction_Potential (Binary: 0=No, 1=Yes)")

        # Show distribution
        liq_counts = self.df['Liquefaction_Potential'].value_counts()
        print(f"\n  Liquefaction Distribution:")
        print(
            f"    Non-Liquefiable (0): {liq_counts.get(0, 0)} ({liq_counts.get(0, 0)/len(self.df)*100:.1f}%)")
        print(
            f"    Liquefiable (1): {liq_counts.get(1, 0)} ({liq_counts.get(1, 0)/len(self.df)*100:.1f}%)")

        print(f"\n✓ Total parameters created: {len(parameters_created)}")
        return parameters_created

    def calculate_settlement_target(self):
        """Calculate post-liquefaction settlement using Tokimatsu & Seed (1987)"""
        print("\n" + "=" * 70)
        print("STEP 6: SETTLEMENT CALCULATION (Target Variable)")
        print("=" * 70)

        N1_60_col = 'Corrected SPT-N Value (N1(60))'
        CSR_col = 'Cyclic Stress Ratio (CSR)'

        required = [N1_60_col, CSR_col, 'Liquefaction_Potential']
        missing = [col for col in required if col not in self.df.columns]

        if missing:
            print(f"⚠ Cannot calculate settlement. Missing columns: {missing}")
            return False

        liquefiable_mask = self.df['Liquefaction_Potential'] == 1
        n_liquefiable = liquefiable_mask.sum()

        print(
            f"\nCalculating settlement for {n_liquefiable} liquefiable soil layers...")

        if n_liquefiable == 0:
            print("⚠ No liquefiable soils found. Settlement = 0 for all layers.")
            self.df['Settlement_m'] = 0.0
            return True

        def calculate_volumetric_strain(N160, CSR, FS):
            if FS >= 1.0:
                return 0.0
            if N160 < 5:
                ev_max = 4.0
            elif N160 < 10:
                ev_max = 3.0
            elif N160 < 15:
                ev_max = 2.0
            elif N160 < 20:
                ev_max = 1.0
            else:
                ev_max = 0.5
            ev = ev_max * min(CSR / 0.3, 1.0)
            return ev / 100

        self.df['Volumetric_Strain'] = self.df.apply(
            lambda row: calculate_volumetric_strain(
                row[N1_60_col], row[CSR_col], row['FS_Liquefaction']
            ) if row['Liquefaction_Potential'] == 1 else 0.0,
            axis=1
        )

        layer_thickness = 1.5
        self.df['Settlement_m'] = self.df['Volumetric_Strain'] * layer_thickness

        print("✓ Created Settlement_m (target variable for ML)")

        settlement_stats = self.df[self.df['Settlement_m']
                                   > 0]['Settlement_m'].describe()
        print(f"\nSettlement Statistics (liquefiable layers only):")
        print(f"  Count: {(self.df['Settlement_m'] > 0).sum()}")
        print(f"  Mean:  {settlement_stats['mean']:.4f} m")
        print(f"  Max:   {settlement_stats['max']:.4f} m")
        print(f"  Min:   {settlement_stats['min']:.4f} m")

        return True

    def calculate_bearing_capacity_target(self):
        """Calculate bearing capacity using Terzaghi (1943)"""
        print("\n" + "=" * 70)
        print("STEP 7: BEARING CAPACITY CALCULATION (Target Variable)")
        print("=" * 70)

        friction_angle_col = 'Internal Friction Angle'
        unit_weight_col = 'Unit Weight (γ)'
        footing_width_col = 'Foundation Width (B)'
        footing_depth_col = 'Foundation Depth (D)'
        N1_60_col = 'Corrected SPT-N Value (N1(60))'

        if friction_angle_col not in self.df.columns:
            if N1_60_col in self.df.columns:
                self.df[friction_angle_col] = 27.5 + 0.3 * self.df[N1_60_col]
                print(f"✓ Estimated {friction_angle_col} from N1(60)")
            else:
                print(f"⚠ Cannot estimate friction angle - missing N1(60)")
                return False

        self.df['Cohesion_kPa'] = 0.0
        print("✓ Set Cohesion = 0 kPa (sandy soils)")

        if footing_width_col not in self.df.columns:
            self.df[footing_width_col] = 1.5
            print(f"✓ Assumed {footing_width_col} = 1.5 m")

        if footing_depth_col not in self.df.columns:
            self.df[footing_depth_col] = 1.5
            print(f"✓ Assumed {footing_depth_col} = 1.5 m")

        print("\nCalculating bearing capacity using Terzaghi (1943)...")

        def calculate_bearing_factors(phi_deg):
            phi_rad = np.radians(phi_deg)
            Nq = np.exp(np.pi * np.tan(phi_rad)) * \
                np.tan(np.radians(45 + phi_deg/2))**2
            Nc = (Nq - 1) / (np.tan(phi_rad) + 0.001)
            Nγ = 2 * (Nq + 1) * np.tan(phi_rad)
            return Nc, Nq, Nγ

        bearing_capacities = []

        for idx, row in self.df.iterrows():
            phi = row[friction_angle_col]
            c = row['Cohesion_kPa']
            gamma = row[unit_weight_col] if unit_weight_col in self.df.columns else 18.0
            B = row[footing_width_col]
            Df = row[footing_depth_col]

            Nc, Nq, Nγ = calculate_bearing_factors(phi)
            qu = 1.3 * c * Nc + gamma * Df * Nq + 0.4 * gamma * B * Nγ
            bearing_capacities.append(qu)

        self.df['Ultimate_Bearing_Capacity_kPa'] = bearing_capacities

        FS_bearing = 3.0
        self.df['Allowable_Bearing_Capacity_kPa'] = (
            self.df['Ultimate_Bearing_Capacity_kPa'] / FS_bearing
        )

        print("✓ Created Ultimate_Bearing_Capacity_kPa")
        print("✓ Created Allowable_Bearing_Capacity_kPa (FS = 3.0)")

        bc_stats = self.df['Allowable_Bearing_Capacity_kPa'].describe()
        print(f"\nBearing Capacity Statistics:")
        print(f"  Mean:  {bc_stats['mean']:.2f} kPa")
        print(f"  Max:   {bc_stats['max']:.2f} kPa")
        print(f"  Min:   {bc_stats['min']:.2f} kPa")

        return True

    def handle_missing_values(self):
        """Handle any remaining missing values"""
        print("\n" + "=" * 70)
        print("STEP 8: HANDLING MISSING VALUES")
        print("=" * 70)

        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        missing_numerical = self.df[numerical_cols].isnull().sum()
        missing_numerical = missing_numerical[missing_numerical > 0]

        if len(missing_numerical) == 0:
            print("✓ No missing values in numerical columns!")
            return True

        print(
            f"\nFound missing values in {len(missing_numerical)} numerical columns:")
        for col, count in missing_numerical.items():
            print(f"  {col}: {count} missing ({count/len(self.df)*100:.1f}%)")

        print("\nFilling missing numerical values with column median...")
        for col in missing_numerical.index:
            median_value = self.df[col].median()
            self.df[col].fillna(median_value, inplace=True)
            print(f"  ✓ {col}: filled with {median_value:.4f}")

        print("\n✓ All missing values handled!")
        return True

    def create_ml_ready_features(self):
        """Identify and organize features for ML"""
        print("\n" + "=" * 70)
        print("STEP 9: ORGANIZING ML FEATURES")
        print("=" * 70)

        input_features = [
            'Latitude', 'Longitude', 'Depth_Midpoint_m', 'Unit Weight (γ)',
            'Corrected SPT-N Value (N1(60))', 'Cyclic Stress Ratio (CSR)', 'CRR',
            'Groundwater Level (m)', 'Peak Ground Acceleration', 'Fines Content',
            'Plasticity Index (PI)', 'Mean Particle Size (D50) (mm)',
            'Effective_Overburden_Stress_kPa', 'FS_Liquefaction',
            'Foundation Width (B)', 'Foundation Depth (D)',
            'Internal Friction Angle', 'Cohesion_kPa'
        ]

        available_features = [
            f for f in input_features if f in self.df.columns]
        missing_features = [
            f for f in input_features if f not in self.df.columns]

        print(f"\n✓ Available Features ({len(available_features)}):")
        for f in available_features:
            print(f"  • {f}")

        if missing_features:
            print(f"\n⚠ Missing Features ({len(missing_features)}):")
            for f in missing_features:
                print(f"  • {f}")

        target_variables = {
            'Liquefaction': 'Liquefaction_Potential',
            'Settlement': 'Settlement_m',
            'Bearing_Capacity': 'Allowable_Bearing_Capacity_kPa'
        }

        print(f"\n✓ Target Variables:")
        for name, col in target_variables.items():
            if col in self.df.columns:
                print(f"  • {name}: {col}")
            else:
                print(f"  ⚠ {name}: {col} (NOT FOUND)")

        return available_features, target_variables

    def export_ml_ready_data_to_memory(self):
        """Export prepared data to Excel IN MEMORY - returns bytes"""
        print("\n" + "=" * 70)
        print("STEP 10: EXPORTING ML-READY DATA (IN MEMORY)")
        print("=" * 70)

        # Create Excel file in memory
        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # 1. Full dataset
            self.df.to_excel(writer, sheet_name='Full_Dataset', index=False)
            print("✓ Exported: Full_Dataset")

            # 2. Feature list
            feature_cols, target_vars = self.create_ml_ready_features()

            def safe_mean(col):
                try:
                    return self.df[col].mean() if pd.api.types.is_numeric_dtype(self.df[col]) else 'N/A'
                except:
                    return 'N/A'

            def safe_std(col):
                try:
                    return self.df[col].std() if pd.api.types.is_numeric_dtype(self.df[col]) else 'N/A'
                except:
                    return 'N/A'

            feature_df = pd.DataFrame({
                'Feature_Name': feature_cols,
                'Data_Type': [str(self.df[f].dtype) for f in feature_cols],
                'Missing_Values': [self.df[f].isnull().sum() for f in feature_cols],
                'Mean': [safe_mean(f) for f in feature_cols],
                'Std': [safe_std(f) for f in feature_cols]
            })
            feature_df.to_excel(writer, sheet_name='Feature_List', index=False)
            print("✓ Exported: Feature_List")

            # 3. Data quality report
            quality_df = pd.DataFrame({
                'Metric': [
                    'Total Records', 'Total Features', 'Numerical Features',
                    'Categorical Features', 'Liquefiable Soils', 'Non-Liquefiable Soils',
                    'Avg Settlement (Liquefiable)', 'Avg Bearing Capacity'
                ],
                'Value': [
                    len(self.df),
                    len(self.df.columns),
                    len(self.df.select_dtypes(include=[np.number]).columns),
                    len(self.df.select_dtypes(include=['object']).columns),
                    (self.df['Liquefaction_Potential'] == 1).sum(
                    ) if 'Liquefaction_Potential' in self.df.columns else 'N/A',
                    (self.df['Liquefaction_Potential'] == 0).sum(
                    ) if 'Liquefaction_Potential' in self.df.columns else 'N/A',
                    f"{self.df[self.df['Liquefaction_Potential'] == 1]['Settlement_m'].mean():.4f} m" if 'Settlement_m' in self.df.columns else 'N/A',
                    f"{self.df['Allowable_Bearing_Capacity_kPa'].mean():.2f} kPa" if 'Allowable_Bearing_Capacity_kPa' in self.df.columns else 'N/A'
                ]
            })
            quality_df.to_excel(
                writer, sheet_name='Data_Quality_Report', index=False)
            print("✓ Exported: Data_Quality_Report")

            # 4. Liquefiable soils only
            if 'Liquefaction_Potential' in self.df.columns:
                liq_df = self.df[self.df['Liquefaction_Potential'] == 1].copy()
                if len(liq_df) > 0:
                    liq_df.to_excel(
                        writer, sheet_name='Liquefiable_Soils_Only', index=False)
                    print(
                        f"✓ Exported: Liquefiable_Soils_Only ({len(liq_df)} records)")

            # 5. Summary statistics
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            stats_df = self.df[numeric_cols].describe().T
            stats_df.to_excel(writer, sheet_name='Summary_Statistics')
            print("✓ Exported: Summary_Statistics")

        # Get bytes from buffer
        excel_buffer.seek(0)
        excel_bytes = excel_buffer.read()

        print(f"\n✓ Excel file created in memory (not saved locally)")
        print(f"✓ File size: {len(excel_bytes) / 1024:.2f} KB")
        return excel_bytes

    def generate_data_report(self):
        """Generate final data preparation report"""
        print("\n" + "=" * 70)
        print("DATA PREPARATION COMPLETE - SUMMARY REPORT")
        print("=" * 70)

        report = {
            'Total Records': len(self.df),
            'Total Columns': len(self.df.columns),
            'Numerical Columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'Categorical Columns': len(self.df.select_dtypes(include=['object']).columns)
        }

        if 'Municipality' in self.df.columns:
            report['Unique Municipalities'] = self.df['Municipality'].nunique()

        if 'Borehole ID' in self.df.columns:
            report['Unique Boreholes'] = self.df['Borehole ID'].nunique()

        if 'Liquefaction_Potential' in self.df.columns:
            liq_counts = self.df['Liquefaction_Potential'].value_counts()
            report['Liquefiable Layers'] = f"{liq_counts.get(1, 0)} ({liq_counts.get(1, 0)/len(self.df)*100:.1f}%)"
            report['Non-Liquefiable Layers'] = f"{liq_counts.get(0, 0)} ({liq_counts.get(0, 0)/len(self.df)*100:.1f}%)"

        if 'Settlement_m' in self.df.columns:
            settlement_data = self.df[self.df['Settlement_m']
                                      > 0]['Settlement_m']
            if len(settlement_data) > 0:
                report['Mean Settlement (Liquefiable)'] = f"{settlement_data.mean():.4f} m"
                report['Max Settlement'] = f"{settlement_data.max():.4f} m"

        if 'Allowable_Bearing_Capacity_kPa' in self.df.columns:
            report['Mean Bearing Capacity'] = f"{self.df['Allowable_Bearing_Capacity_kPa'].mean():.2f} kPa"
            report['Max Bearing Capacity'] = f"{self.df['Allowable_Bearing_Capacity_kPa'].max():.2f} kPa"

        print("\n FINAL STATISTICS:")
        for key, value in report.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 70)
        print(" DATA IS NOW READY FOR MACHINE LEARNING!")
        print("=" * 70)
        print("\nNext Steps:")
        print("  1. Review 'ML_Ready_Data.xlsx' (uploaded to Supabase)")
        print("  2. Check 'Feature_List' sheet for input variables")
        print("  3. Verify target variables are properly calculated")
        print("  4. Proceed to ANN model training")
        print("=" * 70 + "\n")

        return report


# -----------------------------
# Main Execution
# -----------------------------

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("GEOTECHNICAL DATA PREPARATION FOR MACHINE LEARNING (IN-MEMORY)")
    print("Liquefaction, Settlement, and Bearing Capacity Prediction")
    print("Tarlac Province, Philippines")
    print("=" * 70 + "\n")

    # Configuration
    BUCKET_NAME = 'geotechnical-data'
    INPUT_FILE_PATH = 'cleaned/Cleaned_Data.xlsx'  # Source in storage
    OUTPUT_STORAGE_PATH = 'ml_ready/ML_Ready_Data.xlsx'  # Destination in storage

    # Step 1: Download cleaned data from Supabase
    file_bytes = download_file_from_storage(BUCKET_NAME, INPUT_FILE_PATH)
    if not file_bytes:
        print(" Failed to download file from storage. Exiting.")
        return None

    # Step 2: Process data
    prep = GeotechnicalDataPrep(file_bytes)

    if not prep.load_data():
        print(" Failed to load data. Exiting.")
        return None

    prep.analyze_data_quality()
    prep.convert_depth_to_numeric()
    prep.engineer_soil_features()
    prep.calculate_liquefaction_parameters()
    prep.calculate_settlement_target()
    prep.calculate_bearing_capacity_target()
    prep.handle_missing_values()

    # Export to memory (returns bytes)
    excel_bytes = prep.export_ml_ready_data_to_memory()

    prep.generate_data_report()

    # Step 3: Upload ML-ready data to Supabase (from memory)
    upload_to_supabase_storage(excel_bytes, BUCKET_NAME, OUTPUT_STORAGE_PATH)

    print("\n PROCESSING COMPLETED SUCCESSFULLY!")
    print("✓ No local files created - all processing in memory")
    print("="*70)

    return prep


if __name__ == "__main__":
    data_prep = main()
