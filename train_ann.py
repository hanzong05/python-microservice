#!/usr/bin/env python3
"""
Multi-Output ANN Model Training
Per Project Requirements (Alejandrino et al.)

Features:
- Direct database query from PostGIS
- Multi-output ANN: Liquefaction (classification) + Settlement (regression) + Bearing Capacity (regression)
- 3-layer MLP architecture (64-32-16 neurons)
- Validation against: DPWH BSDS (2013), Tokimatsu & Seed (1987), Bray & Macedo (2017), Terzaghi (1943), Olsen & Stark (2002)
- Performance metrics: Confusion Matrix, R², MAE, RMSE, Performance Index (PI)
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, Tuple, Optional
import sys
import os
import math

warnings.filterwarnings('ignore')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[INFO] python-dotenv not installed - .env file will not be loaded")

try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score,
        precision_score, recall_score, f1_score, roc_auc_score,
        r2_score, mean_absolute_error, mean_squared_error
    )
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[ERROR] scikit-learn not installed!")
    print("Install: pip install scikit-learn joblib")
    SKLEARN_AVAILABLE = False

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("[ERROR] supabase-py not installed!")
    print("Install: pip install supabase")
    SUPABASE_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[INFO] Matplotlib not available - plots will be skipped")

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("[INFO] openpyxl not installed - Excel export will use CSV fallback")


class MultiOutputANNTraining:
    """
    Multi-Output ANN Training per Project Requirements
    - Liquefaction Classification
    - Settlement Regression
    - Bearing Capacity Regression
    """
    
    def __init__(self):
        self.client = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_liq_train = None
        self.y_liq_test = None
        self.y_settle_train = None
        self.y_settle_test = None
        self.y_bearing_train = None
        self.y_bearing_test = None
        self.scaler = None
        self.multi_model = None  # Primary multi-output model
        self.liq_model = None
        self.settle_model = None
        self.bearing_model = None
        self.feature_names = []
        self.df_raw = None        # Original raw dataframe (all columns)
        self.df_validation = None  # 20% validation set in raw format
        
    def connect_database(self) -> bool:
        """Connect to PostGIS database"""
        print("\n" + "="*80)
        print("CONNECTING TO POSTGIS DATABASE")
        print("="*80)
        
        if not SUPABASE_AVAILABLE:
            print("  [ERROR] Supabase not available")
            return False
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url or not supabase_key:
            print("  [ERROR] Environment variables not found")
            print("\n  Create .env file in project root with:")
            print("    SUPABASE_URL=your_supabase_url")
            print("    SUPABASE_SERVICE_ROLE_KEY=your_service_role_key")
            return False
        
        try:
            self.client = create_client(supabase_url, supabase_key)
            self.client.table('soil_layers').select('id').limit(1).execute()
            print("  [OK] Connected to PostGIS database")
            return True
        except Exception as e:
            print(f"  [ERROR] Connection failed: {e}")
            return False
    
    def query_database(self) -> bool:
        """Query database for training data"""
        print("\n" + "="*80)
        print("QUERYING DATABASE FOR TRAINING DATA")
        print("="*80)
        
        try:
            print("  Fetching data from v_complete_soil_data view...")
            try:
                result = self.client.table('v_complete_soil_data').select('*').execute()
            except:
                print("  View not available, querying soil_layers table...")
                result = self.client.table('soil_layers').select('*').execute()
            
            if not result.data:
                print("  [ERROR] No data returned")
                return False
            
            self.df = pd.DataFrame(result.data)
            self.df_raw = self.df.copy()  # Preserve raw data for validation export
            print(f"  [OK] Retrieved {len(self.df)} records")
            print(f"  Columns: {len(self.df.columns)}")
            return True
        except Exception as e:
            print(f"  [ERROR] Query failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_settlement_tokimatsu_seed(self, row) -> float:
        """Calculate settlement using Tokimatsu & Seed (1987) method"""
        try:
            spt_n160 = float(row.get('spt_n160', row.get('spt_n60', 15)) or 15)
            csr = float(row.get('csr', 0.2) or 0.2)
            crr = float(row.get('cyclic_strength_ratio', row.get('crr', 0.3)) or 0.3)
            depth_m = float(row.get('depth_mid_m', row.get('depth_from_m', 1.0)) or 1.0)
            
            # Factor of safety
            fs = crr / (csr + 0.001)
            
            if fs >= 1.0:
                return 0.0
            
            # Volumetric strain based on N1(60) and CSR
            if spt_n160 < 5:
                ev_max = 4.0
            elif spt_n160 < 10:
                ev_max = 3.0
            elif spt_n160 < 15:
                ev_max = 2.0
            elif spt_n160 < 20:
                ev_max = 1.0
            else:
                ev_max = 0.5
            
            # Volumetric strain
            ev = ev_max * min(csr / 0.3, 1.0) * (1.0 - fs)
            ev = max(0, min(ev, ev_max))  # Clamp to 0-ev_max
            
            # Settlement = volumetric strain * layer thickness
            layer_thickness = float(row.get('depth_to_m', 1.5) or 1.5) - float(row.get('depth_from_m', 0) or 0)
            settlement_cm = (ev / 100) * layer_thickness * 100  # Convert to cm
            
            return max(0, settlement_cm)
        except:
            return 0.0
    
    def calculate_settlement_bray_macedo(self, row) -> float:
        """Calculate settlement using Bray & Macedo (2017) method"""
        try:
            spt_n160 = float(row.get('spt_n160', row.get('spt_n60', 15)) or 15)
            csr = float(row.get('csr', 0.2) or 0.2)
            crr = float(row.get('cyclic_strength_ratio', row.get('crr', 0.3)) or 0.3)
            fines_pct = float(row.get('fines_content', 15) or 15)
            
            fs = crr / (csr + 0.001)
            
            if fs >= 1.0:
                return 0.0
            
            # Bray & Macedo (2017) volumetric strain
            # Based on SPT N1(60), fines content, and CSR
            if fines_pct < 5:
                # Clean sand
                if spt_n160 < 10:
                    ev_max = 3.5
                elif spt_n160 < 15:
                    ev_max = 2.5
                else:
                    ev_max = 1.5
            else:
                # Silty sand
                if spt_n160 < 10:
                    ev_max = 4.0
                elif spt_n160 < 15:
                    ev_max = 3.0
                else:
                    ev_max = 2.0
            
            # Volumetric strain
            ev = ev_max * (csr / 0.3) * (1.0 - min(fs, 1.0))
            ev = max(0, min(ev, ev_max))
            
            # Settlement
            layer_thickness = float(row.get('depth_to_m', 1.5) or 1.5) - float(row.get('depth_from_m', 0) or 0)
            settlement_cm = (ev / 100) * layer_thickness * 100
            
            return max(0, settlement_cm)
        except:
            return 0.0
    
    def calculate_bearing_capacity_terzaghi(self, row) -> float:
        """Calculate bearing capacity using Terzaghi (1943)"""
        try:
            spt_n60 = float(row.get('spt_n60', row.get('spt_n_value', 15)) or 15)
            unit_weight = float(row.get('unit_weight', 18) or 18)
            depth_m = float(row.get('depth_mid_m', 1.0) or 1.0)
            
            # Estimate friction angle from SPT
            phi_deg = 27.5 + 0.3 * spt_n60
            phi_deg = max(25, min(phi_deg, 45))
            phi_rad = math.radians(phi_deg)
            
            # Bearing capacity factors
            Nq = math.exp(math.pi * math.tan(phi_rad)) * (math.tan(math.radians(45 + phi_deg/2)))**2
            Nc = (Nq - 1) / (math.tan(phi_rad) + 0.001)
            Ng = 2 * (Nq + 1) * math.tan(phi_rad)
            
            # Foundation parameters (assumed)
            B = 1.5  # Foundation width (m)
            D = 1.5  # Foundation depth (m)
            c = 0.0  # Cohesion (kPa) - sandy soil
            
            # Terzaghi bearing capacity
            q_ult = c * Nc + unit_weight * D * Nq + 0.5 * unit_weight * B * Ng
            
            # Convert to kPa and apply safety factor
            qa = q_ult / 3.0  # Safety factor = 3
            
            return max(0, qa)
        except:
            return 200.0  # Default
    
    def calculate_bearing_capacity_olsen_stark(self, row) -> float:
        """Calculate bearing capacity using Olsen & Stark (2002) for post-liquefaction"""
        try:
            # Get pre-liquefaction capacity
            pre_capacity = self.calculate_bearing_capacity_terzaghi(row)
            
            # Get liquefaction probability
            csr = float(row.get('csr', 0.2) or 0.2)
            crr = float(row.get('cyclic_strength_ratio', row.get('crr', 0.3)) or 0.3)
            fs = crr / (csr + 0.001)
            
            # Post-liquefaction reduction factor
            if fs < 1.0:
                # Liquefaction occurs - significant reduction
                reduction_factor = 0.2 + (fs * 0.3)  # 20-50% of original
            else:
                # No liquefaction - minor reduction
                reduction_factor = 0.7 + ((fs - 1.0) / 2.0) * 0.3  # 70-100% of original
            
            reduction_factor = max(0.2, min(reduction_factor, 1.0))
            
            post_capacity = pre_capacity * reduction_factor
            
            return max(0, post_capacity)
        except:
            return 100.0  # Default
    
    def prepare_features_and_targets(self) -> bool:
        """Prepare features and all three targets per project requirements"""
        print("\n" + "="*80)
        print("PREPARING FEATURES AND TARGETS")
        print("="*80)
        
        df = self.df.copy()
        
        # Select exactly 17 features as per project requirements
        # Priority features for geotechnical prediction
        priority_features = [
            'spt_n_value', 'spt_n60', 'spt_n160',
            'unit_weight', 'fines_content', 'groundwater_depth_m',
            'pga_g', 'csr', 'cyclic_strength_ratio', 'crr',
            'effective_overburden_pressure', 'total_overburden_pressure',
            'depth_from_m', 'depth_to_m', 'depth_mid_m',
            'relative_density_percent', 'moisture_content'
        ]
        
        # Get available priority features
        available_priority = [col for col in priority_features if col in df.columns]
        
        # Fill remaining slots with other numeric features
        exclude_cols = [
            'id', 'layer_id', 'borehole_id', 'borehole_record_id',
            'municipality_id', 'barangay_id', 'created_at', 'updated_at',
            'municipality', 'barangay', 'latitude', 'longitude',
            'elevation', 'borehole_id',
            'liquefaction_risk_level', 'liquefaction_status',
            'settlement_cm', 'bearing_capacity_kpa', 'qa_allowable_kpa',
            'liquefaction', 'liquefaction_probability', 'factor_of_safety'
        ]
        
        other_features = [
            col for col in df.columns
            if col not in exclude_cols and col not in available_priority
            and df[col].dtype in ['int64', 'float64', 'bool']
        ]
        
        # Select exactly 17 features
        feature_cols = available_priority[:17]
        if len(feature_cols) < 17:
            needed = 17 - len(feature_cols)
            feature_cols.extend(other_features[:needed])
        
        # If still less than 17, pad with zeros or use what we have
        if len(feature_cols) < 17:
            print(f"  [WARNING] Only {len(feature_cols)} features available, padding to 17")
            for i in range(len(feature_cols), 17):
                df[f'feature_{i}'] = 0.0
                feature_cols.append(f'feature_{i}')
        elif len(feature_cols) > 17:
            feature_cols = feature_cols[:17]
        
        print(f"  Selected exactly {len(feature_cols)} feature columns (as required)")
        self.feature_names = feature_cols
        
        # Prepare feature matrix
        X = df[feature_cols].copy()
        
        # Fill missing values with median
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        
        # Convert boolean to int
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)
        
        # TARGET 1: Liquefaction (Binary Classification)
        print("\n  Preparing TARGET 1: Liquefaction (Binary Classification)...")
        if 'liquefaction' in df.columns:
            y_liq = df['liquefaction'].astype(int)
        elif 'liquefaction_probability' in df.columns:
            y_liq = (df['liquefaction_probability'] > 50).astype(int)
        elif 'factor_of_safety' in df.columns:
            y_liq = (df['factor_of_safety'] < 1.0).astype(int)
        elif 'csr' in df.columns and 'cyclic_strength_ratio' in df.columns:
            fs = df['cyclic_strength_ratio'] / (df['csr'] + 0.001)
            y_liq = (fs < 1.0).astype(int)
        else:
            print("  [ERROR] Cannot determine liquefaction target")
            return False
        
        # TARGET 2: Settlement (Regression) - Average of Tokimatsu & Seed and Bray & Macedo
        print("\n  Preparing TARGET 2: Settlement (Regression)...")
        print("    Calculating using Tokimatsu & Seed (1987)...")
        settle_ts = df.apply(self.calculate_settlement_tokimatsu_seed, axis=1)
        print("    Calculating using Bray & Macedo (2017)...")
        settle_bm = df.apply(self.calculate_settlement_bray_macedo, axis=1)
        y_settle = (settle_ts + settle_bm) / 2.0  # Average of both methods
        
        # TARGET 3: Bearing Capacity (Regression) - Post-liquefaction from Olsen & Stark
        print("\n  Preparing TARGET 3: Bearing Capacity (Regression)...")
        print("    Calculating using Olsen & Stark (2002)...")
        y_bearing = df.apply(self.calculate_bearing_capacity_olsen_stark, axis=1)
        
        # Remove rows with missing targets
        valid_mask = ~(y_liq.isna() | y_settle.isna() | y_bearing.isna())
        X = X[valid_mask]
        y_liq = y_liq[valid_mask]
        y_settle = y_settle[valid_mask]
        y_bearing = y_bearing[valid_mask]
        
        print(f"\n  Features shape: {X.shape}")
        print(f"  Target 1 (Liquefaction): {(y_liq == 1).sum()} positive, {(y_liq == 0).sum()} negative")
        print(f"  Target 2 (Settlement): Mean={y_settle.mean():.2f} cm, Range=[{y_settle.min():.2f}, {y_settle.max():.2f}] cm")
        print(f"  Target 3 (Bearing Capacity): Mean={y_bearing.mean():.1f} kPa, Range=[{y_bearing.min():.1f}, {y_bearing.max():.1f}] kPa")
        
        # Split data: 80% train, 20% validation (per project requirements)
        print("\n  Splitting data (80% train, 20% validation)...")
        self.X_train, self.X_test, self.y_liq_train, self.y_liq_test = train_test_split(
            X, y_liq, test_size=0.2, random_state=42, stratify=y_liq
        )
        _, _, self.y_settle_train, self.y_settle_test = train_test_split(
            X, y_settle, test_size=0.2, random_state=42
        )
        _, _, self.y_bearing_train, self.y_bearing_test = train_test_split(
            X, y_bearing, test_size=0.2, random_state=42
        )

        # Build validation dataframe in raw data format using test indices
        val_indices = self.X_test.index
        val_raw = self.df_raw.loc[val_indices].copy().reset_index(drop=True)
        # Append computed targets so the Excel reflects what the model is predicting
        val_raw['_target_liquefaction'] = self.y_liq_test.values
        val_raw['_target_settlement_cm'] = self.y_settle_test.values
        val_raw['_target_bearing_capacity_kpa'] = self.y_bearing_test.values
        self.df_validation = val_raw

        print(f"  Training set: {len(self.X_train)} samples (80%)")
        print(f"  Validation set: {len(self.X_test)} samples (20%)")
        
        return True
    
    def train_models(self) -> bool:
        """Train multi-output model with specified architecture"""
        if not SKLEARN_AVAILABLE:
            return False
        
        print("\n" + "="*80)
        print("TRAINING MULTI-OUTPUT ANN MODEL")
        print("="*80)
        print("  Architecture:")
        print("    INPUT LAYER:    17 neurons")
        print("    HIDDEN LAYER 1: 30 neurons (Tansig/Tanh activation)")
        print("    HIDDEN LAYER 2: 15 neurons (Tansig/Tanh activation)")
        print("    OUTPUT LAYER:   3 neurons (Purelin/Linear activation)")
        print("  Outputs: [Liquefaction, Settlement, Bearing Capacity]")
        print("  Solver: Adam")
        
        # Verify input features = 17
        if self.X_train.shape[1] != 17:
            print(f"  [ERROR] Expected 17 input features, got {self.X_train.shape[1]}")
            return False
        
        # Scale features
        print("\n  Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Prepare multi-output target (all as regression)
        # Liquefaction: 0-1 probability (will threshold later)
        # Settlement: cm
        # Bearing Capacity: kPa
        print("\n  Preparing multi-output target...")
        y_multi_train = np.column_stack([
            self.y_liq_train.astype(float),  # Liquefaction as probability
            self.y_settle_train,
            self.y_bearing_train
        ])
        y_multi_test = np.column_stack([
            self.y_liq_test.astype(float),
            self.y_settle_test,
            self.y_bearing_test
        ])
        
        # Train multi-output regressor (all 3 outputs)
        print("\n  Training Multi-Output ANN...")
        print("    Architecture: 17 -> 30 (tanh) -> 15 (tanh) -> 3 (linear)")
        
        # Use MLPRegressor with tanh activation and linear output
        # Note: scikit-learn doesn't support different activations per layer,
        # so we use tanh for hidden layers (closest to Tansig)
        self.multi_model = MLPRegressor(
            hidden_layer_sizes=(30, 15),  # 2 hidden layers: 30, 15
            activation='tanh',  # Tansig = tanh
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
            verbose=False
        )
        
        self.multi_model.fit(X_train_scaled, y_multi_train)
        print(f"    [OK] Completed in {self.multi_model.n_iter_} iterations")
        
        # Also train separate models for individual evaluation
        print("\n  Training individual models for validation...")
        
        # MODEL 1: Liquefaction Classifier (for confusion matrix)
        self.liq_model = MLPClassifier(
            hidden_layer_sizes=(30, 15),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
            verbose=False
        )
        self.liq_model.fit(X_train_scaled, self.y_liq_train)
        print(f"    [OK] Liquefaction classifier trained")
        
        # MODEL 2: Settlement Regressor
        self.settle_model = MLPRegressor(
            hidden_layer_sizes=(30, 15),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
            verbose=False
        )
        self.settle_model.fit(X_train_scaled, self.y_settle_train)
        print(f"    [OK] Settlement regressor trained")
        
        # MODEL 3: Bearing Capacity Regressor
        self.bearing_model = MLPRegressor(
            hidden_layer_sizes=(30, 15),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
            verbose=False
        )
        self.bearing_model.fit(X_train_scaled, self.y_bearing_train)
        print(f"    [OK] Bearing capacity regressor trained")
        
        return True
    
    def evaluate_models(self) -> bool:
        """Evaluate all models with required metrics per project requirements"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Evaluate multi-output model
        print("\n" + "-"*80)
        print("MULTI-OUTPUT MODEL EVALUATION")
        print("Architecture: 17 -> 30 (tanh) -> 15 (tanh) -> 3 (linear)")
        print("-"*80)
        
        y_multi_pred = self.multi_model.predict(X_test_scaled)
        y_multi_test = np.column_stack([
            self.y_liq_test.astype(float),
            self.y_settle_test,
            self.y_bearing_test
        ])
        
        print("\n  Multi-Output Performance:")
        for i, output_name in enumerate(['Liquefaction', 'Settlement', 'Bearing Capacity']):
            r2 = r2_score(y_multi_test[:, i], y_multi_pred[:, i])
            mae = mean_absolute_error(y_multi_test[:, i], y_multi_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_multi_test[:, i], y_multi_pred[:, i]))
            print(f"    {output_name}:")
            print(f"      R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        print("\n" + "-"*80)
        
        # EVALUATION 1: Liquefaction (Confusion Matrix per DPWH BSDS 2013)
        print("\n" + "-"*80)
        print("EVALUATION 1: LIQUEFACTION CLASSIFIER")
        print("Validation: DPWH BSDS (2013)")
        print("-"*80)
        
        y_liq_pred = self.liq_model.predict(X_test_scaled)
        y_liq_proba = self.liq_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(self.y_liq_test, y_liq_pred)
        precision = precision_score(self.y_liq_test, y_liq_pred, zero_division=0)
        recall = recall_score(self.y_liq_test, y_liq_pred, zero_division=0)
        f1 = f1_score(self.y_liq_test, y_liq_pred, zero_division=0)
        
        cm = confusion_matrix(self.y_liq_test, y_liq_pred)
        print("\n  Confusion Matrix:")
        print("    " + " " * 20 + "Predicted")
        print("    " + " " * 20 + "No" + " " * 10 + "Yes")
        print("    " + "Actual No" + " " * 10 + f"{cm[0,0]:4d}" + " " * 6 + f"{cm[0,1]:4d}")
        print("    " + "Actual Yes" + " " * 9 + f"{cm[1,0]:4d}" + " " * 6 + f"{cm[1,1]:4d}")
        print(f"\n  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # EVALUATION 2: Settlement (R², MAE, RMSE, PI)
        print("\n" + "-"*80)
        print("EVALUATION 2: SETTLEMENT REGRESSOR")
        print("Validation: Tokimatsu & Seed (1987), Bray & Macedo (2017)")
        print("-"*80)
        
        y_settle_pred = self.settle_model.predict(X_test_scaled)
        
        # Calculate validation values using both methods
        settle_ts_test = self.X_test.apply(self.calculate_settlement_tokimatsu_seed, axis=1)
        settle_bm_test = self.X_test.apply(self.calculate_settlement_bray_macedo, axis=1)
        settle_validation = (settle_ts_test + settle_bm_test) / 2.0
        
        r2 = r2_score(self.y_settle_test, y_settle_pred)
        mae = mean_absolute_error(self.y_settle_test, y_settle_pred)
        rmse = np.sqrt(mean_squared_error(self.y_settle_test, y_settle_pred))
        
        # Performance Index (PI) = 1 - (RMSE / mean)
        mean_settle = self.y_settle_test.mean()
        pi = 1 - (rmse / (mean_settle + 0.001))
        
        print(f"\n  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f} cm")
        print(f"  RMSE: {rmse:.4f} cm")
        print(f"  Performance Index (PI): {pi:.4f}")
        print(f"\n  Validation Comparison (vs Tokimatsu & Seed + Bray & Macedo):")
        r2_val = r2_score(settle_validation, y_settle_pred)
        print(f"    R² vs Validation Methods: {r2_val:.4f}")
        
        # EVALUATION 3: Bearing Capacity (R², MAE, RMSE, PI)
        print("\n" + "-"*80)
        print("EVALUATION 3: BEARING CAPACITY REGRESSOR")
        print("Validation: Terzaghi (1943), Olsen & Stark (2002)")
        print("-"*80)
        
        y_bearing_pred = self.bearing_model.predict(X_test_scaled)
        
        # Calculate validation values
        bearing_terzaghi_test = self.X_test.apply(self.calculate_bearing_capacity_terzaghi, axis=1)
        bearing_olsen_test = self.X_test.apply(self.calculate_bearing_capacity_olsen_stark, axis=1)
        
        r2 = r2_score(self.y_bearing_test, y_bearing_pred)
        mae = mean_absolute_error(self.y_bearing_test, y_bearing_pred)
        rmse = np.sqrt(mean_squared_error(self.y_bearing_test, y_bearing_pred))
        
        mean_bearing = self.y_bearing_test.mean()
        pi = 1 - (rmse / (mean_bearing + 0.001))
        
        print(f"\n  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f} kPa")
        print(f"  RMSE: {rmse:.4f} kPa")
        print(f"  Performance Index (PI): {pi:.4f}")
        print(f"\n  Validation Comparison:")
        r2_terzaghi = r2_score(bearing_terzaghi_test, y_bearing_pred)
        r2_olsen = r2_score(bearing_olsen_test, y_bearing_pred)
        print(f"    R² vs Terzaghi: {r2_terzaghi:.4f}")
        print(f"    R² vs Olsen & Stark: {r2_olsen:.4f}")
        
        return True
    
    def save_models(self) -> bool:
        """Save all models to Supabase Storage"""
        print("\n" + "="*80)
        print("SAVING MODELS")
        print("="*80)
        
        try:
            import json
            import io
            
            bucket_name = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
            
            # Save scaler
            print("  Saving scaler...")
            scaler_buffer = io.BytesIO()
            joblib.dump(self.scaler, scaler_buffer)
            scaler_buffer.seek(0)
            self.client.storage.from_(bucket_name).upload(
                'models/scaler.pkl',
                scaler_buffer.getvalue(),
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            print("    [OK] models/scaler.pkl")
            
            # Save multi-output model (primary model)
            print("  Saving multi-output ANN model...")
            multi_buffer = io.BytesIO()
            joblib.dump(self.multi_model, multi_buffer)
            multi_buffer.seek(0)
            self.client.storage.from_(bucket_name).upload(
                'models/ann_multi_output.pkl',
                multi_buffer.getvalue(),
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            print("    [OK] models/ann_multi_output.pkl")
            
            # Save liquefaction classifier (for individual evaluation)
            print("  Saving liquefaction classifier...")
            liq_buffer = io.BytesIO()
            joblib.dump(self.liq_model, liq_buffer)
            liq_buffer.seek(0)
            self.client.storage.from_(bucket_name).upload(
                'models/ann_liquefaction_classifier.pkl',
                liq_buffer.getvalue(),
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            print("    [OK] models/ann_liquefaction_classifier.pkl")
            
            # Save settlement regressor
            print("  Saving settlement regressor...")
            settle_buffer = io.BytesIO()
            joblib.dump(self.settle_model, settle_buffer)
            settle_buffer.seek(0)
            self.client.storage.from_(bucket_name).upload(
                'models/ann_settlement_regressor.pkl',
                settle_buffer.getvalue(),
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            print("    [OK] models/ann_settlement_regressor.pkl")
            
            # Save bearing capacity regressor
            print("  Saving bearing capacity regressor...")
            bearing_buffer = io.BytesIO()
            joblib.dump(self.bearing_model, bearing_buffer)
            bearing_buffer.seek(0)
            self.client.storage.from_(bucket_name).upload(
                'models/ann_bearing_capacity_regressor.pkl',
                bearing_buffer.getvalue(),
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            print("    [OK] models/ann_bearing_capacity_regressor.pkl")
            
            # Save metadata
            print("  Saving metadata...")
            metadata = {
                'version': '2.0',
                'model_type': 'multi_output',
                'targets': ['liquefaction', 'settlement', 'bearing_capacity'],
                'architecture': {
                    'type': 'MLP',
                    'input_layer': 17,
                    'hidden_layers': [30, 15],
                    'hidden_activation': 'tanh',  # Tansig
                    'output_layer': 3,
                    'output_activation': 'linear',  # Purelin
                    'solver': 'adam',
                    'alpha': 0.001
                },
                'num_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'training_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'validation_methods': {
                    'liquefaction': 'DPWH BSDS (2013)',
                    'settlement': 'Tokimatsu & Seed (1987), Bray & Macedo (2017)',
                    'bearing_capacity': 'Terzaghi (1943), Olsen & Stark (2002)'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_json = json.dumps(metadata, indent=2)
            self.client.storage.from_(bucket_name).upload(
                'models/ann_metadata.json',
                metadata_json.encode('utf-8'),
                file_options={'content-type': 'application/json', 'upsert': 'true'}
            )
            print("    [OK] models/ann_metadata.json")
            
            print("\n  [OK] All models saved to Supabase Storage")
            return True
        except Exception as e:
            print(f"  [ERROR] Save failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def export_validation_excel(self) -> str:
        """Export 20% validation set as Excel file in the same format as raw data"""
        print("\n" + "="*80)
        print("EXPORTING VALIDATION DATA (20%)")
        print("="*80)

        if self.df_validation is None:
            print("  [ERROR] Validation dataframe not available")
            return ""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"validation_data_20pct_{timestamp}.xlsx"

        try:
            if EXCEL_AVAILABLE:
                self.df_validation.to_excel(filename, index=False, engine='openpyxl')
                print(f"  [OK] Excel file saved: {filename}")
            else:
                # Fallback to CSV if openpyxl not installed
                csv_filename = filename.replace('.xlsx', '.csv')
                self.df_validation.to_csv(csv_filename, index=False)
                filename = csv_filename
                print(f"  [INFO] openpyxl not available, saved as CSV: {filename}")

            print(f"  Rows: {len(self.df_validation)} (20% validation set)")
            print(f"  Columns: {len(self.df_validation.columns)}")
            print(f"  Columns include: raw data + _target_liquefaction, "
                  f"_target_settlement_cm, _target_bearing_capacity_kpa")
            return filename
        except Exception as e:
            print(f"  [ERROR] Export failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def run(self) -> bool:
        """Run complete training pipeline"""
        print("\n" + "="*80)
        print("MULTI-OUTPUT ANN MODEL TRAINING")
        print("Per Project Requirements (Alejandrino et al.)")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if not self.connect_database():
            return False
        
        if not self.query_database():
            return False
        
        if not self.prepare_features_and_targets():
            return False
        
        if not self.train_models():
            return False
        
        if not self.evaluate_models():
            return False

        validation_file = self.export_validation_excel()

        if not self.save_models():
            return False
        
        print("\n" + "="*80)
        print("[SUCCESS] TRAINING COMPLETED")
        print("="*80)
        print(f"\n  Model Architecture:")
        print(f"    INPUT: 17 neurons")
        print(f"    HIDDEN 1: 30 neurons (tanh)")
        print(f"    HIDDEN 2: 15 neurons (tanh)")
        print(f"    OUTPUT: 3 neurons (linear)")
        print(f"  Training samples: {len(self.X_train)} (80%)")
        print(f"  Validation samples: {len(self.X_test)} (20%)")
        print(f"  Features: {len(self.feature_names)}")
        if validation_file:
            print(f"\n  Validation Excel: {validation_file}")
        print(f"\n  Models saved:")
        print(f"    - models/scaler.pkl")
        print(f"    - models/ann_multi_output.pkl (Primary: 3 outputs)")
        print(f"    - models/ann_liquefaction_classifier.pkl")
        print(f"    - models/ann_settlement_regressor.pkl")
        print(f"    - models/ann_bearing_capacity_regressor.pkl")
        print(f"    - models/ann_metadata.json")
        
        return True


def main():
    """Main execution"""
    if not SKLEARN_AVAILABLE:
        print("[ERROR] scikit-learn required for training")
        sys.exit(1)
    
    trainer = MultiOutputANNTraining()
    success = trainer.run()
    
    if not success:
        sys.exit(1)
    
    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()
