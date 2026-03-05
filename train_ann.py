#!/usr/bin/env python3
"""
Multi-Output ANN Model Training
Per Project Requirements (Alejandrino et al.)

Features:
- Direct database query from PostGIS
- Multi-output ANN: Foundation Width B (m) + Foundation Depth D (m) + L/B Ratio
- Architecture: 17 inputs -> 30 (tanh) -> 15 (tanh) -> 3 (linear) outputs
- Validation against: Meyerhof (1974), Bowles (1988), DPWH BSDS (2013)
- Performance metrics: R², MAE, RMSE, Performance Index (PI)
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
    - Foundation Width B (m)     [Meyerhof/Bowles — FS_bearing >= 3]
    - Foundation Depth D (m)     [1.5 m standard / 3.0 m liquefiable]
    - L/B Ratio (-)              [1.0 square footing — thesis standard]
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
    
    def compute_foundation_targets(self, row) -> dict:
        """
        Compute the three ANN output targets for a soil layer row:
          - Foundation Width  B   (m)
          - Foundation Depth  D   (m)
          - Length-to-Width   L/B (dimensionless)

        Rules derived from Bowles (1988) / Meyerhof (1974) empirical formula sheet:
          B is chosen so that FS_bearing ≥ 3 for q_actual = 50 kPa.
          D follows standard practice; deeper for liquefiable soils.
          L/B = 1.0 (square footing) as used throughout the thesis.

        Priority: use raw-data column values when available; compute otherwise.
        """
        Q_ACTUAL = 50.0  # kPa (default building contact pressure)
        D_STD    = 1.5   # m   (standard embedment depth)
        D_DEEP   = 3.0   # m   (deeper embedment for liquefiable soils)

        N = max(1.0, float(
            row.get('spt_n160') or row.get('spt_n60') or row.get('spt_n_value') or 15
        ))
        FS = float(row.get('factor_of_safety') or 2.0)

        # ── Foundation Width B ───────────────────────────────────────────────
        # Meyerhof bearing capacity at D=D_STD, small B (Kd capped at 1.33):
        #   Qu ≈ (N/2.5) × 47.88 × 1.33  →  Qa = Qu/3 ≈ N × 8.49 kPa
        # For Qa ≥ Q_ACTUAL = 50 kPa:  N_min ≈ 5.9
        #   N ≥ 6  → B = 2.0 m (standard)
        #   N < 6  → wider footing needed: B = round_0.5(7.83 / N), cap [2.5, 5.0]
        B_raw = row.get('Foundation Width (B)') or row.get('foundation_width_m')
        try:
            B = float(B_raw)
            if B <= 0:
                raise ValueError
        except (TypeError, ValueError):
            if N >= 6.0:
                B = 2.0
            else:
                # Increase width for weaker soils; round to nearest 0.5 m
                B_calc = Q_ACTUAL / (N * 8.49)      # approximate from Qa = N×8.49
                B = max(2.5, round(B_calc * 2) / 2)  # round to 0.5 m steps
                B = min(5.0, B)

        # ── Foundation Depth D ───────────────────────────────────────────────
        D_raw = row.get('Foundation Depth (D)') or row.get('foundation_depth_m')
        try:
            D = float(D_raw)
            if D <= 0:
                raise ValueError
        except (TypeError, ValueError):
            D = D_DEEP if FS < 1.0 else D_STD

        # ── L/B Ratio ────────────────────────────────────────────────────────
        lb_raw = row.get('Length-to-width ratio (L/B)') or row.get('lb_ratio')
        try:
            lb = float(lb_raw)
            if lb <= 0:
                raise ValueError
        except (TypeError, ValueError):
            lb = 1.0   # square footing (thesis standard)

        return {'B': round(B, 2), 'D': round(D, 2), 'lb_ratio': round(lb, 2)}

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
        
        # Priority features — thesis-required inputs first, then derived values.
        # friction_angle, moisture_content, plasticity_index, mean_particle_size_d50
        # are now extracted from Excel so they should be present in the training data.
        priority_features = [
            'spt_n_value', 'spt_n160',
            'unit_weight', 'fines_content', 'groundwater_depth_m',
            'pga_g', 'csr', 'cyclic_strength_ratio',
            'effective_overburden_pressure', 'total_overburden_pressure',
            'depth_mid_m', 'relative_density_percent',
            'friction_angle', 'moisture_content',
            'plasticity_index', 'mean_particle_size_d50',
            'elastic_modulus_es',
        ]

        # Get available priority features (only those present in the dataset)
        available_priority = [col for col in priority_features if col in df.columns]

        # Fill remaining slots with other numeric features if priority list is short
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

        # Use however many priority features are available; pad only if fewer than 17
        feature_cols = list(available_priority)
        if len(feature_cols) < 17:
            needed = 17 - len(feature_cols)
            feature_cols.extend(other_features[:needed])

        if len(feature_cols) < 17:
            print(f"  [WARNING] Only {len(feature_cols)} features available, padding to 17")
            for i in range(len(feature_cols), 17):
                df[f'feature_{i}'] = 0.0
                feature_cols.append(f'feature_{i}')

        print(f"  Selected {len(feature_cols)} feature columns")
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
        
        # ── Compute foundation design targets ────────────────────────────────
        # TARGET 1: Foundation Width  B   (m)   — Bowles/Meyerhof-driven
        # TARGET 2: Foundation Depth  D   (m)   — 1.5 m standard / 3.0 m liquefiable
        # TARGET 3: L/B Ratio              (-)   — 1.0 for square footing (thesis)
        print("\n  Preparing TARGET 1: Foundation Width B (m) ...")
        print("  Preparing TARGET 2: Foundation Depth D (m) ...")
        print("  Preparing TARGET 3: L/B Ratio (-) ...")

        foundation_targets = df.apply(self.compute_foundation_targets, axis=1)
        y_B  = foundation_targets.apply(lambda r: r['B'])
        y_D  = foundation_targets.apply(lambda r: r['D'])
        y_lb = foundation_targets.apply(lambda r: r['lb_ratio'])

        # Remove rows with missing targets (all computed so should be none)
        valid_mask = ~(y_B.isna() | y_D.isna() | y_lb.isna())
        X    = X[valid_mask]
        y_B  = y_B[valid_mask]
        y_D  = y_D[valid_mask]
        y_lb = y_lb[valid_mask]

        print(f"\n  Features shape: {X.shape}")
        print(f"  Target 1 (B):   Mean={y_B.mean():.2f} m,  Range=[{y_B.min():.2f}, {y_B.max():.2f}] m")
        print(f"  Target 2 (D):   Mean={y_D.mean():.2f} m,  Range=[{y_D.min():.2f}, {y_D.max():.2f}] m")
        print(f"  Target 3 (L/B): Mean={y_lb.mean():.2f},   Range=[{y_lb.min():.2f}, {y_lb.max():.2f}]")

        # ── Store for model training and evaluation ───────────────────────────
        self.y_liq_train = self.y_liq_test = None          # no longer used
        self.y_settle_train = self.y_settle_test = None
        self.y_bearing_train = self.y_bearing_test = None

        # Split data: 80% train, 20% validation (per project requirements)
        print("\n  Splitting data (80% train, 20% validation)...")
        self.X_train, self.X_test, self.y_B_train, self.y_B_test = train_test_split(
            X, y_B, test_size=0.2, random_state=42
        )
        _, _, self.y_D_train,  self.y_D_test  = train_test_split(X, y_D,  test_size=0.2, random_state=42)
        _, _, self.y_lb_train, self.y_lb_test = train_test_split(X, y_lb, test_size=0.2, random_state=42)

        # Build validation dataframe in raw data format using test indices
        val_indices = self.X_test.index
        val_raw = self.df_raw.loc[val_indices].copy().reset_index(drop=True)
        val_raw['_target_foundation_width_B_m']  = self.y_B_test.values
        val_raw['_target_foundation_depth_D_m']  = self.y_D_test.values
        val_raw['_target_lb_ratio']              = self.y_lb_test.values
        self.df_validation = val_raw

        print(f"  Training set  : {len(self.X_train)} samples (80%)")
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
        print("  Outputs: [Foundation Width B (m), Foundation Depth D (m), L/B Ratio]")
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
        
        # Multi-output target: [B (m), D (m), L/B ratio]
        print("\n  Preparing multi-output target [B, D, L/B]...")
        y_multi_train = np.column_stack([
            self.y_B_train,
            self.y_D_train,
            self.y_lb_train,
        ])
        y_multi_test = np.column_stack([
            self.y_B_test,
            self.y_D_test,
            self.y_lb_test,
        ])
        
        # Train multi-output regressor (all 3 outputs: B, D, L/B)
        print("\n  Training Multi-Output ANN...")
        print("    Architecture: 17 -> 30 (tanh) -> 15 (tanh) -> 3 (linear)")
        print("    Outputs: Foundation Width B (m) | Foundation Depth D (m) | L/B Ratio")
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
        
        # Also train individual models for per-output evaluation
        print("\n  Training individual models for validation...")

        _mlp_kwargs = dict(
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
            verbose=False,
        )

        # MODEL 1: Foundation Width B
        self.liq_model = MLPRegressor(**_mlp_kwargs)   # reuse slot, now predicts B
        self.liq_model.fit(X_train_scaled, self.y_B_train)
        print(f"    [OK] Foundation Width (B) regressor trained")

        # MODEL 2: Foundation Depth D
        self.settle_model = MLPRegressor(**_mlp_kwargs)  # reuse slot, now predicts D
        self.settle_model.fit(X_train_scaled, self.y_D_train)
        print(f"    [OK] Foundation Depth (D) regressor trained")

        # MODEL 3: L/B Ratio
        self.bearing_model = MLPRegressor(**_mlp_kwargs)  # reuse slot, now predicts L/B
        self.bearing_model.fit(X_train_scaled, self.y_lb_train)
        print(f"    [OK] L/B Ratio regressor trained")
        
        return True
    
    def evaluate_models(self) -> bool:
        """Evaluate all models with required metrics per project requirements"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        X_test_scaled = self.scaler.transform(self.X_test)

        def _reg_metrics(name, y_true, y_pred, unit=''):
            r2   = r2_score(y_true, y_pred)
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            pi   = 1.0 - rmse / (float(y_true.mean()) + 1e-9)
            print(f"\n  {name}:")
            print(f"    R²  : {r2:.4f}")
            print(f"    MAE : {mae:.4f}{unit}")
            print(f"    RMSE: {rmse:.4f}{unit}")
            print(f"    PI  : {pi:.4f}")
            return r2, mae, rmse, pi

        # ── Multi-output model ────────────────────────────────────────────────
        print("\n" + "-"*80)
        print("MULTI-OUTPUT MODEL EVALUATION")
        print("Architecture: 17 -> 30 (tanh) -> 15 (tanh) -> 3 (linear)")
        print("Outputs: Foundation Width B (m) | Foundation Depth D (m) | L/B Ratio")
        print("-"*80)

        y_multi_pred = self.multi_model.predict(X_test_scaled)
        y_multi_test = np.column_stack([self.y_B_test, self.y_D_test, self.y_lb_test])

        print("\n  Multi-Output Performance:")
        for i, (out_name, unit) in enumerate([
            ('Foundation Width B',  ' m'),
            ('Foundation Depth D',  ' m'),
            ('L/B Ratio',           ''),
        ]):
            r2   = r2_score(y_multi_test[:, i], y_multi_pred[:, i])
            mae  = mean_absolute_error(y_multi_test[:, i], y_multi_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_multi_test[:, i], y_multi_pred[:, i]))
            print(f"    {out_name}: R²={r2:.4f}  MAE={mae:.4f}{unit}  RMSE={rmse:.4f}{unit}")

        # ── Individual model evaluations ──────────────────────────────────────
        print("\n" + "-"*80)
        print("EVALUATION 1: FOUNDATION WIDTH B (m)")
        print("Basis: Meyerhof (1974) / Bowles (1988) — FS_bearing >= 3, q_actual=50 kPa")
        print("-"*80)
        y_B_pred = self.liq_model.predict(X_test_scaled)
        _reg_metrics("Foundation Width B", self.y_B_test, y_B_pred, ' m')

        print("\n" + "-"*80)
        print("EVALUATION 2: FOUNDATION DEPTH D (m)")
        print("Basis: Standard 1.5 m / 3.0 m for liquefiable soils")
        print("-"*80)
        y_D_pred = self.settle_model.predict(X_test_scaled)
        _reg_metrics("Foundation Depth D", self.y_D_test, y_D_pred, ' m')

        print("\n" + "-"*80)
        print("EVALUATION 3: L/B RATIO (-)")
        print("Basis: 1.0 square footing (thesis standard)")
        print("-"*80)
        y_lb_pred = self.bearing_model.predict(X_test_scaled)
        _reg_metrics("L/B Ratio", self.y_lb_test, y_lb_pred, '')

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
            
            # Save Foundation Width B regressor
            print("  Saving Foundation Width B regressor...")
            liq_buffer = io.BytesIO()
            joblib.dump(self.liq_model, liq_buffer)
            liq_buffer.seek(0)
            self.client.storage.from_(bucket_name).upload(
                'models/ann_foundation_width_regressor.pkl',
                liq_buffer.getvalue(),
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            print("    [OK] models/ann_foundation_width_regressor.pkl")

            # Save Foundation Depth D regressor
            print("  Saving Foundation Depth D regressor...")
            settle_buffer = io.BytesIO()
            joblib.dump(self.settle_model, settle_buffer)
            settle_buffer.seek(0)
            self.client.storage.from_(bucket_name).upload(
                'models/ann_foundation_depth_regressor.pkl',
                settle_buffer.getvalue(),
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            print("    [OK] models/ann_foundation_depth_regressor.pkl")

            # Save L/B Ratio regressor
            print("  Saving L/B Ratio regressor...")
            bearing_buffer = io.BytesIO()
            joblib.dump(self.bearing_model, bearing_buffer)
            bearing_buffer.seek(0)
            self.client.storage.from_(bucket_name).upload(
                'models/ann_lb_ratio_regressor.pkl',
                bearing_buffer.getvalue(),
                file_options={'content-type': 'application/octet-stream', 'upsert': 'true'}
            )
            print("    [OK] models/ann_lb_ratio_regressor.pkl")
            
            # Save metadata
            print("  Saving metadata...")
            metadata = {
                'version': '2.0',
                'model_type': 'multi_output',
                'targets': ['foundation_width_B', 'foundation_depth_D', 'lb_ratio'],
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
                    'foundation_width_B': 'Meyerhof (1974) / Bowles (1988) — FS_bearing >= 3, q_actual=50 kPa',
                    'foundation_depth_D': 'Standard 1.5 m (normal) / 3.0 m (liquefiable, FS < 1.0)',
                    'lb_ratio': 'Square footing L/B = 1.0 (thesis standard)',
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
        """Export 20% validation set as Excel/CSV and upload to Supabase Storage"""
        print("\n" + "="*80)
        print("EXPORTING VALIDATION DATA (20%)")
        print("="*80)

        if self.df_validation is None:
            print("  [ERROR] Validation dataframe not available")
            return ""

        import io as _io

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bucket_name = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')

        try:
            if EXCEL_AVAILABLE:
                filename = f"validation_data_20pct_{timestamp}.xlsx"
                storage_path = f"validation/{filename}"
                content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

                # Write to in-memory buffer for upload
                buf = _io.BytesIO()
                self.df_validation.to_excel(buf, index=False, engine='openpyxl')
                file_bytes = buf.getvalue()
            else:
                filename = f"validation_data_20pct_{timestamp}.csv"
                storage_path = f"validation/{filename}"
                content_type = 'text/csv'

                buf = _io.StringIO()
                self.df_validation.to_csv(buf, index=False)
                file_bytes = buf.getvalue().encode('utf-8')

            print(f"  Rows: {len(self.df_validation)} (20% validation set)")
            print(f"  Columns: {len(self.df_validation.columns)}")

            # Upload to Supabase Storage
            print(f"  Uploading to Supabase Storage: {storage_path} ...")
            self.client.storage.from_(bucket_name).upload(
                storage_path,
                file_bytes,
                file_options={'content-type': content_type, 'upsert': 'true'}
            )
            print(f"  [OK] Uploaded to bucket '{bucket_name}' -> {storage_path}")

            return filename
        except Exception as e:
            print(f"  [ERROR] Export/upload failed: {e}")
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
            print(f"\n  Validation file (bucket): validation/{validation_file}")
        print(f"\n  Models saved:")
        print(f"    - models/scaler.pkl")
        print(f"    - models/ann_multi_output.pkl (Primary: 3 outputs)")
        print(f"    - models/ann_foundation_width_regressor.pkl")
        print(f"    - models/ann_foundation_depth_regressor.pkl")
        print(f"    - models/ann_lb_ratio_regressor.pkl")
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
