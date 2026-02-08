"""
FIXED ANN-Based Machine Learning Training Pipeline - ALIGNED WITH SINGLE TARGET
Tarlac Province Geotechnical Data - Version 2.0

CRITICAL FIXES v2.0:
- ALIGNED with feature engineering output (single target: qa_allowable_kpa)
- Removed liquefaction and settlement models (no valid data)
- Simplified to single regression model for bearing capacity
- Fixed data loading to match feature engineering column names
- Handle all edge cases for bearing capacity prediction

Previous issues:
1. [FIXED] Feature engineering outputs single target but training expected 3 targets
2. [FIXED] Column name mismatches between feature engineering and training
3. [FIXED] Unnecessary multi-target complexity removed
4. [FIXED] Proper handling of NaN values in target variable

Based on Research Methodology (Chapter 2):
- Phase 2: Model Development and Training (Steps 3-7)
- Phase 3: Performance Evaluation (Steps 8-9)
- Validates against: Terzaghi (1943) bearing capacity theory

Author: Aligned Geotechnical ML Pipeline
Date: 2026-02-04
Version: 2.0-ALIGNED
"""
from datetime import datetime
import sys
import json
import io
import numpy as np
import pandas as pd
import warnings
import os
sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings('ignore')

# Print version information
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

# Check for required libraries
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score
    )
    import sklearn
    import joblib
    SKLEARN_AVAILABLE = True
    print(f"[OK] scikit-learn version: {sklearn.__version__}")
    print(f"[OK] joblib version: {joblib.__version__}")
except ImportError as e:
    print(f"[X] scikit-learn not installed! Error: {e}")
    print("Install with: pip install scikit-learn joblib")
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    print(f"[OK] Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(
        f"[!] Matplotlib/Seaborn not available - plots will be skipped. Error: {e}")
    PLOTTING_AVAILABLE = False

try:
    from supabase_client import get_supabase_client
    SUPABASE_AVAILABLE = True
    print("[OK] Supabase client module loaded")
except ImportError as e:
    print(f"[!] Supabase client not available: {e}")
    print("[!] Running in local mode - storage functions will be skipped")
    SUPABASE_AVAILABLE = False


# -----------------------------
# Supabase Storage Functions
# -----------------------------

def download_file_from_storage(bucket_name, file_path):
    """Download file from Supabase Storage"""
    if not SUPABASE_AVAILABLE:
        print("[!] Supabase not available - skipping download")
        return None

    print(f"  Downloading {file_path} from Supabase Storage...")
    client = get_supabase_client()
    if not client:
        print("[X] Failed to connect to Supabase")
        return None

    try:
        response = client.storage.from_(bucket_name).download(file_path)
        print(f"  [OK] Downloaded {len(response)} bytes")
        return response
    except Exception as e:
        print(f"  [X] Error downloading file: {e}")
        return None


def upload_to_supabase_storage(local_file_path, bucket_name, storage_path):
    """Upload file to Supabase Storage"""
    if not SUPABASE_AVAILABLE:
        print("[!] Supabase not available - skipping upload")
        return False

    print(
        f" Uploading {os.path.basename(local_file_path)} to Supabase Storage...")
    client = get_supabase_client()
    if not client:
        return False

    try:
        with open(local_file_path, 'rb') as f:
            file_data = f.read()

        # Determine content type
        content_type_map = {
            '.pkl': 'application/octet-stream',
            '.joblib': 'application/octet-stream',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.png': 'image/png',
            '.csv': 'text/csv',
        }

        ext = os.path.splitext(local_file_path)[1].lower()
        content_type = content_type_map.get(ext, 'application/octet-stream')

        client.storage.from_(bucket_name).upload(
            storage_path,
            file_data,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )
        print(f"  [OK] Uploaded to {storage_path}")
        return True

    except Exception as e:
        print(f"  [X] Error uploading: {e}")
        return False


# -----------------------------
# FIXED ANN Training Pipeline (Single Target)
# -----------------------------

class FixedBearingCapacityANNPipeline:
    """
    FIXED ANN-Based Machine Learning Pipeline for Bearing Capacity Prediction

    Key Changes v2.0:
    - SINGLE TARGET ONLY: qa_allowable_kpa (Allowable Bearing Capacity)
    - Removed liquefaction and settlement models (no valid data)
    - Aligned with feature engineering output
    - Simplified architecture for single regression task
    - Proper handling of all edge cases
    """

    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.feature_names = None
        self.scaler = None

        # Single model for bearing capacity prediction
        self.ann_bearing_capacity = None

        self.results = {}
        self.feature_importance = {}
        self.cross_val_scores = {}

    def load_data_from_storage(self, bucket_name, base_path='feature_engineering'):
        """Load training data from Supabase Storage - ALIGNED VERSION"""
        print("=" * 80)
        print("LOADING TRAINING DATA FROM SUPABASE STORAGE (ALIGNED)")
        print("=" * 80)

        try:
            # Download train data
            train_bytes = download_file_from_storage(
                bucket_name, f'{base_path}/train_FIXED.csv')
            if not train_bytes:
                return False
            train_df = pd.read_csv(io.BytesIO(train_bytes))

            # Download validation data
            val_bytes = download_file_from_storage(
                bucket_name, f'{base_path}/validation_FIXED.csv')
            if not val_bytes:
                return False
            val_df = pd.read_csv(io.BytesIO(val_bytes))

            # Download test data
            test_bytes = download_file_from_storage(
                bucket_name, f'{base_path}/test_FIXED.csv')
            if not test_bytes:
                return False
            test_df = pd.read_csv(io.BytesIO(test_bytes))

            # SINGLE target column
            target_col = 'qa_allowable_kpa'

            # Check if target exists
            if target_col not in train_df.columns:
                print(f"[X] Target column '{target_col}' not found in data!")
                print(
                    f"Available columns: {train_df.columns.tolist()[:10]}...")
                return False

            # Separate features and target
            self.feature_names = [
                col for col in train_df.columns if col != target_col]

            # Extract features
            self.X_train = train_df[self.feature_names].values
            self.X_val = val_df[self.feature_names].values
            self.X_test = test_df[self.feature_names].values

            # Extract target (bearing capacity)
            self.y_train = train_df[target_col].values
            self.y_val = val_df[target_col].values
            self.y_test = test_df[target_col].values

            print(f"\n[OK] Data loaded successfully!")
            print(f"  - Features: {len(self.feature_names)}")
            print(f"  - Training samples: {len(self.X_train)}")
            print(f"  - Validation samples: {len(self.X_val)}")
            print(f"  - Test samples: {len(self.X_test)}")

            # Check target data quality
            train_valid = ~np.isnan(self.y_train)
            val_valid = ~np.isnan(self.y_val)
            test_valid = ~np.isnan(self.y_test)

            print(f"\n  - Target variable: {target_col}")
            print(
                f"    Training: {train_valid.sum()}/{len(self.y_train)} valid samples")
            if train_valid.sum() > 0:
                print(
                    f"      Mean={np.nanmean(self.y_train):.2f} kPa, Std={np.nanstd(self.y_train):.2f} kPa")
                print(
                    f"      Min={np.nanmin(self.y_train):.2f} kPa, Max={np.nanmax(self.y_train):.2f} kPa")
            else:
                print(f"          WARNING: No valid target data in training set!")

            print(
                f"    Validation: {val_valid.sum()}/{len(self.y_val)} valid samples")
            print(
                f"    Test: {test_valid.sum()}/{len(self.y_test)} valid samples")

            return True

        except Exception as e:
            print(f"[X] Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_data(self, use_robust_scaler=False):
        """Standardize features using StandardScaler or RobustScaler"""
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA WITH ENHANCED SCALING")
        print("=" * 80)

        if use_robust_scaler:
            print("\n   Using RobustScaler (robust to outliers)...")
            self.scaler = RobustScaler()
        else:
            print("\n   Using StandardScaler (standard normalization)...")
            self.scaler = StandardScaler()

        # Fit on training data and transform all sets
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        print("[OK] Features standardized successfully!")
        print(f"  - Scaler type: {type(self.scaler).__name__}")
        print(f"  - Feature shape: {self.X_train.shape}")

        return True

    def load_data_for_target(self, bucket_name, target_name, base_path='feature_engineering'):
        """Load train/val/test for a specific target exported by feature engineering"""
        print("\n" + "=" * 60)
        print(f"LOADING DATA FOR TARGET: {target_name}")
        print("=" * 60)

        try:
            target_file = target_name.lower().replace(' ', '_')
            train_path = f"{base_path}/train_{target_file}_FIXED.csv"
            val_path = f"{base_path}/validation_{target_file}_FIXED.csv"
            test_path = f"{base_path}/test_{target_file}_FIXED.csv"

            train_bytes = download_file_from_storage(bucket_name, train_path)
            if not train_bytes:
                print(f"[X] Could not download {train_path}")
                return False
            train_df = pd.read_csv(io.BytesIO(train_bytes))

            val_bytes = download_file_from_storage(bucket_name, val_path)
            if not val_bytes:
                print(f"[X] Could not download {val_path}")
                return False
            val_df = pd.read_csv(io.BytesIO(val_bytes))

            test_bytes = download_file_from_storage(bucket_name, test_path)
            if not test_bytes:
                print(f"[X] Could not download {test_path}")
                return False
            test_df = pd.read_csv(io.BytesIO(test_bytes))

            # Check target exists
            if target_name not in train_df.columns:
                print(
                    f"[X] Target column '{target_name}' not found in {train_path}!")
                return False

            # Set feature names (all except target)
            self.feature_names = [
                col for col in train_df.columns if col != target_name]

            # Extract arrays
            self.X_train = train_df[self.feature_names].values
            self.X_val = val_df[self.feature_names].values
            self.X_test = test_df[self.feature_names].values

            self.y_train = train_df[target_name].values
            self.y_val = val_df[target_name].values
            self.y_test = test_df[target_name].values

            print(
                f"[OK] Loaded data for {target_name}: features={len(self.feature_names)}, train={len(self.X_train)}")
            return True

        except Exception as e:
            print(f"[X] Failed to load data for {target_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def train_bearing_capacity_ann(self, perform_cv=True):
        """Train ANN for Bearing Capacity Prediction - WITH COMPREHENSIVE ERROR HANDLING"""
        print("\n" + "=" * 80)
        print("TRAINING ANN - ALLOWABLE BEARING CAPACITY (REGRESSION)")
        print("=" * 80)

        # Check for NaN values
        valid_indices = ~np.isnan(self.y_train)
        num_valid = valid_indices.sum()

        print(f"\n   Data quality check:")
        print(f"  - Total training samples: {len(self.y_train)}")
        print(f"  - Valid samples (non-NaN): {num_valid}")
        print(f"  - NaN samples: {len(self.y_train) - num_valid}")

        # CRITICAL CHECK: Handle case where ALL values are NaN
        if num_valid == 0:
            print("\n    CRITICAL WARNING: No valid bearing capacity data available!")
            print("  - All training samples have NaN values for bearing capacity")
            print("  - This model cannot be trained with the current dataset")
            print("  - Pipeline will terminate here")

            # Set model to None
            self.ann_bearing_capacity = None

            # Store empty results
            self.results['bearing_capacity'] = {
                'status': 'failed',
                'reason': 'No valid training data (all NaN values)',
                'total_samples': len(self.y_train),
                'valid_samples': 0
            }

            return None

        # Warning for very few samples
        if num_valid < 10:
            print(f"\n    WARNING: Very few valid samples ({num_valid})")
            print("  - Model training may be unreliable")
            print("  - Results should be interpreted with caution")

        # Use only valid samples
        X_train_clean = self.X_train[valid_indices]
        y_train_clean = self.y_train[valid_indices]

        print(f"\n    Proceeding with {num_valid} valid samples")
        print(f"    - Mean: {y_train_clean.mean():.2f} kPa")
        print(f"    - Std: {y_train_clean.std():.2f} kPa")
        print(f"    - Min: {y_train_clean.min():.2f} kPa")
        print(f"    - Max: {y_train_clean.max():.2f} kPa")

        print("\n   ANN Configuration:")
        print("  - Task: Regression (Allowable Bearing Capacity in kPa)")
        print("  - Hidden layers: (256, 128, 64)")
        print("  - Activation: ReLU (hidden), Identity (output)")
        print("  - Solver: Adam optimizer")
        print("  - Max iterations: 1000")
        print("  - Early stopping: Enabled (patience=10)")
        print("  - Alpha (L2 penalty): 0.0001")

        self.ann_bearing_capacity = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            alpha=0.0001,
            learning_rate='adaptive',
            verbose=False
        )

        print("\n   Training ANN for bearing capacity prediction...")
        self.ann_bearing_capacity.fit(X_train_clean, y_train_clean)

        print("[OK] ANN training completed!")
        print(
            f"  - Final training loss: {self.ann_bearing_capacity.loss_:.6f}")
        print(f"  - Iterations: {self.ann_bearing_capacity.n_iter_}")
        print(
            f"  - Converged: {'Yes' if self.ann_bearing_capacity.n_iter_ < 1000 else 'No (max iterations reached)'}")

        # Cross-validation (only if we have enough samples)
        if perform_cv and num_valid >= 10:
            print("\n   Performing 5-fold cross-validation...")
            cv = KFold(n_splits=min(5, num_valid),
                       shuffle=True, random_state=42)
            try:
                cv_scores = cross_val_score(
                    self.ann_bearing_capacity, X_train_clean, y_train_clean,
                    cv=cv, scoring='r2'
                )
                self.cross_val_scores['bearing_capacity'] = cv_scores
                print(f"  - CV R² scores: {cv_scores}")
                print(
                    f"  - Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            except Exception as e:
                print(f"      Cross-validation failed: {e}")
        else:
            print(
                f"\n    Skipping cross-validation (insufficient valid samples: {num_valid})")

        return self.ann_bearing_capacity

    def train_liquefaction_ann(self, perform_cv=True):
        """Train ANN to predict liquefaction probability (0-100%) as regression"""
        print("\n" + "=" * 80)
        print("TRAINING ANN - LIQUEFACTION PROBABILITY (REGRESSION)")
        print("=" * 80)

        # Check for NaN values
        valid_indices = ~np.isnan(self.y_train)
        num_valid = valid_indices.sum()

        print(
            f"\n   Data quality check: valid samples: {num_valid}/{len(self.y_train)}")
        if num_valid == 0:
            print("  CRITICAL WARNING: No valid liquefaction data available!")
            self.ann_liquefaction = None
            self.results['liquefaction'] = {
                'status': 'failed', 'reason': 'No valid training data'}
            return None

        X_train_clean = self.X_train[valid_indices]
        y_train_clean = self.y_train[valid_indices]

        # Smaller ANN for probability regression
        self.ann_liquefaction = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=800,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            alpha=1e-4,
            learning_rate='adaptive',
            verbose=False
        )

        self.ann_liquefaction.fit(X_train_clean, y_train_clean)
        print("[OK] Liquefaction ANN trained")

        if perform_cv and num_valid >= 10:
            try:
                cv = KFold(n_splits=min(5, num_valid),
                           shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    self.ann_liquefaction, X_train_clean, y_train_clean, cv=cv, scoring='r2')
                self.cross_val_scores['liquefaction'] = cv_scores
                print(f"  - CV R²: {cv_scores}")
            except Exception as e:
                print(f"  CV failed for liquefaction: {e}")

        self.results['liquefaction'] = {'status': 'success'}
        return self.ann_liquefaction

    def train_settlement_ann(self, perform_cv=True):
        """Train ANN to predict settlement in cm (regression)"""
        print("\n" + "=" * 80)
        print("TRAINING ANN - SETTLEMENT (cm) (REGRESSION)")
        print("=" * 80)

        valid_indices = ~np.isnan(self.y_train)
        num_valid = valid_indices.sum()
        print(
            f"\n   Data quality check: valid samples: {num_valid}/{len(self.y_train)}")
        if num_valid == 0:
            print("  CRITICAL WARNING: No valid settlement data available!")
            self.ann_settlement = None
            self.results['settlement'] = {
                'status': 'failed', 'reason': 'No valid training data'}
            return None

        X_train_clean = self.X_train[valid_indices]
        y_train_clean = self.y_train[valid_indices]

        self.ann_settlement = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=800,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            alpha=1e-4,
            learning_rate='adaptive',
            verbose=False
        )

        self.ann_settlement.fit(X_train_clean, y_train_clean)
        print("[OK] Settlement ANN trained")

        if perform_cv and num_valid >= 10:
            try:
                cv = KFold(n_splits=min(5, num_valid),
                           shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    self.ann_settlement, X_train_clean, y_train_clean, cv=cv, scoring='r2')
                self.cross_val_scores['settlement'] = cv_scores
                print(f"  - CV R²: {cv_scores}")
            except Exception as e:
                print(f"  CV failed for settlement: {e}")

        self.results['settlement'] = {'status': 'success'}
        return self.ann_settlement

    def evaluate_bearing_capacity_model(self):
        """Evaluate Bearing Capacity Regression Model"""
        print("\n" + "=" * 80)
        print("EVALUATION - BEARING CAPACITY MODEL")
        print("=" * 80)

        # Check if model was trained
        if self.ann_bearing_capacity is None:
            print("\n    Bearing capacity model was not trained (no valid data)")
            print("  - Skipping evaluation")

            if 'bearing_capacity' not in self.results:
                self.results['bearing_capacity'] = {
                    'status': 'failed',
                    'reason': 'No valid training data'
                }

            return self.results['bearing_capacity']

        # Handle NaN values in validation/test sets
        val_valid = ~np.isnan(self.y_val)
        test_valid = ~np.isnan(self.y_test)

        num_val_valid = val_valid.sum()
        num_test_valid = test_valid.sum()

        print(f"\n   Evaluation data quality:")
        print(
            f"  - Valid validation samples: {num_val_valid}/{len(self.y_val)}")
        print(f"  - Valid test samples: {num_test_valid}/{len(self.y_test)}")

        if num_val_valid == 0 or num_test_valid == 0:
            print(f"\n    Insufficient valid data for evaluation")

            self.results['bearing_capacity'] = {
                'status': 'incomplete',
                'reason': 'Insufficient valid evaluation data',
                'validation': {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'), 'mape': float('nan')},
                'test': {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'), 'mape': float('nan')}
            }

            return self.results['bearing_capacity']

        # Validation set (with valid samples only)
        y_val_pred = self.ann_bearing_capacity.predict(self.X_val[val_valid])
        val_rmse = np.sqrt(mean_squared_error(
            self.y_val[val_valid], y_val_pred))
        val_mae = mean_absolute_error(self.y_val[val_valid], y_val_pred)
        val_r2 = r2_score(self.y_val[val_valid], y_val_pred)
        val_mape = np.mean(np.abs(
            (self.y_val[val_valid] - y_val_pred) / (self.y_val[val_valid] + 1e-10))) * 100

        print("\n[SUCCESS] VALIDATION SET PERFORMANCE:")
        print(f"  RMSE: {val_rmse:.2f} kPa")
        print(f"  MAE:  {val_mae:.2f} kPa")
        print(f"  R²:   {val_r2:.4f}")
        print(f"  MAPE: {val_mape:.2f}%")

        # Test set (with valid samples only)
        y_test_pred = self.ann_bearing_capacity.predict(
            self.X_test[test_valid])
        test_rmse = np.sqrt(mean_squared_error(
            self.y_test[test_valid], y_test_pred))
        test_mae = mean_absolute_error(self.y_test[test_valid], y_test_pred)
        test_r2 = r2_score(self.y_test[test_valid], y_test_pred)
        test_mape = np.mean(np.abs(
            (self.y_test[test_valid] - y_test_pred) / (self.y_test[test_valid] + 1e-10))) * 100

        print("\n[SUCCESS] TEST SET PERFORMANCE:")
        print(f"  RMSE: {test_rmse:.2f} kPa")
        print(f"  MAE:  {test_mae:.2f} kPa")
        print(f"  R²:   {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")

        self.results['bearing_capacity'] = {
            'status': 'success',
            'validation': {
                'rmse': float(val_rmse),
                'mae': float(val_mae),
                'r2': float(val_r2),
                'mape': float(val_mape),
                'valid_samples': int(num_val_valid)
            },
            'test': {
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'r2': float(test_r2),
                'mape': float(test_mape),
                'valid_samples': int(num_test_valid)
            },
            'cross_validation': {
                'mean_r2': float(self.cross_val_scores.get('bearing_capacity', np.array([0])).mean()),
                'std_r2': float(self.cross_val_scores.get('bearing_capacity', np.array([0])).std())
            } if 'bearing_capacity' in self.cross_val_scores else {}
        }

        return self.results['bearing_capacity']

    def analyze_feature_importance(self):
        """Analyze feature importance using connection weights"""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)

        if self.ann_bearing_capacity is None:
            print("    No model available for feature importance analysis")
            return False

        try:
            # Get weights from first layer
            weights_first_layer = np.abs(self.ann_bearing_capacity.coefs_[0])
            feature_importance = weights_first_layer.sum(axis=1)
            feature_importance = feature_importance / feature_importance.sum()

            # Get top 20 features
            top_indices = np.argsort(feature_importance)[-20:][::-1]

            print("\n   Top 20 Most Important Features for Bearing Capacity Prediction:")
            for i, idx in enumerate(top_indices, 1):
                print(
                    f"  {i:2d}. {self.feature_names[idx]:<40s} {feature_importance[idx]:.4f}")

            self.feature_importance['bearing_capacity'] = {
                self.feature_names[i]: float(feature_importance[i])
                for i in top_indices
            }

            return True

        except Exception as e:
            print(f"[!] Could not analyze feature importance: {e}")
            return False

    def save_models(self, output_dir='/mnt/user-data/outputs'):
        """Save trained model and metadata"""
        print("\n" + "=" * 80)
        print("SAVING TRAINED MODEL (VERSION-COMPATIBLE)")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        try:
            # Save scaler
            scaler_file = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_file, protocol=4, compress=3)
            print(f"[OK] Saved scaler: {scaler_file}")
            saved_files.append(scaler_file)

            # Save bearing capacity model (only if trained)
            if self.ann_bearing_capacity is not None:
                model_file = os.path.join(
                    output_dir, 'ann_bearing_capacity.pkl')
                joblib.dump(self.ann_bearing_capacity,
                            model_file, protocol=4, compress=3)
                print(f"[OK] Saved bearing capacity model: {model_file}")
                saved_files.append(model_file)
            else:
                print(f"[!] Skipping model save (not trained - no valid data)")

            # Save liquefaction model
            if hasattr(self, 'ann_liquefaction') and self.ann_liquefaction is not None:
                model_file = os.path.join(output_dir, 'ann_liquefaction.pkl')
                joblib.dump(self.ann_liquefaction, model_file,
                            protocol=4, compress=3)
                print(f"[OK] Saved liquefaction model: {model_file}")
                saved_files.append(model_file)

            # Save settlement model
            if hasattr(self, 'ann_settlement') and self.ann_settlement is not None:
                model_file = os.path.join(output_dir, 'ann_settlement.pkl')
                joblib.dump(self.ann_settlement, model_file,
                            protocol=4, compress=3)
                print(f"[OK] Saved settlement model: {model_file}")
                saved_files.append(model_file)

            # Save metadata
            metadata = {
                'version': 'fixed-v1.2',
                'sklearn_version': sklearn.__version__,
                'numpy_version': np.__version__,
                'joblib_version': joblib.__version__,
                'python_version': sys.version,
                'save_protocol': 4,
                'feature_names': self.feature_names,
                'num_features': len(self.feature_names),
                'training_samples': int(len(self.X_train)),
                'validation_samples': int(len(self.X_val)),
                'test_samples': int(len(self.X_test)),
                'timestamp': datetime.now().isoformat(),
                'target_variable': ['liquefaction_probability_pct', 'settlement_cm_calc', 'qa_allowable_kpa'],
                'problem_type': 'multi_target_regression',
                'model_architecture': {
                    'hidden_layers': [256, 128, 64],
                    'activation': 'relu',
                    'solver': 'adam',
                    'max_iter': 1000,
                    'early_stopping': True,
                    'alpha': 0.0001,
                    'learning_rate': 'adaptive'
                },
                'model_trained': self.ann_bearing_capacity is not None,
                'results': self.results,
                'feature_importance': self.feature_importance,
                'cross_validation_scores': {
                    k: v.tolist() if hasattr(v, 'tolist') else v
                    for k, v in self.cross_val_scores.items()
                }
            }

            metadata_file = os.path.join(output_dir, 'ann_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[OK] Saved metadata: {metadata_file}")
            saved_files.append(metadata_file)

            print(f"\n[SUCCESS] Successfully saved {len(saved_files)} files!")

        except Exception as e:
            print(f"[X] Error saving models: {e}")
            import traceback
            traceback.print_exc()

        return saved_files

    def generate_comprehensive_report(self, output_file='/mnt/user-data/outputs/ann_training_report.txt'):
        """Generate comprehensive training report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE TRAINING REPORT")
        print("=" * 80)

        os.makedirs('/mnt/user-data/outputs', exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ALIGNED ANN-BASED BEARING CAPACITY PREDICTION SYSTEM\n")
            f.write("Single-Target Regression Training Report\n")
            f.write("Tarlac Province, Philippines\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("VERSION INFORMATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Pipeline version: fixed-v1.2\n")
            f.write(f"Python version: {sys.version}\n")
            f.write(f"NumPy version: {np.__version__}\n")
            f.write(f"scikit-learn version: {sklearn.__version__}\n")
            f.write(f"joblib version: {joblib.__version__}\n\n")

            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of features: {len(self.feature_names)}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Validation samples: {len(self.X_val)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n")
            f.write(
                f"Target variable: qa_allowable_kpa (Allowable Bearing Capacity)\n\n")

            # Model architecture
            f.write("ANN MODEL ARCHITECTURE:\n")
            f.write("-" * 80 + "\n")
            f.write("Hidden Layer Structure: (256, 128, 64) neurons\n")
            f.write("Activation Function: ReLU (hidden), Identity (output)\n")
            f.write("Solver: Adam optimizer with adaptive learning rate\n")
            f.write("Max Iterations: 1000\n")
            f.write("Early Stopping: Enabled (patience=10)\n")
            f.write("L2 Regularization (alpha): 0.0001\n\n")

            # Results
            if 'bearing_capacity' in self.results:
                f.write("BEARING CAPACITY PREDICTION RESULTS:\n")
                f.write("-" * 80 + "\n")

                status = self.results['bearing_capacity'].get(
                    'status', 'success')

                if status == 'success':
                    bc_val = self.results['bearing_capacity']['validation']
                    bc_test = self.results['bearing_capacity']['test']

                    f.write("Validation Set:\n")
                    f.write(f"  RMSE: {bc_val['rmse']:.2f} kPa\n")
                    f.write(f"  MAE:  {bc_val['mae']:.2f} kPa\n")
                    f.write(f"  R²:   {bc_val['r2']:.4f}\n")
                    f.write(f"  MAPE: {bc_val['mape']:.2f}%\n")
                    f.write(
                        f"  Valid samples: {bc_val.get('valid_samples', 'N/A')}\n")

                    f.write("\nTest Set:\n")
                    f.write(f"  RMSE: {bc_test['rmse']:.2f} kPa\n")
                    f.write(f"  MAE:  {bc_test['mae']:.2f} kPa\n")
                    f.write(f"  R²:   {bc_test['r2']:.4f}\n")
                    f.write(f"  MAPE: {bc_test['mape']:.2f}%\n")
                    f.write(
                        f"  Valid samples: {bc_test.get('valid_samples', 'N/A')}\n\n")
                else:
                    f.write(f"    MODEL STATUS: {status}\n")
                    f.write(
                        f"Reason: {self.results['bearing_capacity'].get('reason', 'Unknown')}\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"[OK] Report generated: {output_file}")
        return output_file

    def upload_results_to_storage(self, bucket_name, local_dir='/mnt/user-data/outputs', storage_base='ml_models'):
        """Upload all model files to Supabase Storage"""
        print("\n" + "=" * 80)
        print("UPLOADING RESULTS TO SUPABASE STORAGE")
        print("=" * 80)

        uploaded_files = []

        # Get all files in outputs directory
        for filename in os.listdir(local_dir):
            if filename.startswith('ann_') or filename.startswith('scaler'):
                local_path = os.path.join(local_dir, filename)
                if os.path.isfile(local_path):
                    storage_path = f'{storage_base}/{filename}'
                    if upload_to_supabase_storage(local_path, bucket_name, storage_path):
                        uploaded_files.append(storage_path)

        print(
            f"\n[OK] Uploaded {len(uploaded_files)} files to Supabase Storage")
        return uploaded_files


def main():
    """Main execution following research methodology"""
    print("\n" + "=" * 80)
    print("ALIGNED ANN-BASED BEARING CAPACITY PREDICTION SYSTEM")
    print("Single-Target Regression Pipeline - Tarlac Province")
    print("Version: fixed-v1.2")
    print("=" * 80 + "\n")

    if not SKLEARN_AVAILABLE:
        print("[X] scikit-learn is required but not installed")
        print("Run: pip install scikit-learn joblib")
        return None

    # Configuration
    BUCKET_NAME = 'geotechnical-data'

    # Initialize pipeline
    pipeline = FixedBearingCapacityANNPipeline()

    # Phase 1-3: Train models for each target sequentially
    targets = ['liquefaction_probability_pct',
               'settlement_cm_calc', 'qa_allowable_kpa']

    for target in targets:
        print("\n" + "#" * 60)
        print(f"Processing target: {target}")
        print("#" * 60 + "\n")

        # Load target-specific splits exported by feature engineering
        if not pipeline.load_data_for_target(BUCKET_NAME, target):
            print(f"[X] Skipping training for {target} (data load failed)")
            continue

        # Preprocess
        pipeline.preprocess_data(use_robust_scaler=False)

        # Train corresponding model
        if target == 'qa_allowable_kpa':
            pipeline.train_bearing_capacity_ann(perform_cv=True)
            pipeline.evaluate_bearing_capacity_model()
            pipeline.analyze_feature_importance()
        elif target == 'liquefaction_probability_pct':
            pipeline.train_liquefaction_ann(perform_cv=True)
        elif target == 'settlement_cm_calc':
            pipeline.train_settlement_ann(perform_cv=True)

    # Phase 4: Generate report and save/upload all models
    pipeline.generate_comprehensive_report()
    pipeline.save_models()
    pipeline.upload_results_to_storage(BUCKET_NAME)

    print("\n" + "=" * 80)
    print("[SUCCESS] ANN TRAINING PIPELINE COMPLETED!")
    print("=" * 80)

    if 'bearing_capacity' in pipeline.results:
        status = pipeline.results['bearing_capacity'].get('status', 'success')
        print(f"\n   MODEL STATUS: {status.upper()}")

        if status == 'success':
            print("\n   BEARING CAPACITY PREDICTION PERFORMANCE:")
            print(
                f"   Test RMSE: {pipeline.results['bearing_capacity']['test']['rmse']:.2f} kPa")
            print(
                f"   Test R²:   {pipeline.results['bearing_capacity']['test']['r2']:.4f}")
            print(
                f"   Test MAPE: {pipeline.results['bearing_capacity']['test']['mape']:.2f}%")
        else:
            print(
                f"\n    Reason: {pipeline.results['bearing_capacity'].get('reason', 'Unknown')}")

    print("\n\n Results uploaded to Supabase Storage:")
    print(f"  Bucket: {BUCKET_NAME}")
    print("  Path: ml_models/")
    print("\n Files available:")
    # Show available model files
    if any([getattr(pipeline, 'ann_bearing_capacity', None), getattr(pipeline, 'ann_liquefaction', None), getattr(pipeline, 'ann_settlement', None)]):
        if getattr(pipeline, 'ann_bearing_capacity', None) is not None:
            print("  - ann_bearing_capacity.pkl (regressor)")
        if getattr(pipeline, 'ann_liquefaction', None) is not None:
            print("  - ann_liquefaction.pkl (regressor)")
        if getattr(pipeline, 'ann_settlement', None) is not None:
            print("  - ann_settlement.pkl (regressor)")
        print("  - scaler.pkl (feature scaler)")
    else:
        print("  - No model files (training failed - no valid data)")

    print("  - ann_metadata.json (results and configuration)")
    print("  - ann_training_report.txt (detailed report)")
    print("=" * 80 + "\n")

    return pipeline


if __name__ == "__main__":
    aligned_pipeline = main()
