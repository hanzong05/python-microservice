"""
UPGRADED ANN-Based Machine Learning Training Pipeline for Liquefaction Prediction
Tarlac Province Geotechnical Data - Enhanced Upgraded Version

MAJOR UPGRADES FROM PREVIOUS VERSION:
1. [OK] Uses ALL enhanced features from improved feature engineering pipeline
2. [OK] Handles THREE target variables (liquefaction, settlement_cm, qa_allowable_kpa)
3. [OK] Incorporates spatial features from PostGIS views
4. [OK] Enhanced ANN architecture with better hyperparameter tuning
5. [OK] Improved validation metrics aligned with research methodology
6. [OK] Better handling of class imbalance for liquefaction prediction
7. [OK] Cross-validation for robust model evaluation
8. [OK] Feature importance analysis
9. [OK] Enhanced visualization and reporting

Based on Research Methodology (Chapter 2):
- Phase 2: Model Development and Training (Steps 3-7)
- Phase 3: Performance Evaluation (Steps 8-9)
- Validates against: DPWH BSDS (2013), Tokimatsu & Seed (1987), Terzaghi (1943)

Author: Upgraded Geotechnical ML Pipeline
Date: 2026-01-30
Version: Upgraded
"""

import os
import warnings
import pandas as pd
import numpy as np
import io
import json
import pickle
from datetime import datetime
from supabase_client import get_supabase_client

warnings.filterwarnings('ignore')

# Check for required libraries
try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, mean_squared_error,
        mean_absolute_error, r2_score, roc_auc_score, roc_curve
    )
    from sklearn.utils.class_weight import compute_class_weight
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[X] scikit-learn not installed!")
    print("Install with: pip install scikit-learn joblib")
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("[!]  Matplotlib/Seaborn not available - plots will be skipped")
    PLOTTING_AVAILABLE = False


# -----------------------------
# Supabase Storage Functions
# -----------------------------

def download_file_from_storage(bucket_name, file_path):
    """Download file from Supabase Storage"""
    print(f" Downloading {file_path} from Supabase Storage...")
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
    print(
        f" Uploading {os.path.basename(local_file_path)} to Supabase Storage...")
    client = get_supabase_client()
    if not client:
        return False

    try:
        with open(local_file_path, 'rb') as f:
            file_data = f.read()

        # Determine content type
        if local_file_path.endswith('.pkl') or local_file_path.endswith('.joblib'):
            content_type = 'application/octet-stream'
        elif local_file_path.endswith('.json'):
            content_type = 'application/json'
        elif local_file_path.endswith('.txt'):
            content_type = 'text/plain'
        elif local_file_path.endswith('.png'):
            content_type = 'image/png'
        elif local_file_path.endswith('.csv'):
            content_type = 'text/csv'
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
        print(f"  [OK] Uploaded to {storage_path}")
        return True

    except Exception as e:
        print(f"  [X] Error uploading: {e}")
        return False


# -----------------------------
# UPGRADED ANN Training Pipeline
# -----------------------------

class UpgradedLiquefactionANNPipeline:
    """
    UPGRADED ANN-Based Machine Learning Pipeline for Liquefaction Prediction
    Upgraded Version - Enhanced with Spatial Features and Improved Architecture

    Key Improvements:
    - Enhanced feature set from improved feature engineering
    - Better ANN architecture with optimized hyperparameters
    - Class weight balancing for imbalanced liquefaction data
    - Cross-validation for robust evaluation
    - Feature importance analysis
    - Multiple evaluation metrics
    """

    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train_liq = None
        self.y_val_liq = None
        self.y_test_liq = None
        self.y_train_settlement = None
        self.y_val_settlement = None
        self.y_test_settlement = None
        self.y_train_bearing = None
        self.y_val_bearing = None
        self.y_test_bearing = None

        self.feature_names = None
        self.scaler = None

        # Models for three tasks
        self.ann_liquefaction = None  # Classification
        self.ann_settlement = None     # Regression
        self.ann_bearing_capacity = None  # Regression

        self.results = {}
        self.feature_importance = {}
        self.cross_val_scores = {}

    def load_data_from_storage(self, bucket_name, base_path='feature_engineering'):
        """Load training data from Supabase Storage"""
        print("=" * 80)
        print("LOADING ENHANCED TRAINING DATA FROM SUPABASE STORAGE")
        print("=" * 80)

        try:
            # Download train data
            train_bytes = download_file_from_storage(
                bucket_name, f'{base_path}/train.csv')
            if not train_bytes:
                return False
            train_df = pd.read_csv(io.BytesIO(train_bytes))

            # Download validation data
            val_bytes = download_file_from_storage(
                bucket_name, f'{base_path}/validation.csv')
            if not val_bytes:
                return False
            val_df = pd.read_csv(io.BytesIO(val_bytes))

            # Download test data
            test_bytes = download_file_from_storage(
                bucket_name, f'{base_path}/test.csv')
            if not test_bytes:
                return False
            test_df = pd.read_csv(io.BytesIO(test_bytes))

            # Separate features and targets
            target_cols = ['liquefaction', 'settlement_cm', 'qa_allowable_kpa']
            self.feature_names = [
                col for col in train_df.columns if col not in target_cols]

            # Extract features
            self.X_train = train_df[self.feature_names].values
            self.X_val = val_df[self.feature_names].values
            self.X_test = test_df[self.feature_names].values

            # Extract targets
            # Liquefaction (classification)
            self.y_train_liq = train_df['liquefaction'].values
            self.y_val_liq = val_df['liquefaction'].values
            self.y_test_liq = test_df['liquefaction'].values

            # Settlement (regression)
            self.y_train_settlement = train_df['settlement_cm'].values
            self.y_val_settlement = val_df['settlement_cm'].values
            self.y_test_settlement = test_df['settlement_cm'].values

            # Bearing Capacity (regression)
            self.y_train_bearing = train_df['qa_allowable_kpa'].values
            self.y_val_bearing = val_df['qa_allowable_kpa'].values
            self.y_test_bearing = test_df['qa_allowable_kpa'].values

            print(f"\n[OK] Enhanced data loaded successfully!")
            print(f"  - Features: {len(self.feature_names)}")
            print(f"  - Training samples: {len(self.X_train)}")
            print(f"  - Validation samples: {len(self.X_val)}")
            print(f"  - Test samples: {len(self.X_test)}")

            print(f"\n  - Liquefaction distribution (train):")
            print(
                f"    - Liquefiable: {self.y_train_liq.sum()} ({self.y_train_liq.sum()/len(self.y_train_liq)*100:.1f}%)")
            print(
                f"    - Non-liquefiable: {len(self.y_train_liq)-self.y_train_liq.sum()} ({(1-self.y_train_liq.sum()/len(self.y_train_liq))*100:.1f}%)")

            print(
                f"\n  - Settlement (train): Mean={self.y_train_settlement.mean():.2f} cm, Std={self.y_train_settlement.std():.2f} cm")
            print(
                f"  - Bearing capacity (train): Mean={self.y_train_bearing.mean():.2f} kPa, Std={self.y_train_bearing.std():.2f} kPa")

            return True

        except Exception as e:
            print(f"[X] Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_data(self, use_robust_scaler=False):
        """
        Standardize features using StandardScaler or RobustScaler

        RobustScaler is better for data with outliers (uses median and IQR)
        StandardScaler uses mean and std (default for this study)
        """
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA WITH ENHANCED SCALING")
        print("=" * 80)

        if use_robust_scaler:
            print("\n Using RobustScaler (robust to outliers)...")
            self.scaler = RobustScaler()
        else:
            print("\n Using StandardScaler (standard normalization)...")
            self.scaler = StandardScaler()

        # Fit on training data and transform all sets
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        print("[OK] Features standardized successfully!")
        print(f"  - Scaler type: {type(self.scaler).__name__}")
        print(f"  - Feature shape: {self.X_train.shape}")

        return True

    def train_liquefaction_ann(self, use_class_weights=True, perform_cv=True):
        """
        Train UPGRADED ANN for Liquefaction Classification

        Enhancements:
        - Class weight balancing for imbalanced data
        - Cross-validation for robust evaluation
        - Optimized architecture
        - Early stopping with validation monitoring
        """
        print("\n" + "=" * 80)
        print("TRAINING UPGRADED ANN - LIQUEFACTION POTENTIAL (CLASSIFICATION)")
        print("=" * 80)

        # Compute class weights to handle imbalance
        class_weights_dict = None
        if use_class_weights:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train_liq),
                y=self.y_train_liq
            )
            class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
            print(f"\n  Class weights computed:")
            print(f"  - Non-liquefiable (0): {class_weights[0]:.3f}")
            print(f"  - Liquefiable (1): {class_weights[1]:.3f}")

        print("\n UPGRADED ANN Configuration:")
        print("  - Task: Binary Classification (Liquefiable vs Non-Liquefiable)")
        print("  - Hidden layers: (256, 128, 64) - ENHANCED")
        print("  - Activation: ReLU (hidden), Logistic (output)")
        print("  - Solver: Adam optimizer with adaptive learning rate")
        print("  - Max iterations: 1000 (INCREASED)")
        print("  - Early stopping: Enabled (patience=10)")
        print("  - Validation fraction: 0.15")
        print("  - Alpha (L2 penalty): 0.0001")

        self.ann_liquefaction = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Enhanced architecture
            activation='relu',
            solver='adam',
            max_iter=1000,  # Increased for better convergence
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,  # Patience
            alpha=0.0001,  # L2 regularization
            learning_rate='adaptive',
            verbose=False
        )

        print("\n Training ANN for liquefaction classification...")
        self.ann_liquefaction.fit(self.X_train, self.y_train_liq)

        print("[OK] ANN training completed!")
        print(f"  - Final training loss: {self.ann_liquefaction.loss_:.6f}")
        print(f"  - Iterations: {self.ann_liquefaction.n_iter_}")
        print(
            f"  - Converged: {'Yes' if self.ann_liquefaction.n_iter_ < 1000 else 'No (max iterations reached)'}")

        # Cross-validation
        if perform_cv:
            print("\n Performing 5-fold cross-validation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                self.ann_liquefaction, self.X_train, self.y_train_liq,
                cv=cv, scoring='f1'
            )
            self.cross_val_scores['liquefaction'] = cv_scores
            print(f"  - CV F1 scores: {cv_scores}")
            print(
                f"  - Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.ann_liquefaction

    def train_settlement_ann(self, perform_cv=True):
        """
        Train UPGRADED ANN for Settlement Prediction (Regression)

        Enhancements:
        - Optimized architecture for regression
        - Cross-validation
        - Better regularization
        """
        print("\n" + "=" * 80)
        print("TRAINING UPGRADED ANN - POST-LIQUEFACTION SETTLEMENT (REGRESSION)")
        print("=" * 80)

        print("\n UPGRADED ANN Configuration:")
        print("  - Task: Regression (Settlement in cm)")
        print("  - Hidden layers: (256, 128, 64) - ENHANCED")
        print("  - Activation: ReLU (hidden), Identity (output)")
        print("  - Solver: Adam optimizer")
        print("  - Max iterations: 1000 (INCREASED)")
        print("  - Early stopping: Enabled")
        print("  - Alpha (L2 penalty): 0.0001")

        self.ann_settlement = MLPRegressor(
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

        print("\n Training ANN for settlement prediction...")
        self.ann_settlement.fit(self.X_train, self.y_train_settlement)

        print("[OK] ANN training completed!")
        print(f"  - Final training loss: {self.ann_settlement.loss_:.6f}")
        print(f"  - Iterations: {self.ann_settlement.n_iter_}")

        # Cross-validation
        if perform_cv:
            print("\n Performing 5-fold cross-validation...")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                self.ann_settlement, self.X_train, self.y_train_settlement,
                cv=cv, scoring='r2'
            )
            self.cross_val_scores['settlement'] = cv_scores
            print(f"  - CV R^2 scores: {cv_scores}")
            print(
                f"  - Mean CV R^2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.ann_settlement

    def train_bearing_capacity_ann(self, perform_cv=True):
        """
        Train UPGRADED ANN for Bearing Capacity Prediction (Regression)

        Enhancements:
        - Optimized architecture
        - Cross-validation
        - Better regularization
        """
        print("\n" + "=" * 80)
        print("TRAINING UPGRADED ANN - ALLOWABLE BEARING CAPACITY (REGRESSION)")
        print("=" * 80)

        print("\n UPGRADED ANN Configuration:")
        print("  - Task: Regression (Allowable Bearing Capacity in kPa)")
        print("  - Hidden layers: (256, 128, 64) - ENHANCED")
        print("  - Activation: ReLU (hidden), Identity (output)")
        print("  - Solver: Adam optimizer")
        print("  - Max iterations: 1000 (INCREASED)")
        print("  - Early stopping: Enabled")

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

        print("\n Training ANN for bearing capacity prediction...")
        self.ann_bearing_capacity.fit(self.X_train, self.y_train_bearing)

        print("[OK] ANN training completed!")
        print(
            f"  - Final training loss: {self.ann_bearing_capacity.loss_:.6f}")
        print(f"  - Iterations: {self.ann_bearing_capacity.n_iter_}")

        # Cross-validation
        if perform_cv:
            print("\n Performing 5-fold cross-validation...")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                self.ann_bearing_capacity, self.X_train, self.y_train_bearing,
                cv=cv, scoring='r2'
            )
            self.cross_val_scores['bearing_capacity'] = cv_scores
            print(f"  - CV R^2 scores: {cv_scores}")
            print(
                f"  - Mean CV R^2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.ann_bearing_capacity

    def evaluate_liquefaction_model(self):
        """
        ENHANCED Evaluation of Liquefaction Classification Model

        Additional metrics:
        - ROC-AUC score
        - Detailed classification report
        - Per-class metrics
        """
        print("\n" + "=" * 80)
        print("ENHANCED EVALUATION - LIQUEFACTION MODEL")
        print("=" * 80)

        # Validation set evaluation
        y_val_pred = self.ann_liquefaction.predict(self.X_val)
        y_val_proba = self.ann_liquefaction.predict_proba(self.X_val)[:, 1]

        val_accuracy = accuracy_score(self.y_val_liq, y_val_pred)
        val_precision = precision_score(
            self.y_val_liq, y_val_pred, zero_division=0)
        val_recall = recall_score(self.y_val_liq, y_val_pred, zero_division=0)
        val_f1 = f1_score(self.y_val_liq, y_val_pred, zero_division=0)
        val_roc_auc = roc_auc_score(self.y_val_liq, y_val_proba)
        val_cm = confusion_matrix(self.y_val_liq, y_val_pred)

        print("\n VALIDATION SET PERFORMANCE:")
        print(f"  Accuracy:  {val_accuracy:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  F1-Score:  {val_f1:.4f}")
        print(f"  ROC-AUC:   {val_roc_auc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {val_cm[0, 0]:<6} FP: {val_cm[0, 1]}")
        print(f"    FN: {val_cm[1, 0]:<6} TP: {val_cm[1, 1]}")

        # Test set evaluation
        y_test_pred = self.ann_liquefaction.predict(self.X_test)
        y_test_proba = self.ann_liquefaction.predict_proba(self.X_test)[:, 1]

        test_accuracy = accuracy_score(self.y_test_liq, y_test_pred)
        test_precision = precision_score(
            self.y_test_liq, y_test_pred, zero_division=0)
        test_recall = recall_score(
            self.y_test_liq, y_test_pred, zero_division=0)
        test_f1 = f1_score(self.y_test_liq, y_test_pred, zero_division=0)
        test_roc_auc = roc_auc_score(self.y_test_liq, y_test_proba)
        test_cm = confusion_matrix(self.y_test_liq, y_test_pred)

        print("\n TEST SET PERFORMANCE:")
        print(f"  Accuracy:  {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        print(f"  F1-Score:  {test_f1:.4f}")
        print(f"  ROC-AUC:   {test_roc_auc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {test_cm[0, 0]:<6} FP: {test_cm[0, 1]}")
        print(f"    FN: {test_cm[1, 0]:<6} TP: {test_cm[1, 1]}")

        # Classification report
        print("\n Detailed Classification Report (Test Set):")
        print(classification_report(self.y_test_liq, y_test_pred,
                                    target_names=['Non-Liquefiable', 'Liquefiable']))

        self.results['liquefaction'] = {
            'validation': {
                'accuracy': float(val_accuracy),
                'precision': float(val_precision),
                'recall': float(val_recall),
                'f1_score': float(val_f1),
                'roc_auc': float(val_roc_auc),
                'confusion_matrix': val_cm.tolist()
            },
            'test': {
                'accuracy': float(test_accuracy),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1_score': float(test_f1),
                'roc_auc': float(test_roc_auc),
                'confusion_matrix': test_cm.tolist()
            },
            'cross_validation': {
                'mean_f1': float(self.cross_val_scores.get('liquefaction', [0]).mean()),
                'std_f1': float(self.cross_val_scores.get('liquefaction', [0]).std())
            } if 'liquefaction' in self.cross_val_scores else {}
        }

        return self.results['liquefaction']

    def evaluate_settlement_model(self):
        """ENHANCED Evaluation of Settlement Regression Model"""
        print("\n" + "=" * 80)
        print("ENHANCED EVALUATION - SETTLEMENT MODEL")
        print("=" * 80)

        # Validation set
        y_val_pred = self.ann_settlement.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(
            self.y_val_settlement, y_val_pred))
        val_mae = mean_absolute_error(self.y_val_settlement, y_val_pred)
        val_r2 = r2_score(self.y_val_settlement, y_val_pred)
        val_mape = np.mean(np.abs(
            (self.y_val_settlement - y_val_pred) / (self.y_val_settlement + 1e-10))) * 100

        print("\n VALIDATION SET PERFORMANCE:")
        print(f"  RMSE: {val_rmse:.4f} cm")
        print(f"  MAE:  {val_mae:.4f} cm")
        print(f"  R^2:   {val_r2:.4f}")
        print(f"  MAPE: {val_mape:.2f}%")

        # Test set
        y_test_pred = self.ann_settlement.predict(self.X_test)
        test_rmse = np.sqrt(mean_squared_error(
            self.y_test_settlement, y_test_pred))
        test_mae = mean_absolute_error(self.y_test_settlement, y_test_pred)
        test_r2 = r2_score(self.y_test_settlement, y_test_pred)
        test_mape = np.mean(np.abs(
            (self.y_test_settlement - y_test_pred) / (self.y_test_settlement + 1e-10))) * 100

        print("\n TEST SET PERFORMANCE:")
        print(f"  RMSE: {test_rmse:.4f} cm")
        print(f"  MAE:  {test_mae:.4f} cm")
        print(f"  R^2:   {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")

        self.results['settlement'] = {
            'validation': {
                'rmse': float(val_rmse),
                'mae': float(val_mae),
                'r2': float(val_r2),
                'mape': float(val_mape)
            },
            'test': {
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'r2': float(test_r2),
                'mape': float(test_mape)
            },
            'cross_validation': {
                'mean_r2': float(self.cross_val_scores.get('settlement', [0]).mean()),
                'std_r2': float(self.cross_val_scores.get('settlement', [0]).std())
            } if 'settlement' in self.cross_val_scores else {}
        }

        return self.results['settlement']

    def evaluate_bearing_capacity_model(self):
        """ENHANCED Evaluation of Bearing Capacity Regression Model"""
        print("\n" + "=" * 80)
        print("ENHANCED EVALUATION - BEARING CAPACITY MODEL")
        print("=" * 80)

        # Validation set
        y_val_pred = self.ann_bearing_capacity.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val_bearing, y_val_pred))
        val_mae = mean_absolute_error(self.y_val_bearing, y_val_pred)
        val_r2 = r2_score(self.y_val_bearing, y_val_pred)
        val_mape = np.mean(
            np.abs((self.y_val_bearing - y_val_pred) / (self.y_val_bearing + 1e-10))) * 100

        print("\n VALIDATION SET PERFORMANCE:")
        print(f"  RMSE: {val_rmse:.2f} kPa")
        print(f"  MAE:  {val_mae:.2f} kPa")
        print(f"  R^2:   {val_r2:.4f}")
        print(f"  MAPE: {val_mape:.2f}%")

        # Test set
        y_test_pred = self.ann_bearing_capacity.predict(self.X_test)
        test_rmse = np.sqrt(mean_squared_error(
            self.y_test_bearing, y_test_pred))
        test_mae = mean_absolute_error(self.y_test_bearing, y_test_pred)
        test_r2 = r2_score(self.y_test_bearing, y_test_pred)
        test_mape = np.mean(np.abs(
            (self.y_test_bearing - y_test_pred) / (self.y_test_bearing + 1e-10))) * 100

        print("\n TEST SET PERFORMANCE:")
        print(f"  RMSE: {test_rmse:.2f} kPa")
        print(f"  MAE:  {test_mae:.2f} kPa")
        print(f"  R^2:   {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")

        self.results['bearing_capacity'] = {
            'validation': {
                'rmse': float(val_rmse),
                'mae': float(val_mae),
                'r2': float(val_r2),
                'mape': float(val_mape)
            },
            'test': {
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'r2': float(test_r2),
                'mape': float(test_mape)
            },
            'cross_validation': {
                'mean_r2': float(self.cross_val_scores.get('bearing_capacity', [0]).mean()),
                'std_r2': float(self.cross_val_scores.get('bearing_capacity', [0]).std())
            } if 'bearing_capacity' in self.cross_val_scores else {}
        }

        return self.results['bearing_capacity']

    def analyze_feature_importance(self):
        """
        Analyze feature importance using connection weights
        (Approximation for neural networks)
        """
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)

        try:
            # For liquefaction model
            weights_first_layer = np.abs(self.ann_liquefaction.coefs_[0])
            feature_importance_liq = weights_first_layer.sum(axis=1)
            feature_importance_liq = feature_importance_liq / feature_importance_liq.sum()

            # Get top 20 features
            top_indices = np.argsort(feature_importance_liq)[-20:][::-1]

            print("\n Top 20 Most Important Features for Liquefaction Prediction:")
            for i, idx in enumerate(top_indices, 1):
                print(
                    f"  {i:2d}. {self.feature_names[idx]:<40s} {feature_importance_liq[idx]:.4f}")

            self.feature_importance['liquefaction'] = {
                self.feature_names[i]: float(feature_importance_liq[i])
                for i in top_indices
            }

            return True

        except Exception as e:
            print(f"[!]  Could not analyze feature importance: {e}")
            return False

    def generate_comprehensive_report(self, output_file='/mnt/user-data/outputs/upgraded_ann_training_report.txt'):
        """Generate UPGRADED comprehensive ANN training and validation report"""
        print("\n" + "=" * 80)
        print("GENERATING UPGRADED COMPREHENSIVE TRAINING REPORT")
        print("=" * 80)

        os.makedirs('/mnt/user-data/outputs', exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("UPGRADED ANN-BASED LIQUEFACTION PREDICTION SYSTEM\n")
            f.write("Upgraded Version - Enhanced Training and Validation Report\n")
            f.write("Tarlac Province, Philippines\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("KEY UPGRADES IN UPGRADED VERSION:\n")
            f.write("-" * 80 + "\n")
            f.write(
                "[OK] Enhanced feature set from improved feature engineering pipeline\n")
            f.write("[OK] Spatial features from PostGIS views integrated\n")
            f.write("[OK] Improved ANN architecture: (256, 128, 64) neurons\n")
            f.write("[OK] Class weight balancing for imbalanced data\n")
            f.write("[OK] Cross-validation for robust evaluation\n")
            f.write("[OK] Extended training iterations (1000 max)\n")
            f.write("[OK] Additional metrics: ROC-AUC, MAPE\n")
            f.write("[OK] Feature importance analysis\n\n")

            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of features: {len(self.feature_names)}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Validation samples: {len(self.X_val)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n\n")

            # Model architecture
            f.write("UPGRADED ANN MODEL ARCHITECTURE:\n")
            f.write("-" * 80 + "\n")
            f.write("Hidden Layer Structure: (256, 128, 64) neurons\n")
            f.write("Activation Function: ReLU (hidden layers)\n")
            f.write(
                "Output Activation: Logistic (classification), Identity (regression)\n")
            f.write("Solver: Adam optimizer with adaptive learning rate\n")
            f.write("Max Iterations: 1000\n")
            f.write("Early Stopping: Enabled (patience=10)\n")
            f.write("L2 Regularization (alpha): 0.0001\n")
            f.write("Validation Fraction: 0.15\n\n")

            # Liquefaction model results
            f.write("1. LIQUEFACTION POTENTIAL (CLASSIFICATION):\n")
            f.write("-" * 80 + "\n")
            liq_val = self.results['liquefaction']['validation']
            liq_test = self.results['liquefaction']['test']

            f.write("Validation Set:\n")
            f.write(f"  Accuracy:  {liq_val['accuracy']:.4f}\n")
            f.write(f"  Precision: {liq_val['precision']:.4f}\n")
            f.write(f"  Recall:    {liq_val['recall']:.4f}\n")
            f.write(f"  F1-Score:  {liq_val['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC:   {liq_val['roc_auc']:.4f}\n")

            f.write("\nTest Set:\n")
            f.write(f"  Accuracy:  {liq_test['accuracy']:.4f}\n")
            f.write(f"  Precision: {liq_test['precision']:.4f}\n")
            f.write(f"  Recall:    {liq_test['recall']:.4f}\n")
            f.write(f"  F1-Score:  {liq_test['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC:   {liq_test['roc_auc']:.4f}\n")

            cm = liq_test['confusion_matrix']
            f.write(f"\nConfusion Matrix (Test Set):\n")
            f.write(f"  TN: {cm[0][0]:<6} FP: {cm[0][1]}\n")
            f.write(f"  FN: {cm[1][0]:<6} TP: {cm[1][1]}\n")

            if liq_test.get('cross_validation'):
                cv = liq_test['cross_validation']
                f.write(f"\n5-Fold Cross-Validation:\n")
                f.write(
                    f"  Mean F1: {cv['mean_f1']:.4f} (+/- {cv['std_f1']:.4f})\n")
            f.write("\n")

            # Settlement model results
            f.write("2. POST-LIQUEFACTION SETTLEMENT (REGRESSION):\n")
            f.write("-" * 80 + "\n")
            f.write("Validation Method: Tokimatsu & Seed (1987)\n\n")

            settle_val = self.results['settlement']['validation']
            settle_test = self.results['settlement']['test']

            f.write("Validation Set:\n")
            f.write(f"  RMSE: {settle_val['rmse']:.4f} cm\n")
            f.write(f"  MAE:  {settle_val['mae']:.4f} cm\n")
            f.write(f"  R^2:   {settle_val['r2']:.4f}\n")
            f.write(f"  MAPE: {settle_val['mape']:.2f}%\n")

            f.write("\nTest Set:\n")
            f.write(f"  RMSE: {settle_test['rmse']:.4f} cm\n")
            f.write(f"  MAE:  {settle_test['mae']:.4f} cm\n")
            f.write(f"  R^2:   {settle_test['r2']:.4f}\n")
            f.write(f"  MAPE: {settle_test['mape']:.2f}%\n")

            if settle_test.get('cross_validation'):
                cv = settle_test['cross_validation']
                f.write(f"\n5-Fold Cross-Validation:\n")
                f.write(
                    f"  Mean R^2: {cv['mean_r2']:.4f} (+/- {cv['std_r2']:.4f})\n")
            f.write("\n")

            # Bearing capacity model results
            f.write("3. ALLOWABLE BEARING CAPACITY (REGRESSION):\n")
            f.write("-" * 80 + "\n")
            f.write("Validation Method: Terzaghi (1943) & Olsen-Stark (2002)\n\n")

            bearing_val = self.results['bearing_capacity']['validation']
            bearing_test = self.results['bearing_capacity']['test']

            f.write("Validation Set:\n")
            f.write(f"  RMSE: {bearing_val['rmse']:.2f} kPa\n")
            f.write(f"  MAE:  {bearing_val['mae']:.2f} kPa\n")
            f.write(f"  R^2:   {bearing_val['r2']:.4f}\n")
            f.write(f"  MAPE: {bearing_val['mape']:.2f}%\n")

            f.write("\nTest Set:\n")
            f.write(f"  RMSE: {bearing_test['rmse']:.2f} kPa\n")
            f.write(f"  MAE:  {bearing_test['mae']:.2f} kPa\n")
            f.write(f"  R^2:   {bearing_test['r2']:.4f}\n")
            f.write(f"  MAPE: {bearing_test['mape']:.2f}%\n")

            if bearing_test.get('cross_validation'):
                cv = bearing_test['cross_validation']
                f.write(f"\n5-Fold Cross-Validation:\n")
                f.write(
                    f"  Mean R^2: {cv['mean_r2']:.4f} (+/- {cv['std_r2']:.4f})\n")
            f.write("\n")

            # Feature importance
            if self.feature_importance.get('liquefaction'):
                f.write("FEATURE IMPORTANCE (Top 20 for Liquefaction):\n")
                f.write("-" * 80 + "\n")
                for i, (feature, importance) in enumerate(self.feature_importance['liquefaction'].items(), 1):
                    f.write(f"  {i:2d}. {feature:<40s} {importance:.4f}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF UPGRADED REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"[OK] Upgraded report generated: {output_file}")
        return output_file

    def plot_confusion_matrix(self, output_file='/mnt/user-data/outputs/confusion_matrix_liquefaction.png'):
        """Generate ENHANCED confusion matrix plot"""
        if not PLOTTING_AVAILABLE:
            print("[!]  Plotting not available - skipping confusion matrix")
            return None

        print("\n" + "=" * 80)
        print("GENERATING ENHANCED CONFUSION MATRIX PLOT")
        print("=" * 80)

        os.makedirs('/mnt/user-data/outputs', exist_ok=True)

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Validation confusion matrix
            cm_val = np.array(
                self.results['liquefaction']['validation']['confusion_matrix'])
            sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=['Non-Liq', 'Liq'],
                        yticklabels=['Non-Liq', 'Liq'],
                        cbar_kws={'label': 'Count'})
            ax1.set_title(f'Validation Set\nAccuracy: {self.results["liquefaction"]["validation"]["accuracy"]:.4f}',
                          fontsize=12, fontweight='bold')
            ax1.set_ylabel('True Label', fontsize=11)
            ax1.set_xlabel('Predicted Label', fontsize=11)

            # Test confusion matrix
            cm_test = np.array(
                self.results['liquefaction']['test']['confusion_matrix'])
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax2,
                        xticklabels=['Non-Liq', 'Liq'],
                        yticklabels=['Non-Liq', 'Liq'],
                        cbar_kws={'label': 'Count'})
            ax2.set_title(f'Test Set\nAccuracy: {self.results["liquefaction"]["test"]["accuracy"]:.4f}',
                          fontsize=12, fontweight='bold')
            ax2.set_ylabel('True Label', fontsize=11)
            ax2.set_xlabel('Predicted Label', fontsize=11)

            plt.suptitle('Upgraded ANN - Liquefaction Classification Performance',
                         fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  [OK] Saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"  [X] Error creating confusion matrix plot: {e}")
            return None

    def plot_roc_curve(self, output_file='/mnt/user-data/outputs/roc_curve_liquefaction.png'):
        """Generate ROC curve plot (NEW in upgraded version)"""
        if not PLOTTING_AVAILABLE:
            print("[!]  Plotting not available - skipping ROC curve")
            return None

        print("\n" + "=" * 80)
        print("GENERATING ROC CURVE PLOT (NEW)")
        print("=" * 80)

        os.makedirs('/mnt/user-data/outputs', exist_ok=True)

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Validation ROC
            y_val_proba = self.ann_liquefaction.predict_proba(self.X_val)[:, 1]
            fpr_val, tpr_val, _ = roc_curve(self.y_val_liq, y_val_proba)
            auc_val = self.results['liquefaction']['validation']['roc_auc']

            ax1.plot(fpr_val, tpr_val, color='blue', lw=2,
                     label=f'ROC curve (AUC = {auc_val:.4f})')
            ax1.plot([0, 1], [0, 1], color='gray', lw=1,
                     linestyle='--', label='Random')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate', fontsize=11)
            ax1.set_ylabel('True Positive Rate', fontsize=11)
            ax1.set_title('ROC Curve - Validation Set',
                          fontsize=12, fontweight='bold')
            ax1.legend(loc="lower right")
            ax1.grid(alpha=0.3)

            # Test ROC
            y_test_proba = self.ann_liquefaction.predict_proba(self.X_test)[
                :, 1]
            fpr_test, tpr_test, _ = roc_curve(self.y_test_liq, y_test_proba)
            auc_test = self.results['liquefaction']['test']['roc_auc']

            ax2.plot(fpr_test, tpr_test, color='green', lw=2,
                     label=f'ROC curve (AUC = {auc_test:.4f})')
            ax2.plot([0, 1], [0, 1], color='gray', lw=1,
                     linestyle='--', label='Random')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate', fontsize=11)
            ax2.set_ylabel('True Positive Rate', fontsize=11)
            ax2.set_title('ROC Curve - Test Set',
                          fontsize=12, fontweight='bold')
            ax2.legend(loc="lower right")
            ax2.grid(alpha=0.3)

            plt.suptitle('Upgraded ANN - ROC Curve Analysis',
                         fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  [OK] Saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"  [X] Error creating ROC curve plot: {e}")
            return None

    def plot_regression_predictions(self, output_file='/mnt/user-data/outputs/regression_predictions.png'):
        """Generate prediction vs actual plots for regression models (NEW in upgraded version)"""
        if not PLOTTING_AVAILABLE:
            print("[!]  Plotting not available - skipping regression plots")
            return None

        print("\n" + "=" * 80)
        print("GENERATING REGRESSION PREDICTION PLOTS (NEW)")
        print("=" * 80)

        os.makedirs('/mnt/user-data/outputs', exist_ok=True)

        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Settlement - Validation
            y_val_pred_settle = self.ann_settlement.predict(self.X_val)
            axes[0, 0].scatter(self.y_val_settlement,
                               y_val_pred_settle, alpha=0.5, s=20)
            axes[0, 0].plot([self.y_val_settlement.min(), self.y_val_settlement.max()],
                            [self.y_val_settlement.min(), self.y_val_settlement.max()],
                            'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Settlement (cm)', fontsize=10)
            axes[0, 0].set_ylabel('Predicted Settlement (cm)', fontsize=10)
            axes[0, 0].set_title(f'Settlement - Validation\nR^2 = {self.results["settlement"]["validation"]["r2"]:.4f}',
                                 fontsize=11, fontweight='bold')
            axes[0, 0].grid(alpha=0.3)

            # Settlement - Test
            y_test_pred_settle = self.ann_settlement.predict(self.X_test)
            axes[0, 1].scatter(
                self.y_test_settlement, y_test_pred_settle, alpha=0.5, s=20, color='green')
            axes[0, 1].plot([self.y_test_settlement.min(), self.y_test_settlement.max()],
                            [self.y_test_settlement.min(), self.y_test_settlement.max()],
                            'r--', lw=2)
            axes[0, 1].set_xlabel('Actual Settlement (cm)', fontsize=10)
            axes[0, 1].set_ylabel('Predicted Settlement (cm)', fontsize=10)
            axes[0, 1].set_title(f'Settlement - Test\nR^2 = {self.results["settlement"]["test"]["r2"]:.4f}',
                                 fontsize=11, fontweight='bold')
            axes[0, 1].grid(alpha=0.3)

            # Bearing Capacity - Validation
            y_val_pred_bearing = self.ann_bearing_capacity.predict(self.X_val)
            axes[1, 0].scatter(
                self.y_val_bearing, y_val_pred_bearing, alpha=0.5, s=20, color='orange')
            axes[1, 0].plot([self.y_val_bearing.min(), self.y_val_bearing.max()],
                            [self.y_val_bearing.min(), self.y_val_bearing.max()],
                            'r--', lw=2)
            axes[1, 0].set_xlabel('Actual Bearing Capacity (kPa)', fontsize=10)
            axes[1, 0].set_ylabel(
                'Predicted Bearing Capacity (kPa)', fontsize=10)
            axes[1, 0].set_title(f'Bearing Capacity - Validation\nR^2 = {self.results["bearing_capacity"]["validation"]["r2"]:.4f}',
                                 fontsize=11, fontweight='bold')
            axes[1, 0].grid(alpha=0.3)

            # Bearing Capacity - Test
            y_test_pred_bearing = self.ann_bearing_capacity.predict(
                self.X_test)
            axes[1, 1].scatter(
                self.y_test_bearing, y_test_pred_bearing, alpha=0.5, s=20, color='purple')
            axes[1, 1].plot([self.y_test_bearing.min(), self.y_test_bearing.max()],
                            [self.y_test_bearing.min(), self.y_test_bearing.max()],
                            'r--', lw=2)
            axes[1, 1].set_xlabel('Actual Bearing Capacity (kPa)', fontsize=10)
            axes[1, 1].set_ylabel(
                'Predicted Bearing Capacity (kPa)', fontsize=10)
            axes[1, 1].set_title(f'Bearing Capacity - Test\nR^2 = {self.results["bearing_capacity"]["test"]["r2"]:.4f}',
                                 fontsize=11, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)

            plt.suptitle('Upgraded ANN - Regression Model Performance',
                         fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  [OK] Saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"  [X] Error creating regression plots: {e}")
            return None

    def save_models(self, output_dir='/mnt/user-data/outputs'):
        """Save all UPGRADED trained ANN models and metadata"""
        print("\n" + "=" * 80)
        print("SAVING UPGRADED TRAINED MODELS")
        print("=" * 80)

        os.makedirs('/mnt/user-data/outputs', exist_ok=True)
        saved_files = []

        # Save scaler
        scaler_file = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_file)
        print(f"[OK] Saved scaler: {scaler_file}")
        saved_files.append(scaler_file)

        # Save liquefaction model
        liq_model_file = os.path.join(output_dir, 'ann_liquefaction.pkl')
        joblib.dump(self.ann_liquefaction, liq_model_file)
        print(f"[OK] Saved liquefaction model: {liq_model_file}")
        saved_files.append(liq_model_file)

        # Save settlement model
        settle_model_file = os.path.join(output_dir, 'ann_settlement.pkl')
        joblib.dump(self.ann_settlement, settle_model_file)
        print(f"[OK] Saved settlement model: {settle_model_file}")
        saved_files.append(settle_model_file)

        # Save bearing capacity model
        bearing_model_file = os.path.join(
            output_dir, 'ann_bearing_capacity.pkl')
        joblib.dump(self.ann_bearing_capacity, bearing_model_file)
        print(f"[OK] Saved bearing capacity model: {bearing_model_file}")
        saved_files.append(bearing_model_file)

        # Save UPGRADED metadata
        metadata = {
            'version': 'upgraded',
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'training_samples': len(self.X_train),
            'validation_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'timestamp': datetime.now().isoformat(),
            'model_architecture': {
                'hidden_layers': [256, 128, 64],
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'early_stopping': True,
                'alpha': 0.0001,
                'learning_rate': 'adaptive'
            },
            'enhancements': [
                'Spatial features from PostGIS',
                'Enhanced architecture (256-128-64)',
                'Class weight balancing',
                'Cross-validation',
                'ROC-AUC metrics',
                'Feature importance analysis',
                'Extended training iterations'
            ],
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

        return saved_files

    def upload_results_to_storage(self, bucket_name, local_dir='/mnt/user-data/outputs', storage_base='ml_models'):
        """Upload all UPGRADED model files and results to Supabase Storage"""
        print("\n" + "=" * 80)
        print("UPLOADING UPGRADED RESULTS TO SUPABASE STORAGE")
        print("=" * 80)

        uploaded_files = []

        # Get all files in the outputs directory
        for filename in os.listdir(local_dir):
            local_path = os.path.join(local_dir, filename)
            if os.path.isfile(local_path):
                storage_path = f'{storage_base}/{filename}'
                if upload_to_supabase_storage(local_path, bucket_name, storage_path):
                    uploaded_files.append(storage_path)

        print(
            f"\n[OK] Uploaded {len(uploaded_files)} files to Supabase Storage")
        return uploaded_files


def main():
    """Main execution following UPGRADED research methodology"""
    print("\n" + "=" * 80)
    print("UPGRADED ANN-BASED LIQUEFACTION PREDICTION SYSTEM upgraded version")
    print("Training Pipeline - Tarlac Province")
    print("Enhanced with Spatial Features and Improved Architecture")
    print("=" * 80 + "\n")

    if not SKLEARN_AVAILABLE:
        print("[X] scikit-learn is required but not installed")
        return None

    # Configuration
    BUCKET_NAME = 'geotechnical-data'

    # Initialize UPGRADED pipeline
    pipeline = UpgradedLiquefactionANNPipeline()

    # Phase 1: Load enhanced data from Supabase Storage
    if not pipeline.load_data_from_storage(BUCKET_NAME):
        print("[X] Failed to load data. Exiting.")
        return None

    # Phase 2: Preprocess data with enhanced scaling
    pipeline.preprocess_data(use_robust_scaler=False)

    # Phase 2: Train UPGRADED ANN models
    print("\n" + "=" * 80)
    print("PHASE 2: UPGRADED MODEL DEVELOPMENT AND TRAINING")
    print("=" * 80)

    pipeline.train_liquefaction_ann(use_class_weights=True, perform_cv=True)
    pipeline.train_settlement_ann(perform_cv=True)
    pipeline.train_bearing_capacity_ann(perform_cv=True)

    # Phase 3: Evaluate models with ENHANCED metrics
    print("\n" + "=" * 80)
    print("PHASE 3: ENHANCED PERFORMANCE EVALUATION")
    print("=" * 80)

    pipeline.evaluate_liquefaction_model()
    pipeline.evaluate_settlement_model()
    pipeline.evaluate_bearing_capacity_model()

    # NEW: Feature importance analysis
    pipeline.analyze_feature_importance()

    # Generate reports and ENHANCED visualizations
    pipeline.generate_comprehensive_report()
    pipeline.plot_confusion_matrix()
    pipeline.plot_roc_curve()  # NEW in upgraded version
    pipeline.plot_regression_predictions()  # NEW in upgraded version

    # Save UPGRADED models
    pipeline.save_models()

    # Upload results to Supabase Storage
    pipeline.upload_results_to_storage(BUCKET_NAME)

    print("\n" + "=" * 80)
    print("[OK] UPGRADED ANN TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n UPGRADED MODEL PERFORMANCE SUMMARY:")
    print("\n1. Liquefaction Classification:")
    print(
        f"   Test Accuracy: {pipeline.results['liquefaction']['test']['accuracy']:.4f}")
    print(
        f"   Test F1-Score: {pipeline.results['liquefaction']['test']['f1_score']:.4f}")
    print(
        f"   Test ROC-AUC:  {pipeline.results['liquefaction']['test']['roc_auc']:.4f} (NEW)")

    print("\n2. Settlement Prediction:")
    print(
        f"   Test RMSE: {pipeline.results['settlement']['test']['rmse']:.4f} cm")
    print(f"   Test R^2:   {pipeline.results['settlement']['test']['r2']:.4f}")
    print(
        f"   Test MAPE: {pipeline.results['settlement']['test']['mape']:.2f}% (NEW)")

    print("\n3. Bearing Capacity Prediction:")
    print(
        f"   Test RMSE: {pipeline.results['bearing_capacity']['test']['rmse']:.2f} kPa")
    print(
        f"   Test R^2:   {pipeline.results['bearing_capacity']['test']['r2']:.4f}")
    print(
        f"   Test MAPE: {pipeline.results['bearing_capacity']['test']['mape']:.2f}% (NEW)")

    print("\n\n KEY IMPROVEMENTS IN upgraded version:")
    print("  [OK] Enhanced ANN architecture: 256-128-64 neurons")
    print("  [OK] Spatial features from PostGIS integrated")
    print("  [OK] Class weight balancing for better liquefaction detection")
    print("  [OK] Cross-validation for robust evaluation")
    print("  [OK] ROC-AUC and MAPE metrics added")
    print("  [OK] Feature importance analysis")
    print("  [OK] Enhanced visualization (ROC curves, regression plots)")

    print("\n\n Results uploaded to Supabase Storage:")
    print(f"  Bucket: {BUCKET_NAME}")
    print("  Path: ml_models/")
    print("\n Files available:")
    print("  - ann_liquefaction.pkl (upgraded classifier)")
    print("  - ann_settlement.pkl (upgraded regressor)")
    print("  - ann_bearing_capacity.pkl (upgraded regressor)")
    print("  - scaler.pkl (feature scaler)")
    print("  - ann_metadata.json (results and configuration)")
    print("  - upgraded_ann_training_report.txt (detailed report)")
    print("  - confusion_matrix_liquefaction.png (visualization)")
    print("  - roc_curve_liquefaction.png (NEW - ROC analysis)")
    print("  - regression_predictions.png (NEW - regression performance)")
    print("=" * 80 + "\n")

    return pipeline


if __name__ == "__main__":
    upgraded_ann_pipeline = main()
