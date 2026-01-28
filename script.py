"""
ANN-Based Machine Learning Training Pipeline for Liquefaction Prediction
Tarlac Province Geotechnical Data

Based on Research Methodology:
- Trains Artificial Neural Network (Multi-Layer Perceptron) only
- Predicts: Liquefaction Potential, Settlement, and Bearing Capacity
- Validates against: DPWH BSDS (2013), Tokimatsu & Seed (1987), Terzaghi (1943)
- Downloads training data from Supabase Storage
- Uploads trained model and results back to Supabase Storage

Author: Geotechnical ML Pipeline
Date: 2026-01-28
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, mean_squared_error,
        mean_absolute_error, r2_score
    )
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    print("X scikit-learn not installed!")
    print("Install with: pip install scikit-learn joblib")
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠ Matplotlib/Seaborn not available - plots will be skipped")
    PLOTTING_AVAILABLE = False


# -----------------------------
# Supabase Storage Functions
# -----------------------------

def download_file_from_storage(bucket_name, file_path):
    """Download file from Supabase Storage"""
    print(f"📥 Downloading {file_path} from Supabase Storage...")
    client = get_supabase_client()
    if not client:
        print("X Failed to connect to Supabase")
        return None

    try:
        response = client.storage.from_(bucket_name).download(file_path)
        print(f"  ✓ Downloaded {len(response)} bytes")
        return response
    except Exception as e:
        print(f"  X Error downloading file: {e}")
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
        print(f"  ✓ Uploaded to {storage_path}")
        return True

    except Exception as e:
        print(f"  X Error uploading: {e}")
        return False


# -----------------------------
# ANN Training Pipeline
# -----------------------------

class LiquefactionANNPipeline:
    """
    ANN-Based Machine Learning Pipeline for Liquefaction Prediction
    Following Research Methodology: Phase 2 - Model Development and Training
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

    def load_data_from_storage(self, bucket_name, base_path='feature_engineering'):
        """Load training data from Supabase Storage"""
        print("=" * 80)
        print("LOADING TRAINING DATA FROM SUPABASE STORAGE")
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

            # Settlement (regression - only for liquefiable samples)
            self.y_train_settlement = train_df['settlement_cm'].values
            self.y_val_settlement = val_df['settlement_cm'].values
            self.y_test_settlement = test_df['settlement_cm'].values

            # Bearing Capacity (regression)
            self.y_train_bearing = train_df['qa_allowable_kpa'].values
            self.y_val_bearing = val_df['qa_allowable_kpa'].values
            self.y_test_bearing = test_df['qa_allowable_kpa'].values

            print(f"\n✓ Data loaded successfully!")
            print(f"  - Features: {len(self.feature_names)}")
            print(f"  - Training samples: {len(self.X_train)}")
            print(f"  - Validation samples: {len(self.X_val)}")
            print(f"  - Test samples: {len(self.X_test)}")
            print(
                f"\n  - Training liquefaction distribution: Liq={self.y_train_liq.sum()}, Non-liq={len(self.y_train_liq)-self.y_train_liq.sum()}")

            return True

        except Exception as e:
            print(f"X Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_data(self):
        """Standardize features using StandardScaler"""
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA (Phase 2: Step 6)")
        print("=" * 80)

        print("\nStandardizing features using StandardScaler...")
        self.scaler = StandardScaler()

        # Fit on training data and transform all sets
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        print("✓ Features standardized successfully!")
        print(f"  - Mean: ~0.0")
        print(f"  - Std: ~1.0")

        return True

    def train_liquefaction_ann(self):
        """
        Train ANN for Liquefaction Classification
        Phase 2: Step 4-6 - ANN Architecture Design and Training
        """
        print("\n" + "=" * 80)
        print("TRAINING ANN - LIQUEFACTION POTENTIAL (CLASSIFICATION)")
        print("=" * 80)

        print("\nANN Configuration (Following Research Methodology):")
        print("  - Task: Binary Classification (Liquefiable vs Non-Liquefiable)")
        print("  - Hidden layers: (128, 64, 32)")
        print("  - Activation: ReLU (hidden), Logistic (output)")
        print("  - Solver: Adam optimizer")
        print("  - Max iterations: 500")
        print("  - Early stopping: Enabled")

        self.ann_liquefaction = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )

        print("\nTraining ANN for liquefaction classification...")
        self.ann_liquefaction.fit(self.X_train, self.y_train_liq)

        print("✓ ANN training completed!")
        print(f"  - Final training loss: {self.ann_liquefaction.loss_:.4f}")
        print(f"  - Iterations: {self.ann_liquefaction.n_iter_}")

        return self.ann_liquefaction

    def train_settlement_ann(self):
        """
        Train ANN for Settlement Prediction (Regression)
        Validates against Tokimatsu & Seed (1987) method
        """
        print("\n" + "=" * 80)
        print("TRAINING ANN - POST-LIQUEFACTION SETTLEMENT (REGRESSION)")
        print("=" * 80)

        print("\nANN Configuration:")
        print("  - Task: Regression (Settlement in cm)")
        print("  - Hidden layers: (128, 64, 32)")
        print("  - Activation: ReLU (hidden), Identity (output)")
        print("  - Solver: Adam optimizer")
        print("  - Max iterations: 500")
        print("  - Early stopping: Enabled")

        self.ann_settlement = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )

        print("\nTraining ANN for settlement prediction...")
        self.ann_settlement.fit(self.X_train, self.y_train_settlement)

        print("✓ ANN training completed!")
        print(f"  - Final training loss: {self.ann_settlement.loss_:.4f}")
        print(f"  - Iterations: {self.ann_settlement.n_iter_}")

        return self.ann_settlement

    def train_bearing_capacity_ann(self):
        """
        Train ANN for Bearing Capacity Prediction (Regression)
        Validates against Terzaghi (1943) method
        """
        print("\n" + "=" * 80)
        print("TRAINING ANN - ALLOWABLE BEARING CAPACITY (REGRESSION)")
        print("=" * 80)

        print("\nANN Configuration:")
        print("  - Task: Regression (Allowable Bearing Capacity in kPa)")
        print("  - Hidden layers: (128, 64, 32)")
        print("  - Activation: ReLU (hidden), Identity (output)")
        print("  - Solver: Adam optimizer")
        print("  - Max iterations: 500")
        print("  - Early stopping: Enabled")

        self.ann_bearing_capacity = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )

        print("\nTraining ANN for bearing capacity prediction...")
        self.ann_bearing_capacity.fit(self.X_train, self.y_train_bearing)

        print("✓ ANN training completed!")
        print(
            f"  - Final training loss: {self.ann_bearing_capacity.loss_:.4f}")
        print(f"  - Iterations: {self.ann_bearing_capacity.n_iter_}")

        return self.ann_bearing_capacity

    def evaluate_liquefaction_model(self):
        """
        Evaluate Liquefaction Classification Model
        Phase 3: Step 8 - Statistical Analysis using Confusion Matrix
        """
        print("\n" + "=" * 80)
        print("EVALUATING LIQUEFACTION MODEL (Phase 3: Step 8)")
        print("=" * 80)

        # Validation set evaluation
        y_val_pred = self.ann_liquefaction.predict(self.X_val)
        val_accuracy = accuracy_score(self.y_val_liq, y_val_pred)
        val_precision = precision_score(
            self.y_val_liq, y_val_pred, zero_division=0)
        val_recall = recall_score(self.y_val_liq, y_val_pred, zero_division=0)
        val_f1 = f1_score(self.y_val_liq, y_val_pred, zero_division=0)
        val_cm = confusion_matrix(self.y_val_liq, y_val_pred)

        print("\n📊 VALIDATION SET PERFORMANCE:")
        print(f"  Accuracy:  {val_accuracy:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  F1-Score:  {val_f1:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {val_cm[0, 0]:<6} FP: {val_cm[0, 1]}")
        print(f"    FN: {val_cm[1, 0]:<6} TP: {val_cm[1, 1]}")

        # Test set evaluation
        y_test_pred = self.ann_liquefaction.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test_liq, y_test_pred)
        test_precision = precision_score(
            self.y_test_liq, y_test_pred, zero_division=0)
        test_recall = recall_score(
            self.y_test_liq, y_test_pred, zero_division=0)
        test_f1 = f1_score(self.y_test_liq, y_test_pred, zero_division=0)
        test_cm = confusion_matrix(self.y_test_liq, y_test_pred)

        print("\n📊 TEST SET PERFORMANCE:")
        print(f"  Accuracy:  {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        print(f"  F1-Score:  {test_f1:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {test_cm[0, 0]:<6} FP: {test_cm[0, 1]}")
        print(f"    FN: {test_cm[1, 0]:<6} TP: {test_cm[1, 1]}")

        self.results['liquefaction'] = {
            'validation': {
                'accuracy': float(val_accuracy),
                'precision': float(val_precision),
                'recall': float(val_recall),
                'f1_score': float(val_f1),
                'confusion_matrix': val_cm.tolist()
            },
            'test': {
                'accuracy': float(test_accuracy),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1_score': float(test_f1),
                'confusion_matrix': test_cm.tolist()
            }
        }

        return self.results['liquefaction']

    def evaluate_settlement_model(self):
        """
        Evaluate Settlement Regression Model
        Phase 3: Step 8 - Statistical Analysis using RMSE
        """
        print("\n" + "=" * 80)
        print("EVALUATING SETTLEMENT MODEL (Phase 3: Step 8)")
        print("=" * 80)

        # Validation set evaluation
        y_val_pred = self.ann_settlement.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(
            self.y_val_settlement, y_val_pred))
        val_mae = mean_absolute_error(self.y_val_settlement, y_val_pred)
        val_r2 = r2_score(self.y_val_settlement, y_val_pred)

        print("\n📊 VALIDATION SET PERFORMANCE:")
        print(f"  RMSE: {val_rmse:.4f} cm")
        print(f"  MAE:  {val_mae:.4f} cm")
        print(f"  R²:   {val_r2:.4f}")

        # Test set evaluation
        y_test_pred = self.ann_settlement.predict(self.X_test)
        test_rmse = np.sqrt(mean_squared_error(
            self.y_test_settlement, y_test_pred))
        test_mae = mean_absolute_error(self.y_test_settlement, y_test_pred)
        test_r2 = r2_score(self.y_test_settlement, y_test_pred)

        print("\n📊 TEST SET PERFORMANCE:")
        print(f"  RMSE: {test_rmse:.4f} cm")
        print(f"  MAE:  {test_mae:.4f} cm")
        print(f"  R²:   {test_r2:.4f}")

        self.results['settlement'] = {
            'validation': {
                'rmse': float(val_rmse),
                'mae': float(val_mae),
                'r2': float(val_r2)
            },
            'test': {
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'r2': float(test_r2)
            }
        }

        return self.results['settlement']

    def evaluate_bearing_capacity_model(self):
        """
        Evaluate Bearing Capacity Regression Model
        Phase 3: Step 8 - Statistical Analysis using RMSE
        """
        print("\n" + "=" * 80)
        print("EVALUATING BEARING CAPACITY MODEL (Phase 3: Step 8)")
        print("=" * 80)

        # Validation set evaluation
        y_val_pred = self.ann_bearing_capacity.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val_bearing, y_val_pred))
        val_mae = mean_absolute_error(self.y_val_bearing, y_val_pred)
        val_r2 = r2_score(self.y_val_bearing, y_val_pred)

        print("\n📊 VALIDATION SET PERFORMANCE:")
        print(f"  RMSE: {val_rmse:.2f} kPa")
        print(f"  MAE:  {val_mae:.2f} kPa")
        print(f"  R²:   {val_r2:.4f}")

        # Test set evaluation
        y_test_pred = self.ann_bearing_capacity.predict(self.X_test)
        test_rmse = np.sqrt(mean_squared_error(
            self.y_test_bearing, y_test_pred))
        test_mae = mean_absolute_error(self.y_test_bearing, y_test_pred)
        test_r2 = r2_score(self.y_test_bearing, y_test_pred)

        print("\n📊 TEST SET PERFORMANCE:")
        print(f"  RMSE: {test_rmse:.2f} kPa")
        print(f"  MAE:  {test_mae:.2f} kPa")
        print(f"  R²:   {test_r2:.4f}")

        self.results['bearing_capacity'] = {
            'validation': {
                'rmse': float(val_rmse),
                'mae': float(val_mae),
                'r2': float(val_r2)
            },
            'test': {
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'r2': float(test_r2)
            }
        }

        return self.results['bearing_capacity']

    def generate_comprehensive_report(self, output_file='ml_models/ann_training_report.txt'):
        """Generate comprehensive ANN training and validation report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE TRAINING REPORT")
        print("=" * 80)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ANN-BASED LIQUEFACTION PREDICTION SYSTEM\n")
            f.write("Training and Validation Report\n")
            f.write("Tarlac Province, Philippines\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of features: {len(self.feature_names)}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Validation samples: {len(self.X_val)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n\n")

            # Model architecture
            f.write("ANN MODEL ARCHITECTURE:\n")
            f.write("-" * 80 + "\n")
            f.write("Hidden Layer Structure: (128, 64, 32) neurons\n")
            f.write("Activation Function: ReLU\n")
            f.write("Solver: Adam optimizer\n")
            f.write("Max Iterations: 500\n")
            f.write("Early Stopping: Enabled\n\n")

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

            f.write("\nTest Set:\n")
            f.write(f"  Accuracy:  {liq_test['accuracy']:.4f}\n")
            f.write(f"  Precision: {liq_test['precision']:.4f}\n")
            f.write(f"  Recall:    {liq_test['recall']:.4f}\n")
            f.write(f"  F1-Score:  {liq_test['f1_score']:.4f}\n")

            cm = liq_test['confusion_matrix']
            f.write(f"\nConfusion Matrix (Test Set):\n")
            f.write(f"  TN: {cm[0][0]:<6} FP: {cm[0][1]}\n")
            f.write(f"  FN: {cm[1][0]:<6} TP: {cm[1][1]}\n\n")

            # Settlement model results
            f.write("2. POST-LIQUEFACTION SETTLEMENT (REGRESSION):\n")
            f.write("-" * 80 + "\n")
            f.write("Validation Method: Tokimatsu & Seed (1987)\n\n")

            settle_val = self.results['settlement']['validation']
            settle_test = self.results['settlement']['test']

            f.write("Validation Set:\n")
            f.write(f"  RMSE: {settle_val['rmse']:.4f} cm\n")
            f.write(f"  MAE:  {settle_val['mae']:.4f} cm\n")
            f.write(f"  R²:   {settle_val['r2']:.4f}\n")

            f.write("\nTest Set:\n")
            f.write(f"  RMSE: {settle_test['rmse']:.4f} cm\n")
            f.write(f"  MAE:  {settle_test['mae']:.4f} cm\n")
            f.write(f"  R²:   {settle_test['r2']:.4f}\n\n")

            # Bearing capacity model results
            f.write("3. ALLOWABLE BEARING CAPACITY (REGRESSION):\n")
            f.write("-" * 80 + "\n")
            f.write("Validation Method: Terzaghi (1943)\n\n")

            bearing_val = self.results['bearing_capacity']['validation']
            bearing_test = self.results['bearing_capacity']['test']

            f.write("Validation Set:\n")
            f.write(f"  RMSE: {bearing_val['rmse']:.2f} kPa\n")
            f.write(f"  MAE:  {bearing_val['mae']:.2f} kPa\n")
            f.write(f"  R²:   {bearing_val['r2']:.4f}\n")

            f.write("\nTest Set:\n")
            f.write(f"  RMSE: {bearing_test['rmse']:.2f} kPa\n")
            f.write(f"  MAE:  {bearing_test['mae']:.2f} kPa\n")
            f.write(f"  R²:   {bearing_test['r2']:.4f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"✓ Report generated: {output_file}")
        return output_file

    def plot_confusion_matrix(self, output_file='ml_models/confusion_matrix_liquefaction.png'):
        """Generate confusion matrix plot for liquefaction model"""
        if not PLOTTING_AVAILABLE:
            print("⚠ Plotting not available - skipping confusion matrix")
            return None

        print("\n" + "=" * 80)
        print("GENERATING CONFUSION MATRIX PLOT")
        print("=" * 80)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Validation confusion matrix
            cm_val = np.array(
                self.results['liquefaction']['validation']['confusion_matrix'])
            sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=['Non-Liq', 'Liq'],
                        yticklabels=['Non-Liq', 'Liq'])
            ax1.set_title('Liquefaction ANN - Validation Set')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')

            # Test confusion matrix
            cm_test = np.array(
                self.results['liquefaction']['test']['confusion_matrix'])
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax2,
                        xticklabels=['Non-Liq', 'Liq'],
                        yticklabels=['Non-Liq', 'Liq'])
            ax2.set_title('Liquefaction ANN - Test Set')
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"  X Error creating confusion matrix plot: {e}")
            return None

    def save_models(self, output_dir='ml_models'):
        """Save all trained ANN models and metadata"""
        print("\n" + "=" * 80)
        print("SAVING TRAINED MODELS")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        # Save scaler
        scaler_file = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_file)
        print(f"✓ Saved scaler: {scaler_file}")
        saved_files.append(scaler_file)

        # Save liquefaction model
        liq_model_file = os.path.join(output_dir, 'ann_liquefaction.pkl')
        joblib.dump(self.ann_liquefaction, liq_model_file)
        print(f"✓ Saved liquefaction model: {liq_model_file}")
        saved_files.append(liq_model_file)

        # Save settlement model
        settle_model_file = os.path.join(output_dir, 'ann_settlement.pkl')
        joblib.dump(self.ann_settlement, settle_model_file)
        print(f"✓ Saved settlement model: {settle_model_file}")
        saved_files.append(settle_model_file)

        # Save bearing capacity model
        bearing_model_file = os.path.join(
            output_dir, 'ann_bearing_capacity.pkl')
        joblib.dump(self.ann_bearing_capacity, bearing_model_file)
        print(f"✓ Saved bearing capacity model: {bearing_model_file}")
        saved_files.append(bearing_model_file)

        # Save model metadata
        metadata = {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'training_samples': len(self.X_train),
            'validation_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'timestamp': datetime.now().isoformat(),
            'model_architecture': {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 500
            },
            'results': self.results
        }

        metadata_file = os.path.join(output_dir, 'ann_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_file}")
        saved_files.append(metadata_file)

        return saved_files

    def upload_results_to_storage(self, bucket_name, local_dir='ml_models', storage_base='ml_models'):
        """Upload all model files and results to Supabase Storage"""
        print("\n" + "=" * 80)
        print("UPLOADING RESULTS TO SUPABASE STORAGE")
        print("=" * 80)

        uploaded_files = []

        # Get all files in the local directory
        for filename in os.listdir(local_dir):
            local_path = os.path.join(local_dir, filename)
            if os.path.isfile(local_path):
                storage_path = f'{storage_base}/{filename}'
                if upload_to_supabase_storage(local_path, bucket_name, storage_path):
                    uploaded_files.append(storage_path)

        print(f"\n✓ Uploaded {len(uploaded_files)} files to Supabase Storage")
        return uploaded_files


def main():
    """Main execution following research methodology"""
    print("\n" + "=" * 80)
    print("ANN-BASED LIQUEFACTION PREDICTION SYSTEM")
    print("Training Pipeline - Tarlac Province")
    print("Following Research Methodology (Phase 2 & 3)")
    print("=" * 80 + "\n")

    if not SKLEARN_AVAILABLE:
        print("X scikit-learn is required but not installed")
        return None

    # Configuration
    BUCKET_NAME = 'geotechnical-data'

    # Initialize pipeline
    pipeline = LiquefactionANNPipeline()

    # Phase 1: Load data from Supabase Storage
    if not pipeline.load_data_from_storage(BUCKET_NAME):
        print("X Failed to load data. Exiting.")
        return None

    # Phase 2: Preprocess data
    pipeline.preprocess_data()

    # Phase 2: Train ANN models
    print("\n" + "=" * 80)
    print("PHASE 2: MODEL DEVELOPMENT AND TRAINING")
    print("=" * 80)

    pipeline.train_liquefaction_ann()
    pipeline.train_settlement_ann()
    pipeline.train_bearing_capacity_ann()

    # Phase 3: Evaluate models
    print("\n" + "=" * 80)
    print("PHASE 3: PERFORMANCE EVALUATION")
    print("=" * 80)

    pipeline.evaluate_liquefaction_model()
    pipeline.evaluate_settlement_model()
    pipeline.evaluate_bearing_capacity_model()

    # Generate reports and visualizations
    pipeline.generate_comprehensive_report()
    pipeline.plot_confusion_matrix()

    # Save models
    pipeline.save_models()

    # Upload results to Supabase Storage
    pipeline.upload_results_to_storage(BUCKET_NAME)

    print("\n" + "=" * 80)
    print("✓ ANN TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nMODEL PERFORMANCE SUMMARY:")
    print("\n1. Liquefaction Classification:")
    print(
        f"   Test Accuracy: {pipeline.results['liquefaction']['test']['accuracy']:.4f}")
    print(
        f"   Test F1-Score: {pipeline.results['liquefaction']['test']['f1_score']:.4f}")
    print("\n2. Settlement Prediction:")
    print(
        f"   Test RMSE: {pipeline.results['settlement']['test']['rmse']:.4f} cm")
    print(f"   Test R²: {pipeline.results['settlement']['test']['r2']:.4f}")
    print("\n3. Bearing Capacity Prediction:")
    print(
        f"   Test RMSE: {pipeline.results['bearing_capacity']['test']['rmse']:.2f} kPa")
    print(
        f"   Test R²: {pipeline.results['bearing_capacity']['test']['r2']:.4f}")

    print("\n\nResults uploaded to Supabase Storage:")
    print(f"  Bucket: {BUCKET_NAME}")
    print("  Path: ml_models/")
    print("\nFiles available:")
    print("  - ann_liquefaction.pkl (liquefaction classifier)")
    print("  - ann_settlement.pkl (settlement regressor)")
    print("  - ann_bearing_capacity.pkl (bearing capacity regressor)")
    print("  - scaler.pkl (feature scaler)")
    print("  - ann_metadata.json (results and configuration)")
    print("  - ann_training_report.txt (detailed report)")
    print("  - confusion_matrix_liquefaction.png (visualization)")
    print("=" * 80 + "\n")

    return pipeline


if __name__ == "__main__":
    ann_pipeline = main()
