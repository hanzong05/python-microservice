"""
ANN Training Script for Tarlac Liquefaction Prediction
Fetches data directly from Supabase ml_training_data view
"""

import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Supabase connection
DATABASE_URL = os.getenv('DATABASE_URL',
                         'postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT.supabase.co:5432/postgres'
                         )

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Output directory
OUTPUT_DIR = 'models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("TARLAC LIQUEFACTION PREDICTION - ANN TRAINING")
print("="*70)

# ============================================================================
# STEP 1: FETCH DATA FROM SUPABASE
# ============================================================================

print("\n[1/8] Connecting to Supabase...")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

print("[1/8] Fetching training data from ml_training_data view...")
query = "SELECT * FROM ml_training_data"
df = pd.read_sql(query, engine)

print(f"✓ Loaded {len(df)} records")
print(f"✓ Columns: {df.shape[1]}")

# ============================================================================
# STEP 2: FEATURE SELECTION
# ============================================================================

print("\n[2/8] Selecting features and targets...")

# Feature columns
feature_cols = [
    'latitude',
    'longitude',
    'elevation',
    'layer_number',
    'depth_from_m',
    'depth_to_m',
    'depth_mid_m',
    'layer_thickness_m',
    'spt_n_value',
    'spt_n160',
    'unit_weight',
    'moisture_content',
    'plasticity_index',
    'liquid_limit',
    'fines_content',
    'mean_particle_size_d50',
    'friction_angle',
    'cohesion_kpa',
    'effective_overburden_pressure',
    'total_overburden_pressure',
    'groundwater_depth_m',
    'relative_density_percent',
    'pga_g',
    'csr',
    'cyclic_strength_ratio',
    'foundation_width_m',
    'foundation_depth_m',
    'factor_of_safety',
]

# Target columns
target_cols = {
    'liquefaction_binary': 'classification',
    'settlement_cm': 'regression',
    'bearing_capacity_kpa': 'regression',
    'qa_allowable_kpa': 'regression'
}

# Remove rows with missing critical data
required_cols = feature_cols + list(target_cols.keys())
available_features = [col for col in feature_cols if col in df.columns]
df_clean = df[available_features + list(target_cols.keys())].dropna()

print(f"✓ Using {len(available_features)} features")
print(f"✓ Clean dataset: {len(df_clean)} samples")

# Check class balance
liq_count = df_clean['liquefaction_binary'].sum()
print(f"✓ Liquefaction distribution: {liq_count} liquefiable ({liq_count/len(df_clean)*100:.1f}%), "
      f"{len(df_clean)-liq_count} non-liquefiable ({(len(df_clean)-liq_count)/len(df_clean)*100:.1f}%)")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================

print("\n[3/8] Splitting data (80% train, 20% test)...")

X = df_clean[available_features].values
y_liq = df_clean['liquefaction_binary'].values
y_settlement = df_clean['settlement_cm'].values
y_bc = df_clean['bearing_capacity_kpa'].values
y_qa = df_clean['qa_allowable_kpa'].values

# Stratified split based on liquefaction
X_train, X_test, y_liq_train, y_liq_test = train_test_split(
    X, y_liq,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_liq
)

# Use same indices for other targets
_, _, y_set_train, y_set_test = train_test_split(
    X, y_settlement, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_liq
)
_, _, y_bc_train, y_bc_test = train_test_split(
    X, y_bc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_liq
)
_, _, y_qa_train, y_qa_test = train_test_split(
    X, y_qa, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_liq
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================

print("\n[4/8] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 5: BUILD ANN MODEL
# ============================================================================

print("\n[5/8] Building Multi-Output ANN model...")

# Input layer
input_layer = Input(shape=(X_train_scaled.shape[1],), name='input')

# Shared hidden layers
x = Dense(128, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(32, activation='relu')(x)
x = Dropout(0.1)(x)

# Output heads
liquefaction_output = Dense(1, activation='sigmoid', name='liquefaction')(x)
settlement_output = Dense(1, activation='linear', name='settlement')(x)
bearing_capacity_output = Dense(
    1, activation='linear', name='bearing_capacity')(x)
qa_allowable_output = Dense(1, activation='linear', name='qa_allowable')(x)

# Create model
model = Model(
    inputs=input_layer,
    outputs=[liquefaction_output, settlement_output,
             bearing_capacity_output, qa_allowable_output]
)

# Compile with loss weights (prioritize liquefaction prediction)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss={
        'liquefaction': 'binary_crossentropy',
        'settlement': 'mse',
        'bearing_capacity': 'mse',
        'qa_allowable': 'mse'
    },
    loss_weights={
        'liquefaction': 2.0,  # Higher priority
        'settlement': 1.0,
        'bearing_capacity': 1.0,
        'qa_allowable': 0.5
    },
    metrics={
        'liquefaction': ['accuracy', tf.keras.metrics.AUC(name='auc')],
        'settlement': ['mae'],
        'bearing_capacity': ['mae'],
        'qa_allowable': ['mae']
    }
)

print("\n" + "="*70)
model.summary()
print("="*70)

# ============================================================================
# STEP 6: TRAIN MODEL
# ============================================================================

print("\n[6/8] Training model...")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_liquefaction_auc',
        patience=20,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=f'{OUTPUT_DIR}/best_model.keras',
        monitor='val_liquefaction_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# Train
history = model.fit(
    X_train_scaled,
    {
        'liquefaction': y_liq_train,
        'settlement': y_set_train,
        'bearing_capacity': y_bc_train,
        'qa_allowable': y_qa_train
    },
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training complete!")

# ============================================================================
# STEP 7: EVALUATE MODEL
# ============================================================================

print("\n[7/8] Evaluating model on test set...")

# Predictions
predictions = model.predict(X_test_scaled)
liq_pred_prob = predictions[0].flatten()
liq_pred_binary = (liq_pred_prob > 0.5).astype(int)
settlement_pred = predictions[1].flatten()
bc_pred = predictions[2].flatten()
qa_pred = predictions[3].flatten()

# ---- LIQUEFACTION CLASSIFICATION ----
print("\n" + "="*70)
print("LIQUEFACTION CLASSIFICATION RESULTS")
print("="*70)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_liq_test, liq_pred_binary)
print(cm)

print("\nClassification Report:")
print(classification_report(y_liq_test, liq_pred_binary,
                            target_names=['Non-Liquefiable', 'Liquefiable']))

accuracy = accuracy_score(y_liq_test, liq_pred_binary)
precision = precision_score(y_liq_test, liq_pred_binary)
recall = recall_score(y_liq_test, liq_pred_binary)
f1 = f1_score(y_liq_test, liq_pred_binary)

print(f"\nSummary Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# ---- SETTLEMENT REGRESSION ----
print("\n" + "="*70)
print("SETTLEMENT PREDICTION RESULTS")
print("="*70)

rmse_settlement = np.sqrt(mean_squared_error(y_set_test, settlement_pred))
mae_settlement = mean_absolute_error(y_set_test, settlement_pred)
r2_settlement = r2_score(y_set_test, settlement_pred)

print(f"\nSettlement (vs Tokimatsu & Seed 1987):")
print(f"  RMSE: {rmse_settlement:.2f} cm")
print(f"  MAE:  {mae_settlement:.2f} cm")
print(f"  R²:   {r2_settlement:.4f}")

# ---- BEARING CAPACITY REGRESSION ----
print("\n" + "="*70)
print("BEARING CAPACITY PREDICTION RESULTS")
print("="*70)

rmse_bc = np.sqrt(mean_squared_error(y_bc_test, bc_pred))
mae_bc = mean_absolute_error(y_bc_test, bc_pred)
r2_bc = r2_score(y_bc_test, bc_pred)

print(f"\nBearing Capacity (vs Terzaghi 1943):")
print(f"  RMSE: {rmse_bc:.2f} kPa")
print(f"  MAE:  {mae_bc:.2f} kPa")
print(f"  R²:   {r2_bc:.4f}")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("\n[8/8] Saving model and results...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save final model
model.save(f'{OUTPUT_DIR}/tarlac_liquefaction_model_{timestamp}.keras')
print(
    f"✓ Model saved: {OUTPUT_DIR}/tarlac_liquefaction_model_{timestamp}.keras")

# Save scaler
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler_{timestamp}.pkl')
print(f"✓ Scaler saved: {OUTPUT_DIR}/scaler_{timestamp}.pkl")

# Save feature names
with open(f'{OUTPUT_DIR}/feature_names_{timestamp}.txt', 'w') as f:
    f.write('\n'.join(available_features))
print(f"✓ Feature names saved")

# Save evaluation metrics
metrics = {
    'timestamp': timestamp,
    'total_samples': len(df_clean),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'liquefaction_accuracy': float(accuracy),
    'liquefaction_precision': float(precision),
    'liquefaction_recall': float(recall),
    'liquefaction_f1': float(f1),
    'settlement_rmse_cm': float(rmse_settlement),
    'settlement_mae_cm': float(mae_settlement),
    'settlement_r2': float(r2_settlement),
    'bearing_capacity_rmse_kpa': float(rmse_bc),
    'bearing_capacity_mae_kpa': float(mae_bc),
    'bearing_capacity_r2': float(r2_bc),
}

with open(f'{OUTPUT_DIR}/metrics_{timestamp}.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Metrics saved: {OUTPUT_DIR}/metrics_{timestamp}.json")

# Plot training history
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(history.history['liquefaction_accuracy'], label='Train')
plt.plot(history.history['val_liquefaction_accuracy'], label='Val')
plt.title('Liquefaction Accuracy')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(history.history['liquefaction_auc'], label='Train')
plt.plot(history.history['val_liquefaction_auc'], label='Val')
plt.title('Liquefaction AUC')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(history.history['liquefaction_loss'], label='Train')
plt.plot(history.history['val_liquefaction_loss'], label='Val')
plt.title('Liquefaction Loss')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(history.history['settlement_mae'], label='Train')
plt.plot(history.history['val_settlement_mae'], label='Val')
plt.title('Settlement MAE (cm)')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(history.history['bearing_capacity_mae'], label='Train')
plt.plot(history.history['val_bearing_capacity_mae'], label='Val')
plt.title('Bearing Capacity MAE (kPa)')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Total Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_history_{timestamp}.png', dpi=300)
print(f"✓ Training history plot saved")

# Confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Liq', 'Liq'],
            yticklabels=['Non-Liq', 'Liq'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_{timestamp}.png', dpi=300)
print(f"✓ Confusion matrix saved")

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print(f"\nModel files saved in: {OUTPUT_DIR}/")
print(f"  - tarlac_liquefaction_model_{timestamp}.keras")
print(f"  - scaler_{timestamp}.pkl")
print(f"  - feature_names_{timestamp}.txt")
print(f"  - metrics_{timestamp}.json")
print(f"  - training_history_{timestamp}.png")
print(f"  - confusion_matrix_{timestamp}.png")
print("\nNext steps:")
print("1. Review metrics and plots")
print("2. Integrate model into web application")
print("3. Test predictions on new boreholes")
print("4. Compare with empirical methods (DPWH BSDS 2013)")
