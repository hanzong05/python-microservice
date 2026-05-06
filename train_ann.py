#!/usr/bin/env python3
"""
No-Leakage Multi-Output ANN Model Training

Changes made:
- Removed leakage features: factor_of_safety, csr, crr/cyclic_strength_ratio.
- Uses raw soil properties only as inputs.
- Removed L/B output because it was constant at 1.0.12
- Predicts only: Foundation Width B (m), Foundation Depth D (m).
- Strictly cleans invalid rows before training.
- Uses 100% of cleaned data for training; no 80/20 split.

Model:
- Inputs: available raw soil features only
- Hidden layer 1: 30 neurons, tanh
- Hidden layer 2: 15 neurons, tanh
- Output layer: 2 neurons, linear [B, D]
"""

import io
import json
import math
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[INFO] python-dotenv not installed - .env file will not be loaded")

try:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[ERROR] scikit-learn/joblib not installed")
    print("Install: pip install scikit-learn joblib")
    SKLEARN_AVAILABLE = False

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("[ERROR] supabase-py not installed")
    print("Install: pip install supabase")
    SUPABASE_AVAILABLE = False


class MultiOutputANNTraining:
    """
    ANN training without target leakage.

    Inputs are raw/geotechnical properties only.
    Outputs are B and D only.
    L/B is removed because a constant output does not teach the model anything.
    """

    def __init__(self):
        self.client = None
        self.df = None
        self.df_raw = None

        self.feature_names = []
        self.scaler = None
        self.multi_model = None
        self.width_model = None
        self.depth_model = None

        self.X_all = None
        self.y_B_all = None
        self.y_D_all = None

    def connect_database(self) -> bool:
        print("\n" + "=" * 80)
        print("CONNECTING TO POSTGIS DATABASE")
        print("=" * 80)

        if not SUPABASE_AVAILABLE:
            return False

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            print("  [ERROR] Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")
            return False

        try:
            self.client = create_client(supabase_url, supabase_key)
            self.client.table("soil_layers").select("id").limit(1).execute()
            print("  [OK] Connected to PostGIS database")
            return True
        except Exception as e:
            print(f"  [ERROR] Connection failed: {e}")
            return False

    def query_database(self) -> bool:
        print("\n" + "=" * 80)
        print("QUERYING DATABASE FOR TRAINING DATA")
        print("=" * 80)

        try:
            print("  Fetching data from v_complete_soil_data view...")
            try:
                result = self.client.table("v_complete_soil_data").select("*").execute()
            except Exception:
                print("  View not available, querying soil_layers table...")
                result = self.client.table("soil_layers").select("*").execute()

            if not result.data:
                print("  [ERROR] No data returned")
                return False

            self.df = pd.DataFrame(result.data)
            self.df_raw = self.df.copy()
            print(f"  [OK] Retrieved {len(self.df)} records")
            print(f"  Columns: {len(self.df.columns)}")
            return True
        except Exception as e:
            print(f"  [ERROR] Query failed: {e}")
            return False

    @staticmethod
    def _num(series_or_value):
        return pd.to_numeric(series_or_value, errors="coerce")

    def clean_data_strictly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid rows before training."""
        print("\n" + "=" * 80)
        print("STRICT DATA CLEANING")
        print("=" * 80)

        before = len(df)
        df = df.copy()

        numeric_candidates = [
            "spt_n_value", "spt_n60", "spt_n160",
            "unit_weight", "fines_content", "friction_angle",
            "depth_mid_m", "depth_from_m", "depth_to_m",
            "groundwater_depth_m", "moisture_content", "plasticity_index",
            "mean_particle_size_d50", "pga_g", "relative_density_percent",
            "elastic_modulus_es", "foundation_width_m", "foundation_depth_m",
            "Foundation Width (B)", "Foundation Depth (D)",
        ]
        for col in numeric_candidates:
            if col in df.columns:
                df[col] = self._num(df[col])

        if "fines_content" in df.columns:
            df = df[(df["fines_content"] >= 0) & (df["fines_content"] <= 100)]

        if "unit_weight" in df.columns:
            df = df[(df["unit_weight"] >= 10) & (df["unit_weight"] <= 25)]

        # Require at least one SPT column.
        spt_cols = [c for c in ["spt_n_value", "spt_n60", "spt_n160"] if c in df.columns]
        if spt_cols:
            spt_ok = pd.Series(False, index=df.index)
            for col in spt_cols:
                spt_ok = spt_ok | (df[col].notna() & (df[col] > 0))
            df = df[spt_ok]

        # Require usable depth and groundwater when available.
        for col in ["depth_mid_m", "groundwater_depth_m"]:
            if col in df.columns:
                df = df[df[col].notna()]

        after = len(df)
        print(f"  Removed rows: {before - after}")
        print(f"  Remaining rows: {after}")
        return df

    def compute_foundation_targets(self, row) -> dict:
        """
        B and D target preparation.

        Best option: use real target columns if available:
        - foundation_width_m / Foundation Width (B)
        - foundation_depth_m / Foundation Depth (D)

        Fallback is only used when no real target is stored.
        Important: fallback D does NOT use factor_of_safety, CSR, or CRR.
        """
        q_actual = 50.0
        n = row.get("spt_n160")
        if pd.isna(n):
            n = row.get("spt_n60")
        if pd.isna(n):
            n = row.get("spt_n_value")
        try:
            n = max(1.0, float(n))
        except Exception:
            n = 15.0

        # Foundation width target.
        b_raw = row.get("foundation_width_m")
        if b_raw is None or pd.isna(b_raw):
            b_raw = row.get("Foundation Width (B)")
        try:
            b = float(b_raw)
            if b <= 0:
                raise ValueError
        except Exception:
            if n >= 6.0:
                b = 2.0
            else:
                b_calc = q_actual / (n * 8.49)
                b = max(2.5, round(b_calc * 2) / 2)
                b = min(5.0, b)

        # Foundation depth target.
        d_raw = row.get("foundation_depth_m")
        if d_raw is None or pd.isna(d_raw):
            d_raw = row.get("Foundation Depth (D)")
        try:
            d = float(d_raw)
            if d <= 0:
                raise ValueError
        except Exception:
            # No leakage: use raw groundwater/depth condition only, not FS/CSR/CRR.
            gw = row.get("groundwater_depth_m")
            depth_mid = row.get("depth_mid_m")
            try:
                gw = float(gw)
            except Exception:
                gw = 99.0
            try:
                depth_mid = float(depth_mid)
            except Exception:
                depth_mid = 1.5

            d = 3.0 if gw <= depth_mid else 1.5

        return {"B": round(b, 2), "D": round(d, 2)}

    def prepare_features_and_targets(self) -> bool:
        print("\n" + "=" * 80)
        print("PREPARING NO-LEAKAGE FEATURES AND TARGETS")
        print("=" * 80)

        df = self.clean_data_strictly(self.df)

        # Raw soil properties only. No csr, crr/cyclic_strength_ratio, factor_of_safety.
        raw_feature_candidates = [
            "spt_n_value",
            "spt_n60",
            "spt_n160",
            "unit_weight",
            "fines_content",
            "friction_angle",
            "depth_mid_m",
            "depth_from_m",
            "depth_to_m",
            "groundwater_depth_m",
            "moisture_content",
            "plasticity_index",
            "mean_particle_size_d50",
            "relative_density_percent",
            "elastic_modulus_es",
            "pga_g",  # keep only if your thesis allows seismic hazard as raw site input
        ]

        leakage_cols = {
            "factor_of_safety",
            "csr",
            "crr",
            "cyclic_strength_ratio",
            "liquefaction_probability",
            "liquefaction",
            "liquefaction_status",
            "liquefaction_risk_level",
            "bearing_capacity_kpa",
            "qa_allowable_kpa",
            "settlement_cm",
            "effective_overburden_pressure",
            "total_overburden_pressure",
        }

        feature_cols = [
            col for col in raw_feature_candidates
            if col in df.columns and col not in leakage_cols
        ]

        if not feature_cols:
            print("  [ERROR] No valid raw feature columns found")
            return False

        print(f"  Selected raw feature columns ({len(feature_cols)}):")
        for col in feature_cols:
            print(f"    - {col}")

        X = df[feature_cols].copy()
        for col in X.columns:
            X[col] = self._num(X[col])
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        foundation_targets = df.apply(self.compute_foundation_targets, axis=1)
        y_B = foundation_targets.apply(lambda r: r["B"])
        y_D = foundation_targets.apply(lambda r: r["D"])

        valid_mask = ~(y_B.isna() | y_D.isna())
        X = X[valid_mask]
        y_B = y_B[valid_mask]
        y_D = y_D[valid_mask]

        if len(X) < 20:
            print("  [ERROR] Too few valid rows after cleaning")
            return False

        self.feature_names = feature_cols

        # Use the WHOLE cleaned dataset for model fitting.
        # No 80/20 segregation is done here.
        self.X_all = X
        self.y_B_all = y_B
        self.y_D_all = y_D

        print(f"\n  Features shape: {X.shape}")
        print(f"  Target B: Mean={y_B.mean():.2f} m, Range=[{y_B.min():.2f}, {y_B.max():.2f}] m")
        print(f"  Target D: Mean={y_D.mean():.2f} m, Range=[{y_D.min():.2f}, {y_D.max():.2f}] m")
        print(f"  Training set: {len(self.X_all)} samples (100% of cleaned data)")
        print("  Validation split: disabled")
        return True

    def train_models(self) -> bool:
        if not SKLEARN_AVAILABLE:
            return False

        print("\n" + "=" * 80)
        print("TRAINING NO-LEAKAGE ANN MODEL")
        print("=" * 80)
        print(f"  INPUT LAYER: {self.X_all.shape[1]} neurons")
        print("  HIDDEN 1: 30 neurons, tanh")
        print("  HIDDEN 2: 15 neurons, tanh")
        print("  OUTPUT: 2 neurons, linear [B, D]")

        self.scaler = StandardScaler()
        X_all_scaled = self.scaler.fit_transform(self.X_all)

        y_train = np.column_stack([self.y_B_all, self.y_D_all])

        mlp_kwargs = dict(
            hidden_layer_sizes=(30, 15),
            activation="tanh",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=2000,
            early_stopping=False,
            n_iter_no_change=30,
            random_state=42,
            verbose=False,
        )

        self.multi_model = MLPRegressor(**mlp_kwargs)
        self.multi_model.fit(X_all_scaled, y_train)
        print(f"  [OK] Multi-output model completed in {self.multi_model.n_iter_} iterations")

        self.width_model = MLPRegressor(**mlp_kwargs)
        self.width_model.fit(X_all_scaled, self.y_B_all)
        print("  [OK] Foundation width model trained")

        self.depth_model = MLPRegressor(**mlp_kwargs)
        self.depth_model.fit(X_all_scaled, self.y_D_all)
        print("  [OK] Foundation depth model trained")
        return True

    def evaluate_models(self) -> bool:
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)

        print("  NOTE: Since the 80/20 split is disabled, these metrics are training-fit metrics, not independent validation metrics.")
        X_all_scaled = self.scaler.transform(self.X_all)

        def reg_metrics(name, y_true, y_pred, unit=""):
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            pi = 1.0 - rmse / (float(np.mean(y_true)) + 1e-9)
            print(f"\n  {name}:")
            print(f"    R²  : {r2:.4f}")
            print(f"    MAE : {mae:.4f}{unit}")
            print(f"    RMSE: {rmse:.4f}{unit}")
            print(f"    PI  : {pi:.4f}")
            return {"r2": r2, "mae": mae, "rmse": rmse, "pi": pi}

        y_pred = self.multi_model.predict(X_all_scaled)
        reg_metrics("Multi-output Foundation Width B", self.y_B_all, y_pred[:, 0], " m")
        reg_metrics("Multi-output Foundation Depth D", self.y_D_all, y_pred[:, 1], " m")

        y_B_pred = self.width_model.predict(X_all_scaled)
        y_D_pred = self.depth_model.predict(X_all_scaled)
        reg_metrics("Individual Foundation Width B", self.y_B_all, y_B_pred, " m")
        reg_metrics("Individual Foundation Depth D", self.y_D_all, y_D_pred, " m")
        return True

    def export_validation_excel(self) -> str:
        """Validation export disabled because training now uses 100% of cleaned data."""
        print("\n" + "=" * 80)
        print("VALIDATION EXPORT SKIPPED")
        print("=" * 80)
        print("  80/20 segregation is disabled; processing whole cleaned dataset now.")
        return ""

    def save_models(self) -> bool:
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)

        try:
            bucket_name = os.getenv("SUPABASE_STORAGE_BUCKET", "geotechnical-data")

            def upload_joblib(obj, path):
                buf = io.BytesIO()
                joblib.dump(obj, buf)
                buf.seek(0)
                self.client.storage.from_(bucket_name).upload(
                    path,
                    buf.getvalue(),
                    file_options={"content-type": "application/octet-stream", "upsert": "true"},
                )
                print(f"  [OK] {path}")

            upload_joblib(self.scaler, "models/scaler_no_leakage.pkl")
            upload_joblib(self.multi_model, "models/ann_multi_output_BD_no_leakage.pkl")
            upload_joblib(self.width_model, "models/ann_foundation_width_no_leakage.pkl")
            upload_joblib(self.depth_model, "models/ann_foundation_depth_no_leakage.pkl")

            metadata = {
                "version": "3.0-no-leakage",
                "model_type": "multi_output_regression",
                "targets": ["foundation_width_B", "foundation_depth_D"],
                "removed_targets": ["lb_ratio"],
                "removed_leakage_features": [
                    "factor_of_safety", "csr", "crr", "cyclic_strength_ratio",
                    "liquefaction_probability", "liquefaction_status",
                    "bearing_capacity_kpa", "qa_allowable_kpa", "settlement_cm",
                ],
                "feature_names": self.feature_names,
                "num_features": len(self.feature_names),
                "architecture": {
                    "type": "MLPRegressor",
                    "input_layer": len(self.feature_names),
                    "hidden_layers": [30, 15],
                    "hidden_activation": "tanh",
                    "output_layer": 2,
                    "output_activation": "linear",
                    "solver": "adam",
                },
                "training_samples": len(self.X_all),
                "validation_samples": 0,
                "data_split": "none - 100% cleaned data used for training",
                "timestamp": datetime.now().isoformat(),
            }

            self.client.storage.from_(bucket_name).upload(
                "models/ann_metadata_no_leakage.json",
                json.dumps(metadata, indent=2).encode("utf-8"),
                file_options={"content-type": "application/json", "upsert": "true"},
            )
            print("  [OK] models/ann_metadata_no_leakage.json")
            return True
        except Exception as e:
            print(f"  [ERROR] Save failed: {e}")
            return False

    def run(self) -> bool:
        print("\n" + "=" * 80)
        print("NO-LEAKAGE ANN MODEL TRAINING")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        steps = [
            self.connect_database,
            self.query_database,
            self.prepare_features_and_targets,
            self.train_models,
            self.evaluate_models,
        ]

        for step in steps:
            if not step():
                return False

        validation_file = self.export_validation_excel()
        if not self.save_models():
            return False

        print("\n" + "=" * 80)
        print("[SUCCESS] NO-LEAKAGE TRAINING COMPLETED")
        print("=" * 80)
        print(f"  Inputs: {len(self.feature_names)} raw features")
        print("  Outputs: B and D only")
        print("  Removed leakage: factor_of_safety, csr, crr/cyclic_strength_ratio")
        print("  Removed output: L/B ratio")
        print("  Data split: none, 100% cleaned data used for training")
        return True


def main():
    if not SKLEARN_AVAILABLE:
        sys.exit(1)

    trainer = MultiOutputANNTraining()
    success = trainer.run()
    if not success:
        sys.exit(1)

    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()
