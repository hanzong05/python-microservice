#!/usr/bin/env python3
"""
Multi-Output ANN Model Training (v4 — improved targets + proper validation)

Key changes from v3:
- B fallback: physics-based tiered target (varies with N & fines), not constant 2.0
- D fallback: continuous GWL-based formula + N-strength adjustment, not binary 1.5/3.0
- 80/20 train/test split for honest evaluation metrics
- Larger architecture (64, 32) instead of (30, 15)
- Early stopping on validation loss during training
- Reports both training and test-set metrics

Predicts: Foundation Width B (m), Foundation Depth D (m)
Inputs:   raw soil properties only (no leakage features)
"""

import io
import json
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
    from sklearn.model_selection import train_test_split
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

    Inputs:  raw / geotechnical properties only.
    Outputs: Foundation Width B (m), Foundation Depth D (m).
    Targets: physics-based fallback when DB columns are unpopulated.
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

        self.X_train = None
        self.X_test = None
        self.y_B_train = None
        self.y_B_test = None
        self.y_D_train = None
        self.y_D_test = None

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
            # Query soil_layers directly — the view (v_complete_soil_data) omits
            # foundation_width_m, foundation_depth_m, spt_n160, and
            # liquefaction_risk_level which are all needed for target computation.
            print("  Fetching from soil_layers table (includes foundation targets)...")
            page_size = 1000
            all_rows = []
            offset = 0
            while True:
                result = (
                    self.client.table("soil_layers")
                    .select("*")
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                batch = result.data or []
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size

            if not all_rows:
                print("  [ERROR] No data returned from soil_layers")
                return False

            self.df = pd.DataFrame(all_rows)
            self.df_raw = self.df.copy()
            print(f"  [OK] Retrieved {len(self.df)} records")
            print(f"  Columns: {len(self.df.columns)}")

            # Report how many rows already have stored foundation dimensions
            populated_B = self.df["foundation_width_m"].notna().sum() if "foundation_width_m" in self.df.columns else 0
            populated_D = self.df["foundation_depth_m"].notna().sum() if "foundation_depth_m" in self.df.columns else 0
            total = len(self.df)
            print(f"  foundation_width_m : {populated_B}/{total} populated "
                  f"({'will use fallback' if populated_B == 0 else 'OK'})")
            print(f"  foundation_depth_m : {populated_D}/{total} populated "
                  f"({'will use fallback' if populated_D == 0 else 'OK'})")
            if populated_B == 0:
                print("  [TIP] Run populate_foundation_dims.py first for better targets")
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
            "factor_of_safety",
        ]
        for col in numeric_candidates:
            if col in df.columns:
                df[col] = self._num(df[col])

        if "fines_content" in df.columns:
            df = df[(df["fines_content"] >= 0) & (df["fines_content"] <= 100)]

        if "unit_weight" in df.columns:
            df = df[(df["unit_weight"] >= 10) & (df["unit_weight"] <= 25)]

        spt_cols = [c for c in ["spt_n_value", "spt_n60", "spt_n160"] if c in df.columns]
        if spt_cols:
            spt_ok = pd.Series(False, index=df.index)
            for col in spt_cols:
                spt_ok = spt_ok | (df[col].notna() & (df[col] > 0))
            df = df[spt_ok]

        for col in ["depth_mid_m", "groundwater_depth_m"]:
            if col in df.columns:
                df = df[df[col].notna()]

        after = len(df)
        print(f"  Removed rows: {before - after}")
        print(f"  Remaining rows: {after}")
        return df

    @staticmethod
    def _back_calc_B_meyerhof(n: float, q_actual: float, d: float) -> float:
        """
        Find foundation width B such that Meyerhof (1956) SPT formula gives
        qa ≈ q_actual (settlement-limited criterion, SI_allow = 25 mm).

        For B > 1.2 m: qa = 8 * N * ((B+0.3)/B)^2 * Kd
        where Kd = 1 + 0.33*(D/B)

        When the soil is strong enough that qa > q_actual for all B, returns
        a size proportional to 1/sqrt(N) (stronger soil → smaller footing).
        """
        B_MAX = 6.0

        # Check B=1.2 m (transition point) first
        for B_try in np.arange(1.0, B_MAX + 0.5, 0.5):
            Kd = 1.0 + 0.33 * (d / B_try)
            if B_try <= 1.2:
                qa = 12.0 * n * Kd
            else:
                size_factor = ((B_try + 0.3) / B_try) ** 2
                qa = 8.0 * n * size_factor * Kd
            if qa >= q_actual:
                return B_try  # smallest B that satisfies the load

        # Soil too weak even at B_MAX — return max
        return B_MAX

    def compute_foundation_targets(self, row) -> dict:
        """
        B and D target computation.

        Priority:
        1. Use real DB columns (foundation_width_m, foundation_depth_m) when present.
        2. Physics-based fallback that varies with N, fines content, and GWL,
           so the ANN can learn meaningful soil–foundation relationships.

        NOTE: factor_of_safety is used only for TARGET computation here (not as a
        model input), so there is no leakage — the ANN inputs remain raw features.
        """
        q_actual = 150.0  # kPa — representative single/double-storey building load

        # --- SPT N value (best available) ---
        n = row.get("spt_n160")
        if pd.isna(n):
            n = row.get("spt_n60")
        if pd.isna(n):
            n = row.get("spt_n_value")
        try:
            n = max(1.0, float(n))
        except Exception:
            n = 15.0

        fc    = float(row.get("fines_content") or 15.0)

        gwl = row.get("groundwater_depth_m")
        try:
            gwl = float(gwl)
        except Exception:
            gwl = 5.0

        depth_mid = row.get("depth_mid_m")
        try:
            depth_mid = float(depth_mid)
        except Exception:
            depth_mid = 1.5

        # --- Foundation width target ---
        b_raw = row.get("foundation_width_m")
        if b_raw is None or pd.isna(b_raw):
            b_raw = row.get("Foundation Width (B)")
        try:
            b = float(b_raw)
            if b <= 0:
                raise ValueError
        except Exception:
            # Physics-based B: find minimum B where Meyerhof SPT qa >= q_actual.
            # Smaller B → higher qa (settlement criterion). Stronger soil allows
            # smaller footings; weaker soil needs larger footings.
            b = self._back_calc_B_meyerhof(n, q_actual, d=1.5)

            # Fine-grained soils are more compressible → increase B slightly
            if fc >= 35.0:
                b = min(b + 0.5, 6.0)
            elif fc >= 15.0:
                b = min(b + 0.25, 6.0)

            b = round(b * 2) / 2   # round to nearest 0.5 m

        # --- Foundation depth target ---
        d_raw = row.get("foundation_depth_m")
        if d_raw is None or pd.isna(d_raw):
            d_raw = row.get("Foundation Depth (D)")
        try:
            d = float(d_raw)
            if d <= 0:
                raise ValueError
        except Exception:
            # Use stored liquefaction_risk_level (if available) to set D.
            # risk_level IS stored in soil_layers and reflects the per-layer FS.
            # Using it for TARGET computation is not leakage — it does not appear
            # as a model input (raw soil features only are used as inputs).
            risk = str(row.get("liquefaction_risk_level") or "").strip().upper()
            _RISK_ORD = {"VERY HIGH": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "VERY LOW": 1}
            risk_score = _RISK_ORD.get(risk, 0)

            if risk_score >= 4:
                # HIGH or VERY HIGH → liquefiable layer; go deeper for stable bearing
                d = 3.0
            elif risk_score == 3:
                d = 2.5
            elif n < 5.0:
                # Very weak soil — deeper bearing
                d = 3.0
            elif n < 10.0:
                d = 2.0
            elif gwl <= 1.5:
                # Very shallow GWL — keep foundation above water if possible
                d = 1.5
            elif gwl <= 3.0:
                # Place 0.3 m above GWL, clamped to [1.5, 2.5]
                d = max(1.5, min(gwl - 0.3, 2.5))
            else:
                d = 1.5  # normal case: GWL not a constraint

        return {"B": round(float(b), 2), "D": round(float(d), 2)}

    def prepare_features_and_targets(self) -> bool:
        print("\n" + "=" * 80)
        print("PREPARING FEATURES AND TARGETS (80/20 SPLIT)")
        print("=" * 80)

        df = self.clean_data_strictly(self.df)

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
            "pga_g",
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
        X  = X[valid_mask]
        y_B = y_B[valid_mask]
        y_D = y_D[valid_mask]

        if len(X) < 30:
            print("  [ERROR] Too few valid rows after cleaning")
            return False

        self.feature_names = feature_cols

        print(f"\n  Features shape: {X.shape}")
        print(f"  Target B: Mean={y_B.mean():.2f} m, Std={y_B.std():.2f} m, "
              f"Range=[{y_B.min():.2f}, {y_B.max():.2f}] m")
        print(f"  Target D: Mean={y_D.mean():.2f} m, Std={y_D.std():.2f} m, "
              f"Range=[{y_D.min():.2f}, {y_D.max():.2f}] m")

        # 80/20 split — stratify is impractical for continuous targets,
        # so use random_state only for reproducibility.
        test_size = 0.20
        (self.X_train, self.X_test,
         self.y_B_train, self.y_B_test,
         self.y_D_train, self.y_D_test) = train_test_split(
            X, y_B, y_D,
            test_size=test_size,
            random_state=42,
        )

        print(f"\n  Training set: {len(self.X_train)} samples ({100*(1-test_size):.0f}%)")
        print(f"  Test set    : {len(self.X_test)} samples ({100*test_size:.0f}%)")
        return True

    def train_models(self) -> bool:
        if not SKLEARN_AVAILABLE:
            return False

        print("\n" + "=" * 80)
        print("TRAINING ANN MODEL (v4)")
        print("=" * 80)
        print(f"  INPUT LAYER : {self.X_train.shape[1]} neurons")
        print("  HIDDEN 1    : 64 neurons, relu")
        print("  HIDDEN 2    : 32 neurons, relu")
        print("  OUTPUT LAYER: 2 neurons, linear [B, D]")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)

        y_train_stacked = np.column_stack([self.y_B_train, self.y_D_train])

        mlp_kwargs = dict(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=3000,
            early_stopping=True,        # stop when val loss stops improving
            validation_fraction=0.15,   # internal hold-out within training set
            n_iter_no_change=30,
            random_state=42,
            verbose=False,
        )

        # Multi-output model (B + D jointly)
        self.multi_model = MLPRegressor(**mlp_kwargs)
        self.multi_model.fit(X_train_scaled, y_train_stacked)
        print(f"  [OK] Multi-output model: {self.multi_model.n_iter_} iterations "
              f"(best val loss: {self.multi_model.best_validation_score_:.6f})")

        # Individual models (sometimes better per-output accuracy)
        self.width_model = MLPRegressor(**mlp_kwargs)
        self.width_model.fit(X_train_scaled, self.y_B_train)
        print(f"  [OK] Width model (B): {self.width_model.n_iter_} iterations")

        self.depth_model = MLPRegressor(**mlp_kwargs)
        self.depth_model.fit(X_train_scaled, self.y_D_train)
        print(f"  [OK] Depth model (D): {self.depth_model.n_iter_} iterations")

        return True

    def evaluate_models(self) -> bool:
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)

        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled  = self.scaler.transform(self.X_test)

        def reg_metrics(y_true, y_pred):
            r2   = r2_score(y_true, y_pred)
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            return r2, mae, rmse

        def print_split_metrics(name, y_tr, y_tr_pred, y_te, y_te_pred, unit=""):
            r2_tr, mae_tr, rmse_tr = reg_metrics(y_tr, y_tr_pred)
            r2_te, mae_te, rmse_te = reg_metrics(y_te, y_te_pred)
            print(f"\n  {name}:")
            print(f"    {'Metric':<8} {'Train':>10} {'Test':>10}")
            print(f"    {'R²':<8} {r2_tr:>10.4f} {r2_te:>10.4f}")
            print(f"    {'MAE'+unit:<8} {mae_tr:>10.4f} {mae_te:>10.4f}")
            print(f"    {'RMSE'+unit:<8} {rmse_tr:>10.4f} {rmse_te:>10.4f}")
            if r2_te < 0.30:
                print(f"    [WARN] Low test R² — model may need more/better training data")

        # Multi-output model
        y_pred_tr = self.multi_model.predict(X_train_scaled)
        y_pred_te = self.multi_model.predict(X_test_scaled)
        print_split_metrics(
            "Multi-output — Foundation Width B",
            self.y_B_train, y_pred_tr[:, 0],
            self.y_B_test,  y_pred_te[:, 0],
            unit=" m",
        )
        print_split_metrics(
            "Multi-output — Foundation Depth D",
            self.y_D_train, y_pred_tr[:, 1],
            self.y_D_test,  y_pred_te[:, 1],
            unit=" m",
        )

        # Individual models
        yB_pred_tr = self.width_model.predict(X_train_scaled)
        yB_pred_te = self.width_model.predict(X_test_scaled)
        print_split_metrics(
            "Individual    — Foundation Width B",
            self.y_B_train, yB_pred_tr,
            self.y_B_test,  yB_pred_te,
            unit=" m",
        )

        yD_pred_tr = self.depth_model.predict(X_train_scaled)
        yD_pred_te = self.depth_model.predict(X_test_scaled)
        print_split_metrics(
            "Individual    — Foundation Depth D",
            self.y_D_train, yD_pred_tr,
            self.y_D_test,  yD_pred_te,
            unit=" m",
        )

        # Target distribution summary (helps diagnose constant-target issues)
        print("\n  Target distribution (full dataset):")
        all_B = pd.concat([self.y_B_train, self.y_B_test])
        all_D = pd.concat([self.y_D_train, self.y_D_test])
        print(f"    B unique values: {sorted(all_B.unique())}")
        print(f"    D unique values: {sorted(all_D.unique())}")

        return True

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

            upload_joblib(self.scaler,      "models/scaler_no_leakage.pkl")
            upload_joblib(self.multi_model, "models/ann_multi_output_BD_no_leakage.pkl")
            upload_joblib(self.width_model, "models/ann_foundation_width_no_leakage.pkl")
            upload_joblib(self.depth_model, "models/ann_foundation_depth_no_leakage.pkl")

            metadata = {
                "version": "4.0-improved-targets",
                "model_type": "multi_output_regression",
                "targets": ["foundation_width_B", "foundation_depth_D"],
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
                    "hidden_layers": [64, 32],
                    "hidden_activation": "relu",
                    "output_layer": 2,
                    "output_activation": "linear",
                    "solver": "adam",
                    "early_stopping": True,
                },
                "training_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "data_split": "80/20 train/test",
                "target_generation": (
                    "B: Meyerhof (1956) SPT back-calculation + fines correction. "
                    "D: GWL-based + N-strength + stored FS when available."
                ),
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
        print("ANN MODEL TRAINING — v4 (improved targets + 80/20 split)")
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

        if not self.save_models():
            return False

        print("\n" + "=" * 80)
        print("[SUCCESS] TRAINING COMPLETED")
        print("=" * 80)
        print(f"  Features      : {len(self.feature_names)} raw soil properties")
        print(f"  Architecture  : {self.X_train.shape[1]} → 64 → 32 → 2 (relu / linear)")
        print(f"  Training rows : {len(self.X_train)}")
        print(f"  Test rows     : {len(self.X_test)}")
        print(f"  Outputs       : Foundation Width B, Foundation Depth D")
        print(f"  B target      : Meyerhof SPT back-calculation (varies with N & fines)")
        print(f"  D target      : GWL + N-strength + FS when available")
        return True


def main():
    if not SKLEARN_AVAILABLE:
        sys.exit(1)

    trainer = MultiOutputANNTraining()
    success = trainer.run()
    if not success:
        sys.exit(1)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
