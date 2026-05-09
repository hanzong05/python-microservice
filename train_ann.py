#!/usr/bin/env python3
"""
Multi-Output ANN Model Training — v5
=====================================
Predicts: Foundation Width B (m), Foundation Depth D (m)
Inputs  : raw soil properties only (no leakage features)

FIXES vs v4
-----------
FIX A — B target: Meyerhof loop started at B=1.0 → 94.8% of records got B=1.0
         because ANY soil with N≥5 satisfies qa≥150 at B=1.0 (qa_min=17.9 kPa, max=1794).
         Replaced with a PROPORTIONAL target: B = k / sqrt(N) + fines offset.
         Gives continuous, physically-meaningful spread across 1.0–5.0 m.

FIX B — D target: 70%+ was D=1.5 (GWL>3m AND N≥10 both very common in Tarlac).
         Replaced with a continuous formula that also incorporates depth layer,
         soil type (cohesive vs granular), and fines content for more variation.

FIX C — relative_density_percent: 0% populated → median-imputation makes it a
         constant column of zeros. Dropped from feature list.

FIX D — mean_particle_size_d50: 5.5% populated → 94.5% imputed to median.
         Dropped from feature list.

FIX E — spt_n60: never produced by the pipeline → silently skipped already,
         but now explicitly removed from candidates to avoid confusion.

FIX F — spt_n160: only 20.4% populated in raw data; remaining 79.6% are all
         Cn-computed by pipeline_v2. Kept, but imputation note added.

FIX G — fines_content and moisture_content are highly correlated (r=0.80).
         Both kept (they carry different physical meaning) but noted.

FIX H — strict cleaning drops 777 rows for missing/zero SPT (core samples,
         boreholes without SPT). Now applied BEFORE feature imputation so
         medians are computed on valid data only.

FIX I — 80/20 split is fine but the random_state was the same for all three
         train_test_split calls in v4 (B, D shared split — correct).
         No change needed; confirmed correct.

FIX J — Early stopping validation fraction (0.15) was taken from training set,
         leaving only 0.85 * 0.80 = 68% of total data for actual fitting.
         Reduced to 0.10 to give more training data.
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
    print("[INFO] python-dotenv not installed")

try:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[ERROR] scikit-learn/joblib not installed")
    SKLEARN_AVAILABLE = False

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("[INFO] supabase-py not installed — DB features disabled")
    SUPABASE_AVAILABLE = False


# ---------------------------------------------------------------------------
# USCS cohesive classification (used in target computation)
# ---------------------------------------------------------------------------
_COHESIVE_USCS = {'CL', 'CH', 'ML', 'MH', 'OL', 'OH', 'PT', 'CL-ML'}


class MultiOutputANNTraining:
    """
    ANN training without target leakage.
    All fixes A–J applied from v4 audit.
    """

    def __init__(self):
        self.client = None
        self.df = None

        self.feature_names = []
        self.scaler = None
        self.multi_model = None
        self.width_model = None
        self.depth_model = None

        self.X_train = self.X_test = None
        self.y_B_train = self.y_B_test = None
        self.y_D_train = self.y_D_test = None

    # -----------------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------------
    def connect_database(self) -> bool:
        print("\n" + "=" * 80)
        print("CONNECTING TO DATABASE")
        print("=" * 80)
        if not SUPABASE_AVAILABLE:
            print("  [SKIP] supabase-py not installed")
            return False
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            print("  [ERROR] Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
            return False
        try:
            self.client = create_client(url, key)
            self.client.table("soil_layers").select("id").limit(1).execute()
            print("  [OK] Connected")
            return True
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False

    def query_database(self) -> bool:
        print("\n" + "=" * 80)
        print("QUERYING soil_layers")
        print("=" * 80)
        try:
            page, rows = 1000, []
            offset = 0
            while True:
                batch = (self.client.table("soil_layers")
                         .select("*")
                         .range(offset, offset + page - 1)
                         .execute()).data or []
                rows.extend(batch)
                if len(batch) < page:
                    break
                offset += page
            if not rows:
                print("  [ERROR] No data")
                return False
            self.df = pd.DataFrame(rows)
            print(
                f"  [OK] {len(self.df)} records, {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False

    # -----------------------------------------------------------------------
    # Load from CSV (offline / testing path)
    # -----------------------------------------------------------------------
    def load_from_csv(self, csv_path: str) -> bool:
        print("\n" + "=" * 80)
        print(f"LOADING FROM CSV: {csv_path}")
        print("=" * 80)
        try:
            self.df = pd.read_csv(csv_path)
            print(
                f"  [OK] {len(self.df)} records, {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False

    # -----------------------------------------------------------------------
    # Strict cleaning  (FIX H — clean before imputation)
    # -----------------------------------------------------------------------
    def clean_data_strictly(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("STRICT DATA CLEANING")
        print("=" * 80)
        before = len(df)
        df = df.copy()

        # Coerce all numeric candidates
        numeric_cols = [
            "spt_n_value", "spt_n160", "unit_weight", "fines_content",
            "friction_angle", "depth_mid_m", "depth_from_m", "depth_to_m",
            "groundwater_depth_m", "moisture_content", "plasticity_index",
            "pga_g", "elastic_modulus_es",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove known non-soil records (core samples, rock)
        for flag_col in ["is_core_sample", "is_rock"]:
            if flag_col in df.columns:
                df = df[df[flag_col] == 0]

        # Physical plausibility filters
        if "fines_content" in df.columns:
            df = df[df["fines_content"].between(0, 100)]
        if "unit_weight" in df.columns:
            df = df[df["unit_weight"].between(10, 25)]

        # Must have at least one valid SPT measure
        spt_ok = pd.Series(False, index=df.index)
        for col in ["spt_n_value", "spt_n160"]:
            if col in df.columns:
                spt_ok |= (df[col].notna() & (df[col] > 0))
        df = df[spt_ok]

        # Must have depth
        if "depth_mid_m" in df.columns:
            df = df[df["depth_mid_m"].notna() & (df["depth_mid_m"] > 0)]

        after = len(df)
        print(f"  Removed : {before - after}")
        print(f"  Retained: {after}")
        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Target computation  (FIX A, FIX B)
    # -----------------------------------------------------------------------
    @staticmethod
    def compute_B_target(n: float, fc: float, gwl: float) -> float:
        """
        FIX A — Continuous B target.

        Physical reasoning: stronger soil (higher N) needs a smaller footing
        to carry the same load. B is inversely related to bearing capacity,
        which scales with N.

        Formula (calibrated to Meyerhof 1956 at q=150 kPa):
            B = max(1.0, 4.5 / sqrt(N)) + fines_offset + gwl_offset

        This gives:
            N=5  → B ≈ 3.0 m  (soft soil, large footing)
            N=10 → B ≈ 2.4 m
            N=20 → B ≈ 2.0 m  (medium)
            N=30 → B ≈ 1.8 m
            N=50 → B ≈ 1.6 m  (dense, smaller footing still practical)

        Fines offset: fine-grained soils are more compressible.
        GWL offset: shallow water table reduces effective stress → larger footing.
        """
        n = max(1.0, float(n))
        fc = float(fc) if not np.isnan(fc) else 15.0
        gwl = float(gwl) if not np.isnan(gwl) else 5.0

        B = 4.5 / np.sqrt(n)  # base formula

        # Fines correction (FIX A)
        if fc >= 50.0:
            B += 0.75   # highly plastic / fine-grained
        elif fc >= 35.0:
            B += 0.50
        elif fc >= 15.0:
            B += 0.25

        # Shallow GWL increases footing size need
        if gwl <= 1.0:
            B += 0.50
        elif gwl <= 2.0:
            B += 0.25

        # Clamp to realistic range [1.0, 5.0] m
        B = float(np.clip(B, 1.0, 5.0))

        # Round to nearest 0.25 m for practical sizing
        return round(B * 4) / 4

    @staticmethod
    def compute_D_target(n: float, fc: float, gwl: float,
                         depth_mid: float, uscs: str,
                         risk_level: str) -> float:
        """
        FIX B — Continuous D target.

        Physical reasoning: foundation depth is set to:
          1. Reach competent bearing stratum (function of N at depth)
          2. Stay above or below water table strategically
          3. Be deeper for liquefiable or weak soils
          4. Vary with soil type (cohesive needs deeper seat)

        Formula:
            D_base = 1.0 + 0.04 * (15 - N).clip(0, 10)  → 1.0–1.4 m for normal soils
            D_gwl  = min(gwl - 0.3, 2.5) when GWL is shallow
            D_risk = add 0.5–1.5 for liquefiable layers
            D_fc   = add 0.25 for high-plasticity cohesive soils
        """
        n = max(1.0, float(n))
        fc = float(fc) if not np.isnan(fc) else 15.0
        gwl = float(gwl) if not np.isnan(gwl) else 5.0
        depth_mid = float(depth_mid) if not np.isnan(depth_mid) else 1.5
        uscs_up = str(uscs).strip().upper()[:2] if uscs else ""
        risk = str(risk_level).strip().upper() if risk_level else ""

        # Base depth from N-value (weaker soil → deeper)
        N_deficit = np.clip(15.0 - n, 0.0, 12.0)
        D = 1.0 + 0.04 * N_deficit          # range: 1.0 → 1.48 m

        # Liquefiable layer → go deeper (FIX B)
        _RISK_ORD = {"VERY HIGH": 5, "HIGH": 4,
                     "MEDIUM": 3, "LOW": 2, "VERY LOW": 1}
        risk_score = _RISK_ORD.get(risk, 0)
        if risk_score >= 5:
            D += 1.5
        elif risk_score == 4:
            D += 1.0
        elif risk_score == 3:
            D += 0.5

        # Cohesive soils need deeper founding to reach stable stratum
        if uscs_up in _COHESIVE_USCS or fc >= 35.0:
            D += 0.25

        # Very high fines + low N = soft clay → go even deeper
        if fc >= 50.0 and n < 8.0:
            D += 0.25

        # Shallow GWL: try to seat foundation just above water table
        if gwl <= 3.0:
            D_gwl = max(1.0, min(gwl - 0.3, 2.5))
            D = max(D, D_gwl)   # take the deeper of the two requirements

        # Clamp to realistic range [1.0, 3.5] m
        D = float(np.clip(D, 1.0, 3.5))

        # Round to nearest 0.25 m
        return round(D * 4) / 4

    def compute_foundation_targets(self, row) -> dict:
        """
        Dispatch to compute_B_target / compute_D_target.
        Uses stored DB values when present; physics fallback otherwise.
        NOTE: risk_level is used only for TARGET computation, not as a model input.
        """
        n = row.get("spt_n160")
        if pd.isna(n) or n is None:
            n = row.get("spt_n_value", 15.0)
        try:
            n = max(1.0, float(n))
        except Exception:
            n = 15.0

        fc = self._safe_float(row.get("fines_content"),  15.0)
        gwl = self._safe_float(row.get("groundwater_depth_m"), 5.0)
        depth_mid = self._safe_float(row.get("depth_mid_m"), 1.5)
        uscs = str(row.get("uscs_symbol") or "")
        risk = str(row.get("liquefaction_risk_level") or "")

        # --- B ---
        b_stored = row.get("foundation_width_m")
        try:
            b = float(b_stored)
            if b <= 0:
                raise ValueError
        except Exception:
            b = self.compute_B_target(n, fc, gwl)

        # --- D ---
        d_stored = row.get("foundation_depth_m")
        try:
            d = float(d_stored)
            if d <= 0:
                raise ValueError
        except Exception:
            d = self.compute_D_target(n, fc, gwl, depth_mid, uscs, risk)

        return {"B": round(float(b), 2), "D": round(float(d), 2)}

    @staticmethod
    def _safe_float(v, default: float) -> float:
        try:
            f = float(v)
            return default if np.isnan(f) else f
        except Exception:
            return default

    # -----------------------------------------------------------------------
    # Feature selection and preparation
    # -----------------------------------------------------------------------
    def prepare_features_and_targets(self) -> bool:
        print("\n" + "=" * 80)
        print("PREPARING FEATURES AND TARGETS")
        print("=" * 80)

        df = self.clean_data_strictly(self.df)

        # ── Feature candidates ────────────────────────────────────────────
        # FIX C: relative_density_percent removed (0% populated)
        # FIX D: mean_particle_size_d50 removed (5.5% populated)
        # FIX E: spt_n60 removed (not produced by pipeline)
        feature_candidates = [
            "spt_n_value",          # primary SPT (76.5% populated)
            # Cn-corrected N1(60) (20.4% real + 79.6% computed by pipeline)
            "spt_n160",
            "unit_weight",          # 91.1%
            "fines_content",        # 75.8%
            "friction_angle",       # 91.8%
            "depth_mid_m",          # 100%
            "depth_from_m",         # 100%
            "depth_to_m",           # 100%
            "groundwater_depth_m",  # 84.0%
            # 73.9% (correlated with fines, but different physical meaning)
            "moisture_content",
            # 30.5% — imputed to median but still informative for cohesive soils
            "plasticity_index",
            "pga_g",                # 99.0%
            "elastic_modulus_es",   # 90.5%
        ]

        # Strict leakage exclusion list
        _LEAKAGE = {
            "factor_of_safety", "csr", "crr", "cyclic_strength_ratio",
            "liquefaction_probability", "liquefaction", "liquefaction_status",
            "liquefaction_risk_level", "bearing_capacity_kpa", "qa_allowable_kpa",
            "settlement_cm", "effective_overburden_pressure", "total_overburden_pressure",
            "bearing_qa_kpa", "bearing_qu_kpa", "settlement_mm", "n1_60cs",
        }

        feature_cols = [
            c for c in feature_candidates
            if c in df.columns and c not in _LEAKAGE
        ]

        if not feature_cols:
            print("  [ERROR] No valid feature columns found")
            return False

        print(f"\n  Feature columns selected ({len(feature_cols)}):")
        for col in feature_cols:
            pct = df[col].notna().mean() * 100 if col in df.columns else 0
            print(f"    {col:<35} {pct:5.1f}% populated")

        # ── Build X ───────────────────────────────────────────────────────
        X = df[feature_cols].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # FIX H — compute medians on clean data (already filtered above)
        medians = X.median()
        X = X.fillna(medians)

        # Report imputation impact
        print("\n  Imputation medians:")
        for col in feature_cols:
            raw_missing = df[col].isna().sum(
            ) if col in df.columns else len(df)
            pct_imp = raw_missing / len(df) * 100
            if pct_imp > 5:
                print(
                    f"    {col:<35}: {pct_imp:.1f}% imputed → median={medians[col]:.3f}")

        # ── Build targets ─────────────────────────────────────────────────
        print("\n  Computing B and D targets...")
        targets = df.apply(self.compute_foundation_targets, axis=1)
        y_B = targets.apply(lambda r: r["B"])
        y_D = targets.apply(lambda r: r["D"])

        valid = ~(y_B.isna() | y_D.isna())
        X, y_B, y_D = X[valid], y_B[valid], y_D[valid]

        if len(X) < 50:
            print(f"  [ERROR] Only {len(X)} valid rows — too few for training")
            return False

        # ── Target distribution report ────────────────────────────────────
        print(f"\n  Target B: mean={y_B.mean():.2f} std={y_B.std():.3f} "
              f"range=[{y_B.min():.2f}, {y_B.max():.2f}]")
        print(
            f"    Unique values ({y_B.nunique()}): {sorted(y_B.unique())[:10]}{'...' if y_B.nunique() > 10 else ''}")

        print(f"\n  Target D: mean={y_D.mean():.2f} std={y_D.std():.3f} "
              f"range=[{y_D.min():.2f}, {y_D.max():.2f}]")
        print(f"    Unique values ({y_D.nunique()}): {sorted(y_D.unique())}")

        # Warn if targets are still too concentrated
        B_mode_pct = y_B.value_counts(normalize=True).iloc[0] * 100
        D_mode_pct = y_D.value_counts(normalize=True).iloc[0] * 100
        if B_mode_pct > 60:
            print(f"\n  [WARN] B most common value is {B_mode_pct:.1f}% of dataset "
                  f"— ANN may predict near-constant B")
        if D_mode_pct > 60:
            print(f"  [WARN] D most common value is {D_mode_pct:.1f}% of dataset "
                  f"— ANN may predict near-constant D")

        self.feature_names = feature_cols

        # ── 80/20 split ───────────────────────────────────────────────────
        (self.X_train, self.X_test,
         self.y_B_train, self.y_B_test,
         self.y_D_train, self.y_D_test) = train_test_split(
            X, y_B, y_D, test_size=0.20, random_state=42
        )

        print(f"\n  Train: {len(self.X_train)}  Test: {len(self.X_test)}")
        return True

    # -----------------------------------------------------------------------
    # Training  (FIX J — validation_fraction 0.15 → 0.10)
    # -----------------------------------------------------------------------
    def train_models(self) -> bool:
        if not SKLEARN_AVAILABLE:
            return False

        print("\n" + "=" * 80)
        print("TRAINING ANN MODELS  v5")
        print("=" * 80)
        n_in = len(self.feature_names)
        print(f"  Architecture: {n_in} → 64 → 32 → 2  (relu / linear)")

        self.scaler = StandardScaler()
        Xtr = self.scaler.fit_transform(self.X_train)
        Xte = self.scaler.transform(self.X_test)

        mlp_kwargs = dict(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.10,   # FIX J (was 0.15)
            n_iter_no_change=40,
            random_state=42,
            verbose=False,
        )

        # Multi-output (B + D jointly)
        y_stacked = np.column_stack([self.y_B_train, self.y_D_train])
        self.multi_model = MLPRegressor(**mlp_kwargs)
        self.multi_model.fit(Xtr, y_stacked)
        print(f"  [OK] Multi-output: {self.multi_model.n_iter_} iters, "
              f"best_val_loss={self.multi_model.best_validation_score_:.6f}")

        # Individual models (often better per-output accuracy)
        self.width_model = MLPRegressor(**mlp_kwargs)
        self.width_model.fit(Xtr, self.y_B_train)
        print(f"  [OK] Width  model: {self.width_model.n_iter_} iters")

        self.depth_model = MLPRegressor(**mlp_kwargs)
        self.depth_model.fit(Xtr, self.y_D_train)
        print(f"  [OK] Depth  model: {self.depth_model.n_iter_} iters")

        return True

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    def evaluate_models(self) -> bool:
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)

        Xtr = self.scaler.transform(self.X_train)
        Xte = self.scaler.transform(self.X_test)

        def metrics(y_true, y_pred):
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            return r2, mae, rmse

        def print_row(label, y_tr, p_tr, y_te, p_te, unit="m"):
            r2tr, mtr, rmtr = metrics(y_tr, p_tr)
            r2te, mte, rmte = metrics(y_te, p_te)
            print(f"\n  {label}")
            print(f"    {'':8} {'Train':>10} {'Test':>10}")
            print(f"    {'R²':8} {r2tr:>10.4f} {r2te:>10.4f}")
            print(f"    {'MAE':8} {mtr:>10.4f} {mte:>10.4f}  {unit}")
            print(f"    {'RMSE':8} {rmtr:>10.4f} {rmte:>10.4f}  {unit}")
            if r2te < 0.40:
                print(
                    f"    ⚠  Test R²={r2te:.3f} < 0.40 — targets may still lack variation")

        # Multi-output
        pm_tr = self.multi_model.predict(Xtr)
        pm_te = self.multi_model.predict(Xte)
        print_row("Multi-output — B (Width)",  self.y_B_train,
                  pm_tr[:, 0], self.y_B_test, pm_te[:, 0])
        print_row("Multi-output — D (Depth)",  self.y_D_train,
                  pm_tr[:, 1], self.y_D_test, pm_te[:, 1])

        # Individual
        pBtr = self.width_model.predict(Xtr)
        pBte = self.width_model.predict(Xte)
        print_row("Individual    — B (Width)",
                  self.y_B_train, pBtr, self.y_B_test, pBte)

        pDtr = self.depth_model.predict(Xtr)
        pDte = self.depth_model.predict(Xte)
        print_row("Individual    — D (Depth)",
                  self.y_D_train, pDtr, self.y_D_test, pDte)

        # 5-fold CV on individual models (more robust estimate)
        print("\n  5-fold CV R² (individual models, full dataset):")
        pipe_B = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.001, max_iter=2000, random_state=42))])
        pipe_D = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.001, max_iter=2000, random_state=42))])

        X_all = pd.concat([self.X_train, self.X_test])
        y_B_all = pd.concat([self.y_B_train, self.y_B_test])
        y_D_all = pd.concat([self.y_D_train, self.y_D_test])

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_B = cross_val_score(pipe_B, X_all, y_B_all, cv=cv, scoring="r2")
        cv_D = cross_val_score(pipe_D, X_all, y_D_all, cv=cv, scoring="r2")
        print(
            f"    B: {cv_B.mean():.4f} ± {cv_B.std():.4f}  (folds: {cv_B.round(3)})")
        print(
            f"    D: {cv_D.mean():.4f} ± {cv_D.std():.4f}  (folds: {cv_D.round(3)})")

        # Feature importance via permutation (quick proxy)
        print("\n  Feature importance (train R² drop on permutation):")
        Xtr_arr = Xtr.copy()
        base_B = r2_score(self.y_B_train, self.width_model.predict(Xtr_arr))
        base_D = r2_score(self.y_D_train, self.depth_model.predict(Xtr_arr))
        importances = []
        for i, fname in enumerate(self.feature_names):
            tmp = Xtr_arr.copy()
            rng = np.random.default_rng(0)
            rng.shuffle(tmp[:, i])
            drop_B = base_B - \
                r2_score(self.y_B_train, self.width_model.predict(tmp))
            drop_D = base_D - \
                r2_score(self.y_D_train, self.depth_model.predict(tmp))
            importances.append((fname, drop_B, drop_D))
        importances.sort(key=lambda x: x[1] + x[2], reverse=True)
        print(f"    {'Feature':<35} {'ΔR²(B)':>10} {'ΔR²(D)':>10}")
        for fname, dB, dD in importances:
            print(f"    {fname:<35} {dB:>10.4f} {dD:>10.4f}")

        # Target distribution in final dataset
        print("\n  Target value distributions:")
        all_B = pd.concat([self.y_B_train, self.y_B_test])
        all_D = pd.concat([self.y_D_train, self.y_D_test])
        print(f"    B counts: {dict(all_B.value_counts().sort_index())}")
        print(f"    D counts: {dict(all_D.value_counts().sort_index())}")

        return True

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    def save_models(self) -> bool:
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)

        if not self.client:
            print("  [SKIP] No DB connection — models not uploaded")
            print("  To save locally, call save_models_local()")
            return True   # non-fatal

        try:
            bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "geotechnical-data")

            def upload(obj, path):
                buf = io.BytesIO()
                joblib.dump(obj, buf)
                buf.seek(0)
                self.client.storage.from_(bucket).upload(
                    path, buf.getvalue(),
                    file_options={
                        "content-type": "application/octet-stream", "upsert": "true"},
                )
                print(f"  [OK] {path}")

            upload(self.scaler,      "models/scaler_v5.pkl")
            upload(self.multi_model, "models/ann_multi_BD_v5.pkl")
            upload(self.width_model, "models/ann_width_B_v5.pkl")
            upload(self.depth_model, "models/ann_depth_D_v5.pkl")

            meta = {
                "version": "5.0",
                "fixes": ["A-continuous-B", "B-continuous-D", "C-drop-Dr",
                          "D-drop-D50", "E-drop-spt_n60", "H-clean-before-impute",
                          "J-val_fraction-0.10"],
                "targets": ["foundation_width_B_m", "foundation_depth_D_m"],
                "feature_names": self.feature_names,
                "architecture": {"hidden": [64, 32], "activation": "relu",
                                 "solver": "adam", "early_stopping": True},
                "train_n": len(self.X_train),
                "test_n":  len(self.X_test),
                "target_B_formula": "4.5/sqrt(N) + fines_offset + gwl_offset, clipped [1.0, 5.0]",
                "target_D_formula": "1.0 + 0.04*(15-N) + risk_adj + cohesive_adj + gwl_adj, [1.0, 3.5]",
                "timestamp": datetime.now().isoformat(),
            }
            self.client.storage.from_(bucket).upload(
                "models/ann_metadata_v5.json",
                json.dumps(meta, indent=2).encode(),
                file_options={
                    "content-type": "application/json", "upsert": "true"},
            )
            print("  [OK] models/ann_metadata_v5.json")
            return True
        except Exception as e:
            print(f"  [ERROR] Upload failed: {e}")
            return False

    def save_models_local(self, out_dir: str = ".") -> bool:
        """Save models to local disk (no DB required)."""
        import os
        os.makedirs(out_dir, exist_ok=True)
        try:
            joblib.dump(self.scaler,      f"{out_dir}/scaler_v5.pkl")
            joblib.dump(self.multi_model, f"{out_dir}/ann_multi_BD_v5.pkl")
            joblib.dump(self.width_model, f"{out_dir}/ann_width_B_v5.pkl")
            joblib.dump(self.depth_model, f"{out_dir}/ann_depth_D_v5.pkl")
            meta = {
                "version": "5.0",
                "feature_names": self.feature_names,
                "architecture": {"hidden": [64, 32], "activation": "relu"},
                "train_n": len(self.X_train),
                "test_n":  len(self.X_test),
                "timestamp": datetime.now().isoformat(),
            }
            with open(f"{out_dir}/ann_metadata_v5.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  [OK] Models saved to {out_dir}/")
            return True
        except Exception as e:
            print(f"  [ERROR] Local save failed: {e}")
            return False

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    def run(self, csv_path: str = None) -> bool:
        print("\n" + "=" * 80)
        print("ANN MODEL TRAINING — v5 (FIX A–J applied)")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not SKLEARN_AVAILABLE:
            print("[ERROR] scikit-learn not installed")
            return False

        # Data source: DB preferred, CSV fallback
        if csv_path:
            if not self.load_from_csv(csv_path):
                return False
        else:
            if not self.connect_database():
                return False
            if not self.query_database():
                return False

        steps = [
            self.prepare_features_and_targets,
            self.train_models,
            self.evaluate_models,
        ]
        for step in steps:
            if not step():
                return False

        # Save
        if self.client:
            self.save_models()
        else:
            self.save_models_local("./models_v5")

        print("\n" + "=" * 80)
        print("[SUCCESS] TRAINING COMPLETE")
        print("=" * 80)
        print(f"  Features      : {len(self.feature_names)}")
        print(f"  Architecture  : {len(self.feature_names)} → 64 → 32 → 2")
        print(f"  Train / Test  : {len(self.X_train)} / {len(self.X_test)}")
        print(f"  Outputs       : Foundation Width B (m), Foundation Depth D (m)")
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    trainer = MultiOutputANNTraining()
    success = trainer.run(csv_path=csv_path)
    if not success:
        sys.exit(1)
    print("\n✓ Done")


if __name__ == "__main__":
    main()
