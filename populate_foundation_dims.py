#!/usr/bin/env python3
"""
Populate foundation_width_m and foundation_depth_m in soil_layers.

Computes engineering-based B and D that VARY per borehole according to actual
soil conditions (SPT N, fines content, GWL, liquefaction risk level).

These become the ANN training targets and replace the on-the-fly fallback
formula in train_ann.py, which is computed once and stored permanently.

Foundation Width B (m):
  Determined by soil stiffness category (SPT N) and fines correction.
  Stronger soil → smaller footing; weaker soil → wider spread required.
  Range: 1.5 m – 5.0 m.

Foundation Depth D (m):
  Determined by liquefaction risk level and groundwater level.
  Liquefiable layers → deeper to reach stable bearing stratum.
  Range: 1.5 m – 3.0 m.
"""

import math
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from supabase import create_client
except ImportError:
    print("[ERROR] supabase-py not installed.  pip install supabase")
    sys.exit(1)


# ── Engineering formula constants ─────────────────────────────────────────────

RISK_ORDER = {"VERY HIGH": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "VERY LOW": 1}


def design_B(n: float, fc: float) -> float:
    """
    Foundation width B (m) based on SPT N value and fines content.

    Rationale: weaker soil requires a wider spread to limit settlement;
    fine-grained soils are more compressible and need additional spread.

    Tiers are aligned with DPWH BSDS and Bowles (1988) guidance:
      N < 4   → very loose / soft clay   → mat or wide raft  → 4.0 m
      4–6     → loose                    → wide spread       → 3.0 m
      7–11    → medium                   → spread footing    → 2.5 m
      12–19   → medium dense             → compact footing   → 2.0 m
      ≥ 20    → dense / hard             → compact footing   → 1.5 m
    Fines correction (+0.5 m for high-plasticity, +0.25 m for silty):
      FC ≥ 35 % → fine-grained (CL/CH/ML) → +0.5 m
      15 % ≤ FC < 35 % → silty sand      → +0.25 m
    """
    if n < 4.0:
        b = 4.0
    elif n < 7.0:
        b = 3.0
    elif n < 12.0:
        b = 2.5
    elif n < 20.0:
        b = 2.0
    else:
        b = 1.5

    if fc >= 35.0:
        b += 0.5
    elif fc >= 15.0:
        b += 0.25

    return round(min(b, 5.0) * 2) / 2   # round to nearest 0.5 m


def design_D(worst_risk: str, gwl: float, representative_n: float) -> float:
    """
    Foundation depth D (m) based on liquefaction risk and groundwater level.

    Rationale:
      - Liquefiable zones (HIGH/VERY HIGH risk) → foundation must reach a
        non-liquefiable bearing layer, typically 3.0 m.
      - Marginal risk (MEDIUM) → intermediate depth.
      - Non-liquefiable, deep GWL → standard 1.5 m.
      - Shallow GWL: place foundation 0.3 m above water table to avoid
        buoyancy and reduced effective stress, min 1.5 m.
    """
    risk_score = RISK_ORDER.get(worst_risk, 2)

    if risk_score >= 4:         # HIGH or VERY HIGH → liquefiable
        return 3.0
    elif risk_score == 3:       # MEDIUM → marginal
        return 2.5
    elif representative_n < 5.0:
        return 3.0              # very weak soil even without full liquefaction
    elif representative_n < 10.0:
        return 2.0
    elif gwl <= 1.5:
        return 1.5              # very shallow GWL, can't go much deeper
    elif gwl <= 3.0:
        return max(1.5, min(gwl - 0.3, 2.5))   # place just above GWL
    else:
        return 1.5              # GWL deep, no constraint


# ── Database helpers ───────────────────────────────────────────────────────────

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_str(v, default=""):
    if v is None:
        return default
    return str(v).strip().upper()


def connect():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("[ERROR] SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in .env")
        sys.exit(1)
    client = create_client(url, key)
    client.table("soil_layers").select("id").limit(1).execute()
    print("[OK] Connected to database")
    return client


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("POPULATE FOUNDATION DIMENSIONS IN soil_layers")
    print("=" * 70)

    client = connect()

    # ── 1. Fetch all rows we need ─────────────────────────────────────────────
    print("\n[1/4] Fetching soil layers...")

    page_size = 1000
    all_rows = []
    offset = 0
    while True:
        result = (
            client.table("soil_layers")
            .select(
                "id, borehole_id, spt_n_value, spt_n60, spt_n160, "
                "fines_content, groundwater_depth_m, liquefaction_risk_level"
            )
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    print(f"    Fetched {len(all_rows)} soil-layer records")
    if not all_rows:
        print("[ERROR] No records found")
        sys.exit(1)

    # ── 2. Aggregate per borehole ─────────────────────────────────────────────
    print("\n[2/4] Aggregating per borehole...")

    from collections import defaultdict
    borehole_layers: dict = defaultdict(list)
    for row in all_rows:
        borehole_layers[row["borehole_id"]].append(row)

    borehole_params = {}   # borehole_id → {B, D, n_mean, risk, gwl_mean}
    for bh_id, layers in borehole_layers.items():
        n_vals, fc_vals, gwl_vals, risks = [], [], [], []
        for lyr in layers:
            n = safe_float(lyr.get("spt_n160") or lyr.get("spt_n60") or lyr.get("spt_n_value"), 0)
            if n > 0:
                n_vals.append(n)
            fc = safe_float(lyr.get("fines_content"), -1)
            if fc >= 0:
                fc_vals.append(fc)
            gwl = safe_float(lyr.get("groundwater_depth_m"), -1)
            if gwl > 0:
                gwl_vals.append(gwl)
            risk = safe_str(lyr.get("liquefaction_risk_level"), "VERY LOW")
            if risk:
                risks.append(risk)

        n_mean  = sum(n_vals)  / len(n_vals)  if n_vals  else 15.0
        fc_mean = sum(fc_vals) / len(fc_vals) if fc_vals else 15.0
        gwl_mean = sum(gwl_vals) / len(gwl_vals) if gwl_vals else 5.0
        worst_risk = max(risks, key=lambda r: RISK_ORDER.get(r, 1)) if risks else "VERY LOW"

        B = design_B(n_mean, fc_mean)
        D = design_D(worst_risk, gwl_mean, n_mean)

        borehole_params[bh_id] = {
            "B": B, "D": D,
            "n_mean": round(n_mean, 1), "risk": worst_risk, "gwl_mean": round(gwl_mean, 2),
        }

    print(f"    Computed B & D for {len(borehole_params)} boreholes")

    # Quick summary
    B_vals = [v["B"] for v in borehole_params.values()]
    D_vals = [v["D"] for v in borehole_params.values()]
    print(f"    B range: {min(B_vals):.1f} – {max(B_vals):.1f} m  "
          f"(mean {sum(B_vals)/len(B_vals):.2f} m)")
    print(f"    D range: {min(D_vals):.1f} – {max(D_vals):.1f} m  "
          f"(mean {sum(D_vals)/len(D_vals):.2f} m)")
    unique_B = sorted(set(B_vals))
    unique_D = sorted(set(D_vals))
    print(f"    B unique values ({len(unique_B)}): {unique_B}")
    print(f"    D unique values ({len(unique_D)}): {unique_D}")

    # ── 3. Build update payload per layer ─────────────────────────────────────
    print("\n[3/4] Building per-layer update payload...")

    updates = []
    for row in all_rows:
        bh_id = row["borehole_id"]
        params = borehole_params.get(bh_id, {"B": 2.0, "D": 1.5})
        updates.append({
            "id": row["id"],
            "foundation_width_m":  params["B"],
            "foundation_depth_m":  params["D"],
        })

    print(f"    {len(updates)} layer records to update")

    # ── 4. Batch upsert ───────────────────────────────────────────────────────
    print("\n[4/4] Uploading to database...")

    BATCH = 50
    success = 0
    failed  = 0
    for i in range(0, len(updates), BATCH):
        batch = updates[i : i + BATCH]
        try:
            client.table("soil_layers").upsert(batch).execute()
            success += len(batch)
        except Exception as e:
            print(f"    [WARN] Batch {i//BATCH + 1} failed: {e}")
            failed += len(batch)

        if (i // BATCH) % 10 == 0:
            print(f"    Progress: {min(i + BATCH, len(updates))}/{len(updates)}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"  Updated : {success} layer records")
    if failed:
        print(f"  Failed  : {failed} layer records")
    print(f"  Boreholes with new B/D: {len(borehole_params)}")
    print()
    print("Next step: retrain the ANN model.")
    print("  cd geo-be && python train_ann.py")


if __name__ == "__main__":
    main()
