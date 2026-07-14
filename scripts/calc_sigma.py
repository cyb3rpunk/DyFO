"""Calculate sigma (std) of R² across walk-forward windows, excluding window 8."""
import json
import numpy as np

# --- Run with TGAT, TGN, ROLAND, GAT_STATIC (9 windows, rev3) ---
path_4v = "results/bootstrap_eval_tkg_rev3_20260420_141237/bootstrap_summary_tkg_rev3.json"
with open(path_4v, "r") as f:
    data_4v = json.load(f)

# --- Run with persistence, ewma, tgat (9 windows, rev3, 50 epochs) ---
path_3v = "results/bootstrap_eval_tkg_rev3_20260501_200449/bootstrap_summary_tkg_rev3.json"
with open(path_3v, "r") as f:
    data_3v = json.load(f)

print("=" * 70)
print("SIGMA (R²) ANALYSIS — Excluding Window 8")
print("=" * 70)

# Print info about the 4-variant run
cfg = data_4v.get("run_config", {})
print(f"\nSource A: {path_4v}")
print(f"  variants: {cfg.get('variants')}, windows: {cfg.get('n_windows')}, epochs: {cfg.get('epochs')}")

cfg3 = data_3v.get("run_config", {})
print(f"\nSource B: {path_3v}")
print(f"  variants: {cfg3.get('variants')}, windows: {cfg3.get('n_windows')}, epochs: {cfg3.get('epochs')}")

# Combine: use source B for persistence/ewma/tgat, source A for tgn/roland/gat_static
sources = {
    "DyFO (TGAT) [50ep]": ("tgat", data_3v),
    "TGN (Recurrent)": ("tgn", data_4v),
    "ROLAND": ("roland", data_4v),
    "GAT_STATIC": ("gat_static", data_4v),
    "Persistence": ("persistence", data_3v),
    "EWMA": ("ewma", data_3v),
}

print("\n" + "-" * 70)
print(f"{'Model':<25} {'All 9 (mean)':<12} {'std (all 9)':<12} {'Excl W8 (mean)':<16} {'std (excl W8)':<12}")
print("-" * 70)

for label, (variant, data) in sources.items():
    metrics = data["metrics_by_variant"].get(variant, [])
    if not metrics:
        continue
    r2_all = [m["r_squared"] for m in metrics]
    r2_no8 = [r for i, r in enumerate(r2_all) if (i + 1) != 8]

    mean_all = np.mean(r2_all)
    std_all = np.std(r2_all)
    mean_no8 = np.mean(r2_no8)
    std_no8 = np.std(r2_no8)

    print(f"{label:<25} {mean_all:<12.6f} {std_all:<12.6f} {mean_no8:<16.6f} {std_no8:<12.6f}")

# Detailed per-window for DyFO and TGN
print("\n" + "=" * 70)
print("PER-WINDOW DETAIL: DyFO (TGAT) vs TGN (Recurrent)")
print("=" * 70)

tgat_r2 = [m["r_squared"] for m in data_3v["metrics_by_variant"]["tgat"]]
tgn_r2 = [m["r_squared"] for m in data_4v["metrics_by_variant"]["tgn"]]

print(f"{'Window':<10} {'DyFO R²':<12} {'TGN R²':<12} {'Delta':<12} {'Note'}")
print("-" * 60)
for i in range(9):
    tag = "EXCLUDED" if (i + 1) == 8 else ""
    delta = tgat_r2[i] - tgn_r2[i]
    print(f"W{i+1:<9} {tgat_r2[i]:<12.6f} {tgn_r2[i]:<12.6f} {delta:<+12.6f} {tag}")

# Final sigma summary
tgat_no8 = [r for i, r in enumerate(tgat_r2) if (i + 1) != 8]
tgn_no8 = [r for i, r in enumerate(tgn_r2) if (i + 1) != 8]

print(f"\n{'SUMMARY':<25} {'Mean R²':<12} {'std(R²)':<12} {'std(R²) sample':<14}")
print("-" * 65)
print(f"{'DyFO (all 9)':<25} {np.mean(tgat_r2):<12.6f} {np.std(tgat_r2):<12.6f} {np.std(tgat_r2, ddof=1):<14.6f}")
print(f"{'DyFO (excl W8)':<25} {np.mean(tgat_no8):<12.6f} {np.std(tgat_no8):<12.6f} {np.std(tgat_no8, ddof=1):<14.6f}")
print(f"{'TGN  (all 9)':<25} {np.mean(tgn_r2):<12.6f} {np.std(tgn_r2):<12.6f} {np.std(tgn_r2, ddof=1):<14.6f}")
print(f"{'TGN  (excl W8)':<25} {np.mean(tgn_no8):<12.6f} {np.std(tgn_no8):<12.6f} {np.std(tgn_no8, ddof=1):<14.6f}")

print("\nPaper claims:")
print(f"  DyFO sigma_R2 = 0.034  (calculated population: {np.std(tgat_r2):.4f})")
print(f"  TGN  sigma_R2 = 0.077  (calculated population: {np.std(tgn_r2):.4f})")
print(f"  Note: Paper uses 50-epoch run, TGN data here from different run")
