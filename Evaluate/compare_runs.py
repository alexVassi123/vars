"""
Compare multiple training runs side by side.
Shows best val/test metrics for each run to quickly identify which config worked best.

Usage:
  python Evaluate/compare_runs.py VARS/2026-04-08_13-48 VARS/2026-04-09_10-30
  python Evaluate/compare_runs.py VARS/2026-04-08_13-48 VARS/2026-04-09_10-30 --save comparison.png
  python Evaluate/compare_runs.py ... --csv results.csv
"""
import csv
import os
import sys
import argparse
import numpy as np
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate


def evaluate_run(run_dir, dataset_path):
    gt_val = os.path.join(dataset_path, "Valid", "annotations.json")
    gt_test = os.path.join(dataset_path, "Test", "annotations.json")

    val_results = []
    test_results = []

    for f in sorted(os.listdir(run_dir)):
        if not f.endswith(".json") or "epoch" not in f:
            continue
        epoch = int(f.split("epoch_")[1].replace(".json", ""))
        path = os.path.join(run_dir, f)
        if "valid" in f:
            r = evaluate(gt_val, path)
            val_results.append((epoch, r))
        elif "test" in f:
            r = evaluate(gt_test, path)
            test_results.append((epoch, r))

    val_results.sort(key=lambda x: x[0])
    test_results.sort(key=lambda x: x[0])
    return val_results, test_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Paths to VARS run directories")
    parser.add_argument("--dataset", default="/workspace/sn-mvfoul/data/SoccerNet/mvfouls")
    parser.add_argument("--save", default=None, help="Save comparison plot")
    parser.add_argument("--csv", default=None, help="Export results to CSV file")
    args = parser.parse_args()

    print(f"\n{'='*90}")
    print(f" Run Comparison")
    print(f"{'='*90}")

    all_run_data = []

    for run_dir in args.runs:
        # Use parent folder name (e.g. THESIS_mean_s1) if the basename is a timestamp
        basename = os.path.basename(run_dir.rstrip("/"))
        parent = os.path.basename(os.path.dirname(run_dir.rstrip("/")))
        run_name = parent if parent.startswith("THESIS") else basename
        val_results, test_results = evaluate_run(run_dir, args.dataset)

        if not val_results:
            print(f"\n  {run_name}: No validation results found, skipping")
            continue

        best_val = max(val_results, key=lambda x: x[1]["leaderboard_value"])
        best_test = max(test_results, key=lambda x: x[1]["leaderboard_value"]) if test_results else (0, {})

        # Test at best val epoch
        test_at_best_val = None
        for ep, r in test_results:
            if ep == best_val[0]:
                test_at_best_val = r
                break

        total_epochs = max(e for e, _ in val_results)

        all_run_data.append({
            "name": run_name,
            "dir": run_dir,
            "val_results": val_results,
            "test_results": test_results,
            "best_val": best_val,
            "best_test": best_test,
            "test_at_best_val": test_at_best_val,
            "total_epochs": total_epochs,
        })

    # Print comparison table
    print(f"\n{'Run':>25} | {'Epochs':>6} | {'Best Val LB':>11} | {'@Epoch':>6} | {'Test@BestVal':>12} | {'Best Test LB':>12} | {'@Epoch':>6}")
    print("-" * 100)

    for rd in all_run_data:
        bv = rd["best_val"]
        bt = rd["best_test"]
        tabv = rd["test_at_best_val"]
        tabv_str = f"{tabv['leaderboard_value']:>12.2f}" if tabv else "         N/A"
        bt_lb = f"{bt[1]['leaderboard_value']:>12.2f}" if bt[1] else "         N/A"
        bt_ep = f"{bt[0]:>6}" if bt[1] else "   N/A"
        print(f"{rd['name']:>25} | {rd['total_epochs']:>6} | {bv[1]['leaderboard_value']:>11.2f} | {bv[0]:>6} | {tabv_str} | {bt_lb} | {bt_ep}")

    # Detailed per-run breakdown
    for rd in all_run_data:
        bv = rd["best_val"]
        bt = rd["best_test"]
        print(f"\n--- {rd['name']} ---")
        print(f"  Best val  (ep {bv[0]:>2}): LB={bv[1]['leaderboard_value']:.2f}  OffSev={bv[1]['balanced_accuracy_offence_severity']:.2f}  Action={bv[1]['balanced_accuracy_action']:.2f}")
        if bt[1]:
            print(f"  Best test (ep {bt[0]:>2}): LB={bt[1]['leaderboard_value']:.2f}  OffSev={bt[1]['balanced_accuracy_offence_severity']:.2f}  Action={bt[1]['balanced_accuracy_action']:.2f}")
        if rd["test_at_best_val"]:
            t = rd["test_at_best_val"]
            print(f"  Test@best_val:   LB={t['leaderboard_value']:.2f}  OffSev={t['balanced_accuracy_offence_severity']:.2f}  Action={t['balanced_accuracy_action']:.2f}")

        # Val stability: std of last 10 epochs
        last_10_val = [r["leaderboard_value"] for _, r in rd["val_results"][-10:]]
        print(f"  Val LB std (last 10 ep): {np.std(last_10_val):.2f} (lower = more stable)")

    if args.csv and all_run_data:
        fieldnames = [
            "run", "best_val_epoch", "best_val_lb",
            "best_val_offence", "best_val_action",
            "test_at_best_val_epoch", "test_at_best_val_lb",
            "test_at_best_val_offence", "test_at_best_val_action",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rd in all_run_data:
                bv_ep, bv = rd["best_val"]
                tabv = rd["test_at_best_val"]
                writer.writerow({
                    "run": rd["name"],
                    "best_val_epoch": bv_ep,
                    "best_val_lb": round(bv["leaderboard_value"], 4),
                    "best_val_offence": round(bv["balanced_accuracy_offence_severity"], 4),
                    "best_val_action": round(bv["balanced_accuracy_action"], 4),
                    "test_at_best_val_epoch": bv_ep,
                    "test_at_best_val_lb": round(tabv["leaderboard_value"], 4) if tabv else "",
                    "test_at_best_val_offence": round(tabv["balanced_accuracy_offence_severity"], 4) if tabv else "",
                    "test_at_best_val_action": round(tabv["balanced_accuracy_action"], 4) if tabv else "",
                })
        print(f"\nCSV saved to {args.csv}")

    if args.save and len(all_run_data) > 1:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for rd in all_run_data:
            epochs_v = [e for e, _ in rd["val_results"]]
            vals_v = [r["leaderboard_value"] for _, r in rd["val_results"]]
            epochs_t = [e for e, _ in rd["test_results"]]
            vals_t = [r["leaderboard_value"] for _, r in rd["test_results"]]

            axes[0].plot(epochs_v, vals_v, '-o', markersize=3, label=f'{rd["name"]} (val)')
            axes[1].plot(epochs_t, vals_t, '-o', markersize=3, label=f'{rd["name"]} (test)')

        axes[0].set_ylabel("Val Leaderboard Value (%)")
        axes[0].set_title("Validation Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("Test Leaderboard Value (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Test Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.save, dpi=150)
        print(f"\nPlot saved to {args.save}")


if __name__ == "__main__":
    main()
