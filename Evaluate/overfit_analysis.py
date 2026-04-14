"""
Overfitting analysis: compares train vs val metrics, computes generalization gap,
and identifies the optimal stopping point.

Usage:
  python Evaluate/overfit_analysis.py VARS/2026-04-08_13-48
  python Evaluate/overfit_analysis.py VARS/2026-04-08_13-48 --save overfit.png
"""
import os
import sys
import argparse
import numpy as np
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to VARS run directory")
    parser.add_argument("--dataset", default="/workspace/sn-mvfoul/data/SoccerNet/mvfouls")
    parser.add_argument("--save", default=None, help="Save plot to file")
    args = parser.parse_args()

    gt_train = os.path.join(args.dataset, "Train", "annotations.json")
    gt_val = os.path.join(args.dataset, "Valid", "annotations.json")
    gt_test = os.path.join(args.dataset, "Test", "annotations.json")

    train_results = []
    val_results = []
    test_results = []

    for f in sorted(os.listdir(args.run_dir)):
        if not f.endswith(".json") or "epoch" not in f:
            continue
        epoch = int(f.split("epoch_")[1].replace(".json", ""))
        path = os.path.join(args.run_dir, f)

        if "train" in f:
            r = evaluate(gt_train, path)
            train_results.append((epoch, r))
        elif "valid" in f:
            r = evaluate(gt_val, path)
            val_results.append((epoch, r))
        elif "test" in f:
            r = evaluate(gt_test, path)
            test_results.append((epoch, r))

    train_results.sort(key=lambda x: x[0])
    val_results.sort(key=lambda x: x[0])
    test_results.sort(key=lambda x: x[0])

    # Align epochs that have all three splits
    train_map = {e: r for e, r in train_results}
    val_map = {e: r for e, r in val_results}
    test_map = {e: r for e, r in test_results}
    common_epochs = sorted(set(train_map.keys()) & set(val_map.keys()) & set(test_map.keys()))

    if not common_epochs:
        print("No epochs with all three splits found.")
        sys.exit(1)

    print(f"\n{'='*110}")
    print(f" Overfitting Analysis — {args.run_dir}")
    print(f"{'='*110}")

    # Table
    print(f"\n{'Epoch':>5} | {'Train LB':>9} | {'Val LB':>8} | {'Test LB':>8} | {'Gap(T-V)':>9} | {'Gap(T-Te)':>10} | {'Val OffSev':>10} | {'Val Act':>8} | {'Test OffSev':>11} | {'Test Act':>9}")
    print("-" * 120)

    best_val_epoch = common_epochs[0]
    best_val_lb = 0
    for ep in common_epochs:
        t = train_map[ep]
        v = val_map[ep]
        te = test_map[ep]
        gap_tv = t["leaderboard_value"] - v["leaderboard_value"]
        gap_tte = t["leaderboard_value"] - te["leaderboard_value"]

        marker = ""
        if v["leaderboard_value"] > best_val_lb:
            best_val_lb = v["leaderboard_value"]
            best_val_epoch = ep
            marker = " <-- best val"

        print(f"{ep:>5} | {t['leaderboard_value']:>9.2f} | {v['leaderboard_value']:>8.2f} | {te['leaderboard_value']:>8.2f} | {gap_tv:>+9.2f} | {gap_tte:>+10.2f} | {v['balanced_accuracy_offence_severity']:>10.2f} | {v['balanced_accuracy_action']:>8.2f} | {te['balanced_accuracy_offence_severity']:>11.2f} | {te['balanced_accuracy_action']:>9.2f}{marker}")

    # Compute overfitting metrics
    train_lbs = [train_map[ep]["leaderboard_value"] for ep in common_epochs]
    val_lbs = [val_map[ep]["leaderboard_value"] for ep in common_epochs]
    test_lbs = [test_map[ep]["leaderboard_value"] for ep in common_epochs]
    gaps = [t - v for t, v in zip(train_lbs, val_lbs)]

    print(f"\n--- Overfitting Summary ---")
    print(f"Best val epoch:       {best_val_epoch} (LB={best_val_lb:.2f})")
    print(f"Test LB at best val:  {test_map[best_val_epoch]['leaderboard_value']:.2f}")
    print(f"Train LB at best val: {train_map[best_val_epoch]['leaderboard_value']:.2f}")
    print(f"Gap at best val:      {train_map[best_val_epoch]['leaderboard_value'] - best_val_lb:+.2f}")
    print(f"Final gap (last ep):  {gaps[-1]:+.2f}")
    print(f"Max train LB:         {max(train_lbs):.2f} (epoch {common_epochs[np.argmax(train_lbs)]})")
    print(f"Max val LB:           {max(val_lbs):.2f} (epoch {common_epochs[np.argmax(val_lbs)]})")
    print(f"Max test LB:          {max(test_lbs):.2f} (epoch {common_epochs[np.argmax(test_lbs)]})")

    # Detect when overfitting began (gap > 10 and growing)
    for i, ep in enumerate(common_epochs):
        if gaps[i] > 10:
            print(f"Overfitting threshold (gap>10) reached at epoch {ep}")
            break

    # Moving average of val LB to find plateau
    if len(val_lbs) >= 5:
        window = 5
        smoothed = np.convolve(val_lbs, np.ones(window)/window, mode='valid')
        plateau_start = None
        for i in range(1, len(smoothed)):
            if abs(smoothed[i] - smoothed[i-1]) < 0.5:
                if plateau_start is None:
                    plateau_start = common_epochs[i + window - 1]
        if plateau_start:
            print(f"Val LB plateau detected around epoch {plateau_start}")

    if args.save:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(common_epochs, train_lbs, 'g-o', markersize=3, label='Train LB')
        axes[0].plot(common_epochs, val_lbs, 'b-o', markersize=3, label='Val LB')
        axes[0].plot(common_epochs, test_lbs, 'r-o', markersize=3, label='Test LB')
        axes[0].axvline(x=best_val_epoch, color='blue', linestyle='--', alpha=0.5, label=f'Best val (ep {best_val_epoch})')
        axes[0].set_ylabel('Leaderboard Value (%)')
        axes[0].set_title('Train vs Val vs Test — Overfitting Analysis')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(common_epochs, gaps, 'k-o', markersize=3, label='Train-Val Gap')
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overfitting threshold')
        axes[1].fill_between(common_epochs, gaps, alpha=0.2, color='red')
        axes[1].set_ylabel('Generalization Gap (%)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Train-Val Gap (higher = more overfitting)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.save, dpi=150)
        print(f"\nPlot saved to {args.save}")


if __name__ == "__main__":
    main()
