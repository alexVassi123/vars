"""
Analyze class distributions in ground truth and predictions.
Reveals class imbalance issues and prediction bias (does the model over-predict certain classes?).

Usage:
  python Evaluate/class_distribution.py VARS/2026-04-08_13-48 --epoch 18
  python Evaluate/class_distribution.py VARS/2026-04-08_13-48 --epoch best --split valid
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from confusion_matrix import load_and_align

OFFENCE_SEV_CLASSES = ["No offence", "Offence+No card", "Offence+Yellow", "Offence+Red"]
ACTION_CLASSES = ["Tackling", "Standing tackling", "High leg", "Holding", "Pushing", "Elbowing", "Challenge", "Dive"]


def count_distribution(labels, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for l in labels:
        counts[l] += 1
    return counts


def print_distribution_comparison(gt_dist, pred_dist, class_names, title):
    total = gt_dist.sum()
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    print(f"{'Class':>20} | {'GT Count':>8} | {'GT %':>6} | {'Pred Count':>10} | {'Pred %':>7} | {'Bias':>7}")
    print("-" * 70)
    for i, name in enumerate(class_names):
        gt_pct = gt_dist[i] / total * 100 if total > 0 else 0
        pred_pct = pred_dist[i] / total * 100 if total > 0 else 0
        bias = pred_pct - gt_pct
        bias_indicator = "OVER" if bias > 5 else ("UNDER" if bias < -5 else "")
        print(f"{name:>20} | {gt_dist[i]:>8} | {gt_pct:>5.1f}% | {pred_dist[i]:>10} | {pred_pct:>6.1f}% | {bias:>+6.1f}% {bias_indicator}")

    # Compute KL divergence as a single-number prediction bias score
    gt_prob = gt_dist / gt_dist.sum()
    pred_prob = pred_dist / pred_dist.sum()
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl = np.sum(gt_prob * np.log((gt_prob + eps) / (pred_prob + eps)))
    print(f"\nPrediction bias (KL divergence from GT): {kl:.4f} (0 = perfectly matched distribution)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to VARS run directory")
    parser.add_argument("--epoch", default="best", help="Epoch number or 'best'")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--dataset", default="/workspace/sn-mvfoul/data/SoccerNet/mvfouls")
    args = parser.parse_args()

    split_cap = "Valid" if args.split == "valid" else "Test"
    gt_file = os.path.join(args.dataset, split_cap, "annotations.json")

    if args.epoch == "best":
        from SoccerNet.Evaluation.MV_FoulRecognition import evaluate as sn_evaluate
        best_epoch, best_lb = -1, -1
        for f in os.listdir(args.run_dir):
            if f.endswith(".json") and args.split in f and "epoch" in f:
                ep = int(f.split("epoch_")[1].replace(".json", ""))
                r = sn_evaluate(gt_file, os.path.join(args.run_dir, f))
                if r["leaderboard_value"] > best_lb:
                    best_lb = r["leaderboard_value"]
                    best_epoch = ep
        epoch = best_epoch
        print(f"Best {args.split} epoch: {epoch} (LB={best_lb:.2f})")
    else:
        epoch = int(args.epoch)

    pred_file = os.path.join(args.run_dir, f"predicitions_{args.split}_epoch_{epoch}.json")
    if not os.path.exists(pred_file):
        print(f"Not found: {pred_file}")
        sys.exit(1)

    gt_os, pred_os, gt_act, pred_act = load_and_align(gt_file, pred_file)

    gt_os_dist = count_distribution(gt_os, 4)
    pred_os_dist = count_distribution(pred_os, 4)
    gt_act_dist = count_distribution(gt_act, 8)
    pred_act_dist = count_distribution(pred_act, 8)

    print(f"\nEpoch {epoch}, {args.split} split ({len(gt_os)} filtered samples)")

    print_distribution_comparison(gt_os_dist, pred_os_dist, OFFENCE_SEV_CLASSES, "Offence/Severity Distribution")
    print_distribution_comparison(gt_act_dist, pred_act_dist, ACTION_CLASSES, "Action Class Distribution")

    # Also show all splits GT distribution for context
    print(f"\n{'='*70}")
    print(f" Ground Truth Distribution Across All Splits")
    print(f"{'='*70}")
    for split_name in ["Train", "Valid", "Test"]:
        gt = json.load(open(os.path.join(args.dataset, split_name, "annotations.json")))
        action_counts = {}
        for ann in gt["Actions"].values():
            ac = ann.get("Action class", "")
            if ac and ac != "Dont know":
                action_counts[ac] = action_counts.get(ac, 0) + 1
        total = sum(action_counts.values())
        print(f"\n  {split_name} ({total} total):")
        for ac in ACTION_CLASSES:
            c = action_counts.get(ac, 0)
            print(f"    {ac:>20}: {c:>4} ({c/total*100:>5.1f}%)")


if __name__ == "__main__":
    main()
