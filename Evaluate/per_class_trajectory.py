"""
Track per-class recall across epochs to see which classes improve/degrade.
Reveals if the model is sacrificing minority classes for majority ones.

Usage:
  python Evaluate/per_class_trajectory.py VARS/2026-04-08_13-48 --split test
  python Evaluate/per_class_trajectory.py VARS/2026-04-08_13-48 --split valid --save per_class.png
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from confusion_matrix import load_and_align, EVENT_DICTIONARY_action_class

OFFENCE_SEV_CLASSES = ["No offence", "Offence+No card", "Offence+Yellow", "Offence+Red"]
ACTION_CLASSES = ["Tackling", "Standing tackling", "High leg", "Holding", "Pushing", "Elbowing", "Challenge", "Dive"]


def compute_per_class_recall(gt_list, pred_list, num_classes):
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for g, p in zip(gt_list, pred_list):
        total[g] += 1
        if g == p:
            correct[g] += 1
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        recall[i] = correct[i] / total[i] * 100 if total[i] > 0 else 0
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to VARS run directory")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--dataset", default="/workspace/sn-mvfoul/data/SoccerNet/mvfouls")
    parser.add_argument("--save", default=None, help="Save plot to file")
    args = parser.parse_args()

    split_cap = "Valid" if args.split == "valid" else "Test"
    gt_file = os.path.join(args.dataset, split_cap, "annotations.json")

    epochs = []
    os_recalls = []  # (epoch, recall_per_class)
    act_recalls = []

    for f in sorted(os.listdir(args.run_dir)):
        if f.endswith(".json") and args.split in f and "epoch" in f:
            epoch = int(f.split("epoch_")[1].replace(".json", ""))
            pred_file = os.path.join(args.run_dir, f)
            gt_os, pred_os, gt_act, pred_act = load_and_align(gt_file, pred_file)
            os_recall = compute_per_class_recall(gt_os, pred_os, 4)
            act_recall = compute_per_class_recall(gt_act, pred_act, 8)
            epochs.append(epoch)
            os_recalls.append(os_recall)
            act_recalls.append(act_recall)

    # Sort by epoch
    order = np.argsort(epochs)
    epochs = [epochs[i] for i in order]
    os_recalls = np.array([os_recalls[i] for i in order])
    act_recalls = np.array([act_recalls[i] for i in order])

    # Print table
    print(f"\n{'='*100}")
    print(f" Per-class RECALL trajectory ({args.split} split)")
    print(f"{'='*100}")

    print(f"\nOffence/Severity recall (%):")
    header = f"{'Epoch':>5}"
    for c in OFFENCE_SEV_CLASSES:
        header += f" | {c:>16}"
    print(header)
    print("-" * len(header))
    for i, ep in enumerate(epochs):
        row = f"{ep:>5}"
        for j in range(4):
            row += f" | {os_recalls[i, j]:>16.1f}"
        print(row)

    print(f"\nAction recall (%):")
    header = f"{'Epoch':>5}"
    for c in ACTION_CLASSES:
        header += f" | {c:>12}"
    print(header)
    print("-" * len(header))
    for i, ep in enumerate(epochs):
        row = f"{ep:>5}"
        for j in range(8):
            row += f" | {act_recalls[i, j]:>12.1f}"
        print(row)

    # Summary: which classes are always bad?
    print(f"\n--- Summary (mean recall across all epochs) ---")
    print("Offence/Severity:")
    for j, c in enumerate(OFFENCE_SEV_CLASSES):
        mean_r = os_recalls[:, j].mean()
        std_r = os_recalls[:, j].std()
        print(f"  {c:>20}: {mean_r:.1f}% +/- {std_r:.1f}%")

    print("Action:")
    for j, c in enumerate(ACTION_CLASSES):
        mean_r = act_recalls[:, j].mean()
        std_r = act_recalls[:, j].std()
        print(f"  {c:>20}: {mean_r:.1f}% +/- {std_r:.1f}%")

    if args.save:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for j, c in enumerate(OFFENCE_SEV_CLASSES):
            axes[0].plot(epochs, os_recalls[:, j], '-o', markersize=3, label=c)
        axes[0].set_ylabel("Recall (%)")
        axes[0].set_title(f"Per-class Offence/Severity Recall ({args.split})")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        for j, c in enumerate(ACTION_CLASSES):
            axes[1].plot(epochs, act_recalls[:, j], '-o', markersize=3, label=c)
        axes[1].set_ylabel("Recall (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title(f"Per-class Action Recall ({args.split})")
        axes[1].legend(loc="upper right", ncol=2)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.save, dpi=150)
        print(f"\nPlot saved to {args.save}")


if __name__ == "__main__":
    main()
