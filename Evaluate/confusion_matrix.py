"""
Per-class confusion matrices for offence/severity and action classification.
Shows where the model is confusing classes, broken down by epoch.

Usage:
  python Evaluate/confusion_matrix.py VARS/2026-04-08_13-48 --epoch 18
  python Evaluate/confusion_matrix.py VARS/2026-04-08_13-48 --epoch best --split valid
"""
import os
import sys
import json
import argparse
import numpy as np

OFFENCE_SEV_CLASSES = ["No offence", "Offence+No card", "Offence+Yellow", "Offence+Red"]
ACTION_CLASSES = ["Tackling", "Standing tackling", "High leg", "Holding", "Pushing", "Elbowing", "Challenge", "Dive"]

EVENT_DICTIONARY_action_class = {
    "Tackling":0, "Standing tackling":1, "High leg":2, "Holding":3,
    "Pushing":4, "Elbowing":5, "Challenge":6, "Dive":7, "Dont know":8
}


def load_and_align(gt_file, pred_file):
    """Load ground truth and predictions, return aligned lists filtering same way as SoccerNet evaluate."""
    gt = json.load(open(gt_file))
    preds = json.load(open(pred_file))

    gt_offsev = []
    pred_offsev = []
    gt_action = []
    pred_action = []

    for action_id, ann in gt["Actions"].items():
        action_class = ann.get("Action class", "")
        offence_class = ann.get("Offence", "")
        severity_class = ann.get("Severity", "")

        # Apply same filtering as SoccerNet evaluate
        if action_class == "" or action_class == "Dont know":
            continue
        if (offence_class == "" or offence_class == "Between") and action_class != "Dive":
            continue
        if (severity_class == "" or severity_class == "2.0" or severity_class == "4.0") and action_class != "Dive" and offence_class not in ("No offence", "No Offence"):
            continue

        # Normalize
        if offence_class in ("", "Between"):
            offence_class = "Offence"
        if severity_class in ("", "2.0", "4.0"):
            severity_class = "1.0"
        if offence_class in ("No Offence", "No offence"):
            offence_class = "No offence"

        # Map to offence_severity index
        if offence_class == "No offence":
            os_gt = 0
        elif offence_class == "Offence" and severity_class == "1.0":
            os_gt = 1
        elif offence_class == "Offence" and severity_class == "3.0":
            os_gt = 2
        elif offence_class == "Offence" and severity_class == "5.0":
            os_gt = 3
        else:
            continue

        act_gt = EVENT_DICTIONARY_action_class.get(action_class, -1)
        if act_gt < 0 or act_gt >= 8:
            continue

        if action_id not in preds["Actions"]:
            continue

        p = preds["Actions"][action_id]

        # Map prediction offence/severity
        if p["Offence"] in ("No offence", "No Offence"):
            os_pred = 0
        elif p["Offence"] == "Offence" and p["Severity"] == "1.0":
            os_pred = 1
        elif p["Offence"] == "Offence" and p["Severity"] == "3.0":
            os_pred = 2
        elif p["Offence"] == "Offence" and p["Severity"] == "5.0":
            os_pred = 3
        else:
            os_pred = 0

        act_pred = EVENT_DICTIONARY_action_class.get(p.get("Action class", ""), -1)
        if act_pred < 0 or act_pred >= 8:
            act_pred = 0

        gt_offsev.append(os_gt)
        pred_offsev.append(os_pred)
        gt_action.append(act_gt)
        pred_action.append(act_pred)

    return gt_offsev, pred_offsev, gt_action, pred_action


def print_confusion_matrix(gt, pred, class_names, title):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gt, pred):
        cm[g][p] += 1

    # Print header
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    # Find max label length for formatting
    max_len = max(len(c) for c in class_names)
    header = " " * (max_len + 2) + "  ".join(f"{c[:8]:>8}" for c in class_names) + "  | Recall"
    print(f"\n{' ' * (max_len + 2)}{'Predicted':^{8 * n + 2 * (n-1)}}")
    print(header)
    print("-" * len(header))

    for i, name in enumerate(class_names):
        row_total = cm[i].sum()
        recall = cm[i][i] / row_total * 100 if row_total > 0 else 0
        row_str = "  ".join(f"{cm[i][j]:>8}" for j in range(n))
        print(f"{name:>{max_len}}  {row_str}  | {recall:5.1f}%")

    # Precision row
    print("-" * len(header))
    prec_str = ""
    for j in range(n):
        col_total = cm[:, j].sum()
        prec = cm[j][j] / col_total * 100 if col_total > 0 else 0
        prec_str += f"{prec:>7.1f}%  "
    print(f"{'Precision':>{max_len}}  {prec_str}")

    # Overall accuracy
    correct = sum(cm[i][i] for i in range(n))
    total = cm.sum()
    print(f"\nOverall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # Balanced accuracy
    per_class_recall = []
    for i in range(n):
        row_total = cm[i].sum()
        if row_total > 0:
            per_class_recall.append(cm[i][i] / row_total)
    balanced_acc = np.mean(per_class_recall) * 100
    print(f"Balanced accuracy: {balanced_acc:.1f}%")

    # Most confused pairs
    print(f"\nTop confusions:")
    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    for count, true_cls, pred_cls in confusions[:5]:
        print(f"  {true_cls} -> {pred_cls}: {count} ({count/cm[class_names.index(true_cls)].sum()*100:.0f}%)")


def find_best_epoch(run_dir, split, gt_file):
    """Find epoch with best leaderboard value for given split."""
    from SoccerNet.Evaluation.MV_FoulRecognition import evaluate as sn_evaluate

    best_epoch = -1
    best_lb = -1
    for f in os.listdir(run_dir):
        if f.endswith(".json") and split in f:
            epoch = int(f.split("epoch_")[1].replace(".json", ""))
            r = sn_evaluate(gt_file, os.path.join(run_dir, f))
            if r["leaderboard_value"] > best_lb:
                best_lb = r["leaderboard_value"]
                best_epoch = epoch
    return best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to VARS run directory")
    parser.add_argument("--epoch", default="best", help="Epoch number or 'best'")
    parser.add_argument("--split", default="test", choices=["valid", "test"], help="Which split to evaluate")
    parser.add_argument("--dataset", default="/workspace/sn-mvfoul/data/SoccerNet/mvfouls")
    args = parser.parse_args()

    split_cap = "Valid" if args.split == "valid" else "Test"
    gt_file = os.path.join(args.dataset, split_cap, "annotations.json")

    if args.epoch == "best":
        epoch = find_best_epoch(args.run_dir, args.split, gt_file)
        print(f"Best {args.split} epoch: {epoch}")
    else:
        epoch = int(args.epoch)

    pred_file = os.path.join(args.run_dir, f"predicitions_{args.split}_epoch_{epoch}.json")
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        sys.exit(1)

    print(f"Evaluating epoch {epoch} on {args.split} split")
    print(f"Predictions: {pred_file}")

    gt_os, pred_os, gt_act, pred_act = load_and_align(gt_file, pred_file)

    print_confusion_matrix(gt_os, pred_os, OFFENCE_SEV_CLASSES, "Offence / Severity")
    print_confusion_matrix(gt_act, pred_act, ACTION_CLASSES, "Action Classification")


if __name__ == "__main__":
    main()
