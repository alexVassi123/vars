"""
Test-Time Augmentation (TTA) evaluation for VARS foul recognition.

Loads a trained checkpoint and runs inference multiple times per sample with
different temporal shifts and optional horizontal flips, averaging softmax
probabilities across configs before taking argmax. Writes predictions and
evaluates with the official SoccerNet evaluator.

Usage:
    python "VARS model/tta_eval.py" \
        --checkpoint models/VARS/2/r2plus1d_18/0.0001/_B8_F32_S_G0.1_Step3/best_model.pth.tar \
        --path /workspace/dataset \
        --start_frame 20 --end_frame 100 --fps 10 \
        --pooling_type attention
"""
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.video import R2Plus1D_18_Weights
from tqdm import tqdm

from dataset import MultiViewDataset
from model import MVNetwork
from config.classes import INVERSE_EVENT_DICTIONARY
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate


def severity_to_fields(sev_idx):
    if sev_idx == 0:
        return "No offence", ""
    if sev_idx == 1:
        return "Offence", "1.0"
    if sev_idx == 2:
        return "Offence", "3.0"
    return "Offence", "5.0"


def run_tta_on_split(models, split, path_dataset, start, end, fps,
                     tta_configs, output_file, num_workers=8):
    """Run TTA inference on a split, averaging softmax probs across all
    (model, shift, flip) combinations. Writes predictions to JSON."""
    transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()

    dataset = MultiViewDataset(
        path=path_dataset, start=start, end=end, fps=fps,
        split=split, num_views=5, transform_model=transforms_model,
    )

    # action_id -> {"os": summed_probs, "act": summed_probs, "n": count}
    accum = {}

    for shift, flip in tta_configs:
        dataset.tta_temporal_shift = shift
        dataset.tta_flip = flip

        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        for m_idx, model in enumerate(models):
            desc = f"{split} m{m_idx} shift={shift:+d} flip={int(flip)}"
            with torch.no_grad():
                for _, _, mvclips, action in tqdm(loader, desc=desc, leave=False):
                    mvclips = mvclips.cuda().float()
                    out_os, out_act, _ = model(mvclips)

                    # Model may squeeze B=1 away — restore it.
                    if out_os.dim() == 1:
                        out_os = out_os.unsqueeze(0)
                    if out_act.dim() == 1:
                        out_act = out_act.unsqueeze(0)

                    p_os = torch.softmax(out_os, dim=-1).squeeze(0).cpu()
                    p_act = torch.softmax(out_act, dim=-1).squeeze(0).cpu()

                    aid = action[0]
                    if aid not in accum:
                        accum[aid] = {"os": p_os.clone(),
                                      "act": p_act.clone(),
                                      "n": 1}
                    else:
                        accum[aid]["os"] += p_os
                        accum[aid]["act"] += p_act
                        accum[aid]["n"] += 1

    # Reset dataset TTA state so other users of the class see defaults.
    dataset.tta_temporal_shift = 0
    dataset.tta_flip = False

    data = {"Set": split.lower(), "Actions": {}}
    for aid, v in accum.items():
        os_avg = v["os"] / v["n"]
        act_avg = v["act"] / v["n"]
        sev_idx = int(torch.argmax(os_avg).item())
        act_idx = int(torch.argmax(act_avg).item())
        offence, severity = severity_to_fields(sev_idx)
        data["Actions"][aid] = {
            "Action class": INVERSE_EVENT_DICTIONARY["action_class"][act_idx],
            "Offence": offence,
            "Severity": severity,
        }

    with open(output_file, "w") as f:
        json.dump(data, f)
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="One or more checkpoint paths. Multiple checkpoints "
                             "enables seed/model ensembling.")
    parser.add_argument("--path", required=True, help="Dataset root")
    parser.add_argument("--start_frame", type=int, default=20)
    parser.add_argument("--end_frame", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--pooling_type", type=str, default="attention")
    parser.add_argument("--output_dir", type=str, default="tta_predictions")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shifts", type=int, nargs="+",
                        default=[-10, -5, 0, 5, 10])
    parser.add_argument("--no_flip", action="store_true",
                        help="Disable horizontal-flip TTA")
    parser.add_argument("--splits", nargs="+", default=["Valid", "Test"],
                        help="Which splits to evaluate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    flips = [False] if args.no_flip else [False, True]
    tta_configs = [(s, f) for s in args.shifts for f in flips]
    print(f"TTA configs ({len(tta_configs)}): {tta_configs}")
    print(f"Ensembling {len(args.checkpoint)} checkpoint(s)")

    models = []
    for ckpt_path in args.checkpoint:
        m = MVNetwork(net_name="r2plus1d_18", agr_type=args.pooling_type).cuda()
        load = torch.load(ckpt_path, map_location="cuda")
        m.load_state_dict(load["state_dict"])
        m.eval()
        models.append(m)
        print(f"Loaded checkpoint: {ckpt_path}")

    for split in args.splits:
        output_file = os.path.join(
            args.output_dir, f"tta_{split.lower()}_preds.json"
        )
        run_tta_on_split(
            models, split, args.path,
            args.start_frame, args.end_frame, args.fps,
            tta_configs, output_file, args.num_workers,
        )
        annotations = os.path.join(args.path, split, "annotations.json")
        if split == "Chall" or not os.path.exists(annotations):
            print(f"\n=== {split} TTA predictions written to {output_file} "
                  f"(no ground truth — skipping evaluation) ===")
            continue
        results = evaluate(annotations, output_file)
        print(f"\n=== {split} TTA results ===")
        print(results)


if __name__ == "__main__":
    main()
