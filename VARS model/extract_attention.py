

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.video import R2Plus1D_18_Weights

from dataset import MultiViewDataset
from model import MVNetwork
from config.classes import INVERSE_EVENT_DICTIONARY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pooling_type", default="cross_attention",
                        choices=["attention", "cross_attention"])
    parser.add_argument("--path", required=True, help="Dataset root")
    parser.add_argument("--split", default="Test")
    parser.add_argument("--start_frame", type=int, default=20)
    parser.add_argument("--end_frame", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", default="weights/cross_attention_weights.json")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()

    dataset = MultiViewDataset(
        path=args.path, start=args.start_frame, end=args.end_frame,
        fps=args.fps, split=args.split, num_views=5,
        transform_model=transforms_model,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = MVNetwork(net_name="r2plus1d_18", agr_type=args.pooling_type).cuda()
    state = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state["state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint}")

    # Load ground truth for action/offence labels
    gt_path = os.path.join(args.path, args.split, "annotations.json")
    with open(gt_path) as f:
        gt = json.load(f)

    entries = []
    with torch.no_grad():
        for labels_os, labels_act, mvclips, action_id in loader:
            mvclips = mvclips.cuda().float()
            out_os, out_act, attention = model(mvclips)

            # cross_attention: (B=1, num_queries, V) -> store as list of lists
            # attention:        (B=1, V)             -> wrap to [[...]] for shape consistency
            attn = attention.detach().cpu()
            if attn.dim() == 1:
                attn = attn.unsqueeze(0)  # (V,) -> (1, V)
            if attn.dim() == 2 and args.pooling_type == "attention":
                # (B, V) -> (B, 1, V) so downstream code is uniform
                attn = attn.unsqueeze(1)
            attn_np = attn[0].numpy().tolist()

            action_id_str = action_id[0]
            gt_entry = gt["Actions"].get(action_id_str, {})

            entries.append({
                "action_id": action_id_str,
                "gt_action_class": gt_entry.get("Action class", ""),
                "gt_offence": gt_entry.get("Offence", ""),
                "gt_severity": gt_entry.get("Severity", ""),
                "num_views": len(attn_np[0]),
                "attention": attn_np,  # shape (num_queries, V)
                "pred_action_idx": int(torch.argmax(out_act, dim=-1).item()),
                "pred_os_idx": int(torch.argmax(out_os, dim=-1).item()),
            })

    with open(args.output, "w") as f:
        json.dump(entries, f)
    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
