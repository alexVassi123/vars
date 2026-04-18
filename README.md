# VARS Model — Bachelor's Thesis

## Background

### Video Assistant Referee (VAR) in Soccer
Since 2018, professional soccer has used the Video Assistant Referee (VAR) system to help referees review controversial decisions — particularly fouls, penalties, and red card incidents. A human VAR operator manually reviews multiple camera feeds to advise the on-field referee. This process is time-consuming (often 2–5 minutes per review) and still subject to human error and inconsistency.

Automating foul recognition from multi-view video is therefore a practically relevant problem: a reliable system could speed up reviews, reduce inconsistency, and assist referees in lower-budget leagues that cannot afford full VAR infrastructure.

### SoccerNet-MVFoul
SoccerNet-MVFoul (Held et al., 2023) is an open benchmark for this problem. It contains 3,901 real match clips, each showing a foul incident from 2–4 broadcast camera angles. Each clip is annotated with:
- **Action type** (8 classes): tackling, standing tackle, high leg, holding, pushing, elbowing, challenge, dive
- **Offence severity** (4 classes): no offence, no card, yellow card, red card

The dataset is class-imbalanced — most fouls result in no card — so results are measured with **balanced accuracy** (average per-class recall), where a random classifier scores ~25% for severity and ~12.5% for action type.

### The Multi-View Challenge
Using multiple camera angles is essential: a foul may be ambiguous from one angle but clear from another. The key design question is how to combine information from multiple views into a single prediction. This is the **aggregation problem** and is the focus of this thesis.

---

## Related Work

### Video Understanding Backbones
Modern video classification uses 3D CNNs or video transformers as feature extractors:
- **R2Plus1D** (Tran et al., 2018): factorizes 3D convolutions into separate spatial 2D and temporal 1D convolutions. Efficient and strong baseline.
- **R3D / MC3** (Tran et al., 2018): pure 3D convolutions; MC3 mixes 2D and 3D layers.
- **MViT** (Fan et al., 2021): video Vision Transformer with multiscale attention pooling. State-of-the-art but expensive.

All backbones are pretrained on Kinetics-400 (a large-scale action recognition dataset) and fine-tuned on MVFoul.

### Multi-View Aggregation Strategies
How to combine features from V views into one representation is an open research question:

- **Max / Mean pooling**: simple, parameter-free. Takes element-wise max or mean across view features. No inter-view communication.
- **Weighted attention** (original VARS paper): learns a (feat_dim × feat_dim) weight matrix, computes pairwise view scores via matrix multiplication, and produces a weighted sum. Views interact but through a fixed learned matrix, not input-dependent attention.
- **Transformer encoder**: applies standard self-attention (Vaswani et al., 2017) across the V view tokens. Each view attends to all others dynamically based on content. Used in multi-view 3D reconstruction (e.g. NeRF variants) and multi-camera perception (e.g. BEVFormer for autonomous driving).
- **Cross-attention** (this thesis): identical to transformer self-attention when Q=K=V, but produces an explicit (V×V) attention weight matrix that can be visualized and interpreted — showing which camera angles the model relied on for each decision.

### Attention in Vision
The attention mechanism (Bahdanau et al., 2015) allows a model to dynamically weight which parts of the input are most relevant. Applied to multi-view video, it allows the model to learn "view 2 is most informative for this type of foul" rather than averaging everything equally. `nn.MultiheadAttention` in PyTorch implements scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V.

---

## Research Question

**Does cross-attention between camera views produce better foul classification than simpler aggregation strategies?**

The SoccerNet-MVFoul dataset contains ~3,900 soccer foul clips, each filmed from 2–4 camera angles simultaneously. The model must classify:
- **Offence severity** (4 classes: no offence, yellow, red, no card)
- **Action type** (8 classes: tackling, high leg, etc.)

The core question is whether letting views "talk to each other" via cross-attention improves over simpler pooling strategies like max or mean.

---

## Installation

```bash
conda create -n vars python=3.9
conda activate vars

# Install PyTorch with CUDA (check your CUDA version first: nvcc --version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install SoccerNet
pip install av
pip install -r requirements.txt
```

---

## Dataset Setup

```bash
cd data/SoccerNet/mvfouls

# If zips extracted into wrong folder, move them manually:
mv action_* Train/
mv annotations.json Train/

# Unzip each split into its own folder:
cd Valid && unzip ../valid.zip && cd ..
cd Test  && unzip ../test.zip  && cd ..
cd Chall && unzip ../challenge.zip && cd ..
```

Expected structure:
```
data/SoccerNet/mvfouls/
    Train/
        annotations.json
        action_0/
        action_1/
        ...
    Valid/
    Test/
    Chall/
```

---

## Methodology

### Pipeline Overview

```
Input: (B, V, C, T, H, W)  — batch of B clips, V views, C channels, T frames, H×W pixels
         ↓
[batch_tensor]  →  reshape to (B*V, C, T, H, W)
         ↓
[Backbone: R2Plus1D-18]  →  (B*V, feat_dim)   feat_dim=512
         ↓
[unbatch_tensor]  →  reshape to (B, V, feat_dim)
         ↓
[Aggregation]  →  (B, feat_dim)   ← this is what we compare
         ↓
[FC layers]  →  offence logits (4,)  +  action logits (8,)
         ↓
[Loss]  CrossEntropyLoss(offence) + CrossEntropyLoss(action)
```

### Backbone
R2Plus1D-18 is pretrained on Kinetics-400 and its final fully-connected layer is removed, leaving a 512-dimensional feature vector per video clip. All views are passed through the backbone simultaneously by reshaping the batch dimension: (B, V, C, T, H, W) → (B×V, C, T, H, W). This saves memory and ensures all views are processed identically.

### Aggregation Strategies Compared

| Method | How it combines views | Learnable params |
|--------|----------------------|-----------------|
| Max pooling | element-wise max across V | none |
| Mean pooling | element-wise mean across V | none |
| Weighted attention | learned (512×512) weight matrix, input-independent | 512×512 |
| Transformer encoder | self-attention + feedforward across V view tokens | ~2M |
| Cross-attention (proposed) | MultiheadAttention(Q=K=V), explicit V×V matrix | ~1M |

### Cross-Attention Detail
Given view features `aux` of shape (B, V, 512):
```
attended, attn_weights = MultiheadAttention(Q=aux, K=aux, V=aux)
# attn_weights: (B, V, V) — each row is a softmax distribution over views
attended = LayerNorm(attended + aux)   # residual connection
pooled   = mean(attended, dim=1)       # (B, 512)
```
The residual connection (`attended + aux`) preserves the original per-view features, preventing the attention from discarding useful single-view information. The (B, V, V) attention matrix can be visualized after training to show which views the model attended to for each prediction.

### Multi-Task Learning
The model is trained simultaneously on both tasks with equal loss weighting:
```
loss = CrossEntropyLoss(pred_offence, label_offence)
     + CrossEntropyLoss(pred_action, label_action)
```
Class weights are computed from training set frequencies (`--weighted_loss Yes`) to counteract the class imbalance.

### Training Protocol
- Optimizer: AdamW (momentum 0.9/0.999, eps 1e-7)
- LR schedule: StepLR — LR drops by factor 10 every 3 epochs
- During training: 2 random views per clip (data augmentation through view sampling)
- During evaluation: all 5 views per clip (maximum information)
- Data augmentation: random affine, perspective, rotation, color jitter, horizontal flip

---

## Code Contributions

The following aggregation strategies were added to `mvaggregate.py` and wired into `main.py`:

### TransformerAggregate (`--pooling_type transformer`)
Applies a Transformer encoder across view features. Each view attends to all other views via self-attention + feedforward layers. Output is mean-pooled across views.

### CrossAttentionAggregate (`--pooling_type cross_attention`)
Applies `nn.MultiheadAttention` across view features (Q=K=V). Produces an explicit **(B, V, V) attention matrix** showing which views the model relied on. Includes residual connection and LayerNorm. This is the proposed method.

A bug in the original `MVAggregate.__init__` elif chain was also fixed — the `transformer` type previously fell into the `else` branch and constructed `WeightedAggregate` unnecessarily before being overridden.

---

## Experiments

All 5 aggregation strategies are compared under **identical conditions**:
- Backbone: `r2plus1d_18` (pretrained on Kinetics-400)
- Optimizer: AdamW, LR=1e-4, weight_decay=0.001
- Scheduler: StepLR, step_size=3, gamma=0.1
- Epochs: 60
- Batch size: 2
- Frame range: 0–125, fps=25
- Weighted loss: Yes
- Data augmentation: Yes

### Training commands

```bash
# Run all 5 back-to-back in a tmux session (ctrl+B D to detach)
tmux new -s train

python "VARS model/main.py" --path /workspace/sn-mvfoul/data/SoccerNet/mvfouls --pooling_type attention     --pre_model r2plus1d_18 --batch_size 2 --max_epochs 60 --max_num_worker 4 --GPU 0 && \
python "VARS model/main.py" --path /workspace/sn-mvfoul/data/SoccerNet/mvfouls --pooling_type max           --pre_model r2plus1d_18 --batch_size 2 --max_epochs 60 --max_num_worker 4 --GPU 0 && \
python "VARS model/main.py" --path /workspace/sn-mvfoul/data/SoccerNet/mvfouls --pooling_type mean          --pre_model r2plus1d_18 --batch_size 2 --max_epochs 60 --max_num_worker 4 --GPU 0 && \
python "VARS model/main.py" --path /workspace/sn-mvfoul/data/SoccerNet/mvfouls --pooling_type transformer   --pre_model r2plus1d_18 --batch_size 2 --max_epochs 60 --max_num_worker 4 --GPU 0 && \
python "VARS model/main.py" --path /workspace/sn-mvfoul/data/SoccerNet/mvfouls --pooling_type cross_attention --pre_model r2plus1d_18 --batch_size 2 --max_epochs 60 --max_num_worker 4 --GPU 0
```

### Evaluation metric

**Balanced accuracy** for both tasks (handles class imbalance). Reported on the Test set.

| Method | Offence BACC (a_1) | Action BACC (a_2) |
|--------|-------------------|------------------|
| max | | |
| mean | | |
| attention (baseline) | | |
| transformer | | |
| cross_attention (proposed) | | |

---

## Possible Improvements (if results are unsatisfying)

1. **More views during training**: `--num_views 4` or `--num_views 5`
2. **Lower learning rate**: `--LR 1e-5`
3. **More epochs**: `--max_epochs 90`
4. **Stronger backbone**: `mvit_v2_s` (needs more VRAM)
5. **Cross-attention architecture tweaks**: more heads, more layers, projection layer before attention

---

## Hardware

Trained on a rented GPU via Vast.ai (RTX 3090, 24GB VRAM).
Estimated training time: ~2–4 hours per run, ~10–20 hours total for all 5 experiments.

Local machine (NVIDIA RTX A2000, 6GB VRAM) is sufficient for smoke tests only:
```bash
python "VARS model/main.py" --path /path/to/data --pooling_type cross_attention \
  --pre_model r2plus1d_18 --batch_size 1 --max_epochs 1 --max_num_worker 0 --fps 5 --GPU 0
```
