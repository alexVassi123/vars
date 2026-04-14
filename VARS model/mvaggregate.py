from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn


class WeightedAggregate(nn.Module):
    def __init__(self,  model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        num_heads = 8
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()
   


    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))


        ##################### VIEW ATTENTION #####################

        # S = source length 
        # N = batch size
        # E = embedding dimension
        # L = target length

        aux = torch.matmul(aux, self.attention_weights)
        # Dimension S, E for two views (2,512)

        # Dimension N, S, E
        aux_t = aux.permute(0, 2, 1)

        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))

        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))

        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux


class TransformerAggregate(nn.Module):
    def __init__(self, model, feat_dim, num_heads=8, num_layers=1, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape
        # Extract per-view features: (B, V, feat_dim)
        aux = self.lifting_net(unbatch_tensor(
            self.model(batch_tensor(mvimages, dim=1, squeeze=True)),
            B, dim=1, unsqueeze=True
        ))

        # Self-attention + FFN across views
        transformer_output = self.transformer_encoder(aux)

        # Mean pool across views: (B, feat_dim)
        pooled = torch.mean(transformer_output, dim=1)

        return pooled.squeeze(), None


class CrossAttentionAggregate(nn.Module):
    def __init__(self, model, feat_dim, num_heads=8, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape
        # Extract per-view features: (B, V, feat_dim)
        aux = self.lifting_net(unbatch_tensor(
            self.model(batch_tensor(mvimages, dim=1, squeeze=True)),
            B, dim=1, unsqueeze=True
        ))

        # Each view attends to all other views (Q=K=V=aux)
        # attention_weights: (B, V, V) — pairwise view attention matrix
        attended, attention_weights = self.cross_attention(
            query=aux, key=aux, value=aux, need_weights=True, average_attn_weights=True
        )

        # Residual + norm
        attended = self.norm(attended + aux)

        # Mean pool across views: (B, feat_dim)
        pooled = torch.mean(attended, dim=1)

        # How much attention each view receives: (B, V)
        view_importance = attention_weights.sum(dim=1)

        return pooled.squeeze(), view_importance


class MVAggregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim // 2, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim // 2, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "transformer":
            self.aggregation_model = TransformerAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
        elif self.agr_type == "cross_attention":
            self.aggregation_model = CrossAttentionAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
        else:  # "attention"
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages)

        pred_action = self.fc_action(pooled_view)
        pred_offence_severity = self.fc_offence(pooled_view)

        return pred_offence_severity, pred_action, attention
