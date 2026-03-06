import math
from typing import Dict, Tuple, List, Any
import torch
import torch.nn as nn


class GatedCrossAttention(nn.Module):
    """
    Cross-attention with an extra R branch to gate V:
        Q = Wq(q)
        K = Wk(x)
        V = Wv(x)
        R = Wr(x)
        V_eff = sigmoid(R) ⊙ V
        Attn(q, x) = softmax(QK^T / sqrt(dk)) V_eff

    Shapes:
        q: [B, 1, D]
        x: [B, L, D]  (here L=3)
        out: [B, 1, D]
        attn: [B, H, 1, L]
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wr = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, H, L, Dh]
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, Dh] -> [B, L, D]
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self.Wq(q)          # [B,1,D]
        K = self.Wk(x)          # [B,L,D]
        V = self.Wv(x)          # [B,L,D]
        R = self.Wr(x)          # [B,L,D]

        V_eff = torch.sigmoid(R) * V  # [B,L,D]

        Qh = self._split_heads(Q)     # [B,H,1,Dh]
        Kh = self._split_heads(K)     # [B,H,L,Dh]
        Vh = self._split_heads(V_eff) # [B,H,L,Dh]

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,1,L]
        attn = torch.softmax(scores, dim=-1)                                      # [B,H,1,L]
        attn = self.dropout(attn)

        out = torch.matmul(attn, Vh)   # [B,H,1,Dh]
        out = self._merge_heads(out)   # [B,1,D]
        out = self.out_proj(out)       # [B,1,D]

        out = self.norm(q + out)       # residual + norm
        return out, attn


class CMAALayer(nn.Module):
    """
    One CMAA interaction layer:
        Input tokens: [B,4,D]
        For each modality i:
            q = tokens[:,i] as query [B,1,D]
            ctx = concat(other 3)      [B,3,D]
            y_i = GatedCrossAttention(q, ctx)  -> [B,1,D]
        Output tokens: Y = concat(y_1..y_4) -> [B,4,D]
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, share_gca: bool = True):
        super().__init__()
        if share_gca:
            self.gca_blocks = nn.ModuleList([GatedCrossAttention(d_model, num_heads, dropout)])
            self.share_gca = True
        else:
            self.gca_blocks = nn.ModuleList([GatedCrossAttention(d_model, num_heads, dropout) for _ in range(4)])
            self.share_gca = False

    @staticmethod
    def _stack_others(tokens: torch.Tensor, idx: int) -> torch.Tensor:
        # tokens: [B,4,D] -> [B,3,D]
        return torch.cat([tokens[:, :idx, :], tokens[:, idx + 1 :, :]], dim=1)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # tokens: [B,4,D]
        outs: List[torch.Tensor] = []
        attns: List[torch.Tensor] = []

        for i in range(4):
            q = tokens[:, i:i+1, :]         # [B,1,D]
            ctx = self._stack_others(tokens, i)  # [B,3,D]

            gca = self.gca_blocks[0] if self.share_gca else self.gca_blocks[i]
            y, attn = gca(q=q, x=ctx)
            outs.append(y)
            attns.append(attn)

        Y = torch.cat(outs, dim=1)  # [B,4,D]

        aux = {
            "attn_ct_query": attns[0],
            "attn_fs_query": attns[1],
            "attn_tab_query": attns[2],
            "attn_post_query": attns[3],
        }
        return Y, aux


class CMAA(nn.Module):

    def __init__(
        self,

        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
        share_gca: bool = True,
        weight_mode: str = "mlp",      # "mlp" or "dot"
    ):
        super().__init__()
        self.d = 768
        self.d_model = d_model
        self.weight_mode = weight_mode

        self.proj_ct = nn.Sequential(nn.Linear(self.d, d_model), nn.LayerNorm(d_model))
        self.proj_fs = nn.Sequential(nn.Linear(self.d, d_model), nn.LayerNorm(d_model))
        self.proj_tab = nn.Sequential(nn.Linear(self.d, d_model), nn.LayerNorm(d_model))
        self.proj_post = nn.Sequential(nn.Linear(self.d, d_model), nn.LayerNorm(d_model))

        self.layers = nn.ModuleList([
            CMAALayer(d_model=d_model, num_heads=num_heads, dropout=dropout, share_gca=share_gca)
            for _ in range(num_layers)
        ])

        # Dynamic weighting over 4 outputs
        if weight_mode == "mlp":
            self.weight_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.weight_vec = nn.Parameter(torch.randn(d_model))

        self.out_norm = nn.LayerNorm(d_model)

    def _compute_weights(self, Y: torch.Tensor) -> torch.Tensor:
        # Y: [B,4,D] -> w: [B,4]
        if self.weight_mode == "mlp":
            logits = self.weight_mlp(Y).squeeze(-1)           # [B,4]
        else:
            logits = torch.einsum("bld,d->bl", Y, self.weight_vec)  # [B,4]
        return torch.softmax(logits, dim=1)

    from typing import Dict
    import torch
    from torch import Tensor

    def forward(self, feats: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # ---- check keys ----
        required = ["INTRA", "POST", "CT", "TAB"]
        missing = [k for k in required if k not in feats]
        if missing:
            raise KeyError(f"CMAA.forward() missing keys: {missing}, got={list(feats.keys())}")

        z_intra = feats["INTRA"]
        z_post = feats["POST"]
        z_ct = feats["CT"]
        z_tab = feats["TAB"]

        # ---- project -> tokens [B,4,d_model] ----
        ct = self.proj_ct(z_ct)
        intra = self.proj_fs(z_intra)
        tab = self.proj_tab(z_tab)
        post = self.proj_post(z_post)

        tokens = torch.stack([intra, post, ct, tab], dim=1)  # [B,4,Dm]

        # ---- repeated interaction ----
        layer_attns = []
        for layer in self.layers:
            tokens, aux_layer = layer(tokens)
            layer_attns.append(aux_layer)

        # ---- dynamic weighted sum ----
        w = self._compute_weights(tokens)  # [B,4]
        fused = torch.einsum("bld,bl->bd", tokens, w)  # [B,Dm]
        fused = self.out_norm(fused)

        return {
            "fusion_feature": fused,
        }

