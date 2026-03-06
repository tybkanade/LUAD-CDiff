# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn



class CrossModalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        return: [B, N, D]
        """
        return self.encoder(x)


class DynamicModalWeightFusionN(nn.Module):
    """
    输入:
        tokens: [B, N, D]
    输出:
        z: [B, D]
        alpha: [B, N]
    """
    def __init__(self, d_model: int, hidden: Optional[int] = None):
        super().__init__()
        if hidden is None:
            hidden = max(1, d_model // 2)

        # 对每个 token 产生一个标量 score（共享参数，支持 N 变化）
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # init
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: [B, N, D]
        """
        scores = self.scorer(tokens).squeeze(-1)     # [B, N]
        alpha = torch.softmax(scores, dim=-1)        # [B, N]
        z = torch.sum(tokens * alpha.unsqueeze(-1), dim=1)  # [B, D]
        return z, alpha


class MultiModalFusionModule(nn.Module):
    """
    用 dict 输入 N 个模态，支持每个模态不同输入维度：
        feats = {"CT": [B, d_ct], "TAB": [B, d_tab], ...}

    内部会:
      1) 每模态投影到统一 d_model
      2) stack 成 tokens: [B, N, d_model]
      3) Cross-Modal Transformer 交互
      4) 动态权重融合得到 [B, d_model]
    """
    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_model: int,
        dropout: float = 0.1,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        fusion_hidden: Optional[int] = None,
        use_layernorm_on_token: bool = True,
    ):
        super().__init__()
        self.modality_dims = dict(modality_dims)
        self.d_model = d_model
        self.token_names = list(self.modality_dims.keys())

        # 每个模态一个投影层：in_dim -> d_model
        self.proj = nn.ModuleDict()
        for name, in_dim in self.modality_dims.items():
            self.proj[name] = nn.Linear(in_dim, d_model)

        self.token_ln = nn.LayerNorm(d_model) if use_layernorm_on_token else nn.Identity()

        self.cmt = CrossModalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward or d_model * 4,
            dropout=dropout,
        )

        self.fuser = DynamicModalWeightFusionN(d_model=d_model, hidden=fusion_hidden)

    def forward(self, feats: Dict[str, torch.Tensor]):

        tokens_list: List[torch.Tensor] = []
        missing = [n for n in self.token_names if n not in feats]
        if missing:
            raise KeyError(f"Missing modalities in feats: {missing}. Got keys={list(feats.keys())}")

        for name in self.token_names:
            x = feats[name]
            tok = self.proj[name](x)          # [B, D]
            tok = self.token_ln(tok)
            tokens_list.append(tok)

        tokens_before = torch.stack(tokens_list, dim=1)  # [B, N, D]
        tokens_after = self.cmt(tokens_before)           # [B, N, D]
        z_fuse, alpha = self.fuser(tokens_after)         # z: [B,D], alpha: [B,N]

        tokens_after_dict = {
            name: tokens_after[:, i, :]
            for i, name in enumerate(self.token_names)
        }

        return {
            "fusion_feature": z_fuse,
            "modal_alpha" : alpha,
            "token_names" : self.token_names,
            "tokens_before_cmt" : tokens_before,
            "tokens_after_cmt" : tokens_after,
            "tokens_after_cmt_dict" : tokens_after_dict,
        }

