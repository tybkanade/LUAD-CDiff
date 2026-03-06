import torch.nn.functional as F
import torch
import torch.nn as nn
import math

from models.common.common import WeightedModalAttention


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, scale: float = 1.0, use_2pi: bool = False):
        super().__init__()
        assert dim >= 2
        self.dim = dim
        self.base = float(base)
        self.scale = float(scale)
        self.use_2pi = use_2pi

        half = dim // 2
        denom = max(half - 1, 1)
        exponents = torch.arange(half, dtype=torch.float32) * (-math.log(self.base) / denom)
        freqs = torch.exp(exponents)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor):
        t = t.float().view(-1, 1) * self.scale
        if self.use_2pi:
            t = t * (2 * math.pi)
        args = t * self.freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)   # [B,1] or [B,1,1]
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * mask


class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads, mlp_ratio=4.0, dropout=0.0, droppath=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path_attn = DropPath(droppath)

        self.norm2 = nn.LayerNorm(hidden)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden),
            nn.Dropout(dropout),
        )
        self.drop_path_mlp = DropPath(droppath)

    def forward(self, x):
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path_attn(attn)

        h2 = self.norm2(x)
        h2 = self.mlp(h2)
        x = x + self.drop_path_mlp(h2)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.0, droppath=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_k = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path_attn = DropPath(droppath)

        self.norm_out = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.drop_path_ffn = DropPath(droppath)

    def forward(self, x, cond_tokens):
        q = self.norm_q(x)
        k = self.norm_k(cond_tokens)
        v = k

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = x + self.drop_path_attn(attn_out)

        h = self.norm_out(x)
        h = self.ffn(h)
        x = x + self.drop_path_ffn(h)
        return x


class CondDenoiser_Prior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.m = cfg.m  # x_t 向量维度
        self.hidden = cfg.hidden
        self.depth = cfg.depth

        # ===============================
        # 1) 时间 embedding
        # ===============================
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(cfg.time_emb_dim),
            nn.Linear(cfg.time_emb_dim, cfg.time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.time_emb_dim * 2, self.hidden),
        )

        # ===============================
        # 2) x_t 输入映射
        # ===============================
        self.x_in = nn.Linear(cfg.m, self.hidden)

        # ===============================
        # 3) 多条件 token 输入
        #    cond_dim 现在是“每个 token 的维度”
        # ===============================
        self.cond_project = nn.Linear(cfg.cond_dim, self.hidden)

        # ===============================
        # 4) FiLM：只基于融合条件
        # ===============================
        self.film_scale = nn.Linear(self.hidden, self.hidden)
        self.film_shift = nn.Linear(self.hidden, self.hidden)

        self.modal_attn = WeightedModalAttention(
            dim=self.hidden,
            weights=(0.4, 0.4, 0.2)  # 初始权重可以自定义
        )

        # ===============================
        # 5) 可选 cls token
        # ===============================
        self.use_cls = (cfg.cls_num is not None and cfg.cls_num > 0)
        if self.use_cls:
            self.cls_emb = nn.Embedding(cfg.cls_num, self.hidden)

        self.cond_norm = nn.LayerNorm(self.hidden)

        # ===============================
        # 6) 主干 Transformer
        # ===============================
        self.self_blocks = nn.ModuleList([
            TransformerBlock(
                hidden=self.hidden,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                droppath=cfg.droppath * (i + 1) / cfg.depth,
            )
            for i in range(cfg.depth)
        ])

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                hidden=self.hidden,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                droppath=cfg.droppath * (i + 1) / cfg.depth,
            )
            for i in range(cfg.depth)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, cfg.m),
        )
        nn.init.xavier_uniform_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)


    def forward(self, x_t, t, cond_tokens, cls=None):
        """
        x_t:    [B, m]
        cond_tokens: List[Tensor] 或 Tensor [B, L, cond_dim]
                     例如：
                     [
                        ct_vec,     # [B, cond_dim]
                        frozen_vec,
                        table_vec,
                        fusion_vec
                     ]
        """
        # print(cls.shape)

        # -------- x_t reshape --------
        if x_t.dim() == 3 and x_t.size(1) == 1:
            x_t = x_t.squeeze(1)

        # -------- 1) 编码输入 x_t --------
        h = self.x_in(x_t)  # [B,H]

        # -------- 2) 多条件 token 处理 --------
        if isinstance(cond_tokens, list):
            cond_tokens = torch.stack(cond_tokens, dim=1)  # [B,L,cond_dim]

        B, L, D = cond_tokens.shape

        # 每个 token 单独投影到 hidden
        cond_tokens = self.cond_project(cond_tokens)  # [B,L,H]

        # -------- 3) FiLM (基于融合 token 或平均) --------
        # fusion_vec = cond_tokens.mean(dim=1)  # [B,H]
        CT_tok = cond_tokens[:, 0, :]  # [B, H]
        WSI_tok = cond_tokens[:, 1, :]
        Table_tok = cond_tokens[:, 2, :]
        fusion_vec = self.modal_attn([CT_tok, WSI_tok, Table_tok])  # [B, H]
        scale = self.film_scale(fusion_vec)
        shift = self.film_shift(fusion_vec)
        h = h * (scale + 1) + shift

        # -------- 4) 时间 embedding 注入 h --------
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.full((B,), int(t), dtype=torch.long, device=x_t.device)

        t_tok = self.time_emb(t)  # [B,H]
        h = h + t_tok

        # -------- 5) 可选 cls token --------
        if cls is not None:
            cls = cls.long()  # 保证 dtype
            cls_tok = self.cls_emb(cls)  # [B, m]
            cls_tok = cls_tok.unsqueeze(1)  # [B, 1, m]

            cond_tokens = torch.cat([cond_tokens, cls_tok], dim=1)

        cond_tokens = self.cond_norm(cond_tokens)  # [B, L(+1), H]

        # -------- 6) 扩展 h 为 seq --------
        h = h.unsqueeze(1)  # [B,1,H]

        # -------- 7) 主干 transformer --------
        for blk_self, blk_cross in zip(self.self_blocks, self.cross_blocks):
            h = blk_self(h)
            h = blk_cross(h, cond_tokens)

        h = h.squeeze(1)

        # -------- 8) 输出 eps --------
        return self.out(h)
