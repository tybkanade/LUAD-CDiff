import torch
import torch.nn as nn
from typing import List, Optional
from config.configs import TableEncoderConfig

# ---------- activations ----------
ACT_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "none": nn.Identity,
}

# ---------- numeric norm ----------
class FeatureNorm(nn.Module):
    """对纯数值列做可学习标准化：y = (x - mu) / (|sigma| + eps) * gamma + beta"""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(d))
        self.sigma = nn.Parameter(torch.ones(d))
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps
    def forward(self, x):  # [B, d]
        z = (x - self.mu) / (self.sigma.abs() + self.eps)
        return z * self.gamma + self.beta

# ---------- residual MLP ----------
class ResMLPBlock(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.1, act="silu"):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.act = ACT_MAP[act.lower()]()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, d)
    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h

# ---------- categorical as tokens ----------
class CatTokenEmbed(nn.Module):
    """每个类别列一个 Embedding，输出 [B, n_cat, emb_dim]"""
    def __init__(self, cardinals: List[int], emb_dim: int, use_col_embedding: bool = True):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinals])
        self.col_embed = nn.Parameter(torch.zeros(len(cardinals), emb_dim)) if use_col_embedding else None
        # nn.init.normal_(self.embs[0].weight, mean=0.0, std=0.02) if self.embs else None
        for emb in self.embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat: torch.Tensor):  # [B, n_cat] (long)
        if x_cat.dtype != torch.long:
            raise TypeError(f"x_cat dtype must be LongTensor, got {x_cat.dtype}")
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]  # list of [B, emb]
        x = torch.stack(tokens, dim=1)  # [B, n_cat, emb_dim]
        if self.col_embed is not None:
            x = x + self.col_embed.unsqueeze(0)
        return x

# ---------- main encoder ----------
class TableEncoder(nn.Module):
    def __init__(self, config: TableEncoderConfig):
        super().__init__()
        self.cfg = config
        self.feature_extraction = config.feature_extraction

        # flags
        self.has_num = (config.num_features is not None) and (config.num_features > 0)
        self.has_cat = (config.cat_cardinalities is not None) and (len(config.cat_cardinalities) > 0)

        act_cls = ACT_MAP[config.activation.lower()]

        # ---- categorical branch (tokens + transformer) ----
        if self.has_cat:
            assert (config.emb_dim % config.n_heads) == 0, \
                f"emb_dim ({config.emb_dim}) must be divisible by n_heads ({config.n_heads})"
            self.cat_tok = CatTokenEmbed(config.cat_cardinalities, config.emb_dim, use_col_embedding=True)
            self.use_cls_token = True  # 这里固定用 CLS，效果更稳
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.emb_dim))

            # torch.nn.TransformerEncoder 只支持 relu/gelu 作为 activation 字符串
            attn_activation = "gelu" if config.activation.lower() not in ("relu", "gelu") else config.activation.lower()
            enc_layer = nn.TransformerEncoderLayer(
                d_model=config.emb_dim,
                nhead=config.n_heads,
                dim_feedforward=max(256, 2 * config.emb_dim),
                dropout=config.dropout,
                activation=attn_activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.n_attn_layers)
            self.cat_post_norm = nn.LayerNorm(config.emb_dim)
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.cat_tok = None
            self.use_cls_token = False
            self.cls_token = None
            self.transformer = None
            self.cat_post_norm = None

        # ---- numeric branch ----
        if self.has_num:
            self.num_norm = FeatureNorm(config.num_features)
            self.num_proj = nn.Sequential(
                nn.Linear(config.num_features, max(32, config.hidden_dims[0] // 2)),
                act_cls(),
            )
            num_out_dim = max(32, config.hidden_dims[0] // 2)
        else:
            self.num_norm = None
            self.num_proj = None
            num_out_dim = 0

        # ---- fuse & MLP backbone ----
        cat_out_dim = config.emb_dim if self.has_cat else 0
        fused_dim = cat_out_dim + num_out_dim
        if fused_dim == 0:
            raise ValueError("No inputs configured: both num_features=0 and cat_cardinalities=None.")

        self.stem = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dims[0]),
            act_cls(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
        )

        d = config.hidden_dims[0]
        self.backbone = nn.Sequential(
            ResMLPBlock(d=d, hidden=max(d, 4*d//3), dropout=config.dropout, act=config.activation),
            ResMLPBlock(d=d, hidden=max(d, 4*d//3), dropout=config.dropout, act=config.activation),
        )

        tail = []
        for i in range(1, len(config.hidden_dims)):
            tail += [nn.LayerNorm(d), nn.Linear(d, config.hidden_dims[i]), act_cls()]
            if config.dropout > 0:
                tail += [nn.Dropout(config.dropout)]
            d = config.hidden_dims[i]
        self.tail = nn.Sequential(*tail) if tail else nn.Identity()

        self.head_norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, config.out_dim)
        self.cls = nn.Linear(config.out_dim, config.n_classes)

        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None
    ):
        parts = []

        # --- categorical path ---
        if self.has_cat:
            if x_cat is None:
                raise ValueError("Configured cat_cardinalities but got x_cat=None.")
            cat_tokens = self.cat_tok(x_cat)  # [B, n_cat, emb_dim]
            if self.use_cls_token:
                cls = self.cls_token.expand(cat_tokens.size(0), -1, -1)  # [B,1,emb]
                cat_tokens = torch.cat([cls, cat_tokens], dim=1)         # [B,1+n_cat,emb]
            cat_tokens = self.transformer(cat_tokens)                    # [B, L, emb]
            cat_tokens = self.cat_post_norm(cat_tokens)
            cat_repr = cat_tokens[:, 0, :] if self.use_cls_token else cat_tokens.mean(dim=1)
            parts.append(cat_repr)

        # --- numeric path ---
        if self.has_num:
            if x_num is None:
                raise ValueError("Configured num_features>0 but got x_num=None.")
            if missing_mask is not None:
                if missing_mask.shape != x_num.shape:
                    raise ValueError(f"missing_mask shape {missing_mask.shape} != x_num shape {x_num.shape}")
            x = self.num_norm(x_num)
            if missing_mask is not None:
                x = x * (1.0 - missing_mask.float())
            parts.append(self.num_proj(x))

        h = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)  # [B, fused]
        h = self.stem(h)
        h = self.backbone(h)
        h = self.tail(h)
        h = self.head_norm(h)
        feats = self.proj(h)
        return feats if self.feature_extraction else self.cls(feats)


if __name__ == "__main__":
    NUM_COLS = ["年龄", "长径cm", "短径cm", "实性成分长径"]
    BINARY_COLS = [
        "圆形、类圆形", "不规则", "分叶", "结节征", "空泡", "空洞",
        "钙化", "支气管征", "清楚", "光滑锐利", "毛刺", "尖角、桃尖",
        "索条", "血管集束征", "胸膜凹陷"
    ]
    PLEURA_COL = "与胸膜的关系"
    GENDER_COL = "性别"
    DENSITY_COL = "密度"

    CLASS_NAMES = ["AAH/AIS", "MIA", "AC"]
    cfg = TableEncoderConfig(
        num_features=len(NUM_COLS),
        cat_cardinalities=[4, 6, 3],
        emb_dim=32, hidden_dims=(256, 256),
        activation="relu", dropout=0.1,
        use_attention=True, n_heads=4, n_attn_layers=4,
        feature_extraction=True,
    )

    model = TableEncoder(cfg)
    x_num = torch.randn(2, 4)
    x_cat = torch.cat([
        torch.randint(0, 2, (2, 1)),
        torch.randint(0, 2, (2, 1)),
        torch.randint(0, 2, (2, 1)),
    ], dim=1)

    y = model(x_num, x_cat)
    print(y.shape)  # [16, 3]
