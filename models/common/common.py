import torch.nn as nn
import torch
from .agg import *


def _proj_block(in_dim: int, out_dim: int):
    # 线性 -> LN -> SiLU，初始化更稳
    blk = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.SiLU(),
    )
    nn.init.xavier_uniform_(blk[0].weight)
    nn.init.zeros_(blk[0].bias)
    return blk

# ------------------------------
# 线性分类头 768 -> 3
# ------------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=3, dropout_rate=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x.float())   # [B, in_dim] -> [B, hidden_dim]
        x = self.relu(x)          # [B, hidden_dim] -> [B, hidden_dim]
        x = self.dropout(x)       # [B, hidden_dim] -> [B, hidden_dim]
        x = self.fc2(x)           # [B, hidden_dim] -> [B, out_dim]
        return x


class TitanWithAggregator(nn.Module):
    """
    处理预计算好的 N 个 768 维特征（每个样本为 N * 768）。
    """

    def __init__(self, agg: str = "max_pooling", norm_out: bool = True):
        super().__init__()
        # 选择不同的聚合方法
        if agg == "max_pooling":
            self.aggregator = MaxPoolingAggregator(dim=768)
        elif agg == "weighted_avg":
            self.aggregator = WeightedAverageAggregator(dim=768)
        elif agg == "gated_attention":
            self.aggregator = GatedAttentionAggregator(dim=768)
        elif agg == "hybrid":
            self.aggregator = HybridAggregator(dim=768)
        elif agg == "new_hybrid":
            self.aggregator = newHybridAggregator(dim=768)
        elif agg == "enhanced_hybrid":
            self.aggregator = EnhancedHybridAggregator(dim=768)

        elif agg == "transmil_lite_gated":
            self.aggregator = TransMILLiteGatedTopKAggregator()

        elif agg == "transmil_graph":
            self.aggregator = TransMILGraphAggregator()

        elif agg == "transmil_multiscale":
            self.aggregator = TransMILMultiScaleAggregator()
        else:
            raise ValueError("Invalid aggregator type. Choose 'max_pooling' or 'weighted_avg'.")

        self.head = LinearHead(in_dim=768, out_dim=3)

    def forward(self, features):
        # 特征输入为 [B, N, 768]，即每个样本有 N 个 768 维的特征
        if isinstance(features, (list, tuple)) and isinstance(features[0], torch.Tensor):
            local_embs = []
            for f in features:
                # 通过 unsqueeze 扩展每个局部特征的维度为 [1, 768]，以便于后续处理
                e = f.unsqueeze(0)  # f 变为 [1, 768]
                # print(e.shape)  # 查看每个局部特征的维度
                local_embs.append(e)
            E = torch.cat(local_embs, dim=0)  # [N, 768]，按第一个维度拼接
            # print('E', E.shape)
            z = self.aggregator(E)  # [1, 768]，聚合
            # print('z', z.shape)
            z_logit = self.head(z)  # 聚合后的特征通过线性分类头
            # print('z', z.shape)
            return z_logit, z

        return None


class Classifier(nn.Module):
    def __init__(self, tri_dim: int, num_classes: int, dropout: float = 0.1, hidden_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(tri_dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(tri_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        for m_ in self.net:
            if isinstance(m_, nn.Linear):
                nn.init.xavier_uniform_(m_.weight)
                nn.init.zeros_(m_.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


class WeightedModalAttention(nn.Module):
    def __init__(self, dim, weights=(0.5, 0.4, 0.1)):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        self.proj = nn.Linear(dim, dim)

    def forward(self, feats):
        w = F.softmax(self.weights, dim=0)  # shape (3,)
        # 分配权重融合
        fused = (
                w[0] * feats[0] +
                w[1] * feats[1] +
                w[2] * feats[2]
        )
        return self.proj(fused)  # [B, dim]

