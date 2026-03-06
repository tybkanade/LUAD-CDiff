import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPoolingAggregator(nn.Module):
    """
    使用最大池化来聚合局部特征。
    每个病例的最终特征由所有局部特征的最大值决定。
    """
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        对每个局部特征进行最大池化，得到最终的特征表示。
        """
        # feats 形状 [N, 768] -> 最大池化 [1, 768]
        # print(feats.shape)
        return torch.max(feats, dim=0, keepdim=True)[0]  # 对每个特征维度进行最大池化


class WeightedAverageAggregator(nn.Module):
    """
    使用加权平均池化来聚合局部特征。
    """
    def __init__(self, dim=768):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        计算每个局部特征的权重，并根据这些权重对特征进行加权平均。
        """
        # feats 形状 [N, 768]
        scores = self.fc(feats)  # [N, 1]，计算每个局部特征的权重

        # 使用 softmax 获取每个局部特征的权重 alpha
        alpha = torch.softmax(scores, dim=0)  # [N, 1]

        # 使用权重对特征进行加权平均
        weighted_feats = torch.sum(alpha * feats, dim=0)  # [768]

        return weighted_feats.unsqueeze(0)


class GatedAttentionAggregator(nn.Module):
    """
    Ilse 2018（Gated-Attention MIL）:
      h = tanh(V e) ⊙ sigmoid(U e)
      α = softmax(w^T h)
      z = Σ α_i e_i
    """
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        使用自注意力机制对每个局部特征进行加权平均。
        """
        # feats 形状 [N, 768]
        query = self.query(feats)  # [N, hidden_dim]
        key = self.key(feats)      # [N, hidden_dim]
        value = self.value(feats)  # [N, hidden_dim]

        attention_scores = torch.matmul(query, key.transpose(0, 1))  # [N, N]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [N, N]

        attended_feats = torch.matmul(attention_weights, value)  # [N, hidden_dim]
        aggregated_feats = self.fc(attended_feats)  # [N, 768]
        return aggregated_feats.mean(dim=0)  # [768]

class HybridAggregator(nn.Module):
    """
    使用混合聚合策略：最大池化 + 加权平均池化
    """
    def __init__(self, dim=768, hidden_dim=256, kernel_size=3):
        super().__init__()
        self.attention = WeightedAverageAggregator(dim=dim)
        self.maxpool = MaxPoolingAggregator(dim=dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        使用混合聚合策略：最大池化 + 加权平均
        """
        # 使用最大池化来选择最严重的局部特征
        max_feats = self.maxpool(feats)  # [1, 768]

        # 使用加权平均池化来综合局部特征
        weighted_feats = self.attention(feats)  # [1, 768]

        # 将最严重的特征和加权平均特征融合
        final_feats = max_feats + weighted_feats  # [1, 768]

        return final_feats.squeeze(0)  # 返回 [768]


class newHybridAggregator(nn.Module):
    """
    使用混合聚合策略：最大池化 + 加权平均池化
    """
    def __init__(self, dim=768, hidden_dim=256, kernel_size=3):
        super().__init__()
        self.attention = WeightedAverageAggregator(dim=dim)
        self.maxpool = MaxPoolingAggregator(dim=dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        使用混合聚合策略：最大池化 + 加权平均
        """
        # 使用最大池化来选择最严重的局部特征
        max_feats = self.maxpool(feats)  # [1, 768]
        # 使用加权平均池化来综合局部特征
        weighted_feats = self.attention(feats)  # [1, 768]
        # 将最严重的特征和加权平均特征融合
        final_feats = max_feats + weighted_feats  # [1, 768]
        return final_feats  # 返回 [768]


class EnhancedHybridAggregator(nn.Module):
    """
    使用混合聚合策略：最大池化 + 加权平均池化 + 自适应加权
    """

    def __init__(self, dim=768, hidden_dim=256, kernel_size=3):
        super().__init__()
        # 加权平均池化
        self.attention = WeightedAverageAggregator(dim=dim)
        # 最大池化
        self.maxpool = MaxPoolingAggregator(dim=dim)
        # 自适应加权
        self.fc_weight = nn.Linear(dim, 1)  # 用于计算每个局部特征的重要性权重

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        使用混合聚合策略：最大池化 + 加权平均 + 自适应加权
        """
        # 使用最大池化来选择最严重的局部特征
        max_feats = self.maxpool(feats)  # [1, 768]

        # 使用加权平均池化来综合局部特征
        weighted_feats = self.attention(feats)  # [1, 768]

        # 计算每个特征的重要性权重（自适应加权）
        weights = torch.sigmoid(self.fc_weight(feats))  # [N, 1]，通过 sigmoid 限制权重在 [0, 1] 范围
        weighted_feats_adaptive = torch.sum(weights * feats, dim=0)  # [768]，通过权重对特征加权

        # 将最严重的特征、加权平均特征和自适应加权特征融合
        final_feats = max_feats + weighted_feats + weighted_feats_adaptive  # [1, 768]

        return final_feats  # 返回 [768]


class TransMILLiteGatedTopKAggregator(nn.Module):
    """
    TransMIL-Lite + Gated Top-K：
    适合 N 很小（只有少数图像有病变信息）
    """
    def __init__(self,
                 dim=768,
                 local_hidden=256,
                 num_heads=4,
                 num_layers=2,
                 top_k=2):
        super().__init__()
        self.dim = dim
        self.top_k = top_k

        # Local MIL
        self.V = nn.Linear(dim, local_hidden)
        self.U = nn.Linear(dim, local_hidden)
        self.w = nn.Linear(local_hidden, 1, bias=False)

        # Global transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim*4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

        # Gated Top-K
        self.score_head = nn.Linear(dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, feats):
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        B, N, D = feats.shape

        # ===== Local MIL =====
        H = torch.tanh(self.V(feats)) * torch.sigmoid(self.U(feats))
        local_scores = self.w(H).squeeze(-1)
        local_attn = torch.softmax(local_scores, dim=1)
        feats_local = torch.bmm(local_attn.unsqueeze(1), feats).squeeze(1)
        feats = feats + feats_local.unsqueeze(1)

        # ===== Global =====
        feats = self.global_encoder(feats)
        feats = self.norm(feats)

        # ===== Gated Top-K =====
        scores = self.score_head(feats).squeeze(-1)
        k = min(self.top_k, N)
        _, topk_idx = torch.topk(scores, k, dim=-1)
        feats_k = torch.gather(feats, 1, topk_idx.unsqueeze(-1).repeat(1, 1, D))

        # gate 控制每个 token 的贡献
        gated = feats_k * self.gate(feats_k)
        z = gated.mean(dim=1)
        return z

class TransMILGraphAggregator(nn.Module):
    """
    TransMIL + Graph Attention
    完整支持 N=1,2,3,... 任意长度
    """
    def __init__(self, dim=768, k=3, num_layers=2, num_heads=4):
        super().__init__()
        self.dim = dim
        self.k = k

        # Graph Attention
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        # Global Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

        self.score_head = nn.Linear(dim, 1)

    def forward(self, feats: torch.Tensor):
        """
        feats: [N, D] 或 [B, N, D]
        return: [B, D]
        """

        # ---- Step 0: 标准化输入 ----
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # [1, N, D]

        B, N, D = feats.shape
        device = feats.device

        # ---- Step 1: Linear proj (Q,K,V)
        Q = self.q(feats)  # [B,N,D]
        KV = self.kv(feats)  # [B,N,2D]
        Kmat, Vmat = KV.chunk(2, dim=-1)  # each: [B,N,D]

        # ---- Step 2: 计算 Cos 距离（注意 small eps）
        norm_q = F.normalize(Q, dim=-1)  # [B,N,D]
        norm_k = F.normalize(Kmat, dim=-1)  # [B,N,D]
        sim = torch.matmul(norm_q, norm_k.transpose(1, 2))  # [B,N,N]
        dist = 1.0 - sim  # [B,N,N]

        # ---- Step 3: 选择 K 个最近邻（排除自己）
        k_eff = min(N - 1, self.k)
        dist = dist + torch.eye(N, device=device).unsqueeze(0) * 1e6  # 遮掉自己

        _, idx = torch.topk(dist, k_eff, dim=-1, largest=False)  # [B,N,k_eff]
        # idx[b, i] = i 的 k 个最近邻 index

        # ---- Step 4: 构造 gather index
        # Kmat: [B,N,D] → [B,1,N,D] → expand → [B,N,N,D]
        Kmat_exp = Kmat.unsqueeze(1).expand(B, N, N, D)
        Vmat_exp = Vmat.unsqueeze(1).expand(B, N, N, D)

        # idx: [B,N,k] → [B,N,k,1] → expand → [B,N,k,D]
        idx_exp = idx.unsqueeze(-1).expand(B, N, k_eff, D)

        # ---- Step 5: gather 取邻居特征
        K_neighbors = torch.gather(Kmat_exp, 2, idx_exp)  # [B,N,k,D]
        V_neighbors = torch.gather(Vmat_exp, 2, idx_exp)  # [B,N,k,D]

        # ---- Step 6: Graph Attention：Q·K_neighbor
        # Q: [B,N,1,D] ; K_neighbor: [B,N,k,D]
        q_exp = Q.unsqueeze(2)  # [B,N,1,D]
        attn = torch.sum(q_exp * K_neighbors, dim=-1)  # [B,N,k]

        # softmax over k neighbors
        attn = torch.softmax(attn / (D ** 0.5), dim=-1)  # [B,N,k]

        # ---- Step 7: 加权聚合邻居特征
        attn_exp = attn.unsqueeze(-1)  # [B,N,k,1]
        out = torch.sum(attn_exp * V_neighbors, dim=2)  # [B,N,D]

        # ---- Step 8: 用 mean pooling 聚合为病例级特征
        z = out.mean(dim=1)  # [B,D]

        return z


class TransMILMultiScaleAggregator(nn.Module):
    """
    多尺度 MIL：Global + Local + Coarse-to-Fine
    """
    def __init__(self, dim=768, scales=[1, 2, 4], num_heads=4):
        super().__init__()

        self.scales = scales
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim*4,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=1
            )
            for _ in scales
        ])

        self.score_heads = nn.ModuleList([
            nn.Linear(dim, 1) for _ in scales
        ])

        self.fuse = nn.Linear(dim * len(scales), dim)

    def forward(self, feats):
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        B, N, D = feats.shape

        outputs = []
        for s, enc, head in zip(self.scales, self.encoders, self.score_heads):
            step = max(1, N // s)
            idx = torch.arange(0, N, step, device=feats.device)
            sub = feats[:, idx]   # downsampled
            sub = enc(sub)
            scores = torch.softmax(head(sub).squeeze(-1), dim=-1)
            pooled = torch.bmm(scores.unsqueeze(1), sub).squeeze(1)
            outputs.append(pooled)

        z = torch.cat(outputs, dim=-1)
        z = self.fuse(z)
        return z