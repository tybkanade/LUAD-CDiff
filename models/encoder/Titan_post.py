import h5py
import torch
import torch.nn as nn
from transformers import AutoModel

from models.common.agg import MaxPoolingAggregator, WeightedAverageAggregator, GatedAttentionAggregator, \
    HybridAggregator, newHybridAggregator, EnhancedHybridAggregator
from models.common.common import LinearHead

local_titan = "path of TITAN"

class TitanWithAggregator(nn.Module):

    def __init__(self, agg: str = "new_hybrid", freeze_backbone: bool = True, norm_out: bool = True):
        super().__init__()
        self.backbone = self.from_pretrained()
        if freeze_backbone:
            self.freeze_backbone()
        self.norm = nn.LayerNorm(768) if norm_out else nn.Identity()
        self.head = LinearHead(768)

        # 聚合方法
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
        else:
            raise ValueError("Invalid aggregator type.")

    def from_pretrained(self):
        backbone = AutoModel.from_pretrained(
            local_titan,
            trust_remote_code=True,
            local_files_only=True
        )
        return backbone

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def encode_local(self, features: torch.Tensor, coords: torch.Tensor, patch_size_lv0: int) -> torch.Tensor:
        """
        用冻结的 TITAN 把一个局部（M×768）编码为 [1,768]
        """
        emb = self.backbone.encode_slide_from_patch_features(features, coords, patch_size_lv0)  # [1,768]
        return emb  # [1,768]

    def forward(self, features, coords, patch_size_lv0):

        if isinstance(features, torch.Tensor):
            # 直接编码成单个局部的 1×768
            local_emb = self.encode_local(features, coords, patch_size_lv0)     # [1,768]
            return self.head(self.norm(local_emb))

        if isinstance(features, (list, tuple)) and isinstance(features[0], torch.Tensor):
            # 对每个局部跑 TITAN，得到 [N,768]
            local_embs = []
            for f, c, ps in zip(features, coords, patch_size_lv0):
                e = self.encode_local(f, c, int(ps))                            # [1,768]
                local_embs.append(e)
            E = torch.cat(local_embs, dim=0)                        # [N,768]
            # 聚合 -> [1,768]
            z = self.aggregator(E)

            z_logit = self.head(z)              # [1,768]
            return z_logit, z

        return None

def read_h5_one_local(h5_path: str, device=None):
    """
    返回：features(torch.FloatTensor[M,768]), coords(torch.LongTensor[M,2]), patch_size_lv0(int)
    """
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:])          # [M,768]
        coords = torch.from_numpy(f['coords'][:])              # [M,2]
        ps_lv0 = int(f['coords'].attrs['patch_size_level0'])
    if device is not None:
        features = features.to(device)
        coords = coords.to(device)
    return features, coords, ps_lv0

