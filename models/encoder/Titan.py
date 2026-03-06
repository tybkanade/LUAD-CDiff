import h5py
import torch
import torch.nn as nn

from models.common.common import LinearHead

local_titan = "path of TITAN"
from transformers import AutoModel


class TitanClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = self.from_pretrained()
        self.head = LinearHead(in_dim=768, hidden_dim=512, out_dim=num_classes)
        if freeze_backbone:
            self.freeze_backbone()

    def from_pretrained(self):
        """从本地目录加载 TITAN backbone"""
        backbone = AutoModel.from_pretrained(
            local_titan,
            trust_remote_code=True,
            local_files_only=True
        )
        return backbone

    # ====== 冻结/解冻 ======
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()

    @torch.no_grad()
    def encode(self, features: torch.Tensor, coords: torch.Tensor, patch_size_lv0: int) -> torch.Tensor:
        """
        返回 slide 级别的 embedding: [1, 768]
        """
        emb = self.backbone.encode_slide_from_patch_features(features, coords, patch_size_lv0)
        return emb

    def forward(self, features, coords, patch_size_lv0):

        if isinstance(features, torch.Tensor):
            emb = self.backbone.encode_slide_from_patch_features(features, coords, patch_size_lv0)  # [1,768]
            logits = self.head(emb)  # [1, num_classes]
            return logits

        logits_list = []
        for f, c, ps in zip(features, coords, patch_size_lv0):
            emb = self.backbone.encode_slide_from_patch_features(f, c, int(ps))  # [1,768]
            logits_list.append(self.head(emb))  # [1, C]
        return torch.cat(logits_list, dim=0)  # [B, C]

if __name__ == "__main__":
    titan = TitanClassifier(num_classes=4, freeze_backbone=True)
    h5_path = 'path of h5'
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:])
        coords = torch.from_numpy(f['coords'][:])
        patch_size_lv0 = int(f['coords'].attrs['patch_size_level0'])
        ps_lv0 = int(f['coords'].attrs['patch_size_level0'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    titan = titan.to(device).eval()

    with torch.autocast(device_type=device, dtype=torch.float16), torch.inference_mode():
        logits = titan(features.to(device), coords.to(device), patch_size_lv0)
        print("logits:", logits.shape)
