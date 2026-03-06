# 你现有 import 保持不变
from typing import Optional
import torch
from torch import nn

from models.common.common import TitanWithAggregator, LinearHead, _proj_block
from models.encoder.ctEncoder import CtEncoder
from models.encoder.tableEncoder import TableEncoder
from models.fusion.CMAA import CMAA
from models.fusion.DynamicFusion import MultiModalFusionModule
from models.triple_fusion import Classifier
from utils.utils import _load_pretrained_agg, _load_pretrained_ct_encoder, _load_pretrained_table_encoder

class Fusion4Modal(nn.Module):
    def __init__(
        self,
        ct_config,
        table_config,
        fusion_cfg,
        wsi_agg: str = "new_hybrid",
        tab_pretrained_path: Optional[str] = None,
        ct_pretrained_path: Optional[str] = None,
        agg_pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.cfg = fusion_cfg
        d = self.cfg.input_dim

        self.TitanAgg = TitanWithAggregator(agg=wsi_agg)
        if agg_pretrained_path:
            _load_pretrained_agg(self.TitanAgg.aggregator, agg_pretrained_path)

        self.CTEncoder = CtEncoder(config=ct_config)
        self.TableEncoder = TableEncoder(config=table_config)

        if ct_pretrained_path:
           _load_pretrained_ct_encoder(self.CTEncoder, ct_pretrained_path)
        if tab_pretrained_path:
            _load_pretrained_table_encoder(self.TableEncoder, tab_pretrained_path)

        modality_dims = {
            "INTRA": d,
            "POST":  d,
            "CT":    d,
            "TAB":   d,
        }

        self.fusion_module = MultiModalFusionModule(
            modality_dims=modality_dims,
            d_model=d,
            dropout=self.cfg.dropout,
            nhead=4,
            num_layers=2,
            dim_feedforward=d * 4,
            fusion_hidden=None,
        )
        # self.fusion_module = CMAA(
        #     d_model=d,
        #     num_heads=4,
        #     dropout=self.cfg.dropout,
        #     num_layers=2,
        #     share_gca=True,
        #     weight_mode="mlp",
        # )

        self.head_fuse = Classifier(
            tri_dim=d,
            num_classes=self.cfg.num_classes,
            dropout=self.cfg.dropout,
            hidden_ratio=self.cfg.classifier_hidden_ratio,
        )

        self.head_ct = LinearHead(in_dim=d, out_dim=self.cfg.num_classes)
        self.head_tab = LinearHead(in_dim=d, out_dim=self.cfg.num_classes)
        self.head_post = LinearHead(in_dim=d, out_dim=self.cfg.num_classes)
        self.head_intra = LinearHead(in_dim=d, out_dim=self.cfg.num_classes)

    def _agg_post(self, post_wsi_list):
        out = []
        for feats in post_wsi_list:
            logit, z = self.TitanAgg(feats)    # z: [1,768]
            out.append(z)
        return torch.cat(out, dim=0)

    def fusion(self, ct_feat, intra_wsi, post_wsi, tab_feat):
        # ======== 统一 dict 输入 ========
        feats = {
            "INTRA": intra_wsi,
            "POST":  post_wsi,
            "CT":    ct_feat,
            "TAB":   tab_feat,
        }
        fusion_out = self.fusion_module(feats)

        logits = self.head_fuse(fusion_out["fusion_feature"])

        return {
            "logits": logits,
            "fusion_feature": fusion_out,
        }

    def forward(self, CT, intra_wsi, post_wsi_list, Table_num, Table_cat):
        device = next(self.parameters()).device
        self.TitanAgg.to(device)
        if hasattr(self.TitanAgg, "aggregator"):
            self.TitanAgg.aggregator.to(device)

        if isinstance(post_wsi_list, list):
            for i in range(len(post_wsi_list)):
                post_wsi_list[i] = [x.to(device) for x in post_wsi_list[i]]
        else:
            if isinstance(post_wsi_list, torch.Tensor):
                post_wsi_list = post_wsi_list.to(device)

        post_wsi = self._agg_post(post_wsi_list)      # [B,768]
        intra_wsi = intra_wsi.to(device)              # [B,768]

        ct_feat = self.CTEncoder(CT)
        tab_feat = self.TableEncoder(Table_num, Table_cat)

        return self.fusion(ct_feat, intra_wsi, post_wsi, tab_feat)