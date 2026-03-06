from typing import Optional, Dict
import math
import torch
import torch.nn.functional as F
from torch import nn
from config.configs import CTEncoderConfig, TableEncoderConfig, TriFusionConfig
from models.common.common import LinearHead, _proj_block, Classifier
from models.fusion.DynamicFusion import MultiModalFusionModule
from utils.utils import _load_pretrained_ct_encoder, _load_pretrained_table_encoder
from models.encoder.ctEncoder import CtEncoder
from models.encoder.tableEncoder import TableEncoder

class Fusion3Modal(nn.Module):
    """
    三模态融合 (WSI + CT + Table)：
    """
    def __init__(
        self,
        ct_config,
        table_config,
        fusion_cfg,
        tab_pretrained_path: Optional[str] = None,
        ct_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = fusion_cfg
        d = self.cfg.input_dim

        in_dim_ct = self.cfg.input_dim
        in_dim_tab = self.cfg.input_dim
        in_dim_wsi = self.cfg.input_dim

        self.CTEncoder = CtEncoder(config=ct_config)
        self.TableEncoder = TableEncoder(config=table_config)

        if ct_pretrained_path:
            _load_pretrained_ct_encoder(self.CTEncoder, ct_pretrained_path)
        if tab_pretrained_path:
            _load_pretrained_table_encoder(self.TableEncoder, tab_pretrained_path)

        modality_dims = {
            "WSI": in_dim_wsi,
            "CT":  in_dim_ct,
            "TAB": in_dim_tab,
        }
        self.fusion_module = MultiModalFusionModule(
            modality_dims=modality_dims,
            d_model=d,
            dropout=self.cfg.dropout,
            nhead=4,
            num_layers=2,
            dim_feedforward=d * 4,
            fusion_hidden=d // 2,
        )

        # --- 分类头 ---
        self.head_fuse = Classifier(
            tri_dim=d,
            num_classes=self.cfg.num_classes,
            dropout=self.cfg.dropout,
            hidden_ratio=self.cfg.classifier_hidden_ratio,
        )

        self.head_wsi = LinearHead(
            in_dim=in_dim_wsi,
            hidden_dim=512,
            out_dim=self.cfg.num_classes,
            dropout_rate=0.5
        )

        self.head_tab = Classifier(
            tri_dim=d,
            num_classes=self.cfg.num_classes,
            dropout=self.cfg.dropout,
            hidden_ratio=self.cfg.classifier_hidden_ratio,
        )

        self.head_ct = Classifier(
            tri_dim=d,
            num_classes=self.cfg.num_classes,
            dropout=self.cfg.dropout,
            hidden_ratio=self.cfg.classifier_hidden_ratio,
        )

    @staticmethod
    def _to_vec(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return x

    def _encode_all(self, wsi_feat, CT, Table_num=None, Table_cat=None):
        """
          wsi_feat: [B, D_wsi]
          ct_feat:  [B, D_ct]
          tab_feat: [B, D_tab]
        """
        ct_feat = self.CTEncoder(CT)                       # [B, D_ct] or [B,1,D_ct]
        tab_feat = self.TableEncoder(Table_num, Table_cat) # [B, D_tab] or [B,1,D_tab]

        ct_feat = self._to_vec(ct_feat)
        tab_feat = self._to_vec(tab_feat)
        wsi_feat = self._to_vec(wsi_feat)

        return wsi_feat, ct_feat, tab_feat

    def forward(self, CT, wsi_feat, Table_num=None, Table_cat=None):
        # 1) 编码得到各模态向量
        wsi_vec, ct_vec, tab_vec = self._encode_all(
            wsi_feat=wsi_feat,
            CT=CT,
            Table_num=Table_num,
            Table_cat=Table_cat
        )

        # 2) 统一融合（投影 + CMT + 动态权重）
        fusion_out = self.fusion_module({
            "WSI": wsi_vec,
            "CT":  ct_vec,
            "TAB": tab_vec,
        })

        z_fuse = fusion_out["fusion_feature"]                 # [B, d]
        alpha = fusion_out["modal_alpha"]                     # [B, 3]
        tok = fusion_out["tokens_after_cmt_dict"]             # name -> [B, d]

        # 3) 分类：融合头
        logits_fuse = self.head_fuse(z_fuse)

        # 4) 单模态头
        logits_wsi = self.head_wsi(wsi_vec)
        logits_tab = self.head_tab(tok["TAB"])
        logits_ct  = self.head_ct(tok["CT"])

        return {
            "fusion_feature": z_fuse,
            "logits": logits_fuse,
            "WSI_logits": logits_wsi,
            "TAB_logits": logits_tab,
            "CT_logits": logits_ct,
            "CT_feature": tok["CT"],
            "WSI_feature": tok["WSI"],
            "Table_feature": tok["TAB"],
            "modal_alpha": alpha,           # [B,3]
            "token_names": fusion_out["token_names"],
        }
