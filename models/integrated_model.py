import json
import os
from typing import Optional, Dict, List, Any
import torch
import torch.nn as nn
from models.FourModalFusionNetwork import Fusion4Modal
from models.common.loss import FocalCrossEntropyLoss
from config.configs import CTEncoderConfig,  TableEncoderConfig, TriFusionConfig, DiffusionConfig, \
     OverallModelConfig, DenoiserPriorConfig
from models.denoiser_DALL_mulcond import CondDenoiser_Prior
from models.generator_mulcond import GaussianDiffusion1D
from models.common.common import LinearHead, TitanWithAggregator, WeightedModalAttention
from models.triple_fusion import Fusion3Modal

criterion = FocalCrossEntropyLoss(gamma=2.0, weight=None, label_smoothing=0.05, reduction="mean")

class PostOpGenAndCls(nn.Module):
    def __init__(self, cfg, ct_cfg, table_cfg, diff_cfg, tri_cfg, use_modal_attention=False):
        super().__init__()
        self.cfg = cfg
        self.use_modal_attention = use_modal_attention

        self.tri_first = Fusion3Modal(
            ct_config=ct_cfg, table_config=table_cfg, fusion_cfg=tri_cfg,
            tab_pretrained_path = None,
            ct_pretrained_path = None,
        )

        self.modal_attn = WeightedModalAttention(dim=cfg.m)

        self.denoiser = CondDenoiser_Prior(DenoiserPriorConfig())

        self.diff = GaussianDiffusion1D(self.denoiser, diff_cfg)

        if diff_cfg.x0_stats_path is not None:
            print(f"[Diffusion] Loading x0 mean/std from {diff_cfg.x0_stats_path}")

            with open(diff_cfg.x0_stats_path, "r") as f:
                stats = json.load(f)

            mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1)
            std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1)

            device = next(self.denoiser.parameters()).device
            mean, std = mean.to(device), std.to(device)

            assert mean.shape[-1] == diff_cfg.m, \
                f"mean/std dim mismatch: {mean.shape[-1]} vs diffusion.m={diff_cfg.m}"

            self.diff.set_norm(mean, std)

            print(f"[Diffusion] mean={mean.mean():.4f}, std={std.mean():.4f}")

        C = cfg.num_classes
        m = cfg.m
        self.register_buffer("class_mean", torch.zeros(C, m))
        self.register_buffer("class_std", torch.ones(C, m))
        class_mean = []
        class_std = []
        self.register_buffer("has_class_stats", torch.tensor(0, dtype=torch.uint8))
        if getattr(diff_cfg, "x0_class_stats_path", None) is not None:
            print(f"[Diffusion] Loading x0 class-wise stats from {diff_cfg.x0_class_stats_path}")
            with open(diff_cfg.x0_class_stats_path, "r") as f:
                class_stats = json.load(f)

            for c in range(C):
                mu = torch.tensor(class_stats[str(c)]["mean"], dtype=torch.float32)
                var = torch.tensor(class_stats[str(c)]["var"], dtype=torch.float32)
                sd = torch.sqrt(var + 1e-6)

                class_mean.append(mu)
                class_std.append(sd)

            class_mean = torch.stack(class_mean, dim=0).to(device)  # [C, m]
            class_std = torch.stack(class_std, dim=0).to(device)  # [C, m]

            self.class_mean.copy_(class_mean)
            self.class_std.copy_(class_std.clamp_min(1e-6))
            self.has_class_stats.fill_(1)

        # --------- Stage-2 生成侧分类器 ---------
        self.gen_classifier = LinearHead(in_dim=cfg.m, hidden_dim=cfg.hidden, out_dim=cfg.num_classes)

        # --------- Stage-3 再融合 ---------
        self.tri_second = Fusion4Modal(
            ct_config=ct_cfg,
            table_config=table_cfg,
            fusion_cfg=tri_cfg,
            wsi_agg="new_hybrid",
            tab_pretrained_path=None,
            ct_pretrained_path=None,
            agg_pretrained_path=None,
        )

        self.TitanAgg = TitanWithAggregator(agg="new_hybrid")


    def build_multi_token_cond(self, CT_feat, WSI_feat, Table_feat):

        if self.use_modal_attention:
            fused = self.modal_attn([CT_feat, WSI_feat, Table_feat])  # [B, D]
            # print(fused.shape)
            return fused.unsqueeze(1)  # [B,1,D]

        tokens = [CT_feat, WSI_feat, Table_feat]

        cond_tokens = torch.stack(tokens, dim=1)
        # print(cond_tokens.shape)

        return cond_tokens

    # ====================== 训练调度 ======================
    def forward(self, *args, stage: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        统一入口：forward(..., stage=1/2/3)
        - stage=1: forward_stage1(CT, WSI_intra, Table, y)
        - stage=2: forward_stage2(WSI_post, CT, WSI_intra, Table, y)
        - stage=3: forward_stage3(WSI_post, CT, WSI_intra, Table, y)
        """
        if stage == 1:
            return self.forward_stage1(*args, **kwargs)
        elif stage == 2:
            return self.forward_stage2(*args, **kwargs)
        elif stage == 3:
            return self.forward_stage3(*args, **kwargs)
        else:
            raise ValueError("stage must be 1|2|3")

    # ---- 三模态融合 → tri_flat, cond_vec ----
    def _forward_first(self, CT, WSI_feat, Table_num, Table_cat):
        fusion_out = self.tri_first(CT, WSI_feat, Table_num, Table_cat)
        return fusion_out

    # ------------------------------------------------------------
    # Stage-1
    # ------------------------------------------------------------
    def forward_stage1(self, CT, WSI_intra, Table_num, Table_cat, label):
        fusion_out = self._forward_first(CT, WSI_intra, Table_num, Table_cat)
        logits = fusion_out["logits"]
        loss = criterion(logits, label)
        return {"total_loss": loss, "logits": logits}

    # ------------------------------------------------------------
    # Stage-2
    # ------------------------------------------------------------
    def forward_stage2(
            self,
            WSI_post,
            CT,
            WSI_intra,
            Table_num,
            Table_cat,
            label,
            epoch,
    ):
        # 1. 编码阶段
        fusion_out = self._forward_first(CT, WSI_intra, Table_num, Table_cat)

        CT_feat = fusion_out["CT_feature"]
        WSI_feat = fusion_out["WSI_feature"]
        Table_feat = fusion_out["Table_feature"]

        # 2. 条件 tokens
        cond_tokens = self.build_multi_token_cond(
            CT_feat, WSI_feat, Table_feat
        )

        # 3. 真实术后 WSI → x0_real
        x0_real = self.TitanAgg(WSI_post)[1].squeeze(1)  # [B, m]

        # 4. Diffusion 训练
        diff_out = self.diff.train_step(
            x0=x0_real,
            cond=cond_tokens,
        )

        diff_loss = diff_out["diff_loss"]
        angle_loss = diff_out["angle_loss"]
        recon_loss = diff_out["recon_loss"]
        x0_gen = diff_out["x0_gen"]  # [B, m]

        # 5. 分类
        logits_gen = self.gen_classifier(x0_gen)

        if epoch >= 30:
            cls_loss = criterion(logits_gen, label)
        else:
            cls_loss = torch.zeros((), device=x0_gen.device)

        # 6. 总损失
        total_loss = (
                self.cfg.w_diff_stage2 * diff_loss
                + self.cfg.w_rec_stage2 * recon_loss
                # + (self.cfg.w_main_stage2 * cls_loss if epoch >= 30 else 0.0)
        )

        return {
            "total_loss": total_loss,
            "diff_loss": diff_loss * self.cfg.w_diff_stage2,
            "recon_loss": recon_loss * self.cfg.w_rec_stage2,
            "cls_loss": cls_loss * self.cfg.w_main_stage2 if epoch >= 30 else torch.zeros_like(cls_loss),
            "logits": logits_gen,
            "x0_gen": x0_gen,
            "CT_feature": CT_feat,
            "Table_feature": Table_feat,
            "WSI_feature": WSI_feat,
        }


    # ------------------------------------------------------------
    # Stage-3
    # ------------------------------------------------------------
    def forward_stage3(self, WSI_post, CT, WSI_intra, Table_num, Table_cat, label):
        fusion_out = self._forward_first(CT, WSI_intra, Table_num, Table_cat)

        CT_feat = fusion_out["CT_feature"]
        WSI_feat = fusion_out["WSI_feature"]
        Table_feat = fusion_out["Table_feature"]
        fusion_feat = fusion_out["fusion_feature"]

        cond_tokens = self.build_multi_token_cond(CT_feat, WSI_feat, Table_feat)

        x0_real = self.TitanAgg(WSI_post)[1].squeeze(1)

        out = self.diff.train_step(x0=x0_real, cond=cond_tokens)
        x0_gen = out["x0_gen"]

        fusion_gen = self.tri_second.fusion(CT_feat, WSI_feat, x0_gen, Table_feat)
        logits_gen = fusion_gen["logits"]
        cls_loss_gen = criterion(logits_gen, label)

        cls_loss = cls_loss_gen

        total_loss = self.cfg.w_main_stage3 * cls_loss

        return {
            "total_loss": total_loss,
            "logits": logits_gen,
            "x0_gen": x0_gen
        }


    @torch.no_grad()
    def inference_stage2(self, CT, WSI_intra, Table_num, Table_cat, steps=50):
        """
        Stage-2 推理：
        - 用 CT + 术中病理 + 表格 生成术后病理特征 x0_gen
        - 用 Stage-2 分类器 gen_classifier 做分类
        """
        # === 1) Tri-modal encoder ===
        fusion_out = self._forward_first(CT, WSI_intra, Table_num, Table_cat)

        CT_feat = fusion_out["CT_feature"]
        WSI_feat = fusion_out["WSI_feature"]
        Table_feat = fusion_out["Table_feature"]

        # === 2) 构建条件 tokens ===
        cond_tokens = self.build_multi_token_cond(CT_feat, WSI_feat, Table_feat)

        # === 3) diffusion sample（生成术后病理特征）===
        x0_gen = self.diff.sample(cond=cond_tokens, steps=steps)  # [B,768]

        # === 4) Stage-2 分类 ===
        logits = self.gen_classifier(x0_gen)  # [B, num_classes]
        probs = logits.softmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "x0_gen": x0_gen,
            "CT_feature": CT_feat,
            "WSI_feature": WSI_feat,
            "Table_feature": Table_feat,
        }

    @torch.no_grad()
    def inference_stage3(self, CT, WSI_intra, Table_num, Table_cat, steps=100):
        fusion_out = self._forward_first(CT, WSI_intra, Table_num, Table_cat)

        CT_feat = fusion_out["CT_feature"]
        WSI_feat = fusion_out["WSI_feature"]
        Table_feat = fusion_out["Table_feature"]
        fusion_feat = fusion_out["fusion_feature"]

        cond_tokens = self.build_multi_token_cond(CT_feat, WSI_feat, Table_feat)

        x0_gen = self.diff.sample(cond=cond_tokens, steps=steps)

        fin = self.tri_second.fusion(CT_feat, WSI_feat, x0_gen, Table_feat)

        return {
            "logits": fin["logits"],
            "x0_gen": x0_gen,
        }

