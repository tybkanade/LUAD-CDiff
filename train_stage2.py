import os
import json
import random
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from data.luadDataset import FourModalFromXLSX
from models.common.loss import FocalCrossEntropyLoss
from models.integrated_model import PostOpGenAndCls
from config.configs import (
    CTEncoderConfig, TableEncoderConfig,
    TriFusionConfig, DiffusionConfig, OverallModelConfig
)
from utils.utils import (
    update_epoch_buffers,
    format_confusion_matrix,
    compute_epoch_metrics_macro,
    save_epoch_metrics,
    split_dataset_stratified_kfold,
    _maybe_save_topk, sanitize_logits_for_metrics, ensure_finite, _get_autocast_dtype, freeze_module,
    unfreeze_classifier,

)

criterion = FocalCrossEntropyLoss()

def four_modal_collate_fn(batch: List[dict]):
    out = {}

    for k in batch[0].keys():
        vals = [b[k] for b in batch]

        if k == "post_wsi_list":
            out[k] = vals
            continue

        if k == "intra_wsi":
            out[k] = torch.stack(vals, 0)
            continue

        if k == "x_cat":
            out[k] = torch.stack(vals, 0) if vals[0] is not None else None
            continue

        if k in ("CT", "x_num", "missing_mask", "y"):
            out[k] = torch.stack(vals, 0)
            continue

        out[k] = vals
    return out


def train_one_epoch_stage2(
    model,
    loader,
    optimizer_diff,
    optimizer_cls,
    device,
    epoch,
    log_dir,
    cls_start_epoch=30,
    lambda_cls_max=1.0,
    lambda_warmup=5,
):
    """
      - epoch < cls_start_epoch：
          * 生成损失(loss_gen=diff+recon...) 只更新 diffusion
          * 不更新分类头
      - epoch >= cls_start_epoch：
          * 生成损失 loss_gen 只更新 diffusion
          * 分类损失 loss_cls 只更新 gen_classifier（通过 x0_gen.detach() 隔离梯度）
    """
    model.train()

    total_loss = 0.0
    total_diff = 0.0
    total_rec  = 0.0
    total_cls  = 0.0

    buf = {"y_true": [], "y_pred": [], "proba": []}

    ac_dtype = _get_autocast_dtype(device)
    ac_ctx = torch.autocast("cuda", dtype=ac_dtype) if ac_dtype else torch.cuda.amp.autocast(enabled=False)

    if epoch < cls_start_epoch:
        lambda_cls = 0.0
    else:
        if lambda_warmup <= 0:
            lambda_cls = float(lambda_cls_max)
        else:
            t = (epoch - cls_start_epoch + 1) / float(lambda_warmup)
            lambda_cls = float(lambda_cls_max) * max(0.0, min(1.0, t))

    for batch in tqdm(loader):
        CT = batch["CT"].to(device, non_blocking=True)
        intra = batch["intra_wsi"].to(device, non_blocking=True)

        post_list = batch["post_wsi_list"]
        features = [p.to(device, non_blocking=True) for p in post_list[0]]

        xnum = batch["x_num"].to(device, non_blocking=True)
        xcat = batch["x_cat"].to(device, non_blocking=True) if batch["x_cat"] is not None else None

        y_1h = batch["y"].to(device, non_blocking=True)
        y_idx = y_1h.argmax(dim=-1)

        # =========================
        # 1) diffusion 更新（只用生成损失）
        # =========================
        optimizer_diff.zero_grad(set_to_none=True)

        with ac_ctx:
            out = model(
                stage=2,
                WSI_post=features,
                CT=CT,
                WSI_intra=intra,
                Table_num=xnum,
                Table_cat=xcat,
                label=y_idx,
                epoch=epoch,
            )

            # 生成损失：只用于 diffusion 更新
            loss_gen = out["total_loss"]
            # 生成出来的特征
            x0_gen = out.get("x0_gen", None)

        # 数值检查
        if not ensure_finite("loss_gen", loss_gen):
            continue
        if x0_gen is None:
            pass

        # backward + step（只作用于 diffusion 参数）
        loss_gen.backward()
        torch.nn.utils.clip_grad_norm_(model.diff.parameters(), 5.0)
        optimizer_diff.step()

        # =========================
        # 2) classifier 更新（只用分类损失，且隔离 diffusion 梯度）
        # =========================
        loss_cls_val = None
        logits_for_metrics = None

        if epoch >= cls_start_epoch and x0_gen is not None:
            optimizer_cls.zero_grad(set_to_none=True)

            x0_detached = x0_gen.detach()

            with ac_ctx:
                logits_cls = model.gen_classifier(x0_detached)
                loss_cls = criterion(logits_cls, y_idx)
                loss_cls = loss_cls * lambda_cls

            if not ensure_finite("logits_cls", logits_cls):
                continue
            if not ensure_finite("loss_cls", loss_cls):
                continue

            loss_cls.backward()
            torch.nn.utils.clip_grad_norm_(model.gen_classifier.parameters(), 5.0)
            optimizer_cls.step()

            loss_cls_val = float(loss_cls.item())
            logits_for_metrics = logits_cls
        else:
            # 前30轮不更新分类头
            if x0_gen is not None:
                with torch.no_grad():
                    logits_for_metrics = model.gen_classifier(x0_gen.detach())

        # =========================
        # logging / metrics
        # =========================
        total_loss += float(loss_gen.item()) + (loss_cls_val if loss_cls_val is not None else 0.0)
        total_diff += float(out["diff_loss"].item()) if "diff_loss" in out else 0.0
        total_rec  += float(out["recon_loss"].item()) if "recon_loss" in out else 0.0
        if loss_cls_val is not None:
            total_cls += loss_cls_val

        if logits_for_metrics is not None:
            logits_sane = sanitize_logits_for_metrics(logits_for_metrics)
            update_epoch_buffers(buf, logits_sane, y_1h)

    n = max(1, len(loader))
    metrics = compute_epoch_metrics_macro(buf, num_classes=3)
    metrics["loss"] = total_loss / n
    metrics["diff_loss"] = total_diff / n
    metrics["recon_loss"] = total_rec / n
    metrics["cls_loss"] = (total_cls / n) if epoch >= cls_start_epoch else float("nan")
    metrics["lambda_cls"] = float(lambda_cls)

    save_epoch_metrics(log_dir, "train", epoch, metrics)
    return metrics

@torch.no_grad()
def validate_one_epoch_stage2(model, loader, device, epoch, log_dir):
    """
    Stage-2 验证：不使用真实 WSI_post，使用生成的 x0_gen 来评估分类性能。
    """
    model.eval()
    total_loss = 0.0
    buf = {"y_true": [], "y_pred": [], "proba": []}
    umap_buf = {
        "x0_real": [],
        "x0_gen": [],
        "label": []
    }

    ac_dtype = _get_autocast_dtype(device)
    ac_ctx = torch.autocast("cuda", dtype=ac_dtype) if ac_dtype else torch.cuda.amp.autocast(enabled=False)

    for batch in tqdm(loader, desc=f"Val Stage2 Ep{epoch}"):
        CT = batch["CT"].to(device)
        intra = batch["intra_wsi"].to(device)
        xnum = batch["x_num"].to(device)
        xcat = batch["x_cat"].to(device) if batch["x_cat"] is not None else None
        y_1h = batch["y"].to(device)
        y_idx = y_1h.argmax(dim=-1)

        with ac_ctx:
            out = model.inference_stage2(
                CT=CT,
                WSI_intra=intra,
                Table_num=xnum,
                Table_cat=xcat,
                steps=64,
            )

            logits = out["logits"]

        # ===== 取生成特征 =====
        x0_gen = out["x0_gen"]  # [B, m]
        y_idx = y_idx  # [B]
        # ===== 真实特征（仅验证集有）=====
        real_feats = [p.to(device, non_blocking=True) for p in batch["post_wsi_list"][0]]
        x0_real = model.TitanAgg(real_feats)[1].squeeze(1)

        umap_buf["x0_gen"].append(x0_gen.detach().cpu())
        umap_buf["x0_real"].append(x0_real.detach().cpu())
        umap_buf["label"].append(y_idx.detach().cpu())

        logits_sane = sanitize_logits_for_metrics(logits)
        update_epoch_buffers(buf, logits_sane, y_1h)

    metrics = compute_epoch_metrics_macro(buf, num_classes=3)
    save_epoch_metrics(log_dir, "val", epoch, metrics)

    return metrics, umap_buf


def main():

    EXCEL = "path of xlsx"
    LOG_DIR = "path of log"
    X0_STATS_ROOT_CLS = ""
    X0_STATS_ROOT = ""


    NUM_COLS = ["年龄", "长径cm", "短径cm", "实性成分长径"]
    BINARY_COLS = [
        "圆形、类圆形","不规则","分叶","结节征","空泡","空洞",
        "钙化","支气管征","清楚","光滑锐利","毛刺","尖角、桃尖",
        "索条","血管集束征","胸膜凹陷"
    ]
    PLEURA_COL = "与胸膜的关系"
    GENDER_COL = "性别"
    DENSITY_COL = "密度"

    CLASS_NAMES = ["AAH/AIS", "MIA", "AC"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    full_ds = FourModalFromXLSX(
        xlsx_path=EXCEL,
        ct_col="__npy_path__",
        intra_wsi_id_col="冰冻切片号",
        intra_wsi_dir="path of the features of the frozen section",
        post_wsi_prefix_col="术后病理前缀",
        post_wsi_root="path of the postoperative pathology",

        label_col="术后病理诊断类型",
        label_names=CLASS_NAMES,

        num_cols=NUM_COLS,
        binary_cols=BINARY_COLS,
        pleura_col=PLEURA_COL,
        gender_col=GENDER_COL,
        density_col=DENSITY_COL,

        filter_missing=True,
        verbose=True,
    )

    folds = split_dataset_stratified_kfold(full_ds, n_splits=5, seed=seed)
    os.makedirs(LOG_DIR, exist_ok=True)

    fold_best_results = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):

        print(f"\n===================== FOLD {fold_id} / 5 =====================")

        fold_dir = os.path.join(LOG_DIR, f"fold_{fold_id}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        img_dir = os.path.join(fold_dir, "umap")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # x0 mean/std JSON
        stats_path = f"{X0_STATS_ROOT}/fold_{fold_id}/x0_mean_std.json"
        class_stats_path = f"{X0_STATS_ROOT_CLS}/fold_{fold_id}/x0_classwise_stats.json"
        print(f"[Fold {fold_id}] Using x0 stats: {stats_path}")
        print(f"[Fold {fold_id}] Using x0 class stats: {class_stats_path}")

        train_set = Subset(full_ds, train_idx)
        val_set   = Subset(full_ds, val_idx)
        batch_size = 1
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True,
            collate_fn=four_modal_collate_fn
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True,
            collate_fn=four_modal_collate_fn
        )

        diff_cfg = DiffusionConfig(
            x0_stats_path=stats_path,
            x0_class_stats_path=class_stats_path,
        )
        TRI_FIRST_CKPT = ""
        AGG_CKPT = ""
        model = PostOpGenAndCls(
            cfg=OverallModelConfig(),
            ct_cfg=CTEncoderConfig(feature_extraction=True),
            table_cfg=TableEncoderConfig(
                num_features=len(NUM_COLS),
                cat_cardinalities=[3]*len(BINARY_COLS) + [4,6,3],
                emb_dim=32, hidden_dims=(256,256),
                activation="relu", dropout=0.1,
                use_attention=True, n_heads=4, n_attn_layers=4,
                feature_extraction=True,
            ),
            diff_cfg=diff_cfg,           # ← ★★★ 传入 Diffusion 配置 ★★★
            tri_cfg=TriFusionConfig(),
        ).to(device)

        # ======= 加载 tri_first 预训练 =======
        load_tri_first_from_ckpt(model, TRI_FIRST_CKPT, device=device)
        # ===== 加载 TitanAgg =====
        model._load_pretrained_agg(AGG_CKPT)


        # 1) 冻结非生成模块
        freeze_module(model.tri_first)
        freeze_module(model.TitanAgg)
        freeze_module(model.gen_classifier)
        # 2) 解冻 diffusion
        for p in model.diff.parameters():
            p.requires_grad_(True)
        model.diff.train()

        optimizer_diff = optim.AdamW(model.diff.parameters(), lr=1e-4, weight_decay=1e-5)
        optimizer_cls = optim.AdamW(model.gen_classifier.parameters(), lr=1e-4, weight_decay=1e-5)

        best_val_acc = 0.0
        best_val_stats = None
        TOPK = 3
        topk_ckpts = []
        num_epochs = 60

        for epoch in range(1, num_epochs+1):
            if epoch == 30:
                print(">>> Unfreeze gen_classifier (probe training)")
                unfreeze_classifier(model)

            print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

            train_stats = train_one_epoch_stage2(
                model, train_loader, optimizer_diff, optimizer_cls, device, epoch, fold_dir
            )
            val_stats, umap_buf = validate_one_epoch_stage2(
                model, val_loader, device, epoch, fold_dir
            )

            print(
                f"Epoch {epoch:02d} | "
                f"train: loss={train_stats['loss']:.4f}  "
                f"acc={train_stats['acc']:.4f}  "
                f"precision={train_stats['precision']:.4f}  "
                f"recall={train_stats['recall']:.4f}  "
                f"f1={train_stats['f1']:.4f}  "
                f"auroc={train_stats['auroc']:.4f} | "
                # f"val: loss={val_stats['loss']:.4f}  "
                f"acc={val_stats['acc']:.4f}  "
                f"precision={val_stats['precision']:.4f}  "
                f"recall={val_stats['recall']:.4f}  "
                f"f1={val_stats['f1']:.4f}  "
                f"auroc={val_stats['auroc']:.4f}"
            )
            print("Train CM:\n", format_confusion_matrix(train_stats["cm"], labels=CLASS_NAMES))
            print("Val   CM:\n", format_confusion_matrix(val_stats["cm"], labels=CLASS_NAMES))

            # ---- 更新 best ----
            if val_stats["acc"] > best_val_acc:
                best_val_acc = val_stats["acc"]
                best_val_stats = {
                    "acc": float(val_stats["acc"]),
                    "precision": float(val_stats["precision"]),
                    "recall": float(val_stats["recall"]),
                    "f1": float(val_stats["f1"]),
                    "auroc": float(val_stats["auroc"]),
                }

            print(f"Best val_acc = {best_val_acc:.4f}")

            # ---- top-K ckpt 保存 ----
            ckpt_payload = {
                "epoch": epoch,
                "val_acc": float(val_stats["acc"]),
                "model": model.state_dict(),
            }
            # _maybe_save_topk(ckpt_dir, TOPK, topk_ckpts, epoch, float(val_stats["acc"]), ckpt_payload)
            torch.save(
                {
                    "x0_gen": torch.cat(umap_buf["x0_gen"], dim=0),
                    "x0_real": torch.cat(umap_buf["x0_real"], dim=0),
                    "label": torch.cat(umap_buf["label"], dim=0),
                },
                os.path.join(img_dir, f"umap_stage2_epoch{epoch}.pt")
            )

        print(f"[Fold {fold_id}] Best Val Acc = {best_val_acc:.4f}")
        fold_best_results.append(best_val_stats)

    summary_path = os.path.join(LOG_DIR, "kfold_summary.json")

    metrics = ["acc", "precision", "recall", "f1", "auroc"]

    summary = {
        "fold_best_results": fold_best_results,
        "mean": {m: float(np.mean([f[m] for f in fold_best_results])) for m in metrics},
        "std":  {m: float(np.std([f[m] for f in fold_best_results]))  for m in metrics},
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== KFold Final Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
