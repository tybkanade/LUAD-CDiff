import json
import os
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score, \
    average_precision_score
from torch.utils.data import DataLoader, Subset, Dataset

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_index_labels(y: torch.Tensor) -> torch.Tensor:
    return y.argmax(dim=-1) if y.ndim == 2 else y

def _softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)

def update_epoch_buffers(buf: dict, logits: torch.Tensor, y: torch.Tensor):
    y_idx = _to_index_labels(y).detach().cpu().numpy()         # [B]
    proba = _softmax_logits(logits).detach().cpu().numpy()     # [B,C]
    y_pred = proba.argmax(axis=-1)                              # [B]
    buf["y_true"].append(y_idx)
    buf["y_pred"].append(y_pred)
    buf["proba"].append(proba)

def compute_epoch_metrics(buf: dict, num_classes: int = 4) -> dict:
    import numpy as np
    if len(buf["y_true"]) == 0:
        return {"acc":0.0,"precision":0.0,"recall":0.0,"f1":0.0,"auroc":0.0,"cm":np.zeros((num_classes,num_classes),dtype=int)}
    y_true = np.concatenate(buf["y_true"], axis=0)   # [N]
    y_pred = np.concatenate(buf["y_pred"], axis=0)   # [N]
    proba  = np.concatenate(buf["proba"],  axis=0)   # [N,C]

    acc = float((y_pred == y_true).mean().item())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes),
        average="macro", zero_division=0
    )
    try:
        auroc = roc_auc_score(
            y_true, proba,
            multi_class="ovr", average="macro",
            labels=np.arange(num_classes)
        )
        auroc = float(auroc)
    except Exception:
        auroc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))  # [C,C], 行是真值，列是预测
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
        "cm": cm,
    }

def compute_epoch_metrics_macro(buf: dict, num_classes: int = 3) -> dict:
    """
       按一个 epoch 缓冲的数据计算各类指标（宏平均），并返回结果字典。
       """
    # 空缓冲保护
    if not buf or len(buf.get("y_true", [])) == 0:
        return {
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
            "bal_acc": 0.0,
            "auprc_macro": 0.0,
            "cm": np.zeros((num_classes, num_classes), dtype=int),
        }

    # 拼接所有 batch
    y_true = np.concatenate(buf["y_true"], axis=0)  # [N], 取值 0..C-1
    y_pred = np.concatenate(buf["y_pred"], axis=0)  # [N]
    proba = np.concatenate(buf["proba"], axis=0)  # [N, C]

    # 基础准确率
    acc = float((y_pred == y_true).mean())

    # 宏平均 P/R/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0
    )

    # 宏平均 AUROC（OvR）
    try:
        auroc = roc_auc_score(
            y_true, proba,
            multi_class="ovr",
            average="macro",
            labels=np.arange(num_classes)
        )
        auroc = float(auroc)
    except Exception:
        auroc = float("nan")

    # Balanced Accuracy（各类 Recall 的平均）
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # 宏平均 AUPRC（OvR：对每一类当正类）
    ap_list = []
    for c in range(num_classes):
        pos = (y_true == c).astype(int)
        if (pos.sum() > 0) and (pos.sum() < len(pos)):
            ap = average_precision_score(pos, proba[:, c])
            ap_list.append(ap)
    auprc_macro = float(np.mean(ap_list)) if len(ap_list) > 0 else float("nan")

    # 混淆矩阵（行=True，列=Pred）
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "bal_acc": float(balanced_acc),
        "auprc_macro": float(auprc_macro),
        "cm": cm,
    }

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_epoch_metrics(log_dir: str, split: str, epoch: int, metrics: dict):
    """
    把所有指标（loss, diff_loss, recon_loss, cls_loss, consistency_loss
    acc, balanced_acc, precision, recall, f1, auroc, auprc_macro, cm）
    以固定宽度格式写入 metrics_{split}.txt。
    """

    import os, numpy as np

    def _ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    _ensure_dir(log_dir)
    path = os.path.join(log_dir, f"metrics_{split}.txt")

    # ----------- 读取指标值（带默认）-----------
    loss = float(metrics.get("loss", float("nan")))
    diff_loss = float(metrics.get("diff_loss", float("nan")))
    recon_loss = float(metrics.get("recon_loss", float("nan")))
    cls_loss = float(metrics.get("cls_loss", float("nan")))
    cda_loss = float(metrics.get("cda_loss", float("nan")))
    angle_loss = float(metrics.get("angle_loss", float("nan")))
    acc = float(metrics.get("acc", float("nan")))
    balanced_acc = float(metrics.get("balanced_acc", float("nan")))
    precision = float(metrics.get("precision", float("nan")))
    recall = float(metrics.get("recall", float("nan")))
    f1 = float(metrics.get("f1", float("nan")))
    auroc = metrics.get("auroc", float("nan"))
    auprc_macro = float(metrics.get("auprc_macro", float("nan")))

    cm = np.asarray(metrics["cm"], dtype=int).reshape(-1)
    num_classes = int(len(cm) ** 0.5)

    # ----------- 列定义（顺序必须与 row_items 一致）-----------
    col_defs = [
        ("epoch", 6, "d"),
        ("loss", 12, ".6f"),
        ("diff_loss", 12, ".6f"),
        ("recon_loss", 12, ".6f"),
        ("cda_loss", 12, ".6f"),
        ("angle_loss", 12, ".6f"),
        ("cls_loss", 12, ".6f"),
        ("acc", 10, ".6f"),
        ("balanced_acc", 14, ".6f"),
        ("precision", 12, ".6f"),
        ("recall", 10, ".6f"),
        ("f1", 10, ".6f"),
        ("auroc", 10, ".6f"),
        ("auprc_macro", 14, ".6f"),
    ]

    cm_width = 6
    cm_cols = [(f"cm_{r}_{c}", cm_width, "d")
               for r in range(num_classes)
               for c in range(num_classes)]

    all_cols = col_defs + cm_cols

    # -------- 写表头 --------
    if not os.path.exists(path):
        header = []
        for name, width, _ in all_cols:
            header.append(f"{name:>{width}}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(" ".join(header) + "\n")

    # -------- 写数据行 --------

    def fmt(val, width, spec):
        if isinstance(val, str):
            s = val
        else:
            try:
                if spec == "d":
                    s = f"{int(val)}"
                else:
                    s = format(float(val), spec)
            except:
                s = str(val)
        return f"{s:>{width}}"

    # 按列顺序写入
    row_vals = [
        epoch,
        loss,
        diff_loss,
        recon_loss,
        cda_loss,
        angle_loss,
        cls_loss,
        acc,
        balanced_acc,
        precision,
        recall,
        f1,
        auroc,
        auprc_macro,
    ]

    row_items = []
    for (name, width, spec), val in zip(col_defs, row_vals):
        row_items.append(fmt(val, width, spec))

    # 添加混淆矩阵
    for i, (name, width, spec) in enumerate(cm_cols):
        row_items.append(fmt(cm[i], width, spec))

    # 写入文件
    with open(path, "a", encoding="utf-8") as f:
        f.write(" ".join(row_items) + "\n")



def split_dataset_stratified(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_val_per_class: int = 1,   # 每类验证集至少保留多少个样本；极小类会自动放宽
    shuffle: bool = True,
    verbose: bool = True,
):
    """
    分层切分 TitanWSIDataset：保证各类别在 train/val 中的占比与总体尽量一致。
    - 对每个类别 c，抽取 round(n_c * val_ratio) 个进 val，其余进 train。
    - 边界处理：
        * 若某类 n_c == 1：全部放 train（避免该类在 train 里为 0）
        * 若某类 n_c > 1：val 至少取 min_val_per_class，但不会把该类全放到 val
    返回：
        Subset(dataset, train_idx), Subset(dataset, val_idx)
    """
    assert hasattr(dataset, "df") and hasattr(dataset, "label_col") and hasattr(dataset, "label2idx"), \
        "dataset 需要包含 df / label_col / label2idx 属性（TitanWSIDataset 已有）"

    rng = np.random.default_rng(seed)

    # 读取标签并映射到索引（0..C-1）
    labels_str = dataset.df[dataset.label_col].astype(str).str.strip().values
    labels = np.array([dataset.label2idx[s] for s in labels_str], dtype=int)

    n = len(dataset)
    all_idx = np.arange(n)
    classes = sorted(dataset.label2idx.values())  # [0,1,2,3]

    train_idx_list = []
    val_idx_list = []

    for c in classes:
        cls_idx = all_idx[labels == c]
        rng.shuffle(cls_idx)
        n_c = len(cls_idx)

        if n_c == 0:
            continue
        elif n_c == 1:
            # 只有 1 个样本：全部放训练集，避免该类在训练中缺失
            n_val_c = 0
        else:
            # 目标 val 数量（四舍五入），至少 min_val_per_class，且保留至少 1 个在 train
            n_val_c = int(round(n_c * val_ratio))
            n_val_c = max(min_val_per_class, n_val_c)
            n_val_c = min(n_val_c, n_c - 1)

        val_c = cls_idx[:n_val_c]
        train_c = cls_idx[n_val_c:]

        val_idx_list.append(val_c)
        train_idx_list.append(train_c)

    # 汇总
    train_idx = np.concatenate(train_idx_list, axis=0) if len(train_idx_list) else np.array([], dtype=int)
    val_idx   = np.concatenate(val_idx_list,   axis=0) if len(val_idx_list)   else np.array([], dtype=int)

    if shuffle:
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

    if verbose:
        # 打印各类分布
        def count_per_split(idx_arr):
            y = labels[idx_arr] if len(idx_arr) else np.array([], dtype=int)
            return {c: int((y == c).sum()) for c in classes}
        total_counts = {c: int((labels == c).sum()) for c in classes}
        train_counts = count_per_split(train_idx)
        val_counts   = count_per_split(val_idx)
        print("[Stratified Split]")
        print("Total:", total_counts)
        print("Train:", train_counts, " | size:", len(train_idx))
        print("Val  :", val_counts,   " | size:", len(val_idx))

    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())

def split_dataset_stratified_kfold(
    dataset: Dataset,
    n_splits: int = 5,
    seed: int = 42,
    min_per_class_each_fold: int = 1,
    shuffle: bool = True,
    verbose: bool = True
):

    assert hasattr(dataset, "df") and hasattr(dataset, "label_col") and hasattr(dataset, "label2idx"), \
        "dataset 需要包含 df / label_col / label2idx 属性"

    rng = np.random.default_rng(seed)

    # 取标签
    labels_str = dataset.df[dataset.label_col].astype(str).str.strip().values
    labels = np.array([dataset.label2idx[s] for s in labels_str], dtype=int)

    n = len(dataset)
    all_idx = np.arange(n)
    classes = sorted(dataset.label2idx.values())  # 类别列表

    # ======= 每一个类别单独进行 K 抽样 =======
    folds_indices = [dict(train=[], val=[]) for _ in range(n_splits)]

    for c in classes:

        cls_idx = all_idx[labels == c]
        n_c = len(cls_idx)

        if n_c == 0:
            continue

        # 随机打乱
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)

        # 计算每折大约多少个样本
        fold_sizes = [n_c // n_splits] * n_splits
        for i in range(n_c % n_splits):
            fold_sizes[i] += 1

        # 划分 samples 到各折 val 集
        start = 0
        for k in range(n_splits):
            end = start + fold_sizes[k]
            val_k = cls_idx[start:end]

            # 至少 min_per_class
            if (len(val_k) < min_per_class_each_fold) and (n_c >= min_per_class_each_fold * n_splits):
                # 如果太少，从全局借样本
                need = min_per_class_each_fold - len(val_k)
                extra = rng.choice(cls_idx, size=need, replace=False)
                val_k = np.concatenate([val_k, extra])

            folds_indices[k]["val"].extend(val_k.tolist())
            start = end

    # ======= 生成每折 train/val =======
    results = []

    for k in range(n_splits):
        val_idx = np.array(list(set(folds_indices[k]["val"])))
        train_idx = np.array([idx for idx in all_idx if idx not in val_idx])

        # 全局 shuffle
        if shuffle:
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)

        # 统计输出
        if verbose:
            print(f"\n=== Fold {k+1}/{n_splits} ===")
            def count(idx_arr):
                y = labels[idx_arr]
                return {c: int((y == c).sum()) for c in classes}

            print("Train:", count(train_idx), " | size:", len(train_idx))
            print("Val  :", count(val_idx),   " | size:", len(val_idx))

        results.append((train_idx.tolist(), val_idx.tolist()))

    return results


def format_confusion_matrix(cm, labels= ["AAH/AIS", "MIA", "AC"], col_width=6):
    """
    将 2D 混淆矩阵格式化为字符串（行=True标签，列=Pred标签）。
    返回值可直接 print。
    """
    cm = np.asarray(cm)
    # 头部
    header = " " * (col_width + 2) + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    # 每一行
    rows = []
    for i, lbl in enumerate(labels):
        row_vals = "".join(f"{int(cm[i, j]):>{col_width}}" for j in range(len(labels)))
        rows.append(f"{lbl:>{col_width}} |{row_vals}")
    return header + "\n" + "\n".join(rows)


def _maybe_save_topk(ckpt_dir: str, k: int, topk_list: list, cur_epoch: int, cur_acc: float, payload: dict):
    """
    维护一个长度<=k 的 topk 列表（降序按 acc），并在需要时保存/删除 checkpoint 文件。
    topk_list: List[{"epoch": int, "acc": float, "path": str}]
    """
    # 生成当次待保存文件名
    safe_acc = f"{cur_acc:.4f}"
    save_path = os.path.join(ckpt_dir, f"ckpt_epoch{cur_epoch:03d}_acc{safe_acc}.pth")

    for i, it in enumerate(topk_list):
        if it["epoch"] == cur_epoch:
            try:
                if os.path.exists(it["path"]) and it["path"] != save_path:
                    os.remove(it["path"])
            except Exception:
                pass
            topk_list.pop(i)
            break

    # 如果还没达到 k 个，或者当前 acc 比最差的更高，就纳入
    need_add = (len(topk_list) < k) or (cur_acc > min(topk_list, key=lambda x: x["acc"])["acc"])

    if need_add:
        # 先保存当前 checkpoint
        torch.save(payload, save_path)

        # 放入列表并按 acc 降序排序
        topk_list.append({"epoch": cur_epoch, "acc": float(cur_acc), "path": save_path})
        topk_list.sort(key=lambda x: (-x["acc"], x["epoch"]))

        # 如果超过 k 个，删除最差的权重文件
        while len(topk_list) > k:
            worst = topk_list.pop(-1)
            try:
                if os.path.exists(worst["path"]):
                    os.remove(worst["path"])
            except Exception:
                pass

        # 同步写一个 JSON
        meta_path = os.path.join(ckpt_dir, "best_top3.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"epoch": it["epoch"], "acc": round(it["acc"], 6), "path": os.path.basename(it["path"])} for it in topk_list],
                f, ensure_ascii=False, indent=2
            )


def _finite_min_max(x: torch.Tensor):
    if not torch.is_tensor(x):
        return None, None
    mask = torch.isfinite(x)
    if not mask.any():
        return None, None
    x = x[mask]
    return float(x.min().item()), float(x.max().item())


def ensure_finite(name: str, t: torch.Tensor) -> bool:
    if not torch.is_tensor(t):
        return True
    bad = ~torch.isfinite(t)
    if bad.any():
        n = int(bad.sum().item())
        tmin, tmax = _finite_min_max(t)
        print(f"[NaNGuard] {name}: {n} non-finite (min={tmin}, max={tmax})")
        return False
    return True


def sanitize_logits_for_metrics(logits: torch.Tensor, clamp_val=30.0):
    x = logits.float().detach()
    mask = torch.isfinite(x)
    x = torch.where(mask, x, torch.zeros_like(x))
    return x.clamp(-clamp_val, clamp_val)


def _get_autocast_dtype(device: str):
    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


def set_bn_eval_and_freeze(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.eval()
            if mod.affine:
                mod.weight.requires_grad_(False)
                mod.bias.requires_grad_(False)

def _load_pretrained_ct_encoder(model, ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        print(f"WARNING: 找不到 CT 预训练权重: {ckpt_path}")
        return

    print(f"加载 CT 预训练权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    has_prefix = any(k.startswith("CTEncoder.") for k in sd.keys())
    if has_prefix:
        new_sd = {}
        prefix = "CTEncoder."
        for k, v in sd.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                new_sd[new_k] = v
        sd = new_sd

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"CTEncoder missing keys: {len(missing)}")
    if unexpected:
        print(f"CTEncoder unexpected keys: {len(unexpected)}")

# ---------------- 预训练加载：Table ----------------
def _load_pretrained_table_encoder(model, ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        print(f"WARNING: 找不到表格预训练权重: {ckpt_path}")
        return

    print(f"加载表格预训练权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    has_prefix = any(k.startswith("TableEncoder.") for k in sd.keys())
    if has_prefix:
        new_sd = {}
        prefix = "TableEncoder."
        for k, v in sd.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                new_sd[new_k] = v
        sd = new_sd

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"TableEncoder missing keys: {len(missing)}")
    if unexpected:
        print(f"TableEncoder unexpected keys: {len(unexpected)}")


def _load_pretrained_agg(model, path: str):
    if not os.path.isfile(path):
        print(f"[WARN] Aggregator pretrained not found: {path}")
        return

    print(f"[INFO] Loading Aggregator pretrained: {path}")
    ckpt = torch.load(path, map_location="cpu")

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # 只保留 TitanAgg.aggregator 下的权重
    agg_sd = {}
    for k, v in sd.items():
        if k.startswith("aggregator."):
            agg_sd[k[len("aggregator."):]] = v
        elif k.startswith("TitanAgg.aggregator."):
            agg_sd[k[len("TitanAgg.aggregator."):]] = v

    missing, unexpected = model.load_state_dict(
        agg_sd, strict=False
    )

    print(f"[INFO] Aggregator weights loaded.")
    print(f"[INFO] → missing: {missing}")
    print(f"[INFO] → unexpected: {unexpected}")

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_classifier(model):
    for p in model.gen_classifier.parameters():
        p.requires_grad_(True)
    model.gen_classifier.train()

def set_trainable(module: nn.Module, trainable: bool):
    for p in module.parameters():
        p.requires_grad_(trainable)
    module.train() if trainable else module.eval()