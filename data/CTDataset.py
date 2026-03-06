# dataset_ct_from_xlsx.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class CTNPYFromXLSX(Dataset):
    """
    从 xlsx 的【编号】-> .npy 文件名映射构建数据集。
    标签列：术后病理诊断类型
    返回：
        x: torch.FloatTensor [1, 49, 57, 57]  (C=1, T=49, H=57, W=57)
        y: torch.FloatTensor [num_classes]  (one-hot of label_names)
    """

    def __init__(
        self,
        npy_dir: str = "path of npy",
        xlsx_path: str = "path of xlsx",
        id_col: str = "编号",
        label_col: str = "术后病理诊断类型",
        label_names = ("AAH/AIS", "MIA", "AC"),
        # 预处理相关
        use_zscore: bool = True,
        window: tuple | None = None,
        to_112: bool = False,
        # 归一化相关
        global_mean: float | None = None,
        global_std: float | None = None,
        minmax_range: tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
    ):
        super().__init__()
        self.npy_dir   = npy_dir
        self.xlsx_path = xlsx_path
        self.id_col    = id_col
        self.label_col = label_col

        # 固定标签空间 & 映射
        self.label_names = [str(s) for s in label_names]
        self.label2idx = {name: i for i, name in enumerate(self.label_names)}
        self.idx2label = {i: name for name, i in self.label2idx.items()}

        # 读取表
        df = pd.read_excel(self.xlsx_path)
        if id_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Excel 缺少必要列：{id_col} / {label_col}")

        # 标准化两列（去空白，转字符串）
        df[id_col] = df[id_col].astype(str).str.strip().str.upper()
        df[label_col] = df[label_col].astype(str).str.strip()

        # 构造文件路径
        df["__npy_path__"] = df[id_col].apply(lambda s: os.path.join(self.npy_dir, f"{s}.npy"))

        # 过滤：存在文件、标签合法
        df["__exists__"] = df["__npy_path__"].apply(os.path.isfile)
        df["__label_ok__"] = df[label_col].isin(self.label_names)
        kept = df[df["__exists__"] & df["__label_ok__"]].copy()

        # 记录过滤信息
        if verbose:
            miss_file = (df["__exists__"] == False).sum()
            bad_label = (df["__label_ok__"] == False).sum()
            print(f"[CTNPYFromXLSX] 总行数: {len(df)}；保留: {len(kept)}；缺文件: {miss_file}；非法标签: {bad_label}")

        # 必备属性（供 split_dataset_stratified 使用）
        self.df = kept.reset_index(drop=True)
        self.paths = self.df["__npy_path__"].tolist()

        # 预处理配置
        self.use_zscore   = use_zscore
        self.window       = window
        self.to_112       = to_112
        self.global_mean  = global_mean
        self.global_std   = global_std
        self.minmax_range = minmax_range

    def __len__(self):
        return len(self.df)

    def _load_npy_as_tensor(self, path: str) -> torch.Tensor:
        vol = np.load(path).astype("float32")           # (H, W, D)

        # 可选 window（clip）
        if self.window is not None:
            lo, hi = self.window
            vol = np.clip(vol, lo, hi)

        # ----------------
        # 归一化
        # ----------------
        if self.use_zscore:
            # 优先使用全局 mean/std；若未提供则按每个 volume 单独计算
            if (self.global_mean is not None) and (self.global_std is not None):
                m, s = float(self.global_mean), float(self.global_std)
            else:
                m, s = float(vol.mean()), float(vol.std())
            vol = (vol - m) / (s + 1e-6)
        else:
            # min-max 到指定区间（默认 [0,1]）
            vmin, vmax = float(vol.min()), float(vol.max())
            if vmax > vmin:
                a, b = self.minmax_range
                vol = (vol - vmin) / (vmax - vmin)          # [0, 1]
                vol = vol * (b - a) + a                     # [a, b]

        vol = np.moveaxis(vol, -1, 0)
        x = torch.from_numpy(vol)[None]

        # 如需放大空间尺寸到 112
        if self.to_112:
            # 保持 T 不变，仅放大 H,W
            T = x.shape[1]
            x = F.interpolate(x, size=(T, 112, 112), mode="trilinear", align_corners=False)

        return x.contiguous()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["__npy_path__"]
        label_str = str(row[self.label_col]).strip()

        # x: [1, 49, 57, 57]
        x = self._load_npy_as_tensor(path)

        # y: one-hot [num_classes]
        y_idx = self.label2idx[label_str]
        y_onehot = F.one_hot(
            torch.tensor(y_idx, dtype=torch.long),
            num_classes=len(self.label_names)
        ).to(torch.float32)

        return x, y_onehot


