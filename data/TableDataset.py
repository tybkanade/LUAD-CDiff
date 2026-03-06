import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, List

class TabularDataset(Dataset):

    def __init__(self,
                 xlsx_path: str,
                 num_cols: List[str],
                 binary_cols: List[str],
                 pleura_col: str,
                 gender_col: str,
                 density_col: str,
                 label_col: Optional[str] = None):
        super().__init__()
        df = pd.read_excel(xlsx_path)

        # -------- 连续数值部分 --------
        for c in num_cols:
            if c not in df.columns:
                raise ValueError(f"找不到连续列：{c}")
        x_num_df = df[num_cols].copy()

        # 缺失掩码（== -1）
        num_missing_mask_df = (x_num_df == -1)
        # 缺失位置输入置 0.0（语义由 missing_mask 告知模型）
        x_num_df = x_num_df.mask(num_missing_mask_df, other=0.0)
        x_num = torch.tensor(x_num_df.values, dtype=torch.float32)                # [B, n_num]
        missing_mask = torch.tensor(num_missing_mask_df.values, dtype=torch.bool) # [B, n_num]

        # -------- 离散类别部分 --------
        cat_cols_order: List[str] = []
        cat_arrays: List[torch.Tensor] = []
        cat_cardinalities: List[int] = []

        # 1) 二分类：0/1；-1 -> 2
        for c in binary_cols:
            if c not in df.columns:
                raise ValueError(f"找不到二分类列：{c}")
            s = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
            s = s.replace({-1: 2})
            cat_cols_order.append(c)
            cat_arrays.append(torch.tensor(s.values, dtype=torch.long))
            cat_cardinalities.append(3)

        # 2) 密度：文本 -> 数字；-1 -> 3
        if density_col not in df.columns:
            raise ValueError(f"找不到列：{density_col}")
        def map_density(v):
            if pd.isna(v):
                return -1
            s = str(v).strip()
            mapping = {"磨玻璃": 0, "部分实性": 1, "实性": 2}
            if s in mapping:
                return mapping[s]
            # 若原本是数字字符
            try:
                iv = int(float(s))
                return iv if iv in (0, 1, 2) else -1
            except:
                return -1
        den = df[density_col].apply(map_density).astype(int).replace({-1: 3})
        cat_cols_order.append(density_col)
        cat_arrays.append(torch.tensor(den.values, dtype=torch.long))
        cat_cardinalities.append(4)

        # 3) 与胸膜的关系：0..4；-1 -> 5
        if pleura_col not in df.columns:
            raise ValueError(f"找不到列：{pleura_col}")
        pleura = pd.to_numeric(df[pleura_col], errors="coerce").fillna(-1).astype(int)
        pleura = pleura.clip(lower=-1, upper=4).replace({-1: 5})
        cat_cols_order.append(pleura_col)
        cat_arrays.append(torch.tensor(pleura.values, dtype=torch.long))
        cat_cardinalities.append(6)

        # 4) 性别：女=0，男=1；-1/其他 -> 2
        if gender_col not in df.columns:
            raise ValueError(f"找不到列：{gender_col}")
        def map_gender(v):
            if pd.isna(v): return -1
            s = str(v).strip().lower()
            if s in ["男", "male", "m", "1"]: return 1
            if s in ["女", "female", "f", "0"]: return 0
            try:
                iv = int(float(s))
                return iv if iv in (0, 1) else -1
            except:
                return -1
        g = df[gender_col].apply(map_gender).astype(int).replace({-1: 2})
        cat_cols_order.append(gender_col)
        cat_arrays.append(torch.tensor(g.values, dtype=torch.long))
        cat_cardinalities.append(3)

        # 拼接 x_cat
        x_cat = torch.stack(cat_arrays, dim=1) if len(cat_arrays) > 0 else None  # [B, n_cat]

        # -------- 标签 y（one-hot）--------
        y = None
        label2idx = {"AAH/AIS": 0, "MIA": 1, "AC": 2}
        if label_col is not None:
            if label_col not in df.columns:
                raise ValueError(f"找不到标签列：{label_col}")

            df[label_col] = (
                df[label_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"0": "AAH/AIS", "1": "MIA", "2": "AC"})
            )

            idx = torch.tensor(df[label_col].map(label2idx).fillna(-1).astype(int).values, dtype=torch.long)
            # 无效标签 -> 全零 one-hot
            valid = idx >= 0
            idx_clamped = idx.clamp(min=0)
            y = F.one_hot(idx_clamped, num_classes=3).to(torch.float32)
            y[~valid] = 0.0

        self.x_num = x_num
        self.x_cat = x_cat
        self.missing_mask = missing_mask
        self.y = y

        self.num_cols = num_cols
        self.cat_cols = cat_cols_order
        self.cat_cardinalities = cat_cardinalities

        self.df = df.copy()
        self.label_col = label_col        # "术后病理诊断类型"
        self.label2idx = label2idx

    def __len__(self):
        return self.x_num.size(0)

    def __getitem__(self, idx):
        item = {
            "x_num": self.x_num[idx],
            "x_cat": self.x_cat[idx],
            "missing_mask": self.missing_mask[idx],
        }
        if self.y is not None:
            item["y"] = self.y[idx]  # one-hot: [4], dtype=float32
        return item
