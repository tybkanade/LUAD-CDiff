# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Sequence, Dict, Any
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FourModalFromXLSX(Dataset):
    """
    四模态融合：
        1) CT 体积
        2) 表格特征（数值 + 分类）
        3) 术中病理（冰冻切片）单向量 [768]
        4) 术后病理（前缀目录下多个 h5）List[Tensor(768)]

    输出：
        {
            "CT": FloatTensor [1,T,H,W]
            "intra_wsi": FloatTensor [768]
            "post_wsi_list": List[FloatTensor(768)]
            "x_num": FloatTensor [n_num]
            "x_cat": LongTensor [n_cat] or None
            "missing_mask": BoolTensor [n_num]
            "y": FloatTensor one-hot[C]  (可选)
        }
    """

    # ----------------------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------------------
    def __init__(self,
                 # ============ XLSX =============
                 xlsx_path: str,
                 ct_col: str = "__npy_path__",

                 # ============ 术中病理 ============
                 intra_wsi_id_col: str = "冰冻切片号",
                 intra_wsi_dir: str = "path of the features of the frozen section",

                 # ============ 术后病理 ============
                 post_wsi_prefix_col: str = "术后病理前缀",
                 post_wsi_root: str = "path of the postoperative pathology",
                 post_glob: str = "*.h5",
                 post_sort: bool = True,
                 post_raise_empty: bool = True,

                 # ============ 标签 =============
                 label_col: Optional[str] = "术后病理诊断类型",
                 label_names: Sequence[str] = ("AAH/AIS", "MIA", "AC"),

                 # ============ 表格特征 =============
                 num_cols: List[str] = (),
                 binary_cols: List[str] = (),
                 pleura_col: Optional[str] = None,
                 gender_col: Optional[str] = None,
                 density_col: Optional[str] = None,

                 # ============ CT 预处理 ============
                 use_zscore: bool = True,
                 window: Optional[Tuple[float, float]] = None,
                 to_112: bool = False,

                 filter_missing: bool = True,
                 verbose: bool = True,
                 ):
        super().__init__()

        self.xlsx_path = xlsx_path
        self.df = pd.read_excel(self.xlsx_path)

        # ------------------------------------------------------------------
        # 关键列检查
        # ------------------------------------------------------------------
        require_cols = [ct_col, intra_wsi_id_col, post_wsi_prefix_col]
        if label_col:
            require_cols.append(label_col)
        for c in require_cols:
            if c not in self.df.columns:
                raise ValueError(f"Excel 缺少必需列：{c}")

        self.ct_col = ct_col
        self.intra_wsi_id_col = intra_wsi_id_col
        self.intra_wsi_dir = intra_wsi_dir

        self.post_wsi_prefix_col = post_wsi_prefix_col
        self.post_root = Path(post_wsi_root)
        self.post_glob = post_glob
        self.post_sort = post_sort
        self.post_raise_empty = post_raise_empty

        # 标签信息
        self.label_col = label_col
        self.label_names = [str(s) for s in label_names]
        self.label2idx = {n: i for i, n in enumerate(self.label_names)}

        # ------------------------------------------------------------------
        # 预处理 Excel
        # ------------------------------------------------------------------
        df = self.df.copy()
        df[intra_wsi_id_col] = df[intra_wsi_id_col].astype(str).str.strip()
        df[post_wsi_prefix_col] = df[post_wsi_prefix_col].astype(str).str.strip()
        if label_col:
            df[label_col] = df[label_col].astype(str).str.strip()

        # --- 路径构造 ---
        df["__ct_path__"] = df[ct_col].astype(str).str.strip()
        df["__intra_h5__"] = df[intra_wsi_id_col].apply(
            lambda s: os.path.join(intra_wsi_dir, f"{s}.h5")
        )

        exists_ct = df["__ct_path__"].apply(os.path.isfile)
        exists_intra = df["__intra_h5__"].apply(os.path.isfile)

        label_ok = True
        if label_col:
            label_ok = df[label_col].isin(self.label_names)

        keep = exists_ct & exists_intra & label_ok if filter_missing else (exists_ct & exists_intra)
        df = df[keep].copy().reset_index(drop=True)

        if verbose:
            print(f"[FourModalFromXLSX] 总 {len(self.df)} 条 → 保留 {len(df)} 条")
            print(f"   缺CT: {(~exists_ct).sum()}  缺术中WSI: {(~exists_intra).sum()}"
                  + (f" 非法标签: {(~label_ok).sum()}" if label_col else ""))

        self.df = df
        self.paths_ct = df["__ct_path__"].tolist()
        self.paths_intra = df["__intra_h5__"].tolist()

        # ------------------------------------------------------------------
        # 表格特征
        # ------------------------------------------------------------------
        # 数值列
        if num_cols:
            xnum_df = df[num_cols].copy()
            mask_df = (xnum_df == -1)
            xnum_df = xnum_df.mask(mask_df, other=0.0)
            self.x_num = torch.tensor(xnum_df.values, dtype=torch.float32)
            self.missing_mask = torch.tensor(mask_df.values, dtype=torch.bool)
        else:
            self.x_num = torch.zeros((len(df), 0), dtype=torch.float32)
            self.missing_mask = torch.zeros((len(df), 0), dtype=torch.bool)

        # 分类列
        cat_list = []
        # 1) 二分类列
        for col in binary_cols:
            arr = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
            arr = arr.replace({-1: 2})
            cat_list.append(torch.tensor(arr.values, dtype=torch.long))

        # 2) 密度
        if density_col:
            def map_density(v):
                if pd.isna(v): return -1
                mp = {"磨玻璃": 0, "部分实性": 1, "实性": 2}
                s = str(v).strip()
                if s in mp: return mp[s]
                try:
                    iv = int(float(s))
                    return iv if iv in (0, 1, 2) else -1
                except:
                    return -1
            arr = df[density_col].apply(map_density).astype(int).replace({-1: 3})
            cat_list.append(torch.tensor(arr.values, dtype=torch.long))

        # 3) 胸膜关系
        if pleura_col:
            arr = pd.to_numeric(df[pleura_col], errors="coerce").fillna(-1).astype(int)
            arr = arr.clip(-1, 4).replace({-1: 5})
            cat_list.append(torch.tensor(arr.values, dtype=torch.long))

        # 4) 性别
        if gender_col:
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
            arr = df[gender_col].apply(map_gender).astype(int).replace({-1: 2})
            cat_list.append(torch.tensor(arr.values, dtype=torch.long))

        self.x_cat = torch.stack(cat_list, dim=1) if len(cat_list) > 0 else None

        # ------------------------------------------------------------------
        # y
        # ------------------------------------------------------------------
        if label_col:
            idx = torch.tensor(df[label_col].map(self.label2idx).fillna(-1).astype(int).values)
            valid = idx >= 0
            idx2 = idx.clamp(min=0)
            y = F.one_hot(idx2, num_classes=len(self.label_names)).float()
            y[~valid] = 0.0
            self.y = y
        else:
            self.y = None

        # ------------------------------------------------------------------
        # CT
        # ------------------------------------------------------------------
        self.use_zscore = use_zscore
        self.window = window
        self.to_112 = to_112

    # ----------------------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------------------
    def _load_ct(self, path: str) -> torch.Tensor:
        vol = np.load(path).astype("float32")
        if self.window:
            lo, hi = self.window
            vol = np.clip(vol, lo, hi)
        if self.use_zscore:
            m, s = float(vol.mean()), float(vol.std())
            vol = (vol - m) / (s + 1e-6)
        vol = np.moveaxis(vol, -1, 0)   # (T,H,W)
        x = torch.from_numpy(vol)[None]
        if self.to_112:
            T = x.shape[1]
            x = F.interpolate(x, size=(T,112,112), mode="trilinear", align_corners=False)
        return x.contiguous()

    # ------------ 术中病理单向量 ------------
    def _load_intra(self, h5_path: str) -> torch.Tensor:
        with h5py.File(h5_path, "r") as f:
            arr = f["features"][()]
        arr = np.asarray(arr).squeeze()
        return torch.tensor(arr, dtype=torch.float32)

    # ------------ 术后病理多 h5 列表 ----------
    def _load_post_list(self, prefix: str) -> List[torch.Tensor]:
        case_dir = self.post_root / prefix
        if not case_dir.exists():
            if self.post_raise_empty:
                raise FileNotFoundError(f"未找到术后病理目录: {case_dir}")
            else:
                return []
        h5_list = glob.glob(str(case_dir / self.post_glob))
        h5_list = sorted(h5_list) if self.post_sort else h5_list
        if len(h5_list) == 0:
            if self.post_raise_empty:
                raise FileNotFoundError(f"{prefix} 下无 h5 文件")
            else:
                return []
        out = []
        for p in h5_list:
            with h5py.File(p, "r") as f:
                arr = torch.from_numpy(f["features"][:]).float()
            out.append(arr)

        return out


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        ct = self._load_ct(self.paths_ct[idx])

        intra_wsi = self._load_intra(self.paths_intra[idx])

        post_list = self._load_post_list(row[self.post_wsi_prefix_col])

        out = {
            "CT": ct,
            "intra_wsi": intra_wsi,                 # [768]
            "post_wsi_list": post_list,            # List[768]
            "x_num": self.x_num[idx],
            "x_cat": self.x_cat[idx] if self.x_cat is not None else None,
            "missing_mask": self.missing_mask[idx],
        }
        if self.y is not None:
            out["y"] = self.y[idx]
        return out

    def __len__(self):
        return len(self.df)

