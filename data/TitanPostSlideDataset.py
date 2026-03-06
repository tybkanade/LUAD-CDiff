# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Dict, List, Any
import glob

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

CLASS_NAMES = ["AAH/AIS", "MIA", "AC"]
N_CLASSES = 3

# 表格至少包含两列：术后病理前缀、术后病理诊断类型
TABLE_PATH = "path of the tabular data"
COL_GROUP  = "术后病理前缀"          # 病例ID/子目录名
COL_LABEL  = "术后病理诊断类型"       # 标签列

# h5 根目录：/ROOT_DIR/<术后病理前缀>/*.h5
ROOT_DIR         = "path of the postoperative pathology"
H5_GLOB_PATTERN  = "*.h5"
SORT_H5          = True
RAISE_IF_EMPTY   = True

class PostOpWSIPrefixDataset(Dataset):

    LABELS = CLASS_NAMES

    def __init__(self,
                 table_path: str = TABLE_PATH,
                 root_dir: str = ROOT_DIR,
                 group_col: str = COL_GROUP,
                 label_col: str = COL_LABEL,
                 h5_glob: str = H5_GLOB_PATTERN,
                 sort_h5: bool = SORT_H5,
                 raise_if_empty: bool = RAISE_IF_EMPTY):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.group_col = group_col
        self.label_col = label_col
        self.h5_glob = h5_glob
        self.sort_h5 = sort_h5
        self.raise_if_empty = raise_if_empty

        # 读表（支持 xlsx/csv）
        ext = Path(table_path).suffix.lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(table_path)
        elif ext in [".csv", ".txt"]:
            df = pd.read_csv(table_path)
        else:
            raise ValueError(f"不支持的表格格式: {table_path}")

        # 基础清洗 & 列检查
        for col in [group_col, label_col]:
            if col not in df.columns:
                raise ValueError(f"缺少列: {col}")
            df[col] = df[col].astype(str).str.strip()

        # 若表里同一个前缀重复出现，只保留一条
        df = df.drop_duplicates(subset=[group_col]).reset_index(drop=True)

        # 扫描每个 prefix 的 h5 清单
        groups: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            prefix = row[group_col]
            label  = row[label_col]

            case_dir = self.root_dir / prefix
            if not case_dir.exists():
                raise FileNotFoundError(f"未找到病例子目录: {case_dir}")

            h5_list = [Path(p) for p in glob.glob(str(case_dir / self.h5_glob))]
            if self.sort_h5:
                h5_list = sorted(h5_list, key=lambda p: p.name)

            if len(h5_list) == 0:
                msg = f"病例 {prefix} 在 {case_dir} 下未匹配到 {self.h5_glob}"
                if self.raise_if_empty:
                    raise FileNotFoundError(msg)
                else:
                    # 跳过该病例
                    continue

            groups.append({
                "group_id": prefix,
                "paths": [str(p) for p in h5_list],
                "label": label
            })

        if len(groups) == 0:
            raise RuntimeError("未收集到任何病例（groups 为空）。请检查表格或根目录。")

        self.groups = groups

        self.df = pd.DataFrame({
            group_col: [g["group_id"] for g in groups],
            label_col: [g["label"]     for g in groups],
        })
        self.label2idx = {lb: i for i, lb in enumerate(self.LABELS)}

    def __len__(self):
        return len(self.groups)

    @staticmethod
    def _read_one_h5(h5_path: str):
        """返回 (features[M,768] float32, coords[M,2], patch_size_lv0:int)"""
        with h5py.File(h5_path, "r") as f:
            feats = torch.from_numpy(f["features"][:]).float()  # [M,768]
            coords = torch.from_numpy(f["coords"][:])           # [M,2]
            ps_lv0 = int(f["coords"].attrs["patch_size_level0"])
        return feats, coords, ps_lv0

    def _one_hot(self, label: str) -> torch.Tensor:
        if label not in self.label2idx:
            raise ValueError(f"未知标签: {label}（应为 {self.LABELS}）")
        y = torch.zeros(len(self.LABELS), dtype=torch.float32)
        y[self.label2idx[label]] = 1.0
        return y

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item   = self.groups[idx]
        prefix = item["group_id"]
        paths  = item["paths"]
        label  = item["label"]

        feats_list, coords_list, ps_list = [], [], []
        for p in paths:
            f, c, ps = self._read_one_h5(p)
            feats_list.append(f)
            coords_list.append(c)
            ps_list.append(ps)

        return {
            "case_prefix": prefix,
            "features_list": feats_list,           # List[Tensor(M,768)]
            "coords_list": coords_list,            # List[Tensor(M,2)]
            "patch_size_lv0_list": ps_list,        # List[int]
            "label_onehot": self._one_hot(label),  # [C]
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {}
        out["case_prefix"] = [b["case_prefix"] for b in batch]                  # List[str]（B项）
        out["features_list"] = [b["features_list"] for b in batch]              # List[List[Tensor]]
        out["coords_list"] = [b["coords_list"] for b in batch]                  # List[List[Tensor]]
        out["patch_size_lv0_list"] = [b["patch_size_lv0_list"] for b in batch]  # List[List[int]]
        out["label_onehot"] = torch.stack([b["label_onehot"] for b in batch], dim=0)  # (B,C)
        return out

