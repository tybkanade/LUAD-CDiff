from dataclasses import dataclass, field, asdict
from typing import Tuple, Literal, Dict, Any, Optional, List

n_classes = 3
output_dim = 768

@dataclass
class CTEncoderConfig:
    out_dim: int = output_dim
    bn_momentum: float = 0.1
    bn_eps: float = 1e-5
    pretrained: str = None
    dropout: float = 0.0
    output_dir: str = None
    # 训练相关参数
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = False
    sync_bn: bool = False
    n_classes: int = n_classes
    gn_groups: int = 16
    feature_extraction: bool = True
    use_pretrained: bool = True

@dataclass
class TableEncoderConfig:
    # ===== 输入特征相关 =====
    num_features: int = 4
    """连续特征（数值型列）的数量，如年龄、收入等"""
    cat_cardinalities: Optional[List[int]] = (50, 20)
    """类别特征的取值规模列表，比如 [50, 20] 表示有两列类别特征：
       第一列城市有50种取值，第二列职业有20种取值"""
    emb_dim: int = 32
    """每个类别特征的Embedding维度，所有类别列共享该维度"""
    # ===== 模型结构相关 =====
    hidden_dims: Tuple[int, ...] = (256, 256)
    """MLP隐藏层的宽度列表，表示每一层的神经元个数"""
    out_dim: int = output_dim
    """编码后的特征输出维度（feature embedding size），用于下游融合/分类"""
    activation: Literal["relu", "gelu", "none"] = "relu"
    """激活函数类型，推荐 silu/gelu"""
    dropout: float = 0.1
    """Dropout 比例，0~0.5之间，一般0.1~0.3较合适"""
    use_attention: bool = True
    """是否启用 Transformer 注意力层来增强特征交互"""
    n_heads: int = 4
    """多头注意力的头数，仅在 use_attention=True 时有效"""
    n_attn_layers: int = 1
    """Transformer Encoder 堆叠层数，1~2层即可"""
    feature_extraction: bool = True
    """是否只输出特征向量 (out_dim)，
       True=输出 [B, out_dim]，
       False=输出分类预测 [B, n_classes]"""
    # ===== 任务相关 =====
    n_classes: int = 3
    """分类任务的类别数"""
    use_amp: bool = False
    """是否使用混合精度 (Automatic Mixed Precision, AMP)"""
    sync_bn: bool = False
    """是否使用同步 BatchNorm（多GPU时可选）"""
    # ===== 预训练 & 输出 =====
    pretrained_path: Optional[str] = None
    """预训练权重路径（如果有的话可以加载）"""
    output_dir: Optional[str] = None
    """训练日志/模型的输出目录"""

@dataclass
class TriFusionConfig:
    # 共同维度
    input_dim: int = output_dim      # 通道数
    num_heads: int = 4
    dropout: float = 0.1
    num_classes: int = n_classes
    use_confidence: bool = True
    # 分类头
    classifier_hidden_ratio: float = 0.25

from dataclasses import dataclass
from typing import Optional

@dataclass
class DenoiserPriorConfig:
    """
    配置：DALL·E 2 Prior 风格的噪声预测器
    （FiLM + Self-Attn + Cross-Attn + Transformer block）
    """
    m: int = output_dim                 # x_t / x0 的维度（= 目标 embedding 768）
    cond_dim: int = output_dim          # 条件向量维度
    hidden: int = 1024                  # 主体 hidden（推荐大于 m，例如 1024）
    # =====================================================
    # 时间嵌入
    # =====================================================
    time_emb_dim: int = 256             # 时间 embedding 维度
    time_emb_scale: float = 1.0
    time_emb_base: float = 10000.0
    time_emb_use_2pi: bool = True       # 推荐使用 2π 版本来增强周期性

    # =====================================================
    # 深度 Transformer 主干（Self-Attn）
    # =====================================================
    depth: int = 8                      # 主干总深度
    n_heads: int = 16                   # Self-Attn 头数
    mlp_ratio: float = 4.0              # FFN 隐层放大倍数
    attn_dropout: float = 0.1
    dropout: float = 0.1
    droppath: float = 0.1               # StochasticDepth（0.1~0.2）

    # =====================================================
    # Cross-Attention
    # =====================================================
    cross_depth: int = 4                # Cross-Attn 层数
    cross_heads: int = 8                # Cross-Attn 头数
    cond_refine_heads: int = 4          # 条件 tokens 的 self-attn（refine）

    # =====================================================
    # FiLM 调制
    # =====================================================
    film_dim: int = 512                 # cond → gamma/beta 的中间映射维度
    film_dropout: float = 0.0

    # =====================================================
    # Dropout & 正则
    # =====================================================
    layernorm_eps: float = 1e-6
    initializer_range: float = 0.02     # 权重初始化标准差

    # =====================================================
    # 类别条件（可选）
    # =====================================================
    cls_num: int = 3
    cls_emb_dim: int = output_dim                # 用类条件时的 embedding 维度


@dataclass
class DiffusionConfig:
    m : int = output_dim
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"      # "cosine" | "linear"
    loss_type: str = "l2"         # "l1" | "l2"   (噪声回归)
    clip_x0: bool = True          # 采样/训练的 x0_hat/x0_gen 是否 clip 到 [-1,1]
    pred_type: str = "v"  # 预测目标，可选: "eps" 或 "v"
    # min-SNR（可选）
    snr_weighting: bool = True
    snr_gamma: float = 4.0
    # 端到端可微 DDIM 的步数
    train_ddim_steps: int = 64  # 30
    seq_len: int = 3    # 默认序列长度
    x0_stats_path: str = None
    x0_class_stats_path: Optional[str] = None


# ----------------------
# 训练配置
# ----------------------
@dataclass
class TrainingConfig:
    stage: int = 1                        # 1 | 2 | 3
    base_lr: float = 2e-4
    weight_decay: float = 1e-4
    lr_mult: Optional[Dict[str, float]] = field(default=None)  # 模块 lr 倍率
    warmup_epochs: int = 2
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = False
    use_ema: bool = False
    ema_decay: float = 0.9999


@dataclass
class OverallModelConfig:
    tri_dim: int = 768      # TriModalFusion 输出 D
    m: int = 768            # 扩散目标/生成向量维度
    num_classes: int = 3
    hidden: int = 512
    clf_hidden: int = 512
    clf_dropout: float = 0.1
    # 损失权重
    w_diff_stage2: float = 0.4     # 噪声回归
    w_rec_stage2: float = 2.0      # 端到端重建
    w_main_stage2: float = 1.5     # 主分类损失

    w_diff_stage3: float = 0.4  # 噪声回归
    w_rec_stage3: float = 2.0  # 端到端重建
    w_main_stage3: float = 2.0  # 主分类损失

    train_ddim_steps: int = 64  # 20

    clf_pool: str = "attn"
