# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.configs import DiffusionConfig


class GaussianDiffusion1D(nn.Module):
    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.m = cfg.m  # 向量维度

        self.register_buffer("mean", torch.zeros(1, self.m))
        self.register_buffer("std", torch.ones(1, self.m))

        def set_norm(mean: torch.Tensor, std: torch.Tensor):
            self.mean.copy_(mean)
            self.std.copy_(std)

        self.set_norm = set_norm


        # C = getattr(cfg, "num_classes", 3)
        # self.register_buffer("class_mean", torch.zeros(C, self.m))
        # self.register_buffer("class_std", torch.ones(C, self.m))
        # self.register_buffer("has_class_stats", torch.tensor(0, dtype=torch.uint8))
        #
        # def set_class_stats(class_mean: torch.Tensor, class_std: torch.Tensor):
        #     """
        #     class_mean/class_std: [C, m]
        #     """
        #     assert class_mean.dim() == 2 and class_mean.shape[1] == self.m
        #     assert class_std.dim() == 2 and class_std.shape[1] == self.m
        #     self.class_mean.copy_(class_mean)
        #     self.class_std.copy_(class_std)
        #     self.has_class_stats.fill_(1)
        #
        # self.set_class_stats = set_class_stats


        T = cfg.timesteps

        if cfg.schedule == "cosine":
            import math
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float32)
            abar = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            abar = abar / abar[0]
            betas = 1 - (abar[1:] / abar[:-1])
            betas = betas.clamp(1e-8, 0.999)

        elif cfg.schedule == "linear":
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, dtype=torch.float32)

        else:
            raise ValueError("schedule must be 'cosine' or 'linear'")

        alphas = 1 - betas
        abar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', abar)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(abar))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - abar))

        abar_prev = torch.cat([torch.ones(1), abar[:-1]], dim=0)
        posterior_variance = betas * (1 - abar_prev) / (1 - abar)
        self.register_buffer('posterior_variance', posterior_variance)

    def _gather_class_stats(self, cls: torch.Tensor, B: int):
        """
        cls: [B] long
        return mu_y, std_y: [B, m]
        """
        cls = cls.long().view(B)
        mu = self.class_mean.index_select(0, cls)  # [B,m]
        sd = self.class_std.index_select(0, cls)  # [B,m]
        sd = sd.clamp_min(1e-6)
        return mu, sd

    def _gather_coeff(self, buf, t, B):
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            val = buf[int(t)].expand(B)
        else:
            val = buf[t]
        return val.view(B, 1)


    # def _norm_cond_tokens(self, cond_tokens: Union[List[torch.Tensor], torch.Tensor]):
    #     """
    #     cond_tokens:
    #         - list: [ct, wsi_intra, table, ...]，每个 [B,D]
    #         - tensor: [B,L,D]
    #         - tensor: [B,D]
    #
    #     返回:
    #         cond_out: [B,L,D]
    #     """
    #     # list → [B,L,D]
    #     if isinstance(cond_tokens, list):
    #         cond_tokens = torch.stack(cond_tokens, dim=1)  # [B,L,D]
    #
    #     # [B,D] → [B,1,D]
    #     if cond_tokens.dim() == 2:
    #         cond_tokens = cond_tokens.unsqueeze(1)
    #
    #     return cond_tokens

    def _norm_cond_tokens(self, cond_tokens):
        if isinstance(cond_tokens, list):
            cond_tokens = torch.stack(cond_tokens, dim=1)  # [B,L,D]
        return cond_tokens

    # ---------------------------------------------------------
    #              q(x_t|x_0) —— 前向加噪
    # ---------------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        B, m = x0.shape
        if noise is None:
            noise = torch.randn_like(x0)

        fac1 = self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
        fac2 = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
        return fac1 * x0 + fac2 * noise

    # ---------------------------------------------------------
    #            用 eps 还原 x0
    # ---------------------------------------------------------
    def x0_from_xt_eps(self, x_t, t, eps_pred):
        B, m = x_t.shape
        inv = 1.0 / self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
        s1m = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
        x0 = inv * (x_t - s1m * eps_pred)
        return x0.clamp(-1, 1) if self.cfg.clip_x0 else x0

    # ---------------------------------------------------------
    # pred → x0_hat 与 eps_hat
    # ---------------------------------------------------------
    def _pred_to_x0_eps(self, x_t, t, pred, B):
        if self.cfg.pred_type == "eps":
            alpha = self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
            sigma = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
            x0_hat = (x_t - sigma * pred) / (alpha + 1e-12)
            eps_hat = pred

        elif self.cfg.pred_type == "v":
            alpha = self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
            sigma = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
            x0_hat = alpha * x_t - sigma * pred
            eps_hat = sigma * x_t + alpha * pred

        else:
            raise ValueError

        if self.cfg.clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)

        return x0_hat, eps_hat

    # ---------------------------------------------------------
    # 扩散训练 loss
    # ---------------------------------------------------------
    def p_losses(self, x0, cond, cls=None):
        B, m = x0.shape
        device = x0.device

        # normalize x0
        x0_norm = (x0 - self.mean) / self.std

        cond_norm = self._norm_cond_tokens(cond)    # [B,L,D]

        t = torch.randint(0, self.cfg.timesteps, (B,), device=device)
        noise = torch.randn_like(x0_norm)
        x_t = self.q_sample(x0_norm, t, noise=noise)

        pred = self.model(x_t, t, cond_norm, cls)

        if self.cfg.pred_type == "eps":
            target = noise
        else:
            alpha = self._gather_coeff(self.sqrt_alphas_cumprod, t, B)
            sigma = self._gather_coeff(self.sqrt_one_minus_alphas_cumprod, t, B)
            target = alpha * noise - sigma * x0_norm

        elem = (pred - target) ** 2 if self.cfg.loss_type == "l2" else (pred - target).abs()
        return elem.mean()

    # ---------------------------------------------------------
    # DDIM 单步
    # ---------------------------------------------------------
    def _ddim_step_trainable(self, x_t, t, t_prev, cond_norm, cls=None):
        B, m = x_t.shape
        abar_prev = self._gather_coeff(self.alphas_cumprod, t_prev, B)

        pred = self.model(x_t, t, cond_norm, cls)
        x0_hat, eps = self._pred_to_x0_eps(x_t, t, pred, B)

        sqrt_prev = torch.sqrt(abar_prev)
        sqrt_one_minus_prev = torch.sqrt(1 - abar_prev)

        xt_prev = sqrt_prev * x0_hat + sqrt_one_minus_prev * eps
        same_mask = (t_prev == t).view(B, 1)

        return torch.where(same_mask, x0_hat, xt_prev)

    # ---------------------------------------------------------
    # 时间子序列
    # ---------------------------------------------------------
    def _make_subseq(self, steps):
        T = self.cfg.timesteps
        if steps >= T:
            return torch.arange(T - 1, -1, -1)
        idx = torch.round(torch.linspace(T - 1, 0, steps)).long()
        idx = torch.unique_consecutive(idx)
        if idx[0] != T - 1:
            idx = torch.cat([torch.tensor([T - 1]), idx])
        if idx[-1] != 0:
            idx = torch.cat([idx, torch.tensor([0])])
        return idx

    # ---------------------------------------------------------
    # 可微 sample_trainable
    # ---------------------------------------------------------
    def sample_trainable(self, cond_tokens, cls=None, steps=20):

        if isinstance(cond_tokens, list):
            # list = [tensor(B,D), tensor(B,D), ...]
            cond_tokens = torch.stack(cond_tokens, dim=1)  # [B,L,D]

        B, L, D = cond_tokens.shape

        device = self.mean.device

        cond_norm = self._norm_cond_tokens(cond_tokens)

        x_t = torch.randn(B, self.cfg.m, device=device)

        steps = steps or self.cfg.train_ddim_steps
        idx = self._make_subseq(steps).to(device)

        for i in range(len(idx)):
            t = idx[i].repeat(B)
            t_prev = (idx[i] if i == len(idx) - 1 else idx[i + 1]).repeat(B)

            x_t = self._ddim_step_trainable(
                x_t, t, t_prev, cond_norm, cls
            )

        return x_t

    # ---------------------------------------------------------
    # 推理 sample（normalize → denoise → unnorm）
    # ---------------------------------------------------------
    @torch.no_grad()
    def sample(self, cond, cls=None, steps=None):
        """
        cond:
          - list: [B,D] * L
          - tensor: [B,L,D]
          - tensor: [B,D]
        """
        if isinstance(cond, list):
            B = cond[0].shape[0]
        else:
            B = cond.shape[0]

        device = self.mean.device

        cond_norm = self._norm_cond_tokens(cond)

        x_t = torch.randn(B, self.cfg.m, device=device)

        steps = steps or self.cfg.timesteps
        idx = self._make_subseq(steps).to(device)

        for i in range(len(idx)):
            t = idx[i].repeat(B)
            t_prev = (idx[i] if i == len(idx) - 1 else idx[i + 1]).repeat(B)

            pred = self.model(x_t, t, cond_norm, cls)
            x0_hat, eps = self._pred_to_x0_eps(x_t, t, pred, B)

            abar_prv = self._gather_coeff(self.alphas_cumprod, t_prev, B)

            x_t = torch.where(
                (t_prev == t).view(B, 1),
                x0_hat,
                torch.sqrt(abar_prv) * x0_hat + torch.sqrt(1 - abar_prv) * eps
            )

        # 反标准化回原空间
        return x_t * self.std + self.mean

    # ---------------------------------------------------------
    #      train_step
    # ---------------------------------------------------------
    def train_step(self, x0, cond, cls=None, steps_ddim=None):

        diff_loss = self.p_losses(x0, cond, cls)

        x0_gen_norm = self.sample_trainable(cond, cls, steps=steps_ddim or self.cfg.train_ddim_steps)

        x0_gen = x0_gen_norm * self.std + self.mean
        rec_loss = F.mse_loss(x0_gen, x0)

        x0_norm = F.normalize(x0, dim=1)
        x0_gen_norm = F.normalize(x0_gen, dim=1)
        angle_loss = 1.0 - torch.sum(
            x0_norm * x0_gen_norm, dim=1
        ).mean()

        return {
            "diff_loss": diff_loss,
            "recon_loss": rec_loss,
            "x0_gen": x0_gen,
            "angle_loss": angle_loss,
        }