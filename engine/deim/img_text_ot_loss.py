# ot_align_1vM_flat.py
# ------------------------------------------------------------
# 1 ↔ 10  Batched Unbalanced OT for flat batch (P,1,d) ↔ (P,10,d)
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


class OTAlign_Batch(nn.Module):
    """
    Unbalanced Sinkhorn OT for a flat batch of P 正样本：
    每个样本 1 个视觉特征 对 M 个文本/属性 token (1↔M OT)

    forward
    -------
    v_pos        : (P, 1, dim_v)
    a_slots      : (P, 10, dim_t)
    w_col        : (P, 10)   external column‑logits
    valid_mask   : (P, 10) | None     True → 有效；False → 缺失该属性
    Returns
    -------
    loss_ot  : scalar          OT cost(含熵正则, 取样本均值)
    r        : (P, 10)          每样本 softmax 后列先验
    """

    def __init__(
        self,
        *,
        blur: float = 0.05,
        scaling: float = 0.9,
        ent_tau: float = 0.03,
    ):
        super().__init__()
        self.ot = SamplesLoss(
            "sinkhorn", p=2, blur=blur, scaling=scaling, debias=False,
            backend="tensorized"
        )
        self.ent_tau = ent_tau

    # --------------------------------------------------------
    def forward(
        self,
        v_pos: torch.Tensor,          # (P,1,Dv)
        a_slots: torch.Tensor,        # (P,10,Dt)
        *,
        w_col: torch.Tensor,          # (M,) or (P,M)
        valid_mask: torch.Tensor | None = None,   # (P,M) bool, True=valid
    ) -> torch.Tensor:

        P, _, Dv = v_pos.shape
        _, M, Dt = a_slots.shape
        device = v_pos.device

        # ---------- 行分布 μ  (P,1) ----------
        mu = torch.ones(P, 1, device=device)          # 每样本唯一视觉点→权重 1

        v = F.normalize(v_pos, dim=-1)                    # (P,1,d)
        t = F.normalize(a_slots, dim=-1)                  # (P,M,d)

        if w_col.dim() == 1:                              # 全局共享 logits
            r = F.softmax(w_col, dim=0).expand(P, M).clone()
        else:                                             # (P,M) 预先处理好的列权重
            r = w_col

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            r = r * valid_mask
            r = r / r.sum(dim=1, keepdim=True).clamp_min(1e-8)
            t = t * valid_mask.unsqueeze(-1)

        # ---------- 批量 Sinkhorn 1↔10 ----------
        loss_ot = self.ot(mu, v, r, t).mean()   # scalar  已经取平均，相当于/Num_正样本

        return loss_ot
