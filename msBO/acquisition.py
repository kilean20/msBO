import math
import torch
from torch import Tensor
from typing import Optional, Sequence
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, SampleReducingMCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler, IIDNormalSampler
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)

from types import MethodType
from botorch.models.model import FantasizeMixin




# ---------- helpers ----------
def _expand_tasks_concat(X_ctrl: Tensor, task_ids: Sequence[int]) -> Tensor:
    *B, q, d_ctrl = X_ctrl.shape
    Tsel = len(task_ids)
    X_rep = X_ctrl.unsqueeze(-2).expand(*B, q, Tsel, d_ctrl)  # … x q x Tsel x d_ctrl
    t_ids = torch.as_tensor(task_ids, device=X_ctrl.device, dtype=X_ctrl.dtype).view(
        *([1]*len(B)), 1, Tsel, 1
    ).expand(*B, q, Tsel, 1)
    Xt = torch.cat([X_rep, t_ids], dim=-1)                   # … x q x Tsel x (d+1)
    return Xt.reshape(*B, q*Tsel, d_ctrl + 1)                # … x (q*Tsel) x (d+1)

def _split_other_ids(T: int, start: int, J: int):
    left = list(range(0, start))
    right = list(range(start + J, T))
    return left + right, len(left), len(right)


class fixed_state_qUCB(SampleReducingMCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: float,
        S: int,
        J: int,
        s_idx: int,
        objective: MCAcquisitionObjective,
        sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        mc_samples: int = 128,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([mc_samples])
            )
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.beta_prime = math.sqrt(float(beta) * math.pi / 2.0)  # == _get_beta_prime
        self.S = int(S); self.J = int(J); self.s_idx = int(s_idx); self.T = self.S * self.J
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X_ctrl: Tensor) -> Tensor:
        # NOTE: X_ctrl may include pending points appended by the decorator.
        *B, q_tot, d_ctrl = X_ctrl.shape
        m = 0 if self.X_pending is None else self.X_pending.shape[-2]
        q = q_tot - m  # candidate batch size

        start = self.s_idx * self.J
        fixed_ids = list(range(start, start + self.J))

        # 1) Sample ONLY for fixed-state tasks (candidates only)
        X_fix = _expand_tasks_concat(X_ctrl[..., :q, :], fixed_ids)          # … x (qJ) x (d+1)
        post_fix = self.model.posterior(X_fix, observation_noise=False)
        s_fix = self.get_posterior_samples(post_fix).squeeze(-1)            # nmc x … x (qJ)
        s_fix = s_fix.view(s_fix.shape[0], *B, q, self.J)                   # nmc x … x q x J

        # 2) Posterior means for other tasks (candidates only)
        other_ids, nL, nR = _split_other_ids(self.T, start, self.J)
        if other_ids:
            X_oth = _expand_tasks_concat(X_ctrl[..., :q, :], other_ids)
            mu_oth = self.model.posterior(X_oth, observation_noise=False).mean.squeeze(-1)
            mu_oth = mu_oth.view(*B, q, self.T - self.J)                    # … x q x (T-J)
        else:
            mu_oth = None; nL = nR = 0

        # 3) Assemble nmc x … x q x T  (samples for fixed, means elsewhere)
        nmc = s_fix.shape[0]
        Y = s_fix.new_empty(nmc, *B, q, self.T)
        if nL > 0:
            Y[..., :nL] = mu_oth[..., :nL].unsqueeze(0).expand(nmc, *mu_oth[..., :nL].shape)
        Y[..., start:start + self.J] = s_fix
        if nR > 0:
            Y[..., start + self.J:] = mu_oth[..., nL:].unsqueeze(0).expand(nmc, *mu_oth[..., nL:].shape)

        # 4) Composite objective per-sample (shape: nmc x … x q)
        obj = self.objective(Y, X_ctrl[..., :q, :])

        # 5) Apply per-sample UCB transform, then reduce samples (parent handles reduction)
        acq_per_sample = self._sample_forward(obj)     # (nmc x … x q)
        acq_per_sample_max = acq_per_sample.amax(dim=-1)  # (nmc x …)  # max over q first
        reduced_over_samples = acq_per_sample_max.mean(dim=0)  # (…)  # then average over samples
        return reduced_over_samples
        # acq_per_sample = self._sample_forward(obj)     # (nmc x … x q)
        # reduced_over_samples = acq_per_sample.mean(dim=0)  # (… x q)
        # return reduced_over_samples.amax(dim=-1)       # (…)

    # Mirror BoTorch qUCB’s per-sample transform
    def _sample_forward(self, obj: Tensor) -> Tensor:
        mean = obj.mean(dim=0)                                # … x q
        return mean + self.beta_prime * (obj - mean).abs()    # nmc x … x q



class fixed_state_qLogEI(SampleReducingMCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: float,
        S: int,
        J: int,
        s_idx: int,
        objective: MCAcquisitionObjective,
        xi: float = 0.0,
        sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        mc_samples: int = 128,
        eps: float = 1e-12,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([mc_samples]),
            )
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.best_f = float(best_f); self.xi = float(xi); self.eps = float(eps)
        self.S = int(S); self.J = int(J); self.s_idx = int(s_idx); self.T = self.S * self.J
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X_ctrl: Tensor) -> Tensor:
        *B, q_tot, d_ctrl = X_ctrl.shape
        m = 0 if self.X_pending is None else self.X_pending.shape[-2]
        q = q_tot - m

        start = self.s_idx * self.J
        fixed_ids = list(range(start, start + self.J))

        X_fix = _expand_tasks_concat(X_ctrl[..., :q, :], fixed_ids)
        post_fix = self.model.posterior(X_fix, observation_noise=False)
        s_fix = self.get_posterior_samples(post_fix).squeeze(-1)            # nmc x … x (qJ)
        s_fix = s_fix.view(s_fix.shape[0], *B, q, self.J)                   # nmc x … x q x J

        other_ids, nL, nR = _split_other_ids(self.T, start, self.J)
        if other_ids:
            X_oth = _expand_tasks_concat(X_ctrl[..., :q, :], other_ids)
            mu_oth = self.model.posterior(X_oth, observation_noise=False).mean.squeeze(-1)
            mu_oth = mu_oth.view(*B, q, self.T - self.J)
        else:
            mu_oth = None; nL = nR = 0

        nmc = s_fix.shape[0]
        Y = s_fix.new_empty(nmc, *B, q, self.T)
        if nL > 0:
            Y[..., :nL] = mu_oth[..., :nL].unsqueeze(0).expand(nmc, *mu_oth[..., :nL].shape)
        Y[..., start:start + self.J] = s_fix
        if nR > 0:
            Y[..., start + self.J:] = mu_oth[..., nL:].unsqueeze(0).expand(nmc, *mu_oth[..., nL:].shape)

        obj = self.objective(Y, X_ctrl[..., :q, :])                         # nmc x … x q

        per_sample = self._sample_forward(obj)         # improvements (nmc x … x q)
        per_sample_max = per_sample.amax(dim=-1)       # (nmc x …)  # max over q per sample
        mean_ei = per_sample_max.mean(dim=0)           # (…)  # average max over samples
        return (mean_ei + self.eps).log()              # (…)
    
        # per_sample = self._sample_forward(obj)         # improvements (nmc x … x q)
        # mean_ei = per_sample.mean(dim=0)               # (… x q)
        # return (mean_ei + self.eps).log().amax(dim=-1) # (…)

    # return per-sample improvement (no log, no averaging here)
    def _sample_forward(self, obj: Tensor) -> Tensor:
        return (obj - (self.best_f + self.xi)).clamp_min(0.0)



class two_state_qLogEI(MCAcquisitionFunction):
    """
    Joint q-batch LogEI for a MIXED batch across two consecutive states.

    The batch is split:  X_ctrl[..., :q_s, :]  → evaluated at state  s_idx
                         X_ctrl[..., q_s:,  :]  → evaluated at state  ns_idx

    For each sub-group the relevant state's J tasks are drawn from the GP
    posterior via MC sampling; all other tasks use their posterior *mean*
    (same as fixed_state_qLogEI, but applied to two groups independently).
    The composite objective is evaluated jointly over all q candidates and the
    log-EI is taken of the *best* improvement in the batch.

    This replaces a dummy oracle call at X_best when switching states.  The
    ns_idx candidate is chosen by the acquisition itself, so it is
    informative and non-redundant — the acquisition function balances
    exploration/exploitation for next_s given everything already known.

    Parameters
    ----------
    q_s   : number of candidates for the *current* state (s_idx). The
            remaining q - q_s candidates are for next state (ns_idx).
    """

    def __init__(
        self,
        model: Model,
        best_f: float,
        S: int,
        J: int,
        s_idx: int,
        ns_idx: int,
        q_s: int,
        objective: MCAcquisitionObjective,
        xi: float = 0.0,
        sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        mc_samples: int = 128,
        eps: float = 1e-12,
    ) -> None:
        if sampler is None:
            sampler = IIDNormalSampler(sample_shape=torch.Size([mc_samples]))
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.best_f = float(best_f)
        self.xi = float(xi)
        self.eps = float(eps)
        self.S = int(S)
        self.J = int(J)
        self.s_idx  = int(s_idx)
        self.ns_idx = int(ns_idx)
        self.q_s    = int(q_s)
        self.T      = S * J
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X_ctrl: Tensor) -> Tensor:
        *B, q_tot, d_ctrl = X_ctrl.shape
        m  = 0 if self.X_pending is None else self.X_pending.shape[-2]
        q  = q_tot - m
        q_s  = min(self.q_s, q)   # guard if q < q_s
        q_ns = q - q_s

        X_cands   = X_ctrl[..., :q, :]          # (*B, q,    d)
        X_s_cands = X_ctrl[..., :q_s,  :]       # (*B, q_s,  d)
        X_ns_cands= X_ctrl[..., q_s:q, :]       # (*B, q_ns, d)

        s_start  = self.s_idx  * self.J
        ns_start = self.ns_idx * self.J
        s_ids    = list(range(s_start,  s_start  + self.J))
        ns_ids   = list(range(ns_start, ns_start + self.J))
        all_ids  = list(range(self.T))
        # "other" task ids seen from each group's perspective
        s_other_ids  = [t for t in all_ids if t not in s_ids ]
        ns_other_ids = [t for t in all_ids if t not in ns_ids]

        # ── 1. MC sample state-s tasks for the s-group candidates ────────────
        X_s_fix = _expand_tasks_concat(X_s_cands, s_ids)       # (*B, q_s*J, d+1)
        post_s  = self.model.posterior(X_s_fix, observation_noise=False)
        samp_s  = self.get_posterior_samples(post_s).squeeze(-1)        # nmc x *B x (q_s*J)
        nmc     = samp_s.shape[0]
        samp_s  = samp_s.view(nmc, *B, q_s, self.J)                    # nmc x *B x q_s x J

        # ── 2. Posterior means for s-group at other tasks ────────────────────
        X_s_oth  = _expand_tasks_concat(X_s_cands, s_other_ids)        # (*B, q_s*(T-J), d+1)
        mu_s_oth = self.model.posterior(X_s_oth, observation_noise=False).mean.squeeze(-1)
        mu_s_oth = mu_s_oth.view(*B, q_s, self.T - self.J)             # *B x q_s x (T-J)

        # ── 3. MC sample next-state tasks for the ns-group candidates ────────
        if q_ns > 0:
            X_ns_fix = _expand_tasks_concat(X_ns_cands, ns_ids)
            post_ns  = self.model.posterior(X_ns_fix, observation_noise=False)
            samp_ns  = self.get_posterior_samples(post_ns).squeeze(-1)
            samp_ns  = samp_ns.view(nmc, *B, q_ns, self.J)

            X_ns_oth  = _expand_tasks_concat(X_ns_cands, ns_other_ids)
            mu_ns_oth = self.model.posterior(X_ns_oth, observation_noise=False).mean.squeeze(-1)
            mu_ns_oth = mu_ns_oth.view(*B, q_ns, self.T - self.J)

        # ── 4. Assemble Y (nmc, *B, q, T) ────────────────────────────────────
        Y = samp_s.new_empty(nmc, *B, q, self.T)

        # s-group rows: MC samples for s tasks, means for the rest
        Y[..., :q_s, s_other_ids ] = mu_s_oth.unsqueeze(0).expand(nmc, *B, q_s, self.T - self.J)
        Y[..., :q_s, s_start : s_start + self.J] = samp_s

        if q_ns > 0:
            # ns-group rows: MC samples for ns tasks, means for the rest
            Y[..., q_s:, ns_other_ids] = mu_ns_oth.unsqueeze(0).expand(nmc, *B, q_ns, self.T - self.J)
            Y[..., q_s:, ns_start : ns_start + self.J] = samp_ns

        # ── 5. Composite objective → joint LogEI over all q candidates ────────
        obj         = self.objective(Y, X_cands)                         # nmc x *B x q
        improvements= (obj - (self.best_f + self.xi)).clamp_min(0.0)
        max_imp     = improvements.amax(dim=-1)                          # nmc x *B
        return (max_imp.mean(dim=0) + self.eps).log()                   # *B


class two_state_qUCB(MCAcquisitionFunction):
    """
    Joint q-batch UCB for a MIXED batch across two consecutive states.

    Mirrors two_state_qLogEI but uses the UCB per-sample transform.
    X_ctrl[..., :q_s, :] → evaluated at state s_idx
    X_ctrl[..., q_s:,  :] → evaluated at state ns_idx
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        S: int,
        J: int,
        s_idx: int,
        ns_idx: int,
        q_s: int,
        objective: MCAcquisitionObjective,
        sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        mc_samples: int = 128,
    ) -> None:
        if sampler is None:
            sampler = IIDNormalSampler(sample_shape=torch.Size([mc_samples]))
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.beta_prime = math.sqrt(float(beta) * math.pi / 2.0)
        self.S = int(S); self.J = int(J)
        self.s_idx  = int(s_idx)
        self.ns_idx = int(ns_idx)
        self.q_s    = int(q_s)
        self.T      = S * J
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X_ctrl: Tensor) -> Tensor:
        *B, q_tot, d_ctrl = X_ctrl.shape
        m  = 0 if self.X_pending is None else self.X_pending.shape[-2]
        q  = q_tot - m
        q_s  = min(self.q_s, q)
        q_ns = q - q_s

        X_cands    = X_ctrl[..., :q, :]
        X_s_cands  = X_ctrl[..., :q_s,  :]
        X_ns_cands = X_ctrl[..., q_s:q, :]

        s_start  = self.s_idx  * self.J
        ns_start = self.ns_idx * self.J
        s_ids    = list(range(s_start,  s_start  + self.J))
        ns_ids   = list(range(ns_start, ns_start + self.J))
        all_ids  = list(range(self.T))
        s_other_ids  = [t for t in all_ids if t not in s_ids ]
        ns_other_ids = [t for t in all_ids if t not in ns_ids]

        X_s_fix = _expand_tasks_concat(X_s_cands, s_ids)
        post_s  = self.model.posterior(X_s_fix, observation_noise=False)
        samp_s  = self.get_posterior_samples(post_s).squeeze(-1)
        nmc     = samp_s.shape[0]
        samp_s  = samp_s.view(nmc, *B, q_s, self.J)

        X_s_oth  = _expand_tasks_concat(X_s_cands, s_other_ids)
        mu_s_oth = self.model.posterior(X_s_oth, observation_noise=False).mean.squeeze(-1)
        mu_s_oth = mu_s_oth.view(*B, q_s, self.T - self.J)

        if q_ns > 0:
            X_ns_fix  = _expand_tasks_concat(X_ns_cands, ns_ids)
            post_ns   = self.model.posterior(X_ns_fix, observation_noise=False)
            samp_ns   = self.get_posterior_samples(post_ns).squeeze(-1)
            samp_ns   = samp_ns.view(nmc, *B, q_ns, self.J)
            X_ns_oth  = _expand_tasks_concat(X_ns_cands, ns_other_ids)
            mu_ns_oth = self.model.posterior(X_ns_oth, observation_noise=False).mean.squeeze(-1)
            mu_ns_oth = mu_ns_oth.view(*B, q_ns, self.T - self.J)

        Y = samp_s.new_empty(nmc, *B, q, self.T)
        Y[..., :q_s, s_other_ids ] = mu_s_oth.unsqueeze(0).expand(nmc, *B, q_s,  self.T - self.J)
        Y[..., :q_s, s_start : s_start + self.J] = samp_s
        if q_ns > 0:
            Y[..., q_s:, ns_other_ids] = mu_ns_oth.unsqueeze(0).expand(nmc, *B, q_ns, self.T - self.J)
            Y[..., q_s:, ns_start : ns_start + self.J] = samp_ns

        obj = self.objective(Y, X_cands)                 # nmc x *B x q
        mean = obj.mean(dim=0)                           # *B x q
        ucb  = mean + self.beta_prime * (obj - mean).abs()   # nmc x *B x q
        return ucb.amax(dim=-1).mean(dim=0)              # *B

    def _sample_forward(self, obj: Tensor) -> Tensor:
        mean = obj.mean(dim=0)
        return mean + self.beta_prime * (obj - mean).abs()


class CompositePosteriorMean(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective,
        T: int,
        sampler: Optional[MCSampler] = None,
        posterior_transform=None,
        X_pending: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.T = int(T)

    def forward(self, X_ctrl: torch.Tensor) -> torch.Tensor:
        *B, q, _ = X_ctrl.shape
        task_ids = list(range(self.T))
        X_all = _expand_tasks_concat(X_ctrl, task_ids)
        mu = self.model.posterior(X_all).mean.squeeze(-1).view(*B, q, self.T)
        vals_q = self.objective(mu.unsqueeze(0), X_ctrl).squeeze(0)  # ... x q
        return vals_q.amax(dim=-1)  # <-- scalar per t-batch (shape: *B)



def make_expand_fixed_state(J: int, T: int, s_idx: int):
    def expand(X_ctrl: Tensor) -> Tensor:
        *B, q, d_ctrl = X_ctrl.shape
        task_ids = list(range(s_idx * J, s_idx * J + J))
        return _expand_tasks_concat(X_ctrl, task_ids)                  # fantasize ONLY these J tasks
    return expand

def _patch_noiseless_fantasies(model):
    if getattr(model, "_fant_noiseless_patched", False):
        return model
    def _fant_noiseless(self, X, sampler, observation_noise=None, **kwargs):
        post = self.posterior(X, observation_noise=False)
        Y_fantasized = sampler(post)  # nf x batch x n' x m
        return self.condition_on_observations(X=X, Y=Y_fantasized, **kwargs)
    model.fantasize = MethodType(_fant_noiseless, model)
    model._fant_noiseless_patched = True
    return model

def fixed_state_qKG(
    model: Model,
    S: int,
    J: int,
    s_idx: int,
    objective: MCAcquisitionObjective,
    num_fantasies: int = 64,
    current_value: Optional[torch.Tensor | float] = None,
    X_pending: Optional[torch.Tensor] = None,
):
    model = _patch_noiseless_fantasies(model)

    T = int(S) * int(J)
    expand = make_expand_fixed_state(J=J, T=T, s_idx=s_idx)

    def _valfunc_argfac(**_kwargs):
        return {"T": T}

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=num_fantasies,
        objective=objective,           # REQUIRED for multi-output
        X_pending=X_pending,
        current_value=current_value,   # optional baseline
        project=lambda X: X, #X[..., :1, :],   # <-- keep only actual q (you use q=1)
        expand=expand,                 # fantasize only fixed state's J tasks
        valfunc_cls=CompositePosteriorMean,
        valfunc_argfac=_valfunc_argfac,
    )
