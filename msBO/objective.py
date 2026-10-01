from typing import List, Union, Optional, Callable, Tuple
import numpy as np
import logging
import torch
from torch import Tensor
import torch.nn.functional as F



def task_id(s: int, j: int, J: int) -> int:
    return s * J + j

def split_task(t: int, J: int) -> Tuple[int, int]:
    s = t // J
    j = t % J
    return s, j


def regular_simplex_scan_codes(n_factors: int) -> np.ndarray:
    """Return the minimum-size centered orthogonal code for ``n_factors``.

    The returned array has shape ``(n_factors + 1, n_factors)``.  Its columns
    have zero mean, are mutually orthogonal, and have equal squared norm.  A
    single global rescaling makes the largest absolute entry one while
    preserving those properties.

    When the columns represent independently perturbed quadrupoles, the
    state variance of a linear BPM response is proportional to the sum of the
    squared individual-quadrupole responses.  Consequently, simultaneous
    coded scans cannot hide individual responses through cancellation.
    """
    if isinstance(n_factors, bool):
        raise ValueError("n_factors must be a positive integer")
    try:
        n = int(n_factors)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_factors must be a positive integer") from exc
    if n < 1 or n != n_factors:
        raise ValueError("n_factors must be a positive integer")

    # Helmert contrasts form an orthonormal basis for the subspace orthogonal
    # to the all-ones vector.  Their rows are the vertices of a regular
    # simplex centered on the nominal configuration.
    codes = np.zeros((n + 1, n), dtype=float)
    for column in range(n):
        denominator = np.sqrt((column + 1) * (column + 2))
        codes[: column + 1, column] = 1.0 / denominator
        codes[column + 1, column] = -(column + 1) / denominator

    return codes / np.max(np.abs(codes))


def softmin(x, tau: float = 1e-3, dim: int = -1):
    # = -softmax(x/τ)^T x when τ→0 → min
    w = torch.softmax(-x / tau, dim=dim)
    return (w * x).sum(dim=dim)

# Element-wise ELU-based threshold around 0.95
def elu_step(y, center=0.95, width=0.05):
    scaled = (y - center) / width
    return F.elu(scaled) + 1


def beam_loss_penalty(
    transmission_ratios: Tensor,
    *,
    loss_scale: float = 0.1,
    softmin_tau: float = 1e-2,
) -> Tensor:
    """Return a dimensionless penalty based on the worst state transmission.

    ``transmission_ratios`` contains measured BPM magnitude divided by its
    state-specific reference.  The final dimension enumerates machine states.
    A 5% loss produces approximately 0.5 penalty and a 10% loss approximately
    1.0 when ``loss_scale=0.1``.  Ratios at or above reference are unpenalized.
    """
    if loss_scale <= 0:
        raise ValueError("loss_scale must be positive")
    if softmin_tau <= 0:
        raise ValueError("softmin_tau must be positive")
    worst_ratio = softmin(transmission_ratios, tau=softmin_tau, dim=-1)
    return torch.relu(1.0 - worst_ratio) / float(loss_scale)


class BPMvar_minimization:
    def __init__(
        self,
        S: int,
        J: int,
        w_beam_center=0.2,
        w_beam_loss=1.0,
        beam_loss_scale=0.1,
        beam_loss_softmin_tau=1e-2,
    ):
        '''
        S : number of states
        J : number of tasks for each state. 
            Important: All task must be BPM pos except last task. Last task is BPM MAG based contraint. 
        '''
        self.S, self.J = S, J
        self.w_beam_center = w_beam_center
        self.w_beam_loss = w_beam_loss
        self.beam_loss_scale = beam_loss_scale
        self.beam_loss_softmin_tau = beam_loss_softmin_tau

    def _index(self, samples: Tensor, s: int, j: int) -> Tensor:
        # samples: [..., q, T] ; returns [..., q]
        t = task_id(s, j, self.J)
        return samples[..., t]

    def __call__(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # stack across states → shape [..., q, S]
        ys = [torch.stack([self._index(samples, s, t) for s in range(self.S)], dim=-1) for t in range(self.J)]
        losses = [
                                 torch.var (ys[t],    dim=-1, correction=0) +   # (..., q)
            self.w_beam_center * torch.mean(ys[t]**2, dim=-1)     # (..., q)
            for t in range(self.J - 1)
        ]
        if len(losses) > 0:
            # stack -> (..., q, J-1), then sum over the last dim -> (..., q)
            loss_sum = torch.stack(losses, dim=-1).sum(dim=-1)
        else:
            # no variance tasks; create a zero tensor matching the shape we’ll add it to
            loss_sum = torch.zeros_like(ys[-1].min(dim=-1).values)
        loss_penalty = beam_loss_penalty(
            ys[-1],
            loss_scale=self.beam_loss_scale,
            softmin_tau=self.beam_loss_softmin_tau,
        )

        return 1.0 - loss_sum - self.w_beam_loss * loss_penalty


class QuadrupoleCentering:
    """Composite objective for beam-based quadrupole centering.

    A centered beam is insensitive to a quadrupole-strength scan.  The
    objective therefore maximizes the negative, normalized BPM-position
    variance across machine states.  For multiple quadrupoles, use a centered
    full-rank orthogonal state design such as :func:`regular_simplex_scan_codes`;
    moving every quadrupole together measures only one collective mode and can
    hide individual responses through cancellation.  Task ordering must follow
    msBO's convention: all ``J`` BPM readings for state 0, then all readings
    for state 1, and so on.

    Parameters
    ----------
    S:
        Number of quadrupole scan states.
    J:
        Number of tasks returned at each state. This is the number of BPM
        position readings, plus one when ``beam_loss_task=True``.
    norms:
        Per-BPM position scales.  A value of 1.0 means the corresponding BPM
        is measured in the same units as the desired centering tolerance.
    weights:
        Relative per-BPM weights.  The weighted loss is normalized by their
        sum, so multiplying every weight by the same value has no effect.
    beam_loss_task:
        If true, the final task at every state is the minimum BPM magnitude
        ratio relative to a state-specific reference.  It is excluded from the
        position variance and contributes a worst-state beam-loss penalty.
    """

    def __init__(
        self,
        S: int,
        J: int,
        norms: Optional[Union[List[float], Tensor]] = None,
        weights: Optional[Union[List[float], Tensor]] = None,
        beam_loss_task: bool = False,
        beam_loss_weight: float = 1.0,
        beam_loss_scale: float = 0.1,
        beam_loss_softmin_tau: float = 1e-2,
    ):
        self.S = int(S)
        self.J = int(J)
        self.beam_loss_task = bool(beam_loss_task)
        self.position_J = self.J - 1 if self.beam_loss_task else self.J
        self.beam_loss_weight = float(beam_loss_weight)
        self.beam_loss_scale = float(beam_loss_scale)
        self.beam_loss_softmin_tau = float(beam_loss_softmin_tau)
        if self.S < 2:
            raise ValueError("QuadrupoleCentering requires at least two scan states")
        if self.position_J < 1:
            raise ValueError("QuadrupoleCentering requires at least one BPM task")
        if self.beam_loss_weight < 0:
            raise ValueError("beam_loss_weight must be non-negative")
        if self.beam_loss_scale <= 0 or self.beam_loss_softmin_tau <= 0:
            raise ValueError("beam-loss scale and softmin temperature must be positive")

        self.norms = torch.as_tensor(
            [1.0] * self.position_J if norms is None else norms,
            dtype=torch.get_default_dtype(),
        ).view(-1)
        self.weights = torch.as_tensor(
            [1.0] * self.position_J if weights is None else weights,
            dtype=torch.get_default_dtype(),
        ).view(-1)

        if self.norms.numel() != self.position_J or self.weights.numel() != self.position_J:
            raise ValueError("norms and weights must each have one entry per BPM-position task")
        if torch.any(self.norms <= 0):
            raise ValueError("all norms must be positive")
        if torch.any(self.weights < 0) or not bool(self.weights.sum() > 0):
            raise ValueError("weights must be non-negative with a positive sum")

    def __call__(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if samples.shape[-1] != self.S * self.J:
            raise ValueError(
                f"expected {self.S * self.J} state-task values, "
                f"got shape {tuple(samples.shape)}"
            )

        # (..., q, S*J) -> (..., q, S, J)
        y = samples.unflatten(-1, (self.S, self.J))
        y_position = y[..., :self.position_J]
        norms = self.norms.to(device=y.device, dtype=y.dtype)
        weights = self.weights.to(device=y.device, dtype=y.dtype)
        y_scaled = y_position / norms

        # Population variance matches the deterministic multi-condition scan
        # interpretation and remains well-scaled for a two-state scan.
        state_variance = torch.var(y_scaled, dim=-2, correction=0)
        loss = (state_variance * weights).sum(dim=-1) / weights.sum()
        if self.beam_loss_task:
            transmission_ratios = y[..., -1].movedim(-1, -1)
            loss = loss + self.beam_loss_weight * beam_loss_penalty(
                transmission_ratios,
                loss_scale=self.beam_loss_scale,
                softmin_tau=self.beam_loss_softmin_tau,
            )
        return -loss


# class MultiTaskIndexing:
#     def __init__(self,
#                  task_names: List[str],
#                  ):
#         self.task_names = list(task_names)

#     def __call__(self, task_names: str):
#         return self.get_task_index(task_names)
        
#     def get_task_index(self, task_names):
#         return [self.task_names.index(pv) for pv in task_names]
        

# class CompositeObjective:
#     def __init__(self, objfuncs, y_index, weights=None, input_names=None, output_names=None):
#         assert len(y_index) == len(objfuncs), "Mismatch between number of tasks and index groups."
#         self.objfuncs = objfuncs
#         self.y_index = y_index
#         if input_names is not None:
#             assert len(input_names) == len(objfuncs), "Mismatch between number of tasks and index groups."
#             for y_i, name_i in zip(y_index, input_names):
#                 assert len(name_i) == len(y_i), "Mismatch between number of tasks and index groups."
#             self.input_names = list(input_names)
#         if output_names is not None:
#             assert len(objfuncs) == len(output_names), "Mismatch between number of tasks and index groups."
#             self.output_names = list(output_names)
        
#         self.n_obj = len(objfuncs)  
#         assert self.n_obj == len(y_index), "Mismatch between number of tasks and index groups."

#         if weights is None:
#             self.weights = torch.ones(len(objfuncs)) / self.n_obj
#         else:
#             assert self.n_obj == len(weights), "Weights should match number of tasks."
#             if not isinstance(weights, torch.Tensor):  
#                 with torch.no_grad():
#                     self.weights = torch.tensor(weights, dtype=torch.float32)
#                     self.weights = self.weights / self.weights.sum()
#             else:
#                 self.weights = weights / weights.sum()

#     def __call__(self, y, X=None):
#         # print("y.shape",y.shape)
#         # print("y",y)
#         # objs = torch.stack([self.weights[i] * self.objfuncs[i](y[..., self.y_index[i]]) for i in range(self.n_obj)])
#         # print("objs.shape",objs.shape)
#         # print("objs",objs)
#         # objtot = torch.sum(objs,dim=0)  
#         # print('objtot.shape',objtot.shape)    
#         # print('objtot',objtot)  
#         # return objtot
#         return torch.sum(
#             torch.stack([self.weights[i] * self.objfuncs[i](y[..., self.y_index[i]]) for i in range(self.n_obj)]),
#             dim=0
#         )
