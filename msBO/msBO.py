import time
from copy import deepcopy as copy

import math
import numpy as np
import concurrent.futures as cf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors


import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from typing import Callable, Dict, List, Optional, Tuple, Union

from botorch.acquisition import qUpperConfidenceBound, qLogExpectedImprovement
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.analytic import PosteriorMean

from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.optimize import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from .utils import proximal_ordered_init_sampler
from .models import train_mtgp
from .dataset import MultiStateDataset, _expand_tasks_all_states, _expand_tasks_fixed_state
from .acquisition import fixed_state_qUCB, fixed_state_qLogEI, fixed_state_qKG


class MultiStateBO:
    def __init__(self,
                 states: list[str],
                 tasks: list[str],
                 control_min: Union[List[float], np.ndarray],
                 control_max: Union[List[float], np.ndarray],
                 multistate_oracle_evaluator: Callable,
                 composite_objective_function: Callable,
                 local_bound_size=None,
                 use_prior_data=False,
                 asynchronous=False,
                 acq_backend: str = "scipy",
                 acq_restarts: int = 4,
                 acq_raw_samples: int = 32,
                 acq_maxiter: int = 50,
                 acq_repeats: int = 1,
                 acq_lr: float = 0.2,
                 acq_patience: int = 15,
                 acq_rel_tol: float = 1e-3,
                 acq_min_iter: int = 15,
                 acq_seed: Optional[int] = None,
                 kg_num_fantasies: int = 64,
                 fixed_mc_samples: int = 512,    # for fixed_state_qUCB/qLogEI
                 TurBO_failure_tolerance=999,
                 TurBO_success_tolerance=2,
                 TurBO_success_threshold=0.95,
                 control_names: Optional[List[str]] = None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):
        self.states = states
        self.tasks = tasks
        self.S, self.J = len(states), len(tasks)
        self.control_min = np.asarray(control_min)
        self.control_max = np.asarray(control_max)
        if control_names is None:
            self.control_names = [f"x[{i}]" for i in range(len(control_min))]
        else:
            self.control_names = control_names
        self.ndim = len(self.control_min)
        assert self.ndim == len(self.control_max)
        assert np.all(self.control_min <= self.control_max)
        self.multistate_oracle_evaluator = multistate_oracle_evaluator
        self.composite_objective_function = composite_objective_function
        self.mc_objective = GenericMCObjective(composite_objective_function)
        if local_bound_size is None:
            self.local_bound_size_ref = 0.1 * (self.control_max - self.control_min)
        else:
            self.local_bound_size_ref = local_bound_size
        self.local_bound_size = copy(self.local_bound_size_ref)
        self.local_bound_size_min = 2e-2 * (self.control_max - self.control_min)

        self.bounds = np.vstack((control_min, control_max)).T  # Shape bounds as (n, 2)
        self.use_prior_data = use_prior_data
        self.asynchronous = asynchronous
        # Acquisition optimization settings
        self.acq_backend = str(acq_backend).lower()
        if self.acq_backend not in ("torch", "scipy"):
            raise ValueError(f"acq_backend must be 'torch' or 'scipy', got {acq_backend}")

        self.acq_restarts = int(acq_restarts)
        self.acq_raw_samples = int(acq_raw_samples)
        self.acq_maxiter = int(acq_maxiter)
        self.acq_repeats = int(acq_repeats)
        self.acq_lr = float(acq_lr)
        self.acq_patience = int(acq_patience)
        self.acq_rel_tol = float(acq_rel_tol)
        self.acq_min_iter = int(acq_min_iter)
        self.acq_seed = None if acq_seed is None else int(acq_seed)

        self.kg_num_fantasies = int(kg_num_fantasies)
        self.fixed_mc_samples = int(fixed_mc_samples)

        self.TurBO_success_threshold = float(TurBO_success_threshold)
        self.TurBO_failure_tolerance = TurBO_failure_tolerance
        self.TurBO_success_tolerance = TurBO_success_tolerance
        self.TurBO_failure_counter = 0
        self.TurBO_success_counter = 0

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float64

        self.dataset = MultiStateDataset(x_dim=self.ndim, S=self.S, J=self.J, dtype=self.dtype, device=self.device)
        if self.use_prior_data:
            self.prior_dataset = MultiStateDataset(x_dim=self.ndim, S=self.S, J=self.J, dtype=self.dtype,
                                                   device=self.device)

        oracle = self.multistate_oracle_evaluator()  # get current x,s as well as y when no argument passedW
        self.x0 = np.asarray(oracle['x']).astype(float)

        self.executor = cf.ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.X_pending = None
        self.S_pending = None

        self.prior_model = None
        self.model = None
        self.X_best = None
        self.Y_best = None

        self.history = {
            'model_train_loss': [],
            'prior_model_train_loss': [],
            'time_cost': {'model_train': [],
                          'query': [],
                          'prior_model_train': [],
                          },
            'acq_opt': [],
            'predictions': []  # Store snapshots of model predictions
        }

    def _ingest_oracle(self, oracle: Dict = None) -> None:
        if oracle is None:
            assert self.future is not None, "No future to get result from."
            oracle = self.future.result()
            self.X_pending = None
            self.S_pending = None
            self.future = None
        """Append oracle data into datasets."""
        self.dataset.concat_data(x=oracle["x"], s=self.states.index(oracle["state"]), y=oracle["y"])
        if self.use_prior_data and self.prior_dataset is not None and "ramping_x" in oracle:
            self.prior_dataset.concat_data(x=oracle["ramping_x"], s=self.states.index(oracle["ramping_state"]),
                                           y=oracle["ramping_y"])

    def _get_model_based_X_best(self) -> np.ndarray:
        """
        Argmax over observed unique X of the composite objective evaluated at
        the model posterior MEAN for all tasks (T = S*J). Vectorized & shape-safe.
        """
        if self.model is None or len(self.dataset._x) == 0:
            return self.x0.copy()

        # unique controls we have actually tried
        Xobs = torch.stack(self.dataset._x, dim=0).to(self.device, self.dtype)  # (m, d)
        Xuniq = torch.unique(Xobs, dim=0)  # (n, d)

        n = Xuniq.shape[0]
        T = self.S * self.J
        t_ids = torch.arange(T, device=self.device, dtype=self.dtype).view(1, T, 1)  # (1, T, 1)
        Xrep = Xuniq.view(n, 1, self.ndim).repeat(1, T, 1)  # (n, T, d)
        Xt = torch.cat([Xrep, t_ids.repeat(n, 1, 1)], dim=-1).reshape(n * T, self.ndim + 1)

        with torch.no_grad():
            mu = self.model.posterior(Xt).mean.view(n, T)  # (n, T)

        vals = self.composite_objective_function(mu, None).view(n)  # (n,)
        i_best = torch.argmax(vals)
        return Xuniq[i_best].detach().cpu().numpy(), vals[i_best].item()

    def train_model(self):
        if self.use_prior_data and len(self.prior_dataset) > 0:
            t0 = time.monotonic()
            self.prior_model, prior_loss_history = train_mtgp(self.prior_dataset)
            t1 = time.monotonic()
            self.history['prior_model_train_loss'].append(prior_loss_history)
            self.history['time_cost']['prior_model_train'].append(t1 - t0)
        else:
            self.prior_model = None

        t0 = time.monotonic()
        self.model, loss_history = train_mtgp(self.dataset, prior_model=self.prior_model)
        t1 = time.monotonic()
        self.history['model_train_loss'].append(loss_history)
        self.history['time_cost']['model_train'].append(t1 - t0)
        self.X_best, self.Y_best = self._get_model_based_X_best()
        self._update_turbo_counters_and_trust_region()

        # Snapshot predictions after training
        self.snapshot_predictions()

    def init(self, n_init, local_optimization=True):

        if local_optimization:
            lower_bound = np.maximum(self.x0 - 0.5 * self.local_bound_size, self.control_min)
            upper_bound = np.minimum(self.x0 + 0.5 * self.local_bound_size, self.control_max)
            bounds = np.vstack((lower_bound, upper_bound)).T  # Shape bounds as (n, 2)
        else:
            bounds = self.bounds

        init_x = proximal_ordered_init_sampler(
            n_init - 1,
            bounds=bounds,
            x0=self.x0,
            ramping_rate=None,
            polarity_change_time=None,
            method='sobol',
            seed=None)
        init_x = np.vstack((self.x0.reshape(1, -1), init_x))

        self.X_pending = init_x[0]
        self.S_pending = self.states[0]
        self.future = self.executor.submit(self.multistate_oracle_evaluator,
                                           x=self.X_pending, s=self.S_pending)
        is_first = True
        for s in self.states:
            for x in init_x:
                if is_first:
                    is_first = False
                    continue  # first one already submitted
                # wait for pending result, then submit the next
                self._ingest_oracle()
                # submit next
                self.X_pending = x
                self.S_pending = s
                self.future = self.executor.submit(
                    self.multistate_oracle_evaluator, x=self.X_pending, s=self.S_pending
                )
        if not self.asynchronous:
            self._ingest_oracle()
            self.train_model()

    def step(self,
             s,
             local_optimization: Optional = True,
             acq_type: Optional[str] = None,
             beta: Optional[float] = None,
             X_pending: Optional[Tensor] = None,
             fix_acq_state: Optional[bool] = True,
             ):

        if self.asynchronous:
            self.train_model()
            if X_pending is None:
                X_pending = self.X_pending

        if fix_acq_state:
            fixed_state = s
        else:
            fixed_state = None

        if local_optimization:
            lower_bound = np.maximum(self.X_best - 0.5 * self.local_bound_size, self.control_min)
            upper_bound = np.minimum(self.X_best + 0.5 * self.local_bound_size, self.control_max)
            botorch_bounds = torch.tensor(np.vstack((lower_bound, upper_bound)), device=self.device, dtype=self.dtype)
        else:
            botorch_bounds = torch.tensor(self.bounds.T, device=self.device, dtype=self.dtype)
        candidate = self.query_candidate(botorch_bounds, X_pending=X_pending, fixed_state=fixed_state, acq_type=acq_type,
                                         beta=beta)

        if self.asynchronous:
            self._ingest_oracle()

        self.X_pending = candidate
        self.S_pending = s
        self.future = self.executor.submit(
            self.multistate_oracle_evaluator, x=self.X_pending, s=self.S_pending
        )

        if not self.asynchronous:
            self._ingest_oracle()
            self.train_model()

    def _posterior_mean_composite_at(self, X_ctrl: torch.Tensor):
        device, dtype = self.device, self.dtype
        T = int(self.S * self.J)
        X_ctrl = X_ctrl.to(device=device, dtype=dtype)
        *B, q, d_ctrl = X_ctrl.shape
        t_ids = torch.arange(T, device=device, dtype=dtype)
        X_rep = X_ctrl.unsqueeze(-2).expand(*B, q, T, d_ctrl)
        t_col = t_ids.view(*([1] * len(B)), 1, T, 1).expand(*B, q, T, 1)
        X_all = torch.cat([X_rep, t_col], dim=-1).reshape(*B, q * T, d_ctrl + 1)
        with torch.no_grad():
            mu = self.model.posterior(X_all).mean.squeeze(-1).view(*B, q, T)
            vals = self.mc_objective(mu.unsqueeze(0), X_ctrl)  # 1 x ... x q
        vals = vals.squeeze(0)
        return float(vals.squeeze().item()) if vals.numel() == 1 else vals

    def _get_acquisition(self,
                         botorch_bounds: Optional[Tensor] = None,
                         acq_type: Optional[str] = None,
                         beta: Optional[float] = None,
                         X_pending: Optional[Tensor] = None,
                         fixed_state: Optional[str] = None):

        acq_type = acq_type if acq_type is not None else "KG"

        if X_pending is None and self.X_pending is not None:
            X_pending = torch.as_tensor(self.X_pending, device=self.device, dtype=self.dtype).view(1, 1, -1)

        if fixed_state is None:
            if acq_type in ("LogEI", "EI", "qLogEI", "qEI", None):
                return qLogExpectedImprovement(model=self.model, best_f=self.Y_best, objective=self.mc_objective,
                                               X_pending=X_pending)

            if acq_type in ("qUCB", "UCB"):
                if beta is None:
                    beta = 0.0
                return qUpperConfidenceBound(model=self.model, beta=float(beta), objective=self.mc_objective,
                                             X_pending=X_pending)

        else:
            if isinstance(fixed_state, int):
                s_idx = fixed_state
            else:
                s_idx = self.states.index(fixed_state)
            if not (0 <= s_idx < self.S):
                raise ValueError(f"s_idx {s_idx} out of range for S={self.S}")

            if acq_type in ("qUCB", "UCB"):
                if beta is None:
                    beta = 0.0
                return fixed_state_qUCB(
                    model=self.model,
                    objective=self.mc_objective,
                    X_pending=X_pending,
                    S=self.S, J=self.J, s_idx=s_idx, beta=float(beta), mc_samples=self.fixed_mc_samples
                )

            if acq_type in ("LogEI", "EI", "qLogEI", "qEI", None):
                return fixed_state_qLogEI(
                    model=self.model,
                    best_f=self.Y_best,
                    xi=0.01,
                    objective=self.mc_objective,
                    X_pending=X_pending,
                    mc_samples=self.fixed_mc_samples,
                    S=self.S, J=self.J, s_idx=s_idx,
                )

            if acq_type in ("qKG", "KG"):
                with torch.no_grad():
                    Xbest_ctrl = torch.as_tensor(self.X_best, device=self.device, dtype=self.dtype).view(1, 1, -1)
                    current_value = self._posterior_mean_composite_at(Xbest_ctrl)  # optional baseline

                return fixed_state_qKG(
                    model=self.model,
                    objective=self.mc_objective,
                    S=self.S, J=self.J, s_idx=s_idx,
                    num_fantasies=self.kg_num_fantasies,
                    current_value=current_value,
                    X_pending=X_pending,
                )

        raise ValueError(f"Unknown acq_type: {acq_type}")

    def _optimize_acqf_torch_with_history(
        self,
        acq_function,
        bounds: torch.Tensor,
        num_restarts: int,
        raw_samples: int,
        maxiter: int,
        lr: float,
        patience: int,
        rel_tol: float,
        min_iter: int,
        seed: Optional[int] = None,
    ):
        """Torch-based acquisition optimization for q=1 with box bounds.

        Records per-iteration best acquisition value and loss=-best(acq).
        """
        device, dtype = self.device, self.dtype

        # bounds can be (2, d) or (d, 2)
        if bounds.ndim != 2:
            raise ValueError(f"bounds must be 2D, got {bounds.shape}")
        if bounds.shape[0] == 2:
            lb = bounds[0].to(device=device, dtype=dtype)
            ub = bounds[1].to(device=device, dtype=dtype)
        elif bounds.shape[1] == 2:
            lb = bounds[:, 0].to(device=device, dtype=dtype)
            ub = bounds[:, 1].to(device=device, dtype=dtype)
        else:
            raise ValueError(f"bounds must be (2,d) or (d,2), got {bounds.shape}")

        d = int(lb.numel())
        if d != int(self.ndim):
            raise ValueError(f"bounds dim {d} != self.ndim {self.ndim}")

        # Freeze model parameters (we only need gradients wrt X)
        model_params = list(self.model.parameters()) if self.model is not None else []
        req_grad = [p.requires_grad for p in model_params]
        try:
            for p in model_params:
                p.requires_grad_(False)
            if hasattr(acq_function, "eval"):
                acq_function.eval()

            # ----- 1) raw sampling (Sobol in [0,1]^d) -----
            engine = SobolEngine(dimension=d, scramble=True, seed=seed)
            X_raw01 = engine.draw(int(raw_samples)).to(device=device, dtype=dtype)
            X_raw = lb + (ub - lb) * X_raw01  # (raw_samples, d)

            with torch.no_grad():
                vals_raw = acq_function(X_raw.view(int(raw_samples), 1, d)).view(-1)

            k = min(int(num_restarts), int(raw_samples))
            topk = torch.topk(vals_raw, k=k)
            X0 = X_raw[topk.indices].view(k, 1, d)

            # ----- 2) unconstrained parameterization via logit -----
            eps = 1e-7
            denom = (ub - lb).clamp_min(eps)
            frac = (X0 - lb.view(1, 1, d)) / denom.view(1, 1, d)
            frac = frac.clamp(eps, 1 - eps)
            Z0 = torch.log(frac / (1 - frac))  # logit

            Z = torch.nn.Parameter(Z0.clone())
            opt = torch.optim.Adam([Z], lr=float(lr))

            best_curve: List[float] = []
            loss_curve: List[float] = []
            best_val = -float("inf")
            best_X = None
            it_ran = 0

            for it in range(int(maxiter)):
                it_ran = it + 1

                if (self.future is not None) and self.future.done():
                    break

                opt.zero_grad(set_to_none=True)

                X = lb.view(1, 1, d) + (ub - lb).view(1, 1, d) * torch.sigmoid(Z)  # (k,1,d)
                vals = acq_function(X).view(-1)  # (k,)

                loss = -vals.sum()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    vals_det = vals.detach()
                    v_it = float(vals_det.max().item())
                    best_curve.append(v_it)
                    loss_curve.append(-v_it)

                    if v_it > best_val or best_X is None:
                        best_val = v_it
                        best_X = X[torch.argmax(vals_det)].detach().clone()  # (1,d)

                    # if it_ran >= int(min_iter) and int(patience) > 0 and len(best_curve) > int(patience):
                    #     loss_rng = max(loss_curve[-patience:]) - min(loss_curve[-patience:])
                    #     prev_best = max(best_curve[-int(patience)-1:-1])
                    #     # thresh = float(rel_tol) * abs(prev_best)
                    #     thresh = rel_tol * max(1e-11, loss_rng)
                    #     if (best_curve[-1] - prev_best) <= thresh:
                    #         break

                    if it_ran >= min_iter and patience > 0 and len(best_curve) > patience:
                        prev = best_curve[-patience-1]     # best-so-far then
                        curr = best_curve[-1]              # best-so-far now
                        thresh = rel_tol * max(1.0, abs(prev))
                        if (curr - prev) <= thresh:
                            break

            if best_X is None:
                best_X = X0[0].detach().view(1, d)
                best_val = float(vals_raw[topk.indices[0]].item())

            candidate = best_X.view(1, 1, d)
            value = torch.tensor([best_val], device=device, dtype=dtype)

            info = {
                "backend": "torch",
                "raw_samples": int(raw_samples),
                "num_restarts": int(num_restarts),
                "maxiter": int(maxiter),
                "lr": float(lr),
                "patience": int(patience),
                "rel_tol": float(rel_tol),
                "min_iter": int(min_iter),
                "iters_ran": int(it_ran),
                "best_acq": float(best_val),
                "best_curve": best_curve,
                "loss_curve": loss_curve,
            }
            return candidate, value, info
        finally:
            for p, rg in zip(model_params, req_grad):
                p.requires_grad_(rg)

    def query_candidate(self,
                        botorch_bounds=None,
                        acq_function=None,
                        beta: Optional[float] = None,
                        X_pending: Optional[Tensor] = None,
                        fixed_state: Optional[str] = None,
                        acq_type: Optional[str] = None
                        ):
        assert self.model is not None, "Call init() first."
        t0 = time.monotonic()

        if acq_function is None:
            acq_function = self._get_acquisition(
                beta=beta, botorch_bounds=botorch_bounds, X_pending=X_pending,
                fixed_state=fixed_state, acq_type=acq_type
            )

        best_candidate_t = None
        best_val = -float("inf")
        best_info = None

        call_idx = len(self.history["time_cost"]["query"])
        base_seed = self.acq_seed if self.acq_seed is not None else (12345 + call_idx)

        for r in range(self.acq_repeats):
            if self.acq_backend == "torch":
                candidate, value, info = self._optimize_acqf_torch_with_history(
                    acq_function=acq_function,
                    bounds=botorch_bounds,
                    num_restarts=self.acq_restarts,
                    raw_samples=self.acq_raw_samples,
                    maxiter=self.acq_maxiter,
                    lr=self.acq_lr,
                    patience=self.acq_patience,
                    rel_tol=self.acq_rel_tol,
                    min_iter=self.acq_min_iter,
                    seed=base_seed + r,
                )
            else:
                candidate, value = optimize_acqf(
                    acq_function=acq_function,
                    bounds=botorch_bounds,
                    q=1,
                    num_restarts=self.acq_restarts,
                    raw_samples=self.acq_raw_samples,
                    options={"maxiter": self.acq_maxiter, "ftol": self.acq_rel_tol}
                )
                info = {
                    "backend": "scipy",
                    "raw_samples": int(self.acq_raw_samples),
                    "num_restarts": int(self.acq_restarts),
                    "maxiter": int(self.acq_maxiter),
                    "best_acq": float(value.item()),
                    "best_curve": [],
                    "loss_curve": [],
                    "iters_ran": None,
                }

            v = float(value.item())
            if v > best_val or best_candidate_t is None:
                best_val = v
                best_candidate_t = candidate.detach()
                best_info = info

            if (self.future is not None) and self.future.done():
                break

        dt = time.monotonic() - t0
        self.history['time_cost']['query'].append(dt)

        self.history.setdefault('acq_opt', []).append({
            "acq_type": acq_type,
            "fixed_state": fixed_state,
            "time_sec": float(dt),
            "best_val": float(best_val),
            "info": best_info,
        })

        best_candidate = best_candidate_t.view(-1).cpu().numpy()
        return best_candidate

    def _update_turbo_counters_and_trust_region(self):
        """TurBO counters & trust-region update using the last task's value vs a threshold."""
        if len(self.dataset._x) == 0:
            return

        # Read last measurement vector y (shape: J,) and take its last element
        y_last = self.dataset._y[-1]
        if torch.is_tensor(y_last):
            val = float(y_last[-1].detach().cpu().item())
        else:
            val = float(np.asarray(y_last, dtype=float)[-1])

        thr = float(self.TurBO_success_threshold)
        is_success = (val >= thr)

        # Update counters
        if is_success:
            self.TurBO_success_counter += 1
            self.TurBO_failure_counter = 0
        else:
            self.TurBO_failure_counter += 1
            self.TurBO_success_counter = 0

        grew = False
        shrank = False

        # Adjust trust region on threshold hits
        if self.TurBO_success_counter >= self.TurBO_success_tolerance:
            self.local_bound_size = np.minimum(
                self.local_bound_size * 2.0,
                self.local_bound_size_ref * 8.0,
            )
            grew = True
            self.TurBO_success_counter = 0
            self.TurBO_failure_counter = 0

        elif self.TurBO_failure_counter >= self.TurBO_failure_tolerance:
            self.local_bound_size = np.maximum(
                self.local_bound_size / 2.0,
                self.local_bound_size_ref / 8.0,
            )
            shrank = True
            self.TurBO_success_counter = 0
            self.TurBO_failure_counter = 0

        # Optional breadcrumb
        if hasattr(self, "history"):
            self.history.setdefault("trust_region", []).append({
                "last_task_value": val,
                "threshold": thr,
                "success_counter": int(self.TurBO_success_counter),
                "failure_counter": int(self.TurBO_failure_counter),
                "grew": grew,
                "shrank": shrank,
                "local_bound_size": self.local_bound_size.copy(),
            })

    # ---------------------------------------------------------
    #  New / Updated Visualization Methods
    # ---------------------------------------------------------
    def snapshot_predictions(self):
        """
        Evaluates the current model at the current best X (self.X_best)
        across all states and all tasks.
        Stores x_best, mean, and std for later re-evaluation.
        """
        if self.model is None or self.X_best is None:
            return

        # Ensure X_eval is 3D (batch=1, q=1, d)
        X_eval = torch.tensor(self.X_best, device=self.device, dtype=self.dtype).view(1, 1, -1)

        # Expand to all T tasks (S states * J tasks)
        T = self.S * self.J
        Xt_all = _expand_tasks_all_states(X_eval, T, self.device, self.dtype)  # (1, T, d+1)

        with torch.no_grad():
            posterior = self.model.posterior(Xt_all)
            mean = posterior.mean.view(-1).cpu().numpy()  # (T,)
            variance = posterior.variance.view(-1).cpu().numpy()
            std = np.sqrt(variance)

        # Reshape to (S, J)
        mean = mean.reshape(self.S, self.J)
        std = std.reshape(self.S, self.J)

        # Use dataset._y length for iteration count
        current_iter = len(self.dataset._y)

        self.history['predictions'].append({
            'iteration': current_iter,
            'mean': mean,
            'std': std,
            'x_best': copy(self.X_best)  # Save control inputs for re-computation
        })

    def recompute_prediction_history(self):
        """
        Re-evaluates all historical 'x_best' snapshots using the *current* model.
        This ensures that the plot reflects the model's latest belief about
        past decision points, rather than the belief held at the time.

        Returns:
            List of dicts: Updated prediction history.
        """
        if not self.history['predictions'] or self.model is None:
            return []

        # 1. Collect all historical X_best
        # Stack them into a tensor of shape (N, 1, d)
        history_records = self.history['predictions']
        x_bests_np = np.stack([rec['x_best'] for rec in history_records])
        N = len(x_bests_np)

        X_batch = torch.tensor(x_bests_np, device=self.device, dtype=self.dtype).view(N, 1, -1)

        # 2. Expand tasks for the entire batch at once
        # Output shape: (N, T, d+1)
        T = self.S * self.J
        Xt_batch = _expand_tasks_all_states(X_batch, T, self.device, self.dtype)

        # 3. Run posterior on the whole batch (efficient)
        with torch.no_grad():
            posterior = self.model.posterior(Xt_batch)
            # mean shape: (N, T, 1) -> (N, T)
            means = posterior.mean.squeeze(-1).cpu().numpy()
            # var shape: (N, T, 1) -> (N, T)
            variances = posterior.variance.squeeze(-1).cpu().numpy()
            stds = np.sqrt(variances)

        # 4. Re-pack into dictionary structure
        recomputed_history = []
        for i, rec in enumerate(history_records):
            mu_i = means[i].reshape(self.S, self.J)
            std_i = stds[i].reshape(self.S, self.J)

            recomputed_history.append({
                'iteration': rec['iteration'],
                'mean': mu_i,
                'std': std_i,
                'x_best': rec['x_best']
            })

        return recomputed_history
    
    def recompute_dataset_prediction_history(
        self,
        unique_controls: bool = True,
        round_decimals: int = 12,
    ):
        """
        Re-evaluates model predictions at the control points in self.dataset._x
        using the *current* model.

        This is what you want if you want to plot ALL iterations, including
        initialization points (which occur before the first snapshot_predictions()).

        Args:
            unique_controls: If True, deduplicate repeated controls (e.g. init loops over states).
            round_decimals: Dedup key uses np.round(x, round_decimals) to be robust to float noise.

        Returns:
            List of dicts with keys: iteration, mean (S,J), std (S,J), x (d,)
        """
        if self.model is None or len(self.dataset._x) == 0:
            return []

        # 1) Gather controls in chronological order
        xs = []
        iters = []
        seen = set()

        for k, xk in enumerate(self.dataset._x):
            if torch.is_tensor(xk):
                x_np = xk.detach().cpu().numpy()
            else:
                x_np = np.asarray(xk, dtype=float)

            x_np = np.asarray(x_np, dtype=float).reshape(-1)
            if x_np.size != self.ndim:
                raise ValueError(f"dataset x dim mismatch: got {x_np.size}, expected {self.ndim}")

            if unique_controls:
                key = tuple(np.round(x_np, int(round_decimals)))
                if key in seen:
                    continue
                seen.add(key)

            xs.append(x_np)
            iters.append(k + 1)  # oracle-call index (1-based)

        if len(xs) == 0:
            return []

        x_bests_np = np.stack(xs, axis=0)  # (N, d)
        N = x_bests_np.shape[0]

        # 2) Batch-evaluate posterior at all tasks for all controls (vectorized)
        X_batch = torch.as_tensor(x_bests_np, device=self.device, dtype=self.dtype).view(N, 1, -1)
        T = self.S * self.J
        Xt_batch = _expand_tasks_all_states(X_batch, T, self.device, self.dtype)  # (N, T, d+1)

        with torch.no_grad():
            posterior = self.model.posterior(Xt_batch)
            means = posterior.mean.squeeze(-1).cpu().numpy()        # (N, T)
            variances = posterior.variance.squeeze(-1).cpu().numpy() # (N, T)
            stds = np.sqrt(np.maximum(variances, 0.0))

        # 3) Pack results
        out = []
        for i in range(N):
            out.append({
                "iteration": int(iters[i]),
                "mean": means[i].reshape(self.S, self.J),
                "std": stds[i].reshape(self.S, self.J),
                "x": xs[i],   # control point used for this record
            })
        return out
    

    def plot_state_predictions_history(
        self,
        iterations: Optional[List[int]] = None,
        update_w_recent_model: bool = True,
        ylim: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        fname: str = None,
        source: str = "dataset",
        unique_controls: bool = True,
        round_decimals: int = 12,
        n_skip_iter: Optional[int] = None,   # <<< NEW
    ):
        """
        Plots model predictions across states for each task.

        Key behavior:
        - Oldest (init) and newest (final) are emphasized with alpha=1.
        - All other iterations are drawn with an auto-scaled small alpha that
        decreases as the number of lines increases.
        - Layering is: middle (faint) -> init (strong) -> final (strong, top).

        New behavior (when iterations is None):
        - Optionally subsample iterations so the plot stays readable:
            * keep the 1st and last always
            * keep every n_skip_iter-th point in between
        If n_skip_iter is None, default to round(total/16) when total > 16.

        Args:
            iterations:
                If None, plot all available iterations from the chosen source,
                optionally subsampled by n_skip_iter.
                If provided, selects closest snapshots to these iteration numbers.
            update_w_recent_model:
                If True, recompute predictions using the current model.
            ylim:
                Either a single (ymin, ymax) applied to all tasks, or
                a list of per-task (ymin, ymax) of length self.J.
            fname:
                If provided, save figure.
            source:
                "dataset"  -> evaluate current model at all controls in dataset._x
                            (includes initialization points)
                "snapshots"-> use self.history['predictions'] (x_best snapshots)
            unique_controls:
                Only used for source="dataset". Deduplicate repeated x (common during init).
            round_decimals:
                Only used for source="dataset". Dedup key rounding precision.
            n_skip_iter:
                When iterations is None: keep every n_skip_iter-th snapshot (plus first/last).
                If None and total>16, use round(total/16). If total<=16, no skipping.

        Returns:
            (fig, axes)
        """
        source = str(source).lower()
        if source not in ("dataset", "snapshots"):
            raise ValueError(f"source must be 'dataset' or 'snapshots', got {source}")

        # --------------------------
        # 0) Build full_history
        # --------------------------
        full_history = []
        if source == "dataset":
            if hasattr(self, "recompute_dataset_prediction_history"):
                full_history = self.recompute_dataset_prediction_history(
                    unique_controls=unique_controls,
                    round_decimals=round_decimals,
                )
            else:
                if not self.history.get("predictions", []):
                    print("No prediction history available (no dataset-based history function, and no snapshots).")
                    return None, None
                full_history = self.recompute_prediction_history() if update_w_recent_model else self.history["predictions"]
        else:
            if not self.history.get("predictions", []):
                print("No prediction history available in history['predictions'].")
                return None, None
            full_history = self.recompute_prediction_history() if update_w_recent_model else self.history["predictions"]

        if not full_history:
            print("No prediction history available.")
            return None, None

        # Ensure full_history is sorted by iteration (helps subsampling logic)
        full_history = sorted(full_history, key=lambda r: int(r["iteration"]))

        # --------------------------
        # 1) Select snapshots
        # --------------------------
        if iterations is None:
            snapshots = list(full_history)

            # --- NEW: subsample using n_skip_iter (keep first/last always) ---
            total = len(snapshots)
            if total > 2:
                if n_skip_iter is None and total > 16:
                    n_skip_iter_eff = int(round(total / 16.0))
                    n_skip_iter_eff = max(1, n_skip_iter_eff)
                elif n_skip_iter is None:
                    n_skip_iter_eff = 1
                else:
                    n_skip_iter_eff = int(n_skip_iter)
                    if n_skip_iter_eff < 1:
                        n_skip_iter_eff = 1

                if n_skip_iter_eff > 1:
                    first = snapshots[0]
                    last = snapshots[-1]
                    middle = snapshots[1:-1]
                    middle_kept = middle[::n_skip_iter_eff]
                    snapshots = [first] + middle_kept + [last]
        else:
            snapshots = []
            for target_iter in iterations:
                best_rec = min(full_history, key=lambda x: abs(int(x["iteration"]) - int(target_iter)))
                snapshots.append(best_rec)

        # Remove duplicates by iteration
        seen_iters = set()
        unique = []
        for r in snapshots:
            itn = int(r["iteration"])
            if itn not in seen_iters:
                unique.append(r)
                seen_iters.add(itn)
        snapshots = unique

        # Sort ascending by iteration (oldest -> newest)
        snapshots = sorted(snapshots, key=lambda r: int(r["iteration"]))
        num_snaps = len(snapshots)
        if num_snaps == 0:
            print("No snapshots to plot after filtering.")
            return None, None

        # --------------------------
        # 2) Colormap + colorbar mode
        # --------------------------
        use_colorbar = num_snaps > 6

        iter_vals = [int(rec["iteration"]) for rec in snapshots]
        min_iter, max_iter = min(iter_vals), max(iter_vals)

        if min_iter == max_iter:
            norm = mcolors.Normalize(vmin=min_iter - 1, vmax=max_iter)
        else:
            norm = mcolors.Normalize(vmin=min_iter, vmax=max_iter)

        cmap = plt.cm.viridis
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])

        # --------------------------
        # 3) Auto alpha scaling
        # --------------------------
        # NOTE: you had fixed values; here is a robust auto-scaling.
        # If you prefer fixed, just overwrite these two lines after this block.
        mid_line_alpha = float(np.clip(8.0 / max(num_snaps, 1), 0.05, 0.65))
        mid_band_alpha = float(np.clip(0.25 * mid_line_alpha, 0.01, 0.12))

        # endpoints: always fully visible
        end_line_alpha = 1.0
        end_band_alpha = 0.10 if use_colorbar else 0.14

        lw_mid = 1.4 if num_snaps > 10 else 2.0
        lw_end = 2.8

        # --------------------------
        # 4) Layout
        # --------------------------
        n_tasks = self.J
        max_cols = 4
        n_cols = min(n_tasks, max_cols)
        n_rows = (n_tasks + n_cols - 1) // n_cols

        fig_width = 4 * n_cols + (1.0 if use_colorbar else 0.0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 3.5 * n_rows), sharex=True)

        if n_tasks == 1:
            ax_flat = np.array([axes])
        else:
            ax_flat = np.array(axes).flatten()

        state_indices = np.arange(self.S)

        # Parse ylim
        ylim_is_per_task = isinstance(ylim, (list, tuple)) and len(ylim) == self.J and isinstance(ylim[0], (list, tuple))
        ylim_is_single = isinstance(ylim, (list, tuple)) and len(ylim) == 2 and not ylim_is_per_task

        # Identify endpoints + middle
        oldest = snapshots[0]
        newest = snapshots[-1]
        middle = snapshots[1:-1]

        # --------------------------
        # 5) Plotting
        # --------------------------
        for j in range(n_tasks):
            ax = ax_flat[j]
            ax.set_title(f"Task: {self.tasks[j]}")
            ax.set_xticks(state_indices)
            ax.set_xticklabels(self.states, rotation=45)

            # (A) middle: faint, drawn first
            for rec in middle:
                itn = int(rec["iteration"])
                mu = np.asarray(rec["mean"][:, j], dtype=float)
                sd = np.asarray(rec["std"][:, j], dtype=float)
                color = cmap(norm(itn))

                ax.fill_between(
                    state_indices, mu - 2 * sd, mu + 2 * sd,
                    color=color, alpha=mid_band_alpha, zorder=1
                )
                ax.plot(
                    state_indices, mu,
                    marker="o", color=color, alpha=mid_line_alpha, lw=lw_mid,
                    zorder=2, label=None
                )

            # (B) oldest: strong, 2nd top (same color mapping: darkest)
            it0 = int(oldest["iteration"])
            mu0 = np.asarray(oldest["mean"][:, j], dtype=float)
            sd0 = np.asarray(oldest["std"][:, j], dtype=float)
            c0 = cmap(norm(it0))

            ax.fill_between(
                state_indices, mu0 - 2 * sd0, mu0 + 2 * sd0,
                color=c0, alpha=end_band_alpha, zorder=8
            )
            ax.plot(
                state_indices, mu0,
                marker="o", color=c0, alpha=end_line_alpha, lw=lw_end,
                zorder=9,
                label=(f"Init (iter {it0})" if (j == 0 and not use_colorbar) else None)
            )

            # (C) newest: strong, top (same color mapping: brightest)
            it1 = int(newest["iteration"])
            mu1 = np.asarray(newest["mean"][:, j], dtype=float)
            sd1 = np.asarray(newest["std"][:, j], dtype=float)
            c1 = cmap(norm(it1))

            ax.fill_between(
                state_indices, mu1 - 2 * sd1, mu1 + 2 * sd1,
                color=c1, alpha=end_band_alpha, zorder=10
            )
            ax.plot(
                state_indices, mu1,
                marker="o", color=c1, alpha=end_line_alpha, lw=lw_end,
                zorder=11,
                label=(f"Final (iter {it1})" if (j == 0 and not use_colorbar) else None)
            )

            if j % n_cols == 0:
                ax.set_ylabel("Prediction")

            if ylim is not None:
                if ylim_is_per_task:
                    ax.set_ylim(*ylim[j])
                elif ylim_is_single:
                    ax.set_ylim(*ylim)

        # Hide unused subplots
        for k in range(n_tasks, len(ax_flat)):
            ax_flat[k].axis("off")

        # --------------------------
        # 6) Legend / colorbar
        # --------------------------
        if use_colorbar:
            fig.tight_layout(rect=[0, 0, 0.92, 1])
            cbar_ax = fig.add_axes([0.92, 0.2, 0.012, 0.7])
            cbar = fig.colorbar(scalar_mappable, cax=cbar_ax)
            cbar.set_label("Iteration")
        else:
            fig.tight_layout()
            ax_flat[0].legend(title="Emphasis", loc="best")

        if fname:
            plt.savefig(fname, bbox_inches="tight", dpi=128)

        return fig, axes



    def plot_acq_optimization_history(self, last_n: int = 10, fname: str = None):
        """Plot per-iteration acquisition optimization traces (torch backend only)."""
        recs = self.history.get("acq_opt", [])
        if not recs:
            print("No acquisition optimization history available.")
            return None, None

        torch_recs = [r for r in recs if (r.get("info", {}) or {}).get("backend") == "torch" and (
                    r["info"].get("best_curve") or [])]
        if not torch_recs:
            print("No torch-based acquisition traces found.")
            return None, None

        torch_recs = torch_recs[-int(last_n):]

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for k, r in enumerate(torch_recs):
            info = r["info"]
            ax[0].plot(info["best_curve"], alpha=0.7, label=f"query {-len(torch_recs) + k + 1}")
            ax[1].plot(info["loss_curve"], alpha=0.7, label=f"query {-len(torch_recs) + k + 1}")

        ax[0].set_title("Acq opt: best acquisition vs iter")
        ax[0].set_xlabel("optimizer iter");
        ax[0].set_ylabel("best acq value")
        ax[1].set_title("Acq opt: loss=-best(acq) vs iter")
        ax[1].set_xlabel("optimizer iter");
        ax[1].set_ylabel("loss")

        ax[0].legend(loc="best", fontsize=8)
        ax[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname, bbox_inches="tight", dpi=128)
        return fig, ax

    #========================================
    #      Legacy plotting tools
    #========================================
    def plot_tasks_1d(
            self,
            dim: int = 0,
            states=None,
            tasks=None,
            X_best: float = None,
            X_fix: Union[List[float], np.ndarray] = None,
            X_current: float = None,
            S_current: int = None,
            num_mc: int = 256,
            fname: str = None,
    ):
        """
        For each task j, overlay curves for *all* states on a single axis.
        The final axis shows the composite objective mean +/- std dev (from MC).
        1-D control only. Uses the current model.
        """
        S, J = self.S, self.J
        T = S * J
        # Resolve state indices
        if states is None:
            state_idx = list(range(S))
        else:
            state_idx = []
            for s in states:
                if isinstance(s, (int, np.integer)):
                    state_idx.append(int(s))
                else:
                    state_idx.append(self.states.index(s))

        # Resolve task indices
        if tasks is None:
            task_idx = list(range(J))
        else:
            task_idx = []
            for t in tasks:
                if isinstance(t, (int, np.integer)):
                    task_idx.append(int(t))
                else:
                    task_idx.append(self.tasks.index(t))

        # Grid along the selected dimension
        n_grid = 128
        x_grid = np.linspace(float(self.control_min[dim]), float(self.control_max[dim]), n_grid)

        # Layout: one column per task, PLUS one for composite objective
        n_tasks_plot = len(task_idx)
        n_plots = n_tasks_plot + 1
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3.5), squeeze=False)
        axes = axes[0]  # axes is 1D array of plot axes

        if self.model is not None:

            # --- Resolve default best / pending x and pending state from object ---
            if X_best is None:
                try:
                    # self.X_best is (d,) array
                    X_best = float(np.asarray(self.X_best).flat[dim])
                except Exception:
                    X_best = None

            if X_current is None:
                if self.X_pending is not None:
                    # self.X_pending is (d,) array or list
                    X_current = float(np.asarray(self.X_pending).flat[dim])
                    S_current = self.S_pending  # S_current is state *name*
                else:
                    try:
                        X_current = self.dataset._x[-1][dim]
                        S_current = self.states[self.dataset._s[-1]]  # S_current is state *name*
                    except Exception:
                        X_current = None
                        S_current = None

            # --- Build Xbase (control points) for the 1D slice ---
            Xbase_np = np.zeros((n_grid, self.ndim), dtype=float)
            Xbase_np[:, dim] = x_grid
            default_fix = np.zeros(self.ndim, dtype=float)
            if X_fix is not None:
                default_fix = np.asarray(X_fix, dtype=float)

            # Fill other dimensions
            for d_fix in range(self.ndim):
                if d_fix != dim:
                    Xbase_np[:, d_fix] = default_fix[d_fix]

            Xbase = torch.as_tensor(Xbase_np, device=self.device, dtype=self.dtype)  # (n_grid, d)

            # --- 1. Plot individual tasks (first n_tasks_plot axes) ---
            for c, j_idx in enumerate(task_idx):
                ax = axes[c]

                # Overlay states on the same subplot for this task j_idx
                for s_i, s_idx in enumerate(state_idx):
                    # Build Xt for this (state, task) across x_grid
                    t_id = float(s_idx * J + j_idx)
                    t_col = torch.full((n_grid, 1), t_id, device=self.device, dtype=self.dtype)
                    Xt = torch.cat([Xbase, t_col], dim=-1)  # (n_grid, d+1)

                    with torch.no_grad():
                        post = self.model.posterior(Xt)  # , observation_noise=False)
                        mu = post.mean.view(n_grid).detach().cpu().numpy()
                        sd = post.variance.view(n_grid).sqrt().detach().cpu().numpy()

                    color = f"C{int(s_idx)}"
                    lbl = (self.states[s_idx] if c == 0 else None)  # legend once
                    ax.plot(x_grid, mu, lw=2.0, alpha=0.9, color=color, linestyle="--", label=lbl)
                    ax.fill_between(x_grid, mu - sd, mu + sd, alpha=0.16, linewidth=0, color=color)

                    # Training data for this (state, task)
                    if hasattr(self.dataset, "get_state_data"):
                        # Get data *only* for this state
                        x_s, y_s, *_ = self.dataset.get_state_data(s_idx, stack=True)
                        if x_s is not None and x_s.numel() > 0:
                            # Check which points are on the 1D slice
                            on_slice_mask = torch.ones(x_s.shape[0], dtype=torch.bool, device=self.device)
                            for d_fix in range(self.ndim):
                                if d_fix != dim:
                                    on_slice_mask &= torch.isclose(
                                        x_s[:, d_fix],
                                        torch.tensor(default_fix[d_fix], device=self.device, dtype=self.dtype)
                                    )

                            x_np = x_s[on_slice_mask, dim].detach().cpu().numpy()
                            y_np = y_s[on_slice_mask, j_idx].detach().cpu().numpy()
                            if x_np.shape[0] > 0:
                                ax.scatter(
                                    x_np, y_np, s=30, marker="o", facecolors="none",
                                    edgecolors=color, linewidths=1.0, zorder=5,
                                    label=None if c > 0 else "data (on slice)"
                                )

                    # If this is the pending state and X_current is set, place a FILLED marker at the model μ
                    if (S_current is not None) and (X_current is not None) and (
                            s_idx == self.states.index(S_current)) and np.isfinite(X_current):
                        t_id_p = float(s_idx * J + j_idx)
                        X_curr_np = Xbase_np[0].copy()  # Get a (d,) array
                        X_curr_np[dim] = X_current
                        Xt_p = torch.as_tensor(np.hstack([X_curr_np, [t_id_p]]), device=self.device,
                                               dtype=self.dtype).view(1, -1)

                        with torch.no_grad():
                            mu_p = self.model.posterior(Xt_p).mean.view(-1).detach().cpu().numpy()[0]
                        ax.scatter(
                            [X_current], [mu_p], s=70, marker="o",
                            facecolors=color, edgecolors="k", linewidths=0.8, zorder=6,
                            label=None if c > 0 else "pending (filled)"
                        )

                # Vertical reference lines (gray)
                if X_best is not None and np.isfinite(X_best):
                    ax.axvline(X_best, ls="--", lw=1.5, alpha=0.8, color="gray",
                               label="best x (model)" if c == 0 else None)
                if X_current is not None and np.isfinite(X_current):
                    ax.axvline(X_current, ls=":", lw=2.0, alpha=0.9, color="C1", label="pending x" if c == 0 else None)

            # --- 2. Plot Composite Objective (last axis) ---
            ax_comp = axes[-1]

            # Expand Xbase to all T tasks
            # Xbase is (n_grid, d)
            Xrep = Xbase.view(n_grid, 1, self.ndim).repeat(1, T, 1)  # (n_grid, T, d)
            t_ids = torch.arange(T, device=self.device, dtype=self.dtype).view(1, T, 1)  # (1, T, 1)
            t_ids_rep = t_ids.repeat(n_grid, 1, 1)  # (n_grid, T, 1)

            Xt = torch.cat([Xrep, t_ids_rep], dim=-1)  # (n_grid, T, d+1)
            Xt_flat = Xt.reshape(n_grid * T, self.ndim + 1)  # (n_grid*T, d+1)

            with torch.no_grad():
                # Get posterior samples for all tasks
                post = self.model.posterior(Xt_flat, observation_noise=False)
                # samples shape (num_mc, n_grid*T, 1)
                samples_flat = post.rsample(sample_shape=torch.Size([num_mc]))
                # reshape to (num_mc, n_grid, T)
                samples = samples_flat.view(num_mc, n_grid, T)

                # Evaluate composite objective
                obj_samples = self.mc_objective(samples, Xbase)  # (num_mc, n_grid)

                # Get mean and std
                comp_mean = obj_samples.mean(dim=0).detach().cpu().numpy()
                comp_std = obj_samples.std(dim=0, correction=0).detach().cpu().numpy()

            # Plot composite mean and std dev
            ax_comp.plot(x_grid, comp_mean, lw=2.0, alpha=0.9, color="k", linestyle="--", label="Composite Obj")
            ax_comp.fill_between(x_grid, comp_mean - comp_std, comp_mean + comp_std, alpha=0.16, linewidth=0,
                                 color="k", label="±1σ")
            ax_comp.set_xlabel(self.control_names[dim])
            ax_comp.set_ylabel("Composite Objective")
            ax_comp.set_title("Composite Objective")

            # Plot data points (composite mean) on the slice
            if len(self.dataset._x) > 0:
                Xobs = torch.stack(self.dataset._x, dim=0).to(self.device, self.dtype)
                Xuniq = torch.unique(Xobs, dim=0)
                n = Xuniq.shape[0]

                # Get composite obj mean for all unique points
                t_ids_all = torch.arange(T, device=self.device, dtype=self.dtype).view(1, T, 1)
                Xrep_all = Xuniq.view(n, 1, self.ndim).repeat(1, T, 1)
                Xt_all = torch.cat([Xrep_all, t_ids_all.repeat(n, 1, 1)], dim=-1).reshape(n * T, self.ndim + 1)

                with torch.no_grad():
                    mu_all_tasks = self.model.posterior(Xt_all).mean.view(n, T)  # (n, T)
                    vals_all_uniq = self.composite_objective_function(mu_all_tasks, None).view(n)  # (n,)

                # Find which points are on the 1D slice
                on_slice_mask = torch.ones(n, dtype=torch.bool, device=self.device)
                for d_fix in range(self.ndim):
                    if d_fix != dim:
                        on_slice_mask &= torch.isclose(
                            Xuniq[:, d_fix],
                            torch.tensor(default_fix[d_fix], device=self.device, dtype=self.dtype)
                        )

                x_on_slice = Xuniq[on_slice_mask, dim].detach().cpu().numpy()
                y_on_slice = vals_all_uniq[on_slice_mask].detach().cpu().numpy()

                if x_on_slice.shape[0] > 0:
                    ax_comp.scatter(
                        x_on_slice, y_on_slice, s=30, marker="o", facecolors="none",
                        edgecolors="k", linewidths=1.0, zorder=5, label="data μ (composite)"
                    )

            # Vertical reference lines
            if X_best is not None and np.isfinite(X_best):
                ax_comp.axvline(X_best, ls="--", lw=1.5, alpha=0.8, color="gray", label="best x (model)")
            if X_current is not None and np.isfinite(X_current):
                ax_comp.axvline(X_current, ls=":", lw=2.0, alpha=0.9, color="C1", label="pending x")

            ax_comp.legend(loc="best")

            # --- 3. Final plot cleanup ---
            for c, j_idx in enumerate(task_idx):
                ax = axes[c]
                ax.set_xlabel(self.control_names[dim])
                ax.set_ylabel(f"Task: {self.tasks[j_idx]}")
                if c == 0:
                    ax.legend(loc="best")

            fig.tight_layout(rect=[0, 0, 1, 0.92])

            if fname is not None:
                fig.savefig(fname, bbox_inches="tight", dpi=128)
            return fig, axes

    def virtual_composite_history(
            self,
            num_mc: int = 256,
    ):
        if self.model is None:
            raise RuntimeError("Model is None. Call init()/train_model() before plotting.")
        if len(self.dataset._x) == 0:
            raise RuntimeError("No data to plot. Run init()/step() to gather data.")

        T = self.S * self.J
        n = len(self.dataset._x)
        means = np.zeros(n, dtype=float)
        stds = np.zeros(n, dtype=float)
        with torch.no_grad():
            for k in range(n):
                xk = self.dataset._x[k]
                xk = xk.detach().cpu().numpy() if torch.is_tensor(xk) else np.asarray(xk, dtype=float)
                xk = xk.reshape(1, -1)
                Xt_list = [np.hstack([xk, np.array([[float(t)]], dtype=float)]) for t in range(T)]
                Xt = torch.as_tensor(np.vstack(Xt_list), device=self.device, dtype=self.dtype)
                post = self.model.posterior(Xt, observation_noise=False)
                samples = post.rsample(sample_shape=torch.Size([num_mc])).view(num_mc, 1, T)
                obj_samples = self.mc_objective(samples, None).view(num_mc).detach().cpu().numpy()
                means[k], stds[k] = float(obj_samples.mean()), float(obj_samples.std(ddof=0))
        iters = np.arange(1, n + 1, dtype=int)
        return iters, means, stds

    def plot_composite_objective(
            self,
            num_mc: int = 256,
            fname: str = None,
    ):
        iters, means, stds = self.virtual_composite_history(num_mc=num_mc)

        # ---------- panel layout ----------
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        # Panel 1: composite objective over time
        ax[0].plot(iters, means, '--', lw=2, label="model μ (composite)")
        ax[0].plot(iters, np.maximum.accumulate(means), color='C0', label="best μ so far")
        ax[0].fill_between(iters, means - stds, means + stds, alpha=0.2, label="±1σ")
        ax[0].set_xlabel("oracle calls (time)");
        ax[0].set_ylabel("composite objective")
        ax[0].set_title("Composite over time");
        ax[0].legend(loc="best")

        # ---------- panel 2: time costs ----------
        mt = self.history["time_cost"].get("model_train", [])
        qt = self.history["time_cost"].get("query", [])
        if len(mt) or len(qt):
            ax[1].plot(mt, label="train time")
            ax[1].plot(qt, label="query time")
            ax[1].set_title("Time cost per iteration")
            ax[1].set_xlabel("iteration");
            ax[1].set_ylabel("seconds")
            ax[1].legend()

        # ---------- panel 3: training loss histories ----------
        for loss_hist in self.history.get("model_train_loss", []):
            ax[2].plot(loss_hist)
        ax[2].set_title("Model train loss")
        ax[2].set_xlabel("epoch");
        ax[2].set_ylabel("loss")

        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname, bbox_inches="tight", dpi=128)
        return fig, ax, means


    def plot_acq_loss_history(self, alpha=0.6, fname=None):
        recs = self.history.get("acq_opt", [])
        torch_recs = [
            r for r in recs
            if (r.get("info", {}) or {}).get("backend") == "torch"
            and r["info"].get("loss_curve")
        ]
        if not torch_recs:
            print("No torch acquisition loss history found.")
            return None, None

        curves = [r["info"]["loss_curve"] for r in torch_recs]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        for c in curves:
            ax.plot(c, alpha=alpha)
        ax.set_xlabel("optimizer iter (within query)")
        ax.set_ylabel("loss = -best(acq)")
        ax.set_title(f"Acq opt loss (overlay), {len(curves)} queries")
        ax.set_yscale('log')

        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname, bbox_inches="tight", dpi=128)
        return fig, ax