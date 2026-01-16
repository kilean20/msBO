import time
from copy import deepcopy as copy

import numpy as np
import concurrent.futures as cf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors


import torch
from torch import Tensor
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
                 states : list[str],
                 tasks  : list[str],
                 control_min     : Union[List[float], np.ndarray],
                 control_max     : Union[List[float], np.ndarray],
                 multistate_oracle_evaluator : Callable,
                 composite_objective_function: Callable,
                 local_bound_size = None,
                 use_prior_data = False,
                 asynchronous = False,
                 acq_restarts: int = 2,
                 acq_raw_samples: int = 32,
                 acq_maxiter: int = 75,
                 acq_repeats: int = 2,          
                 kg_num_fantasies: int = 64,
                 fixed_mc_samples: int = 512,    # for fixed_state_qUCB/qLogEI
                 TurBO_failure_tolerance = 999,
                 TurBO_success_tolerance = 2,
                 TurBO_success_threshold = 0.95,
                 control_names: Optional[List[str]] = None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
    ):
        self.states = states
        self.tasks  = tasks
        self.S, self.J = len(states), len(tasks)
        self.control_min   = np.asarray(control_min)
        self.control_max   = np.asarray(control_max)
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
            self.local_bound_size_ref = 0.1*(self.control_max - self.control_min)
        else: 
            self.local_bound_size_ref = local_bound_size
        self.local_bound_size = copy(self.local_bound_size_ref)
        self.local_bound_size_min = 2e-2 * (self.control_max - self.control_min)
        
        self.bounds = np.vstack((control_min, control_max)).T  # Shape bounds as (n, 2)
        self.use_prior_data = use_prior_data
        self.asynchronous = asynchronous

        self.acq_restarts = int(acq_restarts)
        self.acq_raw_samples = int(acq_raw_samples)
        self.acq_maxiter = int(acq_maxiter)
        self.acq_repeats = int(acq_repeats)
        self.kg_num_fantasies = int(kg_num_fantasies)
        self.fixed_mc_samples = int(fixed_mc_samples)

        self.TurBO_success_threshold = float(TurBO_success_threshold)
        self.TurBO_failure_tolerance = TurBO_failure_tolerance
        self.TurBO_success_tolerance = TurBO_success_tolerance
        self.TurBO_failure_counter = 0
        self.TurBO_success_counter = 0

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float64

        self.dataset  = MultiStateDataset(x_dim=self.ndim, S=self.S, J=self.J, dtype=self.dtype, device=self.device)
        if self.use_prior_data:
            self.prior_dataset = MultiStateDataset(x_dim=self.ndim, S=self.S, J=self.J, dtype=self.dtype, device=self.device)

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
            'model_train_loss':[],
            'prior_model_train_loss':[],
            'time_cost' : {'model_train':[],
                          'query':[],
                          'prior_model_train':[],
                          },
            'predictions': []  # Store snapshots of model predictions
        }
        

    def _ingest_oracle(self, oracle: Dict=None) -> None:
        if oracle is None:
            assert self.future is not None, "No future to get result from."
            oracle = self.future.result()
            self.X_pending = None
            self.S_pending = None
            self.future = None
        """Append oracle data into datasets."""
        self.dataset.concat_data(x=oracle["x"], s=self.states.index(oracle["state"]), y=oracle["y"]) 
        if self.use_prior_data and self.prior_dataset is not None and "ramping_x" in oracle:
            self.prior_dataset.concat_data(x=oracle["ramping_x"], s=self.states.index(oracle["ramping_state"]), y=oracle["ramping_y"])


    
    def _get_model_based_X_best(self) -> np.ndarray:
        """
        Argmax over observed unique X of the composite objective evaluated at
        the model posterior MEAN for all tasks (T = S*J). Vectorized & shape-safe.
        """
        if self.model is None or len(self.dataset._x) == 0:
            return self.x0.copy()

        # unique controls we have actually tried
        Xobs = torch.stack(self.dataset._x, dim=0).to(self.device, self.dtype)  # (m, d)
        Xuniq = torch.unique(Xobs, dim=0)                                       # (n, d)

        n = Xuniq.shape[0]
        T = self.S * self.J
        t_ids = torch.arange(T, device=self.device, dtype=self.dtype).view(1, T, 1)   # (1, T, 1)
        Xrep = Xuniq.view(n, 1, self.ndim).repeat(1, T, 1)                             # (n, T, d)
        Xt = torch.cat([Xrep, t_ids.repeat(n, 1, 1)], dim=-1).reshape(n * T, self.ndim + 1)

        with torch.no_grad():
            mu = self.model.posterior(Xt).mean.view(n, T)                              # (n, T)

        vals = self.composite_objective_function(mu, None).view(n)               # (n,)
        i_best = torch.argmax(vals)
        return Xuniq[i_best].detach().cpu().numpy(), vals[i_best].item()



    def train_model(self):
        if self.use_prior_data and len(self.prior_dataset) > 0:
            t0 = time.monotonic()
            self.prior_model, prior_loss_history = train_mtgp(self.prior_dataset)
            t1 = time.monotonic()
            self.history['prior_model_train_loss'].append(prior_loss_history)
            self.history['time_cost']['prior_model_train'].append(t1-t0)
        else:
            self.prior_model = None

        t0 = time.monotonic()
        self.model, loss_history = train_mtgp(self.dataset,prior_model=self.prior_model)
        t1 = time.monotonic()
        self.history['model_train_loss'].append(loss_history)
        self.history['time_cost']['model_train'].append(t1-t0)
        self.X_best, self.Y_best = self._get_model_based_X_best()
        self._update_turbo_counters_and_trust_region()
        
        # Snapshot predictions after training
        self.snapshot_predictions()



    def init(self, n_init, local_optimization=True):
        
        if local_optimization:        
            lower_bound = np.maximum(self.x0 -0.5*self.local_bound_size, self.control_min)
            upper_bound = np.minimum(self.x0 +0.5*self.local_bound_size, self.control_max)
            bounds = np.vstack((lower_bound, upper_bound)).T  # Shape bounds as (n, 2)
        else:
            bounds = self.bounds

        init_x = proximal_ordered_init_sampler(
            n_init-1,
            bounds=bounds,
            x0=self.x0,
            ramping_rate=None,
            polarity_change_time=None, 
            method='sobol',
            seed=None)
        init_x = np.vstack((self.x0.reshape(1,-1),init_x))

        self.X_pending = init_x[0]
        self.S_pending = self.states[0]
        self.future = self.executor.submit(self.multistate_oracle_evaluator, 
                                           x=self.X_pending, s=self.S_pending)
        is_first  = True
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
             beta: Optional[float]=None, 
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
            lower_bound = np.maximum(self.X_best -0.5*self.local_bound_size, self.control_min)
            upper_bound = np.minimum(self.X_best +0.5*self.local_bound_size, self.control_max)
            botorch_bounds = torch.tensor(np.vstack((lower_bound, upper_bound)), device=self.device, dtype=self.dtype)  
        else:
            botorch_bounds = torch.tensor(self.bounds.T, device=self.device, dtype=self.dtype)
        candidate = self.query_candidate(botorch_bounds,X_pending=X_pending,fixed_state=fixed_state,acq_type=acq_type,beta=beta)

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
        t_col = t_ids.view(*([1]*len(B)), 1, T, 1).expand(*B, q, T, 1)
        X_all = torch.cat([X_rep, t_col], dim=-1).reshape(*B, q*T, d_ctrl+1)
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
                return qLogExpectedImprovement(model=self.model,best_f=self.Y_best,objective=self.mc_objective, X_pending=X_pending)
            
            if acq_type in ("qUCB", "UCB"):
                if beta is None:
                    beta = 0.0
            return qUpperConfidenceBound(model=self.model, beta=beta, objective=self.mc_objective, X_pending=X_pending)
        
        else:
            if isinstance(fixed_state, int):
                s_idx = fixed_state
            else:
                s_idx = self.states.index(fixed_state)
            if not (0 <= s_idx < self.S):
                raise ValueError(f"s_idx {s_idx} out of range for S={self.S}")

            if acq_type in ("qUCB", "UCB"):
                return fixed_state_qUCB(
                    model=self.model,
                    objective=self.mc_objective,
                    X_pending=X_pending,
                    mc_samples=self.fixed_mc_samples,
                    S=self.S, J=self.J, s_idx=s_idx, beta=float(beta)
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
            


        
    def query_candidate(self,
                        botorch_bounds=None,
                        acq_function=None,
                        beta: Optional[float]=None, 
                        X_pending: Optional[Tensor] = None,
                        fixed_state: Optional[str] = None,
                        acq_type: Optional[str] = None
                        ):
        assert self.model is not None, "Call init() first."
        t0 = time.monotonic()
        if acq_function is None:
            acq_function = self._get_acquisition(beta=beta, botorch_bounds=botorch_bounds, X_pending=X_pending, fixed_state=fixed_state, acq_type=acq_type)
        # Optimize acquisition function to get next candidate (q=1 for single candidate)
        best_candidate_t = None
        best_val = -float("inf")

        for i in range(self.acq_repeats):
            candidate, value = optimize_acqf(
                acq_function=acq_function,
                bounds=botorch_bounds,
                q=1,
                num_restarts=self.acq_restarts,
                raw_samples=self.acq_raw_samples,
                options={"maxiter": self.acq_maxiter},
            )

            val_scalar = float(value.item())  # (sync unavoidable, but OK)
            if val_scalar > best_val:
                best_val = val_scalar
                best_candidate_t = candidate.detach()  # keep torch, no cpu/numpy yet
            if (self.future is not None) and self.future.done():
                break  # terminate acq search early when async job finished

        self.history['time_cost']['query'].append(time.monotonic()-t0)
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
        Xt_all = _expand_tasks_all_states(X_eval, T, self.device, self.dtype) # (1, T, d+1)
        
        with torch.no_grad():
            posterior = self.model.posterior(Xt_all)
            mean = posterior.mean.view(-1).cpu().numpy()      # (T,)
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



    def plot_state_predictions_history(self, iterations: Optional[List[int]]= None, 
                                       update_w_recent_model: Optional[bool] = True, 
                                       ylim: Optional[Tuple[float, float]] = None,
                                       fname: str = None):
        """
        Plots model predictions across states for each task.
        - Max 4 columns of subplots.
        - Vertical colorbar on the right (outside plots) if > 6 snapshots.
        """
        if not self.history['predictions']:
            print("No prediction history available.")
            return
        
        if update_w_recent_model:
            full_history = self.recompute_prediction_history()
        else:
            full_history = self.history['predictions']

        # 1. Find closest snapshots
        snapshots = []
        if iterations is None:
            iterations = [rec['iteration'] for rec in full_history]
        for target_iter in iterations:
            best_rec = min(full_history, key=lambda x: abs(x['iteration'] - target_iter))
            snapshots.append(best_rec)
            
        # 2. Remove duplicates
        unique_snapshots = []
        seen_iters = set()
        for s in snapshots:
            if s['iteration'] not in seen_iters:
                unique_snapshots.append(s)
                seen_iters.add(s['iteration'])
        snapshots = unique_snapshots
        
        # 3. Determine visualization mode
        num_snaps = len(snapshots)
        use_colorbar = num_snaps > 6

        # 4. Setup Colors
        iter_vals = [rec['iteration'] for rec in snapshots]
        min_iter, max_iter = min(iter_vals), max(iter_vals)
        
        if min_iter == max_iter:
            norm = mcolors.Normalize(vmin=min_iter - 1, vmax=max_iter)
        else:
            norm = mcolors.Normalize(vmin=min_iter, vmax=max_iter)
            
        cmap = plt.cm.viridis
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])

        # 5. Setup Grid Layout
        n_tasks = self.J
        max_cols = 4
        n_cols = min(n_tasks, max_cols)
        n_rows = (n_tasks + n_cols - 1) // n_cols  # Ceiling division

        # Adjust figure size
        # Make figure wider to accommodate colorbar if needed
        fig_width = 4 * n_cols + (1 if use_colorbar else 0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 3.5 * n_rows), sharex=True)
        
        # Flatten axes array
        if n_tasks == 1:
            ax_flat = np.array([axes])
        else:
            ax_flat = axes.flatten()

        state_indices = np.arange(self.S)

        # 6. Plotting Loop
        for j in range(n_tasks):
            ax = ax_flat[j]
            ax.set_title(f"Task: {self.tasks[j]}")
            ax.set_xticks(state_indices)
            ax.set_xticklabels(self.states, rotation=45)
            
            init = True
            for rec in snapshots:
                iter_num = rec['iteration']
                mu    = rec['mean'][:, j]   # (S,)
                sigma = rec['std' ][:, j] # (S,)
                
                color = cmap(norm(iter_num))
                lbl = f"Iter {iter_num}" if not use_colorbar else None

                ax.plot(state_indices, mu, marker='o', color=color, label=lbl, alpha=0.7)
                ax.fill_between(state_indices, mu - 2*sigma, mu + 2*sigma, color=color, alpha=0.1)

            if len(snapshots) > 0:
                # First snapshot
                first_rec = snapshots[0]
                first_mu = first_rec['mean'][:, j]
                first_color = cmap(norm(first_rec['iteration']))
                ax.plot(state_indices, first_mu, marker='o', color=first_color, alpha=1.0)
                
                # Last snapshot
                last_rec = snapshots[-1]
                last_mu = last_rec['mean'][:, j]
                last_color = cmap(norm(last_rec['iteration']))
                ax.plot(state_indices, last_mu, marker='o', color=last_color, alpha=1.0)

            if j % n_cols == 0:
                ax.set_ylabel("Prediction")
            if ylim is not None:
                ax.set_ylim(*ylim[j])

        # 7. Hide unused subplots
        for k in range(n_tasks, len(ax_flat)):
            ax_flat[k].axis('off')

        # 8. Legend / Colorbar
        if use_colorbar:
            # Use tight_layout rect to reserve space on the right (0 to 0.9)
            fig.tight_layout(rect=[0, 0, 0.92, 1])
            
            # Add a dedicated axis for the colorbar [left, bottom, width, height] in figure coords
            cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.7])
            cbar = fig.colorbar(scalar_mappable, cax=cbar_ax)
            cbar.set_label('Iteration')
        else:
            fig.tight_layout()
            # Use legend on the first subplot if few lines
            ax_flat[0].legend(title="Iteration", loc='best')

        if fname:
            plt.savefig(fname)
        return fig, axes

    #========================================
    #      Legacy plotting tools
    #========================================
    def plot_tasks_1d(
            self,
            dim: int = 0,
            states=None,
            tasks=None,
            X_best: float = None,
            X_fix : Union[List[float], np.ndarray] = None,
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
            axes = axes[0] # axes is 1D array of plot axes

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
                        S_current = self.S_pending # S_current is state *name*
                    else:
                        try:
                            X_current = self.dataset._x[-1][dim]
                            S_current = self.states[self.dataset._s[-1]] # S_current is state *name*
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
                
                Xbase = torch.as_tensor(Xbase_np, device=self.device, dtype=self.dtype) # (n_grid, d)


                # --- 1. Plot individual tasks (first n_tasks_plot axes) ---
                for c, j_idx in enumerate(task_idx):
                    ax = axes[c]

                    # Overlay states on the same subplot for this task j_idx
                    for s_i, s_idx in enumerate(state_idx):
                        # Build Xt for this (state, task) across x_grid
                        t_id = float(s_idx * J + j_idx)
                        t_col = torch.full((n_grid, 1), t_id, device=self.device, dtype=self.dtype)
                        Xt = torch.cat([Xbase, t_col], dim=-1) # (n_grid, d+1)

                        with torch.no_grad():
                            post = self.model.posterior(Xt)#, observation_noise=False)
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
                        if (S_current is not None) and (X_current is not None) and (s_idx == self.states.index(S_current)) and np.isfinite(X_current):
                            t_id_p = float(s_idx * J + j_idx)
                            X_curr_np = Xbase_np[0].copy() # Get a (d,) array
                            X_curr_np[dim] = X_current
                            Xt_p = torch.as_tensor(np.hstack([X_curr_np, [t_id_p]]), device=self.device, dtype=self.dtype).view(1, -1)
                            
                            with torch.no_grad():
                                mu_p = self.model.posterior(Xt_p).mean.view(-1).detach().cpu().numpy()[0]
                            ax.scatter(
                                [X_current], [mu_p], s=70, marker="o",
                                facecolors=color, edgecolors="k", linewidths=0.8, zorder=6,
                                label=None if c > 0 else "pending (filled)"
                            )

                    # Vertical reference lines (gray)
                    if X_best is not None and np.isfinite(X_best):
                        ax.axvline(X_best, ls="--", lw=1.5, alpha=0.8, color="gray", label="best x (model)" if c == 0 else None)
                    if X_current is not None and np.isfinite(X_current):
                        ax.axvline(X_current, ls=":", lw=2.0, alpha=0.9, color="C1", label="pending x" if c == 0 else None)


                # --- 2. Plot Composite Objective (last axis) ---
                ax_comp = axes[-1]
                
                # Expand Xbase to all T tasks
                # Xbase is (n_grid, d)
                Xrep = Xbase.view(n_grid, 1, self.ndim).repeat(1, T, 1) # (n_grid, T, d)
                t_ids = torch.arange(T, device=self.device, dtype=self.dtype).view(1, T, 1) # (1, T, 1)
                t_ids_rep = t_ids.repeat(n_grid, 1, 1) # (n_grid, T, 1)

                Xt = torch.cat([Xrep, t_ids_rep], dim=-1) # (n_grid, T, d+1)
                Xt_flat = Xt.reshape(n_grid * T, self.ndim + 1) # (n_grid*T, d+1)
                
                with torch.no_grad():
                    # Get posterior samples for all tasks
                    post = self.model.posterior(Xt_flat, observation_noise=False)
                    # samples shape (num_mc, n_grid*T, 1)
                    samples_flat = post.rsample(sample_shape=torch.Size([num_mc]))
                    # reshape to (num_mc, n_grid, T)
                    samples = samples_flat.view(num_mc, n_grid, T)

                    # Evaluate composite objective
                    # samples: (num_mc, n_grid, T) -> (sample_shape, q, m)
                    # Xbase: (n_grid, d) -> (q, d)
                    obj_samples = self.mc_objective(samples, Xbase) # (num_mc, n_grid)
                    
                    # Get mean and std
                    comp_mean = obj_samples.mean(dim=0).detach().cpu().numpy()
                    comp_std = obj_samples.std(dim=0, correction=0).detach().cpu().numpy() # <-- FIXED

                # Plot composite mean and std dev
                ax_comp.plot(x_grid, comp_mean, lw=2.0, alpha=0.9, color="k", linestyle="--", label="Composite Obj")
                ax_comp.fill_between(x_grid, comp_mean - comp_std, comp_mean + comp_std, alpha=0.16, linewidth=0, color="k", label="±1σ")
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
                        mu_all_tasks = self.model.posterior(Xt_all).mean.view(n, T) # (n, T)
                        vals_all_uniq = self.composite_objective_function(mu_all_tasks, None).view(n) # (n,)
                    
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
                fig.savefig(fname, bbox_inches="tight",dpi=128)
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
        stds  = np.zeros(n, dtype=float)
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
        iters = np.arange(1, n+1, dtype=int)
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
        ax[0].set_xlabel("oracle calls (time)"); ax[0].set_ylabel("composite objective")
        ax[0].set_title("Composite over time"); ax[0].legend(loc="best")


        # ---------- panel 2: time costs ----------
        mt = self.history["time_cost"].get("model_train", [])
        qt = self.history["time_cost"].get("query", [])
        if len(mt) or len(qt):
            ax[1].plot(mt, label="train time")
            ax[1].plot(qt, label="query time")
            ax[1].set_title("Time cost per iteration")
            ax[1].set_xlabel("iteration"); ax[1].set_ylabel("seconds")
            ax[1].legend()

        # ---------- panel 3: training loss histories ----------
        for loss_hist in self.history.get("model_train_loss", []):
            ax[2].plot(loss_hist)
        ax[2].set_title("Model train loss")
        ax[2].set_xlabel("epoch"); ax[2].set_ylabel("loss")

        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname, bbox_inches="tight", dpi=128)
        return fig, ax, means