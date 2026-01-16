import contextlib
import copy
import numpy as np
import torch
from torch import Tensor
import gpytorch
from botorch.models import MultiTaskGP   
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from typing import Union, List, Optional,Callable
from .dataset import MultiStateDataset



class PriorMean(gpytorch.means.Mean):
    def __init__(self, prior_model, x_untransform_fn=None, y_standardize_fn=None, detach=True):
        super().__init__()
        self.prior_model = prior_model
        # store plain callables (not Modules) to avoid double registration
        self._x_untransform = x_untransform_fn or (lambda X: X)
        self._y_standardize = y_standardize_fn or (lambda Y: Y)
        self.detach = detach
        self._cache_key = None
        self._cache_mu = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        key = (id(X), X.shape)  # safer than data_ptr for caching identity
        if self._cache_key == key and self._cache_mu is not None:
            return self._cache_mu

        X_raw = self._x_untransform(X)

        ctx = torch.no_grad() if self.detach else contextlib.nullcontext()
        self.prior_model.eval()
        with ctx:
            post = self.prior_model.posterior(X_raw)
            mu = post.mean.view(-1, 1)

        mu_std = self._y_standardize(mu)
        mu_out = mu_std.squeeze(-1)
        self._cache_key, self._cache_mu = key, mu_out
        return mu_out



def train_mtgp(
    dataset: MultiStateDataset,
    lr: float = 0.2,
    epochs: int = 200,
    prior_model: Optional[MultiTaskGP] = None,
):
    """
    Train a (FixedNoise) MultiTaskGP on the MultiStateDataset.
    - Excludes the last column (task id) from normalization.
    - If prior_model is provided, uses its posterior mean as a (detached) prior mean,
      aligning input/output spaces via the *current* model's transforms through
      non-registered callables (avoids double-registration of Modules).
    - Enforces all-or-none yvar policy at dataset level.

    Returns:
        model: trained BoTorch model in eval() mode
        loss_history: np.ndarray of training losses
    """
    # ----------------------------
    # Build training tensors
    # ----------------------------
    train_X = dataset.X
    train_Y = dataset.Y
    Yvar_all = dataset.Yvar
    has_yvar = Yvar_all.numel() > 0 and Yvar_all.shape[0] == train_Y.shape[0]
    train_Yvar = Yvar_all if has_yvar else None

    # ----------------------------
    # Align device/dtype with prior model if provided
    # ----------------------------
    target_device = train_X.device
    target_dtype = train_X.dtype
    if prior_model is not None:
        pm_param = next(prior_model.parameters(), None)
        if pm_param is not None:
            target_device = pm_param.device
            target_dtype = pm_param.dtype

    train_X = train_X.to(target_device, target_dtype)
    train_Y = train_Y.to(target_device, target_dtype)
    if has_yvar:
        train_Yvar = train_Yvar.to(target_device, target_dtype)

    # ----------------------------
    # Transforms: normalize controls only; leave task index alone.
    # ----------------------------
    input_tf = Normalize(d=dataset.x_dim + 1, indices=list(range(dataset.x_dim)))
    outcome_tf = Standardize(m=1)

    # ----------------------------
    # Optional prior mean via callables (no module re-registration)
    # ----------------------------
    mean_module = None
    if prior_model is not None:
        def x_untransform_fn(X):
            return input_tf.untransform(X) if input_tf is not None else X

        def y_standardize_fn(Y):
            if outcome_tf is None:
                return Y
            Yt = outcome_tf(Y)
            return Yt[0] if isinstance(Yt, tuple) else Yt

        mean_module = PriorMean(
            prior_model=prior_model,
            x_untransform_fn=x_untransform_fn,
            y_standardize_fn=y_standardize_fn,
            detach=True,
        )

    # ----------------------------
    # Build & train model
    # ----------------------------
    common_kwargs = dict(
        task_feature=-1,                   # last column is task id
        input_transform=input_tf,
        outcome_transform=outcome_tf,
    )
    if mean_module is not None:
        common_kwargs["mean_module"] = mean_module

    if has_yvar:
        model = MultiTaskGP(
            train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar, **common_kwargs
        )
    else:
        model = MultiTaskGP(train_X=train_X, train_Y=train_Y, **common_kwargs)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    opt = torch.optim.Adam(mll.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                                    max_lr=lr,
                                                    div_factor=2.5,
                                                    pct_start=0.1, 
                                                    final_div_factor=5,
                                                    epochs=epochs, steps_per_epoch=1)


    loss_history: List[float] = []

    best_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    best_lik_state = copy.deepcopy(model.likelihood.state_dict())
    for _ in range(epochs):
        opt.zero_grad()
        out = model(*model.train_inputs)            # inputs auto-transformed
        loss = -mll(out, model.train_targets)       # scalar already
        loss.backward()
        opt.step()
        scheduler.step()
        loss = loss.item()
        loss_history.append(loss)

        if loss < best_loss - 1e-9:
            best_loss = loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_lik_state = copy.deepcopy(model.likelihood.state_dict())

    model.load_state_dict(best_model_state)
    model.likelihood.load_state_dict(best_lik_state) 
    model.eval()
    return model, np.array(loss_history, dtype=float)
