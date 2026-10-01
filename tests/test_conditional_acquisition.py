import torch
import pytest
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import MultiTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.base import MCSampler
from gpytorch.distributions import MultivariateNormal

from msBO.acquisition import conditional_state_qLogEI, fixed_state_qLogEI


class FixedObservationSampler(MCSampler):
    def __init__(self, value: float):
        super().__init__(sample_shape=torch.Size([1]), seed=0)
        self.value = float(value)

    def forward(self, posterior):
        return torch.full(
            posterior._extended_shape(self.sample_shape),
            self.value,
            device=posterior.device,
            dtype=posterior.dtype,
        )


class TwoTaskGaussianModel(Model):
    """Exact two-task posterior with configurable independent observation noise."""

    def __init__(self, observation_noise: float):
        super().__init__()
        self.observation_noise = float(observation_noise)
        self.register_buffer("task_mean", torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.register_buffer(
            "task_cov",
            torch.tensor([[4.0, 3.0], [3.0, 9.0]], dtype=torch.float64),
        )

    @property
    def num_outputs(self):
        return 1

    def posterior(self, X, observation_noise=False, **kwargs):
        task = X[..., -1].long()
        mean = self.task_mean[task] + 0.0 * X[..., 0]
        covariance = self.task_cov[
            task.unsqueeze(-1), task.unsqueeze(-2)
        ]
        if observation_noise:
            eye = torch.eye(
                task.shape[-1], device=X.device, dtype=X.dtype
            ).expand(*task.shape[:-1], task.shape[-1], task.shape[-1])
            covariance = covariance + self.observation_noise * eye
        return GPyTorchPosterior(MultivariateNormal(mean, covariance))


def test_conditional_mean_uses_noisy_observation_covariance():
    # Observe task 0 as y=3.  With latent mean [1,2], latent covariance
    # [[4,3],[3,9]], and observation-noise variance 1, the Gaussian update is
    # [1,2] + [4,3] / (4+1) * (3-1) = [2.6, 3.2].
    acquisition = conditional_state_qLogEI(
        model=TwoTaskGaussianModel(observation_noise=1.0),
        best_f=0.0,
        S=2,
        J=1,
        s_idx=0,
        objective=GenericMCObjective(lambda samples, X=None: samples[..., 0]),
        sampler=FixedObservationSampler(3.0),
    )
    X = torch.zeros(1, 1, 1, dtype=torch.float64)

    updated = acquisition._conditional_mean_samples(X)

    expected = torch.tensor([2.6, 3.2], dtype=torch.float64)
    assert updated.shape == (1, 1, 1, 2)
    assert torch.allclose(updated[0, 0, 0], expected, atol=1e-7, rtol=0.0)


def test_conditional_forward_is_finite_and_differentiable():
    acquisition = conditional_state_qLogEI(
        model=TwoTaskGaussianModel(observation_noise=0.25),
        best_f=0.0,
        S=2,
        J=1,
        s_idx=0,
        objective=GenericMCObjective(lambda samples, X=None: -samples.var(dim=-1)),
        mc_samples=16,
    )
    X = torch.tensor([[[0.25]]], dtype=torch.float64, requires_grad=True)

    value = acquisition(X)
    value.sum().backward()

    assert torch.isfinite(value).all()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


@pytest.mark.parametrize("fixed_noise", [False, True])
def test_conditional_supports_real_multitaskgp_likelihoods(fixed_noise):
    # BoTorch 0.11 MultiTaskGP raises NotImplementedError for
    # posterior(..., observation_noise=True).  Exercise msBO's compatibility
    # path with both learned homoskedastic and supplied fixed-noise likelihoods.
    train_X = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
         [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    train_Y = torch.tensor(
        [[0.0], [0.4], [1.0], [0.2], [0.7], [1.2]], dtype=torch.float64
    )
    kwargs = {}
    if fixed_noise:
        kwargs["train_Yvar"] = torch.tensor(
            [[0.01], [0.02], [0.03], [0.04], [0.05], [0.06]],
            dtype=torch.float64,
        )
    model = MultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        outcome_transform=Standardize(m=1),
        **kwargs,
    ).eval()
    acquisition = conditional_state_qLogEI(
        model=model,
        best_f=-1.0,
        S=2,
        J=1,
        s_idx=0,
        objective=GenericMCObjective(lambda samples, X=None: -samples.var(dim=-1)),
        mc_samples=8,
    )
    X = torch.tensor([[[0.25]]], dtype=torch.float64, requires_grad=True)

    value = acquisition(X)
    value.sum().backward()

    assert torch.isfinite(value).all()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_legacy_fixed_state_logei_constructs_and_evaluates():
    train_X = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    train_Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float64)
    model = MultiTaskGP(train_X, train_Y, task_feature=-1).eval()
    acquisition = fixed_state_qLogEI(
        model=model,
        best_f=-1.0,
        S=2,
        J=1,
        s_idx=0,
        objective=GenericMCObjective(lambda samples, X=None: -samples.var(dim=-1)),
        mc_samples=8,
    )

    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.float64))

    assert torch.isfinite(value).all()
