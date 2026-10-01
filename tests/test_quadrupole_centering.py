import numpy as np
import pytest
import torch

from msBO.objective import (
    BPMvar_minimization,
    QuadrupoleCentering,
    regular_simplex_scan_codes,
)


@pytest.mark.parametrize("n_factors", [1, 2, 4])
def test_regular_simplex_scan_codes_are_minimal_centered_and_orthogonal(n_factors):
    codes = regular_simplex_scan_codes(n_factors)

    assert codes.shape == (n_factors + 1, n_factors)
    assert codes.mean(axis=0) == pytest.approx(0.0, abs=1e-12)
    assert abs(codes).max() == pytest.approx(1.0)
    gram = codes.T @ codes
    assert gram == pytest.approx(np.eye(n_factors) * gram[0, 0], abs=1e-12)
    assert np.linalg.matrix_rank(codes) == n_factors


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, True, None])
def test_regular_simplex_scan_codes_rejects_invalid_dimension(bad_n):
    with pytest.raises(ValueError, match="positive integer"):
        regular_simplex_scan_codes(bad_n)


def test_quadrupole_centering_is_zero_for_state_invariant_orbit():
    objective = QuadrupoleCentering(S=3, J=2)
    samples = torch.tensor([[[1.5, -2.0, 1.5, -2.0, 1.5, -2.0]]])

    value = objective(samples)

    assert value.shape == (1, 1)
    assert value.item() == pytest.approx(0.0)


def test_quadrupole_centering_applies_norms_and_weights():
    objective = QuadrupoleCentering(S=3, J=2, norms=[1.0, 2.0], weights=[1.0, 3.0])
    # After normalization, both BPMs have state values [0, 1, 2], whose
    # population variance is 2/3. Weight normalization therefore leaves 2/3.
    samples = torch.tensor([[[0.0, 0.0, 1.0, 2.0, 2.0, 4.0]]])

    value = objective(samples)

    assert value.item() == pytest.approx(-2.0 / 3.0)


def test_quadrupole_centering_rejects_wrong_task_count():
    objective = QuadrupoleCentering(S=2, J=2)

    with pytest.raises(ValueError, match="expected 4"):
        objective(torch.zeros(1, 1, 3))


def test_quadrupole_centering_penalizes_worst_state_beam_loss():
    objective = QuadrupoleCentering(
        S=3,
        J=2,
        beam_loss_task=True,
        beam_loss_weight=1.0,
        beam_loss_scale=0.1,
        beam_loss_softmin_tau=1e-4,
    )
    # One position task is invariant. The final task is the magnitude ratio;
    # a worst-state ratio of 0.90 contributes approximately one unit penalty.
    samples = torch.tensor([[[2.0, 1.0, 2.0, 0.9, 2.0, 1.0]]])

    value = objective(samples)

    assert value.item() == pytest.approx(-1.0, abs=1e-4)


def test_bpmvar_minimization_uses_beam_loss_weight_and_correct_sign():
    objective = BPMvar_minimization(
        S=2,
        J=2,
        w_beam_center=0.0,
        w_beam_loss=2.0,
        beam_loss_scale=0.1,
        beam_loss_softmin_tau=1e-4,
    )
    no_loss = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
    ten_percent_loss = torch.tensor([[[0.0, 1.0, 0.0, 0.9]]])

    assert objective(no_loss).item() == pytest.approx(1.0)
    assert objective(ten_percent_loss).item() == pytest.approx(-1.0, abs=1e-4)
