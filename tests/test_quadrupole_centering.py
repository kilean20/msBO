import pytest
import torch

from msBO.objective import QuadrupoleCentering


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
