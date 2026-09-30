import concurrent.futures as cf
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from msBO import MultiStateBO


def make_switch_harness(states=("A", "B", "C", "D")):
    bo = MultiStateBO.__new__(MultiStateBO)
    bo.states = list(states)
    bo.ndim = 2
    bo.bounds = np.array([[0.0, 10.0], [0.0, 100.0]])
    bo.control_min = bo.bounds[:, 0]
    bo.control_max = bo.bounds[:, 1]
    bo.local_bound_size = bo.control_max - bo.control_min
    bo.X_best = np.array([0.5, 0.5])
    bo.device = torch.device("cpu")
    bo.dtype = torch.float64
    bo.executor = cf.ThreadPoolExecutor(max_workers=1)
    bo.future = None
    bo.X_pending = None
    bo.S_pending = None
    bo._switch_prefetch = None
    bo.history = {"time_cost": {}}
    bo.model = object()
    bo.dataset = SimpleNamespace(_x=[])
    bo._model_raw_count = 0

    measurements = []
    queries = []

    def oracle(x, s):
        result = {"x": np.asarray(x).copy(), "state": s, "y": np.array([0.0])}
        measurements.append(result)
        return result

    def ingest(result=None):
        if result is None:
            result = bo.future.result()
            bo.future = None
            bo.X_pending = None
            bo.S_pending = None
        bo.dataset._x.append(np.asarray(result["x"]).copy())
        return result

    def train_model():
        bo._model_raw_count = len(bo.dataset._x)

    def query_q(q, botorch_bounds, fixed_state=None, acq_type=None,
                beta=None, X_pending=None, condition_on_state=None):
        pending_n = 0 if X_pending is None else int(X_pending.shape[-2])
        queries.append({
            "q": int(q),
            "fixed_state": fixed_state,
            "condition_on_state": condition_on_state,
            "pending_n": pending_n,
            "machine_pending_state": bo.S_pending,
            "dataset_count": len(bo.dataset._x),
            "model_count": bo._model_raw_count,
        })

        acquisition_state = condition_on_state if condition_on_state is not None else fixed_state
        state_idx = bo.states.index(acquisition_state) if acquisition_state is not None else 8
        is_cross_state_bridge = (
            int(q) == 1
            and acquisition_state is not None
            and bo.S_pending is not None
            and acquisition_state != bo.S_pending
        )
        if is_cross_state_bridge:
            return np.array([[state_idx + 0.5, 99.0]])
        return np.array([
            [state_idx + 0.1 * (i + 1), float(i)] for i in range(int(q))
        ]).reshape(int(q), 2)

    bo.multistate_oracle_evaluator = oracle
    bo._ingest_oracle = ingest
    bo._query_q_batch = query_q
    bo.train_model = train_model
    return bo, measurements, queries


def test_prefetch_is_next_state_only_and_follows_three_state_route():
    bo, measurements, queries = make_switch_harness()
    try:
        bo.step_batch_with_switch("A", "B", q=3, prefetch_next=True, fix_acq_state=True)

        # The A->B switch may query A and B, but it must not select a C point.
        assert all(row["fixed_state"] != "C" for row in queries)

        bo.step_batch_with_switch("B", "C", q=3, prefetch_next=False, fix_acq_state=True)
    finally:
        bo.executor.shutdown(wait=True)

    bridges = [
        (row["state"], row["x"][0])
        for row in measurements
        if row["x"][1] == 99.0
    ]
    assert bridges == [("B", 1.5), ("C", 2.5)]
    assert [(row["q"], row["fixed_state"], row["pending_n"]) for row in queries] == [
        (2, "A", 0),       # initial current-state batch
        (1, "B", 0),       # bridge, while the last A point is pending
        (2, "B", 1),       # B-only prefetch, with the B bridge pending
        (1, "C", 0),       # bridge chosen only after the B batch
    ]
    assert bo.history["time_cost"]["switch_overlap"][1]["prefetch_used"] is True


def test_current_state_prefetch_survives_outgoing_destination_change():
    bo, _, queries = make_switch_harness()
    try:
        bo.step_batch_with_switch("A", "B", q=3, prefetch_next=True, fix_acq_state=True)
        bo.step_batch_with_switch("B", "D", q=3, prefetch_next=False, fix_acq_state=True)
    finally:
        bo.executor.shutdown(wait=True)

    # The prefetched points are B-only, so changing B's eventual destination
    # from C to D does not invalidate them or cause a premature D query.
    assert bo.history["time_cost"]["switch_overlap"][1]["prefetch_used"] is True
    assert [row["fixed_state"] for row in queries].count("B") == 2
    assert queries[-1]["fixed_state"] == "D"


def test_batch_size_mismatch_discards_prefetch_and_preserves_budget():
    bo, measurements, queries = make_switch_harness()
    try:
        bo.step_batch_with_switch("A", "B", q=4, prefetch_next=True, fix_acq_state=True)
        bo.step_batch_with_switch("B", "C", q=2, prefetch_next=False, fix_acq_state=True)
    finally:
        bo.executor.shutdown(wait=True)

    assert len(measurements) == 6
    assert bo.history["time_cost"]["switch_overlap"][1]["prefetch_used"] is False
    assert any(row["q"] == 1 and row["fixed_state"] == "B" for row in queries)


def test_bridge_uses_latest_ingested_data_with_only_one_pending_read():
    bo, _, queries = make_switch_harness()
    try:
        bo.step_batch_with_switch("A", "B", q=3, prefetch_next=True, fix_acq_state=True)
        bo.step_batch_with_switch("B", "C", q=3, prefetch_next=False, fix_acq_state=True)
    finally:
        bo.executor.shutdown(wait=True)

    bridge_c = [
        row for row in queries
        if row["q"] == 1 and row["fixed_state"] == "C"
    ][0]
    assert bridge_c["machine_pending_state"] == "B"
    assert bridge_c["dataset_count"] == bridge_c["model_count"]
    assert bridge_c["dataset_count"] == 4  # B bridge + first B batch point included


def test_global_ei_mode_is_forwarded_to_each_single_state_query():
    bo, _, queries = make_switch_harness()
    try:
        bo.step_batch_with_switch(
            "A", "B", q=3, prefetch_next=True, fix_acq_state=False
        )
    finally:
        bo.executor.shutdown(wait=True)

    assert [(row["q"], row["fixed_state"], row["pending_n"]) for row in queries] == [
        (2, None, 0),
        (1, None, 1),
        (2, None, 1),
    ]


def test_conditional_mode_tracks_measurement_state_and_same_state_pending_only():
    bo, _, queries = make_switch_harness()
    try:
        bo.step_batch_with_switch(
            "A", "B", q=3, prefetch_next=True,
            acq_state_mode="conditional",
        )
    finally:
        bo.executor.shutdown(wait=True)

    assert [
        (row["q"], row["condition_on_state"], row["fixed_state"], row["pending_n"])
        for row in queries
    ] == [
        (2, "A", None, 0),  # current-state batch
        (1, "B", None, 0),  # A pending cannot be represented as a B observation
        (2, "B", None, 1),  # pending bridge is a B observation
    ]
    overlap = bo.history["time_cost"]["switch_overlap"][0]
    assert overlap["acq_state_mode"] == "conditional"


def test_conditional_is_the_switch_scheduler_default():
    bo, _, queries = make_switch_harness()
    try:
        bo.step_batch_with_switch("A", "B", q=2, prefetch_next=False)
    finally:
        bo.executor.shutdown(wait=True)

    assert [row["condition_on_state"] for row in queries] == ["A", "B"]
    assert all(row["fixed_state"] is None for row in queries)
    overlap = bo.history["time_cost"]["switch_overlap"][0]
    assert overlap["acq_state_mode"] == "conditional"
    assert overlap["fix_acq_state"] is None


def test_invalid_acquisition_state_mode_is_rejected():
    bo, _, _ = make_switch_harness()
    try:
        with pytest.raises(ValueError, match="acq_state_mode"):
            bo.step_batch_with_switch("A", "B", acq_state_mode="fantasy")
    finally:
        bo.executor.shutdown(wait=True)


def test_acquisition_mode_change_invalidates_prefetch():
    bo, _, queries = make_switch_harness()
    try:
        bo.step_batch_with_switch(
            "A", "B", q=3, prefetch_next=True,
            acq_state_mode="conditional",
        )
        n_queries_after_first = len(queries)
        bo.step_batch_with_switch(
            "B", "C", q=3, prefetch_next=False,
            acq_state_mode="global",
        )
    finally:
        bo.executor.shutdown(wait=True)

    assert bo.history["time_cost"]["switch_overlap"][1]["prefetch_used"] is False
    # A fresh global B batch plus the C bridge must be queried on the second call.
    second_call = queries[n_queries_after_first:]
    assert [(row["q"], row["fixed_state"], row["condition_on_state"])
            for row in second_call] == [(2, None, None), (1, None, None)]


def test_following_state_remains_a_backward_compatible_prefetch_marker():
    bo, _, _ = make_switch_harness()
    try:
        bo.step_batch_with_switch("A", "B", q=2, following_s="C", fix_acq_state=True)
    finally:
        bo.executor.shutdown(wait=True)

    row = bo.history["time_cost"]["switch_overlap"][0]
    assert row["prefetch_next"] is True
    assert row["prefetch_created"] is True


def test_pending_result_on_entry_is_ingested_before_switch_scheduler_starts():
    bo, measurements, _ = make_switch_harness()
    bo.X_pending = np.array([0.25, 0.5])
    bo.S_pending = "A"
    bo.future = bo.executor.submit(
        bo.multistate_oracle_evaluator, x=bo.X_pending, s=bo.S_pending
    )
    try:
        bo.step_batch_with_switch("A", "B", q=1, prefetch_next=False, fix_acq_state=True)
    finally:
        bo.executor.shutdown(wait=True)

    assert len(measurements) == 2
    assert len(bo.dataset._x) == 2
    assert bo.history["time_cost"]["switch_overlap"][0]["ingested_pending_on_entry"] is True


def test_acquisition_failure_collects_inflight_reading_and_clears_future():
    bo, _, queries = make_switch_harness()
    query_q = bo._query_q_batch

    def fail_on_bridge(*args, **kwargs):
        if len(queries) == 1:
            raise RuntimeError("synthetic acquisition failure")
        return query_q(*args, **kwargs)

    bo._query_q_batch = fail_on_bridge
    try:
        try:
            bo.step_batch_with_switch("A", "B", q=3, prefetch_next=True, fix_acq_state=True)
        except RuntimeError as exc:
            assert "synthetic acquisition failure" in str(exc)
        else:
            raise AssertionError("expected acquisition failure")
    finally:
        bo.executor.shutdown(wait=True)

    assert bo.future is None
    assert bo.X_pending is None
    assert bo.S_pending is None
    assert bo._switch_prefetch is None
    assert len(bo.dataset._x) == 2


def test_init_forwards_seed_to_qmc_sampler():
    bo, _, _ = make_switch_harness(states=("A", "B"))
    bo.x0 = np.array([0.5, 0.5])
    bo.asynchronous = False
    sampled = np.array([[0.1, 0.2], [0.8, 0.9]])
    try:
        with patch("msBO.msBO.proximal_ordered_init_sampler", return_value=sampled) as sampler:
            bo.init(n_init=3, local_optimization=False, seed=123)
    finally:
        bo.executor.shutdown(wait=True)

    assert sampler.call_args.kwargs["seed"] == 123


def test_failed_oracle_future_is_cleared():
    bo = MultiStateBO.__new__(MultiStateBO)
    bo.states = ["A"]
    bo.X_pending = np.array([0.5])
    bo.S_pending = "A"
    bo.future = cf.Future()
    bo.future.set_exception(RuntimeError("synthetic oracle failure"))
    bo.dataset = SimpleNamespace(concat_data=lambda **kwargs: None)
    bo.use_prior_data = False

    with pytest.raises(RuntimeError, match="synthetic oracle failure"):
        bo._ingest_oracle()

    assert bo.future is None
    assert bo.X_pending is None
    assert bo.S_pending is None


@pytest.mark.parametrize("n_states,q", [(2, 2), (3, 2), (3, 3), (4, 3)])
def test_closed_cycle_adds_exactly_q_measurements_per_state(n_states, q):
    states = tuple(chr(ord("A") + i) for i in range(n_states))
    bo, measurements, _ = make_switch_harness(states=states)
    route = list(states) + [states[0]]
    try:
        for i in range(n_states):
            bo.step_batch_with_switch(
                route[i], route[i + 1], q=q,
                prefetch_next=i < n_states - 1,
                fix_acq_state=True,
            )
    finally:
        bo.executor.shutdown(wait=True)

    assert Counter(row["state"] for row in measurements) == Counter({s: q for s in states})
