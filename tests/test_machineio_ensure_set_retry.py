import os
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Keep any PyEPICS discovery performed at import on localhost.
os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
MACHINEIO_PROJECT = Path(__file__).resolve().parents[1] / "machineIO"
sys.path.insert(0, str(MACHINEIO_PROJECT))

from machineIO.construct_machineIO import (
    AbstractMachineIO,
    StatefulOracleEvaluator,
    construct_machineIO,
)


class SequencedMachine(AbstractMachineIO):
    def __init__(self, statuses):
        super().__init__(
            ensure_set_timeout=0.01,
            ensure_set_timewait_after_ramp=0.0,
            ensure_set_exception_waittime=0.0,
        )
        self.statuses = list(statuses)
        self.attempts = 0

    def _caget(self, pvname):
        return 0.0

    def _caput(self, pvname, value):
        return None

    def _ensure_set(self, setpoint_pv, readback_pv, goal, tol, **kwargs):
        status = self.statuses[self.attempts]
        self.attempts += 1
        return status, pd.DataFrame([{"attempt": self.attempts}])

    def _fetch_data(self, pvlist, time_span, sample_interval, **kwargs):
        return pd.DataFrame([{pv: 0.0 for pv in pvlist}])


def call_ensure_set(machine):
    return machine.ensure_set(
        ["X:I_CSET"],
        ["X:I_RD"],
        [1.0],
        [0.1],
    )


def test_successful_first_attempt_is_not_repeated():
    machine = SequencedMachine(["PutFinish"])

    status, data = call_ensure_set(machine)

    assert status == "PutFinish"
    assert machine.attempts == 1
    assert data.iloc[0]["attempt"] == 1
    assert machine.history[-1]["attempt_statuses"] == ["PutFinish"]


def test_timeout_is_retried_once_and_second_success_is_returned():
    machine = SequencedMachine(["Timeout", "PutFinish"])

    with pytest.warns(RuntimeWarning, match="attempt 1 timed out"):
        status, data = call_ensure_set(machine)

    assert status == "PutFinish"
    assert machine.attempts == 2
    assert data.iloc[0]["attempt"] == 2
    assert machine.history[-1]["attempt_statuses"] == ["Timeout", "PutFinish"]
    assert not machine.history[-1]["continued_after_timeout"]


def test_two_timeouts_warn_and_continue_without_raising():
    machine = SequencedMachine(["Timeout", "Timeout"])

    with pytest.warns(RuntimeWarning) as warning_records:
        status, data = call_ensure_set(machine)

    messages = [str(record.message) for record in warning_records]
    assert any("attempt 1 timed out" in message for message in messages)
    assert any("continuing without confirmed readback tolerance" in message for message in messages)
    assert status == "Timeout"
    assert machine.attempts == 2
    assert data.iloc[0]["attempt"] == 2
    assert machine.history[-1]["continued_after_timeout"]
    assert len(machine.last_ensure_set_timing["machineio_ensure_set_attempts"]) == 2


def test_exception_waittime_is_validated_and_serialized():
    with pytest.raises(ValueError, match="non-negative finite"):
        SequencedMachine(["PutFinish"]).ensure_set_exception_waittime = float("nan")

    machine = construct_machineIO(test=True, ensure_set_exception_waittime=7.5)
    payload = machine.to_dump_dict(include_history=False)
    restored = construct_machineIO.from_dump_dict(payload, test=True)
    assert restored.ensure_set_exception_waittime == pytest.approx(7.5)


def test_stateful_async_read_infers_state_from_readback_without_x_context():
    evaluator = object.__new__(StatefulOracleEvaluator)
    evaluator.state_CSETs = ["Q:I_CSET"]
    evaluator.state_RDs = ["Q:I_RD"]
    evaluator.state_tols = np.asarray([0.1])
    evaluator.state_key_vals = {"low": [1.0], "high": [2.0]}
    evaluator.state_df_manipulators = None
    evaluator.oracle_key_names = {"x": ["X:I_RD"], "state": ["state"]}

    future = Future()
    # Deliberately disagreeing CSET proves that state detection uses the
    # measured readback. No variable named x is available to get_result().
    future.set_result(pd.DataFrame({
        "Q:I_CSET": [1.0, 1.0],
        "Q:I_RD": [2.01, 1.99],
        "X:I_RD": [3.0, 3.2],
    }))
    future._state_context = None

    result = evaluator.get_result(future)

    assert result["state"].tolist() == ["high"]
    assert result["x"].tolist() == pytest.approx([3.1])
