"""External localhost IOC fixture for testing the live-only FS1 notebooks.

The production notebooks never import this module. The separate validation
harness injects it into an in-memory notebook copy before PyEPICS or machineIO
is imported, preventing accidental contact with live IOCs during testing.
"""

from __future__ import annotations

import atexit
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_EPICS_ROOT = REPO_ROOT / "simEPICS"
if str(SIM_EPICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_EPICS_ROOT))

from sim_ioc_lib import create_interactive_ioc, run_ioc_background


_DECISION_CSETS = [
    "FS1_CSS:PSC2_D2276:I_CSET",
    "FS1_CSS:PSC2_D2351:I_CSET",
    "FS1_CSS:PSC2_D2367:I_CSET",
    "FS1_CSS:PSC2_D2381:I_CSET",
    "FS1_BBS:PSTC_D2435:I_CSET",
    "FS1_BBS:PSTC_D2453:I_CSET",
]

_STATE_CSETS = [
    "FS1_CSS:PSQ_D2356:I_CSET",
    "FS1_CSS:PSQ_D2362:I_CSET",
    "FS1_CSS:PSQ_D2372:I_CSET",
    "FS1_CSS:PSQ_D2377:I_CSET",
    "FS1_BBS:PSQ_D2416:I_CSET",
    "FS1_BBS:PSQ_D2424:I_CSET",
    "FS1_BBS:PSQ_D2463:I_CSET",
    "FS1_BBS:PSQ_D2472:I_CSET",
]

_OBJECTIVE_PVS = [
    "FS1_BBS:BPM_D2421:XPOS_RD",
    "FS1_BBS:BPM_D2466:XPOS_RD",
    "FS1_BMS:BPM_D2502:XPOS_RD",
    "FS1_BMS:BPM_D2537:XPOS_RD",
]

_MAGNITUDE_PVS = [pv.replace(":XPOS_RD", ":MAG_RD") for pv in _OBJECTIVE_PVS]


def _input_pvs():
    config = {
        pv: {"initial": 0.0, "ramping_rate": 2_000.0}
        for pv in _DECISION_CSETS
    }
    # Nonzero nominal quadrupoles exercise the production percentage-based
    # amplitude used by the upstream regular-simplex scan designs.
    config.update(
        {
            pv: {"initial": 20.0 + 2.0 * index, "ramping_rate": 2_000.0}
            for index, pv in enumerate(_STATE_CSETS)
        }
    )
    config["FS1_BBS:PSD_D2394:I_CSET"] = {
        "initial": 100.0,
        "ramping_rate": 2_000.0,
    }
    return config


def _output_pvs():
    config = {pv: {"min": -8.0, "max": 8.0} for pv in _OBJECTIVE_PVS}
    config.update({pv: {"min": 0.9, "max": 1.1} for pv in _MAGNITUDE_PVS})
    return config


def _static_pvs():
    return {
        "REA_EXP:ELMT": {"value": "Ar", "read_only": True},
        "ACS_DIAG:CHP:STATE_RD": {"value": 3.0, "read_only": True},
        "ACS_DIAG:DEST:ACTIVE_ION_SOURCE": {"value": 1.0, "read_only": True},
        "FE_ISRC1:BEAM:ELMT_BOOK": {"value": "Ar", "read_only": True},
        "ACC_OPS:BEAM:Q_STRIP": {"value": 18.0, "read_only": True},
        "FE_ISRC1:BEAM:A_BOOK": {"value": 40.0, "read_only": True},
    }


@dataclass
class FS1Simulator:
    ioc: object
    loop: object
    thread: threading.Thread
    _stopped: bool = False

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=3.0)


def start_fs1_simulator(seed: int = 2026) -> FS1Simulator:
    """Start the local IOC and return an idempotent shutdown handle."""
    ioc = create_interactive_ioc(
        _input_pvs(),
        _output_pvs(),
        static_pv_info=_static_pvs(),
        seed=seed,
    )
    # These are production PV names.  Bind the simulator to loopback only so
    # it cannot answer Channel Access searches from another machine.
    loop, thread = run_ioc_background(ioc, interfaces=["127.0.0.1"])
    simulator = FS1Simulator(ioc=ioc, loop=loop, thread=thread)
    atexit.register(simulator.stop)
    return simulator
