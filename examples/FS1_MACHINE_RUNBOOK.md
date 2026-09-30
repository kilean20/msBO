# FS1 quadrupole-centering notebook runbook

## Live-machine checklist

1. Start a fresh kernel on a host with the production EPICS/PHANTASY
   environment and the intended Channel Access configuration.
2. Run through the read-only preflight. Confirm beam identity, all CSET/RD/BPM
   values, requested control bounds, drive-limit checks, categorical state
   settings, tolerances, and the printed measurement budget.
3. Confirm the configured `machineIO` wait and averaging spans are appropriate
   for current operations. The notebooks preserve the corresponding LSQ values:
   1 s post-ramp wait plus 4 s readings before/after the first dipole, and 1 s
   plus 5 s readings after the third dipole.
4. Only then set both `RUN_OPTIMIZATION = True` and
   `LIVE_MACHINE_CONFIRMED = True` in the operator-gate cell.
5. Stay with the run. If a ramp times out, machineIO retries the same setpoint
   once. If the retry also times out, it waits 30 seconds, emits a prominent
   warning, and continues without raising. Inspect the live readbacks and abort
   manually if the deviation is not acceptable. Independent optimization
   exceptions or interruptions still attempt to restore the captured starting
   steering controls and nominal quadrupoles; a rollback failure is printed as
   requiring operator action.
6. After validation, either retain the recommendation or run the explicit
   rollback cell.

Software validation cannot verify current PV permissions, beam availability,
device limits, interlocks, magnet response, or the suitability of bounds under
current conditions. Those remain operator preflight responsibilities.

The operations environment should provide the EPICS CA repeater executable on
`PATH`; investigate any Channel Access warning before opening the operator gate.
