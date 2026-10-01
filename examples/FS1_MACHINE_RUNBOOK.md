# FS1 quadrupole-centering notebook runbook

## Live-machine checklist

1. Start a fresh kernel on a host with the production EPICS/PHANTASY
   environment and the intended Channel Access configuration.
2. Run through the read-only preflight. Confirm beam identity, all CSET/RD/BPM
   values, requested control bounds, drive-limit checks, regular-simplex state
   settings, tolerances, and the printed measurement budget. Confirm that the
   dimensionless code table is centered and that the actual current table is
   acceptable for every quadrupole.
3. Confirm the configured `machineIO` wait and averaging spans are appropriate
   for current operations. The notebooks preserve the corresponding LSQ values:
   1 s post-ramp wait plus 4 s readings before/after the first dipole, and 1 s
   plus 5 s readings after the third dipole.
4. Only then set both `RUN_OPTIMIZATION = True` and
   `LIVE_MACHINE_CONFIRMED = True` in the operator-gate cell.
5. The machine-moving cell first records a BPM-magnitude reference at the
   captured starting controls for nominal and every coded quadrupole state.
   Confirm that all reference magnitudes are positive and credible. Subsequent
   BO observations use decision-magnet readbacks as their control coordinates
   and include the minimum measured/reference BPM-magnitude ratio as a
   beam-loss task.
6. Stay with the run. If a ramp times out, machineIO retries the same setpoint
   once. If the retry also times out, it waits 30 seconds, emits a prominent
   warning, and continues without raising. Inspect the live readbacks and abort
   manually if the deviation is not acceptable. After successful optimization,
   the notebook applies and retains the recommended controls with nominal
   quadrupoles.
7. After validation, compare both quadrupole-scan sensitivity and BPM-magnitude
   ratios. The decoded-response table and heat maps report the separately
   identified response of every quadrupole at every BPM. A validation exception
   means the scan did not complete; resolve the machine condition and rerun the
   validation cell. The cell returns to the recommended controls and nominal
   quadrupoles in its `finally` block.

The optimization budget is 50 evaluations for the four-quadrupole section and
30 evaluations for either two-quadrupole section. Nominal is a separate entry
and exit configuration, not an additional GP state or optimization evaluation.

Software validation cannot verify current PV permissions, beam availability,
device limits, interlocks, magnet response, or the suitability of bounds under
current conditions. Those remain operator preflight responsibilities.

The operations environment should provide the EPICS CA repeater executable on
`PATH`; investigate any Channel Access warning before opening the operator gate.
