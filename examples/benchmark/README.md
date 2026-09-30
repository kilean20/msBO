# Switch-scheduler acquisition benchmark

The current result is
`switch_acquisition_modes_20seeds_noiseaware.csv` with its matching JSON
summary. It was generated after conditional-state qLogEI was corrected to
condition on noisy averaged readings.

`switch_acquisition_modes_20seeds_stable.*` predates that correction and is
retained only as implementation history; do not use it to judge the current
method.

The figure `conditional_state_qlogei_benchmark.png` shows paired final loss,
query time, and the computation-to-machine-window comparison. Regenerate it
from the current CSV with:

```powershell
python examples/plot_conditional_state_qlogei.py
```
