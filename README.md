# msBO

Multi-state Bayesian optimization for shared controls measured under discrete
machine states.

`step_batch_with_switch()` uses conditional-state qLogEI by default. The method
values a candidate using only the noisy diagnostic readings obtainable in the
scheduled categorical state, while the multi-task GP transfers information to
the beam responses under other states.

- [Conditional-state qLogEI: derivation, limitations, benchmark, and citations](conditional_state_qLogEI.md)
- [Full msBO methodology](msBO_methodology.md)
- [Four-state concurrency validation notebook](examples/[VM]4state-switch-concurrency-test.ipynb)

```python
msbo.step_batch_with_switch(current_state, next_state, q=3)

# Explicit comparison modes:
msbo.step_batch_with_switch(current_state, next_state, q=3,
                            acq_state_mode="global")
msbo.step_batch_with_switch(current_state, next_state, q=3,
                            acq_state_mode="mean")
```

The legacy boolean remains accepted: explicit `fix_acq_state=False` means
global qLogEI and explicit `fix_acq_state=True` means the old scheduled-state
mean-fill approximation.

# TODO:




# Consider:



