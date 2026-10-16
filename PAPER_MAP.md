# Main-text implementation map

| Main-text task | Entry point | Scientific contract |
|---|---|---|
| Duration vs prediction | scripts/run_main.py --axis duration | B=3/8/15 s; 10 Hz; origin=20 s; next 5 s; seven models |
| Rate vs prediction | scripts/run_main.py --axis rate | B=8 s; 1/5/10/25/125 Hz; origins=15/20/24; equal origin weight |
| Physical/PI implementations | revision_physics.py, revision_native.c, revision_koopman.py | two-start output error; weak prior correction; calibration-only scaling |
| Neural implementations | unified_axes_neural.py and compact/staged dependencies | original compact/expanded source-selected candidates, seeds and updates |
| Known-law reconstruction | synthetic_benchmark.py | affine/quadratic; 18 train/12 test trajectories; 8 corruptions; shared dictionary |
| Paired field consistency | claim_rate_consistency.py | 42 trials; [2,10] history; shared noisy parent; 41 clean probes; grouped uncertainty |
| PI residual recognition | run_pi_residual_recognition.py | full nominal 15 s/25 Hz; 30 one-second residual supports; run-disjoint thresholds |
| Data preparation | scripts/rebuild_data.py --input <private-parents>; data/scripts | frozen synchronized parents; geometry and best fixed windows |

Experiment entry points are in `experiments/`; shared model and training modules
are in `src/`. Prediction execution and aggregation read `configs/config.json`.
The recognition demonstration includes the physical baselines in the main-text table.

Timing entry points: `experiments/timing_single_encounter.py` and
`experiments/timing_neural_single.py`; state/input excerpts are in `data/timing/`.
