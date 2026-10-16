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

Paths without a prefix are in `experiments/`. The old affine-coefficient classifier
is not the main demonstration; affine/physical residual controls remain with the
current PI demonstration because it uses them directly. Strong-error predictive
stress and the RTM/hybrid comparison are explicitly supplementary in the selected
latest main text and are excluded as standalone experiments.

Timing entry points: `experiments/timing_single_encounter.py` and
`experiments/timing_neural_single.py`; frozen state/input excerpts are in `data/timing/`.
