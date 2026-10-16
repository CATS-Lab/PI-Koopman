# CATS-CF data dictionary and processing

`data/release/15s` and `30s` contain 42 trials each, 14 conditions × three runs.
ACC1/ACC2 identify ACC following settings; HV1/HV2 identify the two recorded human
drivers. Keep entire encounters together in splits.
The 15-s and 30-s windows can begin at different points in the same original run.

Each directory has `koopman_ready/koopman_{125,25,5,1}hz.csv` and
quality, trial, and cross-validation manifests.
`data/release/metadata/data_dictionary.csv` is the detailed field dictionary.

Core fields: `trial_uid` (encounter), `case_id` (leader/follower condition),
`run_index` (1/2/3), `follower_mode`, `leader_mode`, `t_rel_s` (seconds),
`gap_m` (metres), `follower_longitudinal_velocity_mps` (m/s),
`leader_velocity_in_follower_longitudinal_mps` (m/s). Model coordinates are
z=[gap, follower speed, leader speed minus follower speed].
The yaw diagnostic uses `follower_yaw_rate_corrimu_radps` × `lateral_offset_left_m`.

GPS tags align continuous channels by linear interpolation at 125 Hz. WGS84
geodetic coordinates pass through ECEF and per-trial ENU, then follower-heading
projection. Gap subtracts 3 m from longitudinal sensor spacing. CORRIMU increments
are divided by their source intervals; units are documented in the field dictionary.
25/5/1-Hz files are exact integer decimations, without anti-alias filtering.
10 Hz is interpolated within each observation interval only. Relative speed is
recomputed after resampling.

A D-second release contains D×rate samples: 15 s at 25 Hz contains
375 rows ending at 14.96 s. Prediction histories strictly inside 30-s records are
endpoint-inclusive, e.g. 3 s at 10 Hz gives 31 samples. The principal prediction
origin is 20 s, with the next 5 s scored excluding the common initial state.

Public tables retain trial-local ENU coordinates, headings, relative sample times,
and the motion channels above. Absolute GPS time, latitude/longitude, geographic
ENU origins, and timestamp-bearing source identifiers are removed.

The geometry and window-selection implementation is in `src/data/`.
For data owners with private synchronized parent tables, run:

```sh
python -m scripts.rebuild_data --input /path/to/private/parents --output outputs/rebuilt_data
```

This runs the original geometry pipeline, then applies the public-field export.
Training and evaluation use the included release tables.
