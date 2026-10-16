# OpenACC external validation subset

Official JRC data (CC BY 4.0), raw bytes and copyright notices retained. Cite Makridis et al. (2021), doi:10.1016/j.trc.2021.103047 and the JRC database doi:10.2905/JRC.KMH3D00.

Released 10 Hz grids: AstaZero was acquired above 100 Hz and downsampled; Casale used cubic-spline interpolation to reach 10 Hz and was not filtered, according to the official campaign notes. Non-overlapping 30 s blocks require finite nonnegative velocities, positive bumper gap, 0.1 s spacing, and one constant ACC/Human follower label. At most six qualifying blocks per source file/follower are sampled at deterministic evenly spaced indices. No trajectory-error or excitation threshold was used for selection. Every selected block retains its file, follower and time identity. These blocks are dependent within files, vehicles and collection days; file holdout is not vehicle/day holdout, and this is not a full-database benchmark. Gap is the released IVS, with no extra bumper correction.

284 episodes; {'AstaZero': 240, 'Casale': 44}
