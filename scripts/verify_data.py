"""Check the published CATS-CF trajectories at each analysis rate."""
import numpy as np
import pandas as pd
from src.paths import RELEASE_DIR
from src.data.trials import load_trials, _following_run
from src.data.windows import PAPER_15_START_SAMPLES


def main():
    files = sorted(RELEASE_DIR.glob("*_run??.csv"))
    assert len(files) == 42 and len(list(RELEASE_DIR.rglob("*.csv"))) == 42
    columns = None
    for file in files:
        data = pd.read_csv(file, float_precision="round_trip")
        assert data.shape == (3750, 26), (file, data.shape)
        if columns is None:
            columns = list(data)
        assert list(data) == columns and np.isfinite(data.to_numpy()).all()
        assert columns[-3:] == ["distance_m", "gap_m", "relative_v_mps"]
        np.testing.assert_allclose(data.t_s, np.arange(3750)/125, atol=1e-12, rtol=0)
        heading = np.deg2rad(data.follower_heading_deg)
        east, north = data.leader_x_m-data.follower_x_m, data.leader_y_m-data.follower_y_m
        np.testing.assert_allclose(data.distance_m, east*np.sin(heading)+north*np.cos(heading), atol=1e-10)
        np.testing.assert_allclose(data.gap_m, data.distance_m-3., atol=1e-10)
        leader_along_follower = data.leader_vx_mps*np.sin(heading)+data.leader_vy_mps*np.cos(heading)
        np.testing.assert_allclose(data.relative_v_mps, leader_along_follower-data.follower_v_lon_mps, atol=1e-10)
        for role in ("leader", "follower"):
            angle = np.deg2rad(data[f"{role}_heading_deg"])
            vx, vy = data[f"{role}_vx_mps"], data[f"{role}_vy_mps"]
            np.testing.assert_allclose(data[f"{role}_v_lon_mps"], vx*np.sin(angle)+vy*np.cos(angle), atol=1e-10)
            np.testing.assert_allclose(data[f"{role}_v_lat_mps"], vx*np.cos(angle)-vy*np.sin(angle), atol=1e-10)
            np.testing.assert_allclose(data[f"{role}_v_lon_mps"]**2+data[f"{role}_v_lat_mps"]**2, vx**2+vy**2, atol=1e-10)
    print("42 plain CSV tables: 3750 rows, 26 columns, paired coordinates and velocity projections verified")
    for duration in (30,):
        parent, _ = load_trials(duration, 125)
        for rate in (25, 5, 1):
            trials, _ = load_trials(duration, rate)
            stride = 125 // rate
            for base, trial in zip(parent, trials, strict=True):
                assert trial.uid == base.uid and trial.run == base.run and trial.top2 == base.top2
                np.testing.assert_array_equal(trial.t, base.t[::stride])
                np.testing.assert_array_equal(trial.z, base.z[::stride])
                np.testing.assert_array_equal(trial.corr, base.corr[::stride])
        print(f"{duration} s: {len(parent)} trials, 125/25/5/1 Hz verified")
    contained = 0
    for file in files:
        start = PAPER_15_START_SAMPLES.get(file.stem, 0)
        if start < 0:
            print(f"Original 15 s window pending: {file.stem} starts {-start/125:g} s before its 30 s table; no replacement was selected")
            continue
        window = _following_run(file, 15)
        assert len(window) == 1875
        np.testing.assert_allclose(window.t_s, np.arange(start, start+1875)/125, atol=1e-12)
        contained += 1
    print(f"Original 15 s window containment verified for {contained}/42 trials")


if __name__ == "__main__":
    main()
