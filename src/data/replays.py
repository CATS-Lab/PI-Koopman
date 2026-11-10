"""Generate native-clock follower trajectories driven by recorded leaders."""

import numpy as np
from src.data.trials import Trial

laws = ["CTH", "IDM", "asymmetric", "lagged"]
styles = ["short", "medium", "long"]
params = [
    (0.8, 0.18, 0.75, 1.9, 2.5, 0.35),
    (1.3, 0.12, 0.55, 1.4, 2.0, 0.65),
    (1.8, 0.09, 0.40, 1.0, 1.6, 0.95),
]


ORIGINS = (15, 20, 24)
NOISE = [("clean", 0)] + [
    (kind, seed) for kind in ("white", "correlated") for seed in (1101, 1102, 1103)
]


def field(law, style, z, u):
    tau, ks, kv, amax, b, lag = params[style]
    if law == "lagged" and style == 2:
        # Long-headway lagged followers use a stronger closing-speed response.
        kv = 0.50
    s, v, m = z
    v = np.maximum(0.0, v)
    d = u - v
    e = s - 3 - tau * v
    a = ks * e + kv * d
    if law == "IDM":
        star = 3 + np.maximum(0, v * tau - v * d / (2 * np.sqrt(amax * b)))
        a = amax * (1 - (v / 35) ** 4 - (star / np.maximum(s, 0.05)) ** 2)
    if law == "asymmetric":
        a = ks * np.where(e < 0, 1.5 * e, 0.75 * e) + kv * np.where(
            d < 0, 1.5 * d, 0.75 * d
        )
    command = np.clip(a, -6, 3)
    actual = m if law == "lagged" else command
    actual = np.where(v <= 0, np.maximum(actual, 0), actual)
    return np.stack(
        [u - v, actual, (command - m) / lag if law == "lagged" else np.zeros_like(v)]
    )


def simulate(trials, law, style, sub=1):
    u = np.array([tr.u for tr in trials]).T
    t = trials[0].t
    h = (t[1] - t[0]) / sub
    state = np.array([[tr.z[0, 0], tr.z[0, 1], 0.0] for tr in trials]).T
    zs = []
    acc = []
    for j in range(len(t)):
        zs.append(state.copy())
        acc.append(field(law, style, state, u[j])[1])
        if j == len(t) - 1:
            break
        for i in range(sub):
            left = u[j] + (u[j + 1] - u[j]) * i / sub
            right = u[j] + (u[j + 1] - u[j]) * (i + 1) / sub
            mid = (left + right) / 2
            a = field(law, style, state, left)
            b = field(law, style, state + h * a / 2, mid)
            c = field(law, style, state + h * b / 2, mid)
            d = field(law, style, state + h * c, right)
            state = state + h * (a + 2 * b + 2 * c + d) / 6
            state[1] = np.maximum(state[1], 0)
    return np.array(zs), np.array(acc)


def make10(tr):
    t = np.arange(300) / 10
    z = np.array([np.interp(t, tr.t, tr.z[:, j]) for j in (0, 1)]).T
    u = np.interp(t, tr.t, tr.u)
    return Trial(tr.uid, tr.case, tr.mode, tr.run, t, np.c_[z, u - z[:, 1]], u)


def records(dataset, trials):
    leaders = [tr for tr in trials if tr.mode.startswith("ACC")]
    for tr in leaders:
        yield dataset, "recorded", make10(tr), None
    for law in laws:
        for style in range(len(styles)):
            states, acc = simulate(leaders, law, style)
            if (
                not np.isfinite(states).all()
                or states[:, 0].min() <= 0
                or states[:, 1].min() < 0
            ):
                raise ValueError(
                    f"Invalid trajectory for {dataset} {law} {styles[style]}"
                )
            for i, tr in enumerate(leaders):
                z = np.c_[states[:, 0, i], states[:, 1, i], tr.u - states[:, 1, i]]
                follow = make10(Trial(tr.uid, tr.case, tr.mode, tr.run, tr.t, z, tr.u))
                yield (
                    dataset,
                    f"{law}_{styles[style]}",
                    follow,
                    np.interp(follow.t, tr.t, acc[:, i]),
                )
