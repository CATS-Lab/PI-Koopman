# CATS-CF & PI-Koopman

[![arXiv](https://img.shields.io/badge/arXiv-2610.24588-b31b1b.svg)](https://arxiv.org/abs/2610.24588)

English | [简体中文](README.zh-CN.md)

CATS-CF dataset & PI-Koopman research code for **[Physics-Informed Koopman Learning for Data-Efficient System Identification of Automated Vehicle Dynamics](https://arxiv.org/abs/2610.24588)**.

<details>
<summary><strong>Citation</strong> — Please cite when using CATS-CF or our algorithms/code (BibTeX).</summary>

```bibtex
@misc{li2026physicsinformedkoopmanlearningdataefficient,
  title={Physics-Informed Koopman Learning for Data-Efficient System Identification of Automated Vehicle Dynamics},
  author={Hangyu Li and Yiming Meng and Zhaohui Liang and Yunqing Xiao and Yuxuan Chen and Bofeng Cao and Shixiao Liang and Jiaxi Liu and Haotian Shi and Xiaopeng Li},
  year={2026},
  eprint={2610.24588},
  archivePrefix={arXiv},
  primaryClass={eess.SY},
  url={https://arxiv.org/abs/2610.24588}
}
```

</details>

## Data and code

Hangyu Li maintains the experiment code. The 42 tables in `data/CATS-CF/` contain 30 seconds of synchronized observations at 125 Hz. All experiment times are relative to each table. OpenACC uses 284 eligible 30-second observations from AstaZero and Casale. Use `local` for CATS-CF and `openacc` for OpenACC.

Rebuild the released tables from the paired raw vehicle logs with `python -m scripts.rebuild_data --input RAW --output DEST`. The external trajectories are from the [European Commission JRC OpenACC dataset](https://doi.org/10.2905/JRC.KMH3D00).

### OpenPAV

<a href="https://openpav.github.io/OpenPAV/">
  <img src="https://openpav.github.io/OpenPAV/images/method.png" alt="OpenPAV: data collection, behavior modeling, and performance evaluation" width="640">
</a>

CATS-CF will contribute to [OpenPAV](https://openpav.github.io/OpenPAV/), a broader open-source platform for automated vehicle data, behavior modeling, and evaluation.

## Reproduction

### Setup

From the repository root, use Python 3.12 and a C99 compiler on Linux or macOS (WSL on Windows). Install the dependencies once, build the integration kernel and verify both datasets.

```sh
python -m pip install -e .
python -m scripts.build_native
python -m scripts.verify_data
python -m scripts.download_openacc
python -m scripts.prepare_openacc
```

The checks should report 42 CATS-CF tables and 284 OpenACC encounters. Commands below default to CPU. On a CUDA machine, append `--device cuda` to a prediction, noise or timing command. Generated files go to `outputs/`. Reference values are rounded. Neural fits and nonlinear optimizers can vary slightly across devices and numerical libraries.

### 1. Prediction by history length

Fit each model with 5, 7 or 11 s of history at 10 Hz and forecast the next 5 s from t = 20 s. This run also writes the held-out group settings used by the noise and timing experiments.

```sh
python -m scripts.reproduce duration
```

Each cell gives speed RMSE (m/s) / gap RMSE (m), averaged over ACC encounters.

**CATS-CF** · `outputs/summaries/local_duration.csv`

| Model | 5 s | 7 s | 11 s |
| --- | --- | --- | --- |
| CTH | 0.428 / 0.771 | 0.445 / 0.913 | 0.493 / 1.019 |
| IDM | 0.518 / 0.920 | 0.500 / 0.964 | 0.373 / 0.835 |
| DNN | 0.488 / 0.772 | 0.548 / 0.962 | 0.501 / 0.819 |
| LSTM | 0.501 / 0.797 | 0.497 / 0.784 | 0.483 / 0.736 |
| Deep Koopman | 0.589 / 0.865 | 0.561 / 0.872 | 0.567 / 0.859 |
| PI-Koopman-CTH | 0.420 / 0.743 | 0.423 / 0.868 | 0.479 / 0.989 |
| PI-Koopman-IDM | 0.541 / 0.913 | 0.465 / 0.892 | 0.337 / 0.676 |

**OpenACC** · `outputs/summaries/openacc_duration.csv`

| Model | 5 s | 7 s | 11 s |
| --- | --- | --- | --- |
| CTH | 0.331 / 0.626 | 0.333 / 0.653 | 0.370 / 0.766 |
| IDM | 0.347 / 0.658 | 0.351 / 0.661 | 0.377 / 0.765 |
| DNN | 0.435 / 0.706 | 0.415 / 0.688 | 0.387 / 0.672 |
| LSTM | 0.418 / 0.680 | 0.400 / 0.659 | 0.399 / 0.658 |
| Deep Koopman | 0.840 / 1.226 | 0.646 / 1.041 | 0.635 / 0.985 |
| PI-Koopman-CTH | 0.334 / 0.621 | 0.327 / 0.634 | 0.362 / 0.741 |
| PI-Koopman-IDM | 0.373 / 0.679 | 0.402 / 0.772 | 0.378 / 0.733 |

### 2. Prediction by observation rate

On CATS-CF, use 7 s of history and average forecasts starting at 15, 20 and 24 s. All origins and rates use settings selected at the 7 s, 10 Hz reference condition.

```sh
python -m scripts.reproduce rate
```

Each cell gives speed RMSE (m/s) / gap RMSE (m), averaged over ACC encounters.

**CATS-CF** · `outputs/summaries/local_rate.csv`

| Model | 1 Hz | 5 Hz | 10 Hz | 25 Hz | 125 Hz |
| --- | --- | --- | --- | --- | --- |
| CTH | 0.347 / 0.681 | 0.365 / 0.721 | 0.369 / 0.729 | 0.373 / 0.739 | 0.375 / 0.743 |
| IDM | 0.433 / 0.853 | 0.445 / 0.891 | 0.448 / 0.898 | 0.466 / 0.932 | 0.468 / 0.937 |
| DNN | 0.442 / 0.789 | 0.464 / 0.830 | 0.466 / 0.833 | 0.475 / 0.848 | 0.475 / 0.850 |
| LSTM | 0.463 / 0.779 | 0.423 / 0.717 | 0.416 / 0.705 | 0.431 / 0.725 | 0.444 / 0.759 |
| Deep Koopman | 0.518 / 0.847 | 0.535 / 0.853 | 0.535 / 0.854 | 0.541 / 0.869 | 0.559 / 0.890 |
| PI-Koopman-CTH | 0.342 / 0.674 | 0.352 / 0.694 | 0.356 / 0.701 | 0.357 / 0.700 | 0.359 / 0.706 |
| PI-Koopman-IDM | 0.421 / 0.828 | 0.416 / 0.825 | 0.416 / 0.825 | 0.433 / 0.861 | 0.431 / 0.855 |

### 3. ACC and human-driver recognition

Fit the first 15 s of each CATS-CF encounter and score the variance of the velocity-integral residual. The classification threshold is selected on the other runs.

```sh
python -m scripts.reproduce recognition
```

`outputs/recognition/summary.csv`

| Hz | Model | ACC correct | Human correct | Balanced accuracy | AUC |
| --- | --- | --- | --- | --- | --- |
| 10 | CTH-OE | 20/24 | 18/18 | 0.917 | 0.938 |
| 10 | IDM-OE | 15/24 | 14/18 | 0.701 | 0.824 |
| 10 | PI-Koopman-CTH | 19/24 | 17/18 | 0.868 | 0.940 |
| 10 | PI-Koopman-IDM | 16/24 | 16/18 | 0.778 | 0.847 |
| 25 | CTH-OE | 20/24 | 18/18 | 0.917 | 0.938 |
| 25 | IDM-OE | 15/24 | 14/18 | 0.701 | 0.819 |
| 25 | PI-Koopman-CTH | 19/24 | 17/18 | 0.868 | 0.940 |
| 25 | PI-Koopman-IDM | 16/24 | 16/18 | 0.778 | 0.847 |

### 4. Response under observation noise

Run experiment 1 first for its prediction settings. All methods use the same 7 s histories at 10 Hz, three origins and three noise seeds. Gap noise has SD 0.2 m and each vehicle’s speed noise has SD 0.1 m. Absolute speeds are bounded below by zero before computing relative speed.

```sh
python -m scripts.reproduce noise
```

**Seven-model comparison.** For the 260 recorded encounters, the metric is the RMS change from the clean-fit acceleration response. For 3,120 matched simulations, it is acceleration RMSE against truth. Each cell gives independent / time-correlated noise (m/s²).

`outputs/noise/summary.csv` · `outputs/noise/claim_rate_consistency.pdf`

The reproduction command applies the paper’s 24 held-out simulation PI configurations from `configs/simulation_pi.csv` (two physical priors × twelve followers). To repeat their offline selection on your machine, run `python -m experiments.noise_response` after experiment 1. Selection can change the fitted window and the resulting errors across numerical environments.

| Model | CATS-CF recorded | OpenACC recorded | CATS-CF simulated | OpenACC simulated |
| --- | --- | --- | --- | --- |
| CTH | 0.0405 / 0.0640 | 0.0562 / 0.0707 | 0.0774 / 0.1079 | 0.0852 / 0.1029 |
| IDM | 0.0375 / 0.0551 | 0.0529 / 0.0657 | 0.0487 / 0.0747 | 0.0613 / 0.0765 |
| DNN | 0.1309 / 0.1223 | 0.2118 / 0.1860 | 0.2169 / 0.2070 | 0.2155 / 0.1886 |
| LSTM | 0.2039 / 0.2192 | 0.3527 / 0.3233 | 0.4011 / 0.3652 | 0.3823 / 0.3428 |
| Deep Koopman | 0.1671 / 0.1666 | 0.1580 / 0.1555 | 0.2992 / 0.2968 | 0.2849 / 0.2833 |
| PI-Koopman-CTH | 0.0384 / 0.0586 | 0.0516 / 0.0656 | 0.0719 / 0.0996 | 0.0782 / 0.0941 |
| PI-Koopman-IDM | 0.0348 / 0.0517 | 0.0494 / 0.0624 | 0.0464 / 0.0724 | 0.0569 / 0.0720 |

**Estimator comparison.** FD uses finite differences, Integral uses the integral equation and Exp. weak (ours) uses exponentially weighted weak regression. The table gives simulated truth RMSE (m/s²), independent / time-correlated. The figure subtracts the ordinary weak result from each value. A negative difference indicates lower error.

`outputs/noise/summary.csv` · `outputs/noise/synthetic_field_error.pdf`

| Prior | Estimator | CATS-CF | OpenACC |
| --- | --- | --- | --- |
| CTH | FD | 0.07241 / 0.09803 | 0.07881 / 0.09296 |
| CTH | Integral | 0.07542 / 0.10089 | 0.08093 / 0.09494 |
| CTH | **Exp. weak (ours)** | 0.07190 / 0.09958 | 0.07818 / 0.09415 |
| IDM | FD | 0.04856 / 0.06936 | 0.05895 / 0.06949 |
| IDM | Integral | 0.04957 / 0.07238 | 0.05994 / 0.07243 |
| IDM | **Exp. weak (ours)** | 0.04644 / 0.07235 | 0.05686 / 0.07204 |

### Computational efficiency

Run experiment 1 first. The CPU command times all seven models, and the optional CUDA command times the three neural models. Time one fit and one 5 s forecast per target on six encounters per dataset at each of the three history lengths. Run on each device whose timing you want to measure.

```sh
python -m scripts.reproduce timing
# NVIDIA GPU
python -m scripts.reproduce timing --device cuda
```

Means in ms, reported as fit (+ forecast). The CPU and GPU commands write `outputs/timing.csv` and `outputs/timing_cuda.csv` respectively. Times depend on hardware and software.

| Model | M4 Pro | Threadripper | RTX 6000 Ada |
| --- | --- | --- | --- |
| CTH | 14.42 (+0.02) | 16.21 (+0.02) | — |
| DNN | 4101.87 (+4.98) | 5410.59 (+8.20) | 2231.99 (+8.82) |
| Deep Koopman | 259.48 (+9.12) | 350.97 (+12.49) | 246.91 (+12.89) |
| IDM | 21.80 (+0.03) | 22.70 (+0.03) | — |
| LSTM | 8635.98 (+5.66) | 10809.42 (+9.17) | 4163.01 (+10.00) |
| PI-Koopman-CTH | 15.47 (+0.07) | 17.20 (+0.08) | — |
| PI-Koopman-IDM | 23.06 (+0.08) | 23.71 (+0.10) | — |

## License

Code: [MIT](LICENSE). CATS-CF data: [CC BY 4.0](LICENSE-DATA). OpenACC © European Union, CC BY 4.0 ([AstaZero](data/OpenACC/AstaZero_copyright.txt), [Casale](data/OpenACC/Casale_copyright.txt)).
