# CATS-CF & PI-Koopman

English | [简体中文](README.zh-CN.md)

Self-collected car-following data and implementations for **Physics-Informed Koopman Learning for Data-Efficient System Identification of Automated Vehicle Dynamics**.

Models: CTH, IDM, PI-Koopman-CTH, PI-Koopman-IDM, DNN, LSTM, and projected Deep Koopman.

## Data

- **CATS-CF**: 42 leader–follower trials across 14 conditions, with three repetitions per condition. Includes ACC and human driving, with 15-s and 30-s windows. Public tables retain local coordinates and relative time, with absolute time and geographic location removed. Data live in `data/release/`; see [data structure and fields](DATA.md).
- **OpenACC**: [official JRC download](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/). We use 284 30-s episodes from AstaZero and Casale. See [source and license details](data/external/openacc/README.md).

CLI names: `local` = CATS-CF; `openacc` = OpenACC.

## Installation

Runs on **macOS arm64 (Python 3.12.14)** and **Linux x86_64 (Python 3.12.15)**, using CPU. A C99 compiler (`cc`, or set `CC`) is required.

| Dependency | Version |
|---|---|
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| pandas | 2.2.3 |
| scikit-learn | 1.9.1 |
| PyTorch | 2.14.0 |
| threadpoolctl | 3.7.0 |

Clone the repository, then run:

```sh
git clone https://github.com/CATS-Lab/PI-Koopman.git
cd PI-Koopman
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_native.py
```

## Quick Start

CATS-CF data are included in the repository:

```sh
python scripts/verify_data.py
python scripts/smoke_test.py
python scripts/run_prediction.py --dataset local --budget 3 --rate 10 --origin 20
```

This fits CTH, IDM, and both PI models using 3 s of history at 10 Hz, then forecasts 5 s from a 20-s origin. Results are saved to `outputs/physics/grid/local_B3_Hz10_end20/`.

To prepare OpenACC:

```sh
python scripts/download_openacc.py
python scripts/prepare_openacc.py
```

## Reproduce the paper

**History duration:** 3/8/15-s histories at 10 Hz, followed by a 5-s forecast. Run all seven models and aggregate the results:

```sh
python scripts/run_main.py --axis duration --dataset local --models all
python scripts/summarize_prediction.py --axis duration --dataset local
```

Replace `local` with `openacc` for OpenACC. Use `--models physics` or `--models neural` to run model groups separately.

**Sampling rate:** CATS-CF, 8-s histories at 1/5/10/25/125 Hz, with forecast origins at 15/20/24 s:

```sh
python scripts/run_main.py --axis rate --dataset local --models all
python scripts/summarize_prediction.py --axis rate --dataset local
```

Summaries are written to `outputs/summaries/`. Full neural searches can take hours; for a compact training example:

```sh
python experiments/unified_axes_neural.py --dataset local --budget 3 --rate 10 --origin 20 --stage compact
```

| Experiment | Command |
|---|---|
| Known-law response-field reconstruction | `python experiments/synthetic_benchmark.py` |
| Paired sampling-rate/noise consistency | `python experiments/claim_rate_consistency.py` |
| ACC/manual recognition from PI residuals | `python experiments/run_pi_residual_recognition.py` |
| Physical/PI timing | `python experiments/timing_single_encounter.py --label my-cpu --output outputs/timing_cpu` |
| Neural timing | `python experiments/timing_neural_single.py --device cpu --label my-cpu --output outputs/timing_neural` |

Grid settings: [configs/config.json](configs/config.json). `run_main.py` and `summarize_prediction.py` accept `--config PATH` to use another grid.

## License

Code: [MIT](LICENSE). CATS-CF data: [CC BY 4.0](LICENSE-DATA). OpenACC attribution and copyright notices are in the [data notes](data/external/openacc/README.md).
