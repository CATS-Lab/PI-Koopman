# CATS-CF & PI-Koopman

[![arXiv](https://img.shields.io/badge/arXiv-2610.24588-b31b1b.svg)](https://arxiv.org/abs/2610.24588)

English | [简体中文](README.zh-CN.md)

Self-collected car-following data and implementations for **[Physics-Informed Koopman Learning for Data-Efficient System Identification of Automated Vehicle Dynamics](https://arxiv.org/abs/2610.24588)**.

<details>
<summary><strong>Citation</strong> — Please cite when using CATS-CF or our algorithms/code (BibTeX).</summary>

```bibtex
@misc{li2026physicsinformedkoopmanlearningdataefficient,
  title={Physics-Informed Koopman Learning for Data-Efficient System Identification of Automated Vehicle Dynamics},
  author={Hangyu Li and Yiming Meng and Zhaohui Liang and Yunqing Xiao and Yuxuan Chen and Bofeng Cao and Shixiao Liang and Jiaxi Liu and Haotian Shi and Melkior Ornik and Xiaopeng Li},
  year={2026},
  eprint={2610.24588},
  archivePrefix={arXiv},
  primaryClass={eess.SY},
  url={https://arxiv.org/abs/2610.24588}
}
```

</details>

## Data

- **CATS-CF**: 42 leader–follower trials across 14 conditions, with three repetitions per condition. Includes ACC and human driving, with 15-s and 30-s windows. Tables provide local coordinates and relative time. Data live in `data/release/`; see [data structure and fields](DATA.md).
- **OpenACC**: [official JRC download](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/). We use 284 30-s episodes from AstaZero and Casale. See [source and license details](data/external/openacc/README.md).

CLI names: `local` = CATS-CF; `openacc` = OpenACC.

### OpenPAV

<a href="https://openpav.github.io/OpenPAV/">
  <img src="https://openpav.github.io/OpenPAV/images/method.png" alt="OpenPAV: data collection, behavior modeling, and performance evaluation" width="640">
</a>

CATS-CF will contribute to [OpenPAV](https://openpav.github.io/OpenPAV/), a broader open-source platform for automated vehicle data, behavior modeling, and evaluation.

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
conda create -n pi-koopman python=3.12 pip -y
conda activate pi-koopman
python -m pip install -e .
python -m scripts.build_native
```

## Quick Start

CATS-CF data are included in the repository:

```sh
python -m scripts.verify_data
python -m scripts.smoke_test
python -m experiments.run_prediction --dataset local --budget 5 --rate 10 --origin 20
```

This fits CTH, IDM, and both PI models using 5 s of history at 10 Hz, then forecasts 5 s from a 20-s origin. Results are saved to `outputs/physics/grid/local_B5_Hz10_end20/`.

To prepare OpenACC:

```sh
python -m scripts.download_openacc
python -m scripts.prepare_openacc
```

## Reproduce the paper

**History duration:** 5/7/11-s histories at 10 Hz, with a common forecast origin at 20 s and a 5-s forecast. Run all seven models and aggregate the results:

```sh
python -m experiments.run_main --axis duration --dataset local --models all
python -m experiments.summarize_prediction --axis duration --dataset local
```

Replace `local` with `openacc` for OpenACC. Use `--models physics` or `--models neural` to run model groups separately. Add `--device cuda:0` to `run_main` to train neural models on a CUDA device.

**Sampling rate:** CATS-CF, 7-s histories at 1/5/10/25/125 Hz, with forecast origins at 15/20/24 s:

```sh
python -m experiments.run_main --axis rate --dataset local --models all
python -m experiments.summarize_prediction --axis rate --dataset local
```

Summaries are written to `outputs/summaries/`. Full neural searches can take hours; for a compact training example:

```sh
python -m experiments.neural_prediction --dataset local --budget 5 --rate 10 --origin 20 --stage compact
```

| Experiment | Command |
|---|---|
| Known-law response-field reconstruction | `python -m experiments.synthetic_benchmark` |
| Paired sampling-rate/noise consistency | `python -m experiments.rate_consistency` |
| ACC/manual recognition from PI residuals | `python -m experiments.recognition` |
| Physical/PI timing | `python -m experiments.timing_physics --label my-cpu --output outputs/timing_cpu` |
| Neural timing | `python -m experiments.timing_neural --device cpu --label my-cpu --output outputs/timing_neural` |

Grid settings: [configs/config.json](configs/config.json). `run_main.py` and `summarize_prediction.py` accept `--config PATH` to use another grid.

## Repository layout

All experiment results are written under `outputs/`.

- `src/models/`: Physical, Koopman, and neural implementations
- `src/training/`: Training and source-group model selection
- `src/data/`: Data loading, geometry, window selection, and public export
- `src/evaluation/`: Metrics and recognition helpers
- `experiments/`: Paper experiment and aggregation entry points
- `scripts/`: Native build, data preparation, and checks

## License

Code: [MIT](LICENSE). CATS-CF data: [CC BY 4.0](LICENSE-DATA). OpenACC attribution and copyright notices are in the [data notes](data/external/openacc/README.md).
