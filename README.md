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

```sh
python -m pip install -e .
python -m scripts.build_native
python -m scripts.verify_data
python -m scripts.download_openacc
python -m experiments.run_main --axis duration --dataset both
python -m experiments.summarize_prediction --axis duration --dataset local
python -m experiments.summarize_prediction --axis duration --dataset openacc
python -m experiments.run_main --axis rate --dataset local
python -m experiments.summarize_prediction --axis rate --dataset local
python -m experiments.recognition
python -m experiments.noise_response --workers 4
python -m experiments.noise_neural --prepare --input outputs/noise/observations
python -m experiments.noise_neural --input outputs/noise/observations --device cpu
python -m experiments.summarize_noise
python -m experiments.timing local 7 --device cpu
```

Training DNN and LSTM uses fixed architectures and one grouped selection procedure. The 7-second reference transfers its settings to each rate and origin. Run the two duration summaries before the noise and timing experiments. For a single physical prediction condition, use `python -m experiments.run_prediction --dataset local --budget 7 --rate 10 --origin 20`. The main grid is in `configs/config.json`. The timing command accepts either dataset and budgets 5, 7 and 11. GPU neural fits use `--device cuda`. Results are created in `outputs/`.

The 10 Hz noise experiment uses three paired realizations of independent or time-correlated errors (gap SD 0.2 m, speed SD 0.1 m per vehicle). Each absolute speed has a lower bound of zero. Relative speed is computed from the perturbed speeds. All 3,380 recorded and simulated follower records enter the comparison.

## License

Code: [MIT](LICENSE). CATS-CF data: [CC BY 4.0](LICENSE-DATA). OpenACC © European Union, CC BY 4.0 ([AstaZero](data/OpenACC/AstaZero_copyright.txt), [Casale](data/OpenACC/Casale_copyright.txt)).
