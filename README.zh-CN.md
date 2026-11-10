# CATS-CF & PI-Koopman

[![arXiv](https://img.shields.io/badge/arXiv-2610.24588-b31b1b.svg)](https://arxiv.org/abs/2610.24588)

[English](README.md) | 简体中文

CATS-CF 跟驰数据集与 PI-Koopman 实验代码，配套论文为 **[Physics-Informed Koopman Learning for Data-Efficient System Identification of Automated Vehicle Dynamics](https://arxiv.org/abs/2610.24588)**。

<details>
<summary><strong>引用</strong> — 使用 CATS-CF 数据或本项目算法、代码时，请引用（展开 BibTeX）。</summary>

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

## 数据与代码

实验代码作者为 Hangyu Li。`data/CATS-CF/` 中的 42 张公开表各含 30 秒、125 Hz 的双车同步观测；实验片段均相对于各表的起点。OpenACC 从 AstaZero 和 Casale 选取 284 段 30 秒记录。命令中的 `local` 代表 CATS-CF，`openacc` 代表 OpenACC。

原始双车日志可用 `python -m scripts.rebuild_data --input RAW --output DEST` 重建公开表。OpenACC 来源为[欧盟 JRC](https://doi.org/10.2905/JRC.KMH3D00)。

### OpenPAV

<a href="https://openpav.github.io/OpenPAV/">
  <img src="https://openpav.github.io/OpenPAV/images/method.png" alt="OpenPAV: data collection, behavior modeling, and performance evaluation" width="640">
</a>

CATS-CF 将贡献到 [OpenPAV](https://openpav.github.io/OpenPAV/) 这一综合开源平台，支持自动驾驶车辆的数据共享、行为建模与评估。

## 复现实验

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

DNN、LSTM 各自固定结构和训练流程。采样率实验沿用 7 秒参考条件的组外参数。先为两个数据集各生成历史长度实验汇总，再运行噪声实验和计时。单个物理预测条件可用 `python -m experiments.run_prediction --dataset local --budget 7 --rate 10 --origin 20`；主实验网格位于 `configs/config.json`。计时命令可分别指定两个数据集和 5、7、11 秒历史；神经网络可用 `--device cuda`。输出保存在 `outputs/`。

10 Hz 噪声实验对独立噪声和时间相关噪声各使用三个配对随机种子。车距误差标准差为 0.2 m，两车速度误差标准差各为 0.1 m；分别将两车速度限制为非负，再计算相对速度。全部 3,380 条实车和配套仿真记录参与比较。

## 许可

代码采用 [MIT](LICENSE)，CATS-CF 数据采用 [CC BY 4.0](LICENSE-DATA)。OpenACC © European Union，CC BY 4.0（[AstaZero](data/OpenACC/AstaZero_copyright.txt)、[Casale](data/OpenACC/Casale_copyright.txt)）。
