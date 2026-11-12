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

原始双车日志可用 `python -m scripts.rebuild_data --input RAW` 重建公开表。OpenACC 来源为[欧盟 JRC](https://doi.org/10.2905/JRC.KMH3D00)。

### OpenPAV

<a href="https://openpav.github.io/OpenPAV/">
  <img src="https://openpav.github.io/OpenPAV/images/method.png" alt="OpenPAV: data collection, behavior modeling, and performance evaluation" width="640">
</a>

CATS-CF 将贡献到 [OpenPAV](https://openpav.github.io/OpenPAV/) 这一综合开源平台，支持自动驾驶车辆的数据共享、行为建模与评估。

## 复现实验

### 环境与数据准备

在仓库根目录使用 Python 3.12 和 C99 编译器。支持 Linux、macOS 和 Windows 上的 WSL。安装依赖、编译积分程序并检查两套数据。

```sh
python -m pip install -e .
python -m scripts.build_native
python -m scripts.verify_data
python -m scripts.download_openacc
python -m scripts.prepare_openacc
```

默认使用 CPU。预测、噪声和计时实验支持 `--device cuda`。

### 1. 不同历史长度下的预测

以 10 Hz 观测 5、7 或 11 秒，从第 20 秒起预测未来 5 秒。

```sh
python -m scripts.reproduce duration
```

每格为 ACC 跟驰样本的平均速度 RMSE（m/s）/ 车距 RMSE（m）。

**CATS-CF**

| 模型 | 5 s | 7 s | 11 s |
| --- | --- | --- | --- |
| CTH | 0.428 / 0.771 | 0.445 / 0.913 | 0.493 / 1.019 |
| IDM | 0.518 / 0.920 | 0.500 / 0.964 | 0.373 / 0.835 |
| DNN | 0.488 / 0.772 | 0.548 / 0.962 | 0.501 / 0.819 |
| LSTM | 0.501 / 0.797 | 0.497 / 0.784 | 0.483 / 0.736 |
| Deep Koopman | 0.589 / 0.865 | 0.561 / 0.872 | 0.567 / 0.859 |
| PI-Koopman-CTH | 0.420 / 0.743 | 0.423 / 0.868 | 0.479 / 0.989 |
| PI-Koopman-IDM | 0.541 / 0.913 | 0.465 / 0.892 | 0.337 / 0.676 |

**OpenACC**

| 模型 | 5 s | 7 s | 11 s |
| --- | --- | --- | --- |
| CTH | 0.331 / 0.626 | 0.333 / 0.653 | 0.370 / 0.766 |
| IDM | 0.347 / 0.658 | 0.351 / 0.661 | 0.377 / 0.765 |
| DNN | 0.435 / 0.706 | 0.415 / 0.688 | 0.387 / 0.672 |
| LSTM | 0.418 / 0.680 | 0.400 / 0.659 | 0.399 / 0.658 |
| Deep Koopman | 0.840 / 1.226 | 0.646 / 1.041 | 0.635 / 0.985 |
| PI-Koopman-CTH | 0.334 / 0.621 | 0.327 / 0.634 | 0.362 / 0.741 |
| PI-Koopman-IDM | 0.373 / 0.679 | 0.402 / 0.772 | 0.378 / 0.733 |

### 2. 不同观测频率下的预测

使用 CATS-CF 的 7 秒历史，平均第 15、20、24 秒三个起点的预测结果。各起点和频率共用 7 秒、10 Hz 参考条件选出的配置。

```sh
python -m scripts.reproduce rate
```

每格为 ACC 跟驰样本的平均速度 RMSE（m/s）/ 车距 RMSE（m）。

**CATS-CF**

| 模型 | 1 Hz | 5 Hz | 10 Hz | 25 Hz | 125 Hz |
| --- | --- | --- | --- | --- | --- |
| CTH | 0.347 / 0.681 | 0.365 / 0.721 | 0.369 / 0.729 | 0.373 / 0.739 | 0.375 / 0.743 |
| IDM | 0.433 / 0.853 | 0.445 / 0.891 | 0.448 / 0.898 | 0.466 / 0.932 | 0.468 / 0.937 |
| DNN | 0.442 / 0.789 | 0.464 / 0.830 | 0.466 / 0.833 | 0.475 / 0.848 | 0.475 / 0.850 |
| LSTM | 0.463 / 0.779 | 0.423 / 0.717 | 0.416 / 0.705 | 0.431 / 0.725 | 0.444 / 0.759 |
| Deep Koopman | 0.518 / 0.847 | 0.535 / 0.853 | 0.535 / 0.854 | 0.541 / 0.869 | 0.559 / 0.890 |
| PI-Koopman-CTH | 0.342 / 0.674 | 0.352 / 0.694 | 0.356 / 0.701 | 0.357 / 0.700 | 0.359 / 0.706 |
| PI-Koopman-IDM | 0.421 / 0.828 | 0.416 / 0.825 | 0.416 / 0.825 | 0.433 / 0.861 | 0.431 / 0.855 |

### 3. ACC 与人工驾驶辨识

拟合每条 CATS-CF 轨迹的前 15 秒，以速度积分残差的方差打分，在其他轮次上选择分类阈值。

```sh
python -m scripts.reproduce recognition
```

| Hz | 模型 | ACC 判对 | 人工判对 | 平衡准确率 | AUC |
| --- | --- | --- | --- | --- | --- |
| 10 | CTH-OE | 20/24 | 18/18 | 0.917 | 0.938 |
| 10 | IDM-OE | 15/24 | 14/18 | 0.701 | 0.824 |
| 10 | PI-Koopman-CTH | 19/24 | 17/18 | 0.868 | 0.940 |
| 10 | PI-Koopman-IDM | 16/24 | 16/18 | 0.778 | 0.847 |
| 25 | CTH-OE | 20/24 | 18/18 | 0.917 | 0.938 |
| 25 | IDM-OE | 15/24 | 14/18 | 0.701 | 0.819 |
| 25 | PI-Koopman-CTH | 19/24 | 17/18 | 0.868 | 0.940 |
| 25 | PI-Koopman-IDM | 16/24 | 16/18 | 0.778 | 0.847 |

### 4. 观测噪声下的加速度响应

先完成实验 1。仿真采用论文中按后车对象和物理先验分别选定的组外参数。全部方法使用相同的 7 秒历史、10 Hz 观测、三个起点和三个噪声种子。车距噪声标准差为 0.2 m，两车速度噪声标准差各为 0.1 m。加噪后将两车速度分别限制为非负，再计算相对速度。

```sh
python -m scripts.reproduce noise
```

**七种模型的比较。** 260 条实车记录评价加噪拟合相对干净拟合的加速度响应均方根变化，3,120 条配套仿真评价相对真值的加速度 RMSE。每格依次给出独立噪声 / 时间相关噪声的结果（m/s²）。

| 模型 | CATS-CF 实车 | OpenACC 实车 | CATS-CF 仿真 | OpenACC 仿真 |
| --- | --- | --- | --- | --- |
| CTH | 0.0405 / 0.0640 | 0.0562 / 0.0707 | 0.0774 / 0.1079 | 0.0852 / 0.1029 |
| IDM | 0.0375 / 0.0551 | 0.0529 / 0.0657 | 0.0487 / 0.0747 | 0.0613 / 0.0765 |
| DNN | 0.1309 / 0.1223 | 0.2118 / 0.1860 | 0.2169 / 0.2070 | 0.2155 / 0.1886 |
| LSTM | 0.2039 / 0.2192 | 0.3527 / 0.3233 | 0.4011 / 0.3652 | 0.3823 / 0.3428 |
| Deep Koopman | 0.1671 / 0.1666 | 0.1580 / 0.1555 | 0.2992 / 0.2968 | 0.2849 / 0.2833 |
| PI-Koopman-CTH | 0.0384 / 0.0586 | 0.0516 / 0.0656 | 0.0719 / 0.0996 | 0.0782 / 0.0941 |
| PI-Koopman-IDM | 0.0348 / 0.0517 | 0.0494 / 0.0624 | 0.0464 / 0.0724 | 0.0569 / 0.0720 |

**三种估计方法。** FD 为有限差分，Integral 为积分形式，Exp. weak（ours）为本文采用的指数加权弱形式。下表给出配套仿真的真值 RMSE（m/s²），每格为独立噪声 / 时间相关噪声。图中以普通弱形式为零线，负值表示误差较低。

| 物理先验 | 估计方法 | CATS-CF | OpenACC |
| --- | --- | --- | --- |
| CTH | FD | 0.07241 / 0.09803 | 0.07881 / 0.09296 |
| CTH | Integral | 0.07542 / 0.10089 | 0.08093 / 0.09494 |
| CTH | **Exp. weak (ours)** | 0.07190 / 0.09958 | 0.07818 / 0.09415 |
| IDM | FD | 0.04856 / 0.06936 | 0.05895 / 0.06949 |
| IDM | Integral | 0.04957 / 0.07238 | 0.05994 / 0.07243 |
| IDM | **Exp. weak (ours)** | 0.04644 / 0.07235 | 0.05686 / 0.07204 |

### 计算效率

先完成实验 1。CPU 命令测量全部七种模型；可选的 CUDA 命令测量三种神经模型。在每套数据的六条轨迹上，分别使用三种历史长度，测量单次拟合和一次 5 秒预测。

```sh
python -m scripts.reproduce timing
# NVIDIA GPU
python -m scripts.reproduce timing --device cuda
```

表中为平均拟合时间（+ 预测时间），单位为毫秒。

| 模型 | M4 Pro | Threadripper | RTX 6000 Ada |
| --- | --- | --- | --- |
| CTH | 14.42 (+0.02) | 16.21 (+0.02) | — |
| DNN | 4101.87 (+4.98) | 5410.59 (+8.20) | 2231.99 (+8.82) |
| Deep Koopman | 259.48 (+9.12) | 350.97 (+12.49) | 246.91 (+12.89) |
| IDM | 21.80 (+0.03) | 22.70 (+0.03) | — |
| LSTM | 8635.98 (+5.66) | 10809.42 (+9.17) | 4163.01 (+10.00) |
| PI-Koopman-CTH | 15.47 (+0.07) | 17.20 (+0.08) | — |
| PI-Koopman-IDM | 23.06 (+0.08) | 23.71 (+0.10) | — |

## 许可

代码采用 [MIT](LICENSE)，CATS-CF 数据采用 [CC BY 4.0](LICENSE-DATA)。OpenACC © European Union，CC BY 4.0（[AstaZero](data/OpenACC/AstaZero_copyright.txt)、[Casale](data/OpenACC/Casale_copyright.txt)）。
