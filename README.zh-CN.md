# CATS-CF & PI-Koopman

[English](README.md) | 简体中文

论文 **Physics-Informed Koopman Learning for Data-Efficient System Identification of Automated Vehicle Dynamics** 的自采跟驰数据与核心实现。

包含模型：CTH、IDM、PI-Koopman-CTH、PI-Koopman-IDM、DNN、LSTM 和 projected Deep Koopman。

## 数据

- **CATS-CF**：42 个双车跟驰试验，覆盖 14 种条件，每种重复 3 次，包含 ACC 和人工驾驶，提供 15 秒及 30 秒窗口。公开数据保留局部坐标与相对时间，移除了绝对时间和地理位置。数据位于 `data/release/`，详见[数据结构与字段](DATA.md)。
- **OpenACC**：[JRC 官方下载](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/)。本文使用 AstaZero 和 Casale 的 284 个 30 秒片段，详见[来源与许可说明](data/external/openacc/README.md)。

命令参数中，`local` 表示 CATS-CF，`openacc` 表示 OpenACC。

## 安装

已测试环境：Python **3.12.14**、**macOS arm64 / CPU**。需要 C99 编译器（通过 `cc` 调用，或设置 `CC`）。编译脚本也支持 Linux；Linux 和 CUDA 运行尚未测试。

| 依赖 | 已测试版本 |
|---|---|
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| pandas | 2.2.3 |
| scikit-learn | 1.9.1 |
| PyTorch | 2.14.0 |
| threadpoolctl | 3.7.0 |

在仓库根目录执行：

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python scripts/build_native.py
```

[requirements-tested.txt](requirements-tested.txt) 固定上述版本；[requirements.txt](requirements.txt) 列出依赖版本范围。

## 快速开始

将 CATS-CF 数据放在 `data/release/` 后执行：

```sh
python scripts/verify_data.py
python scripts/smoke_test.py
python scripts/run_prediction.py --dataset local --budget 3 --rate 10 --origin 20
```

该示例使用 10 Hz、3 秒历史拟合 CTH、IDM 和两种 PI 模型，从第 20 秒起预测随后 5 秒。结果位于 `outputs/physics/grid/local_B3_Hz10_end20/`。

准备 OpenACC 数据：

```sh
python experiments/download_openacc.py
python experiments/revision_data.py
```

## 复现论文实验

**历史时长比较**：使用 10 Hz、3/8/15 秒历史，预测随后 5 秒。运行全部七种模型并汇总：

```sh
python scripts/run_main.py --axis duration --dataset local --models all
python scripts/summarize_prediction.py --axis duration --dataset local
```

将 `local` 替换为 `openacc` 即可运行 OpenACC。可用 `--models physics` 或 `--models neural` 分别运行两组模型。

**采样率比较**：CATS-CF，8 秒历史，1/5/10/25/125 Hz，预测起点为第 15/20/24 秒：

```sh
python scripts/run_main.py --axis rate --dataset local --models all
python scripts/summarize_prediction.py --axis rate --dataset local
```

汇总结果位于 `outputs/summaries/`。完整神经搜索可能耗时数小时；紧凑训练示例：

```sh
python experiments/unified_axes_neural.py --dataset local --budget 3 --rate 10 --origin 20 --stage compact
```

| 实验 | 命令 |
|---|---|
| 已知律响应场重建 | `python experiments/synthetic_benchmark.py` |
| 配对采样率／噪声一致性 | `python experiments/claim_rate_consistency.py` |
| PI 残差的 ACC／人工跟驰识别 | `python experiments/run_pi_residual_recognition.py` |
| 对照合成与识别实验的参考结果 | `python scripts/verify_reference.py` |
| 物理／PI 模型计时 | `python experiments/timing_single_encounter.py --label my-cpu --output outputs/timing_cpu` |
| 神经模型计时 | `python experiments/timing_neural_single.py --device cpu --label my-cpu --output outputs/timing_neural` |

详见[实验与代码对应关系](PAPER_MAP.md)、[实验配置](configs/main_text.json)。完整 expanded 神经搜索网格尚未完成验证。

## 许可

代码采用 [MIT](LICENSE)，CATS-CF 数据采用 [CC BY 4.0](LICENSE-DATA)。OpenACC 保留原始许可，详见[第三方说明](THIRD_PARTY.md)。
