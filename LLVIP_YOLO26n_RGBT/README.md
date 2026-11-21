# LLVIP YOLO26n RGBT 实验

本目录保存 LLVIP 配对可见光/红外数据集上的 YOLO26n 对照实验代码与结果。原始数据集、处理后的图像和 4 GB 数据压缩包没有提交到 GitHub；请从 [LLVIP 作者项目](https://github.com/bupt-ai-cz/LLVIP) 获取数据，并按脚本生成本地数据目录。

## 实验配置

- Ultralytics 8.4.215
- PyTorch 2.12.1+cu130
- Python 3.12
- 输入尺寸 640，batch 32，30 epochs，seed 42
- optimizer AdamW，lr0 0.002，workers 8，单卡 CUDA
- 数据划分：train 10436，val 1589，test 3463

## 验证集结果

| 实验 | 输入 | 第四通道初始化 | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| S1 RGB | RGB | — | 0.7795 | 0.3895 | 0.8328 | 0.7005 |
| S2 Thermal | Thermal | — | 0.9282 | 0.5616 | 0.8972 | 0.8654 |
| S3 RGBT | RGB+Thermal | random | **0.9369** | **0.5758** | 0.9252 | 0.8791 |
| S3' RGBT | RGB+Thermal | RGB mean | 0.9351 | 0.5502 | **0.9573** | **0.8829** |

指标来源于各实验目录中的 `baseline_result.json`；模型权重位于对应目录的 `weights/best.pt`。

## 目录说明

- `reproduce/prepare_llvip.py`：建立配对数据、固定划分和 YOLO 数据配置。
- `reproduce/train_yolo26.py`：执行 RGB、Thermal、随机初始化 RGBT、RGB 均值初始化 RGBT 四组实验。
- `reproduce/reference_split.csv`：固定划分清单。
- `results/`：四组训练日志、曲线、验证结果、快照和权重。

## 复现实验

在 `reproduce/` 目录准备好 LLVIP 原始数据后，先执行数据准备：

```bash
python prepare_llvip.py --modes rgb thermal rgbt --workers 8
```

单组训练示例：

```bash
python train_yolo26.py --mode rgbt --t-init random --epochs 30 --batch 32 --imgsz 640 --workers 8 --device 0 --optimizer AdamW --lr0 0.002
```

最终测试集评估可以在训练命令中追加 `--final-test`。当前仓库保留的是四组已经完成的验证集实验结果。
