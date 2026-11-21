# LLVIP RGBT 100 轮基线

2026-11-20 训练完成。YOLO26n 使用 640 × 640 四通道早期融合、batch 32、seed 42、AMP、MuSGD（初始学习率 0.01）。RTX 5090 单卡训练耗时 1.446 小时。

| 评估集 | 图像对数 | mAP50 | mAP50–95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 项目验证集 | 1,589 | 93.0% | 54.9% | 92.8% | 89.1% |
| LLVIP 作者测试集 | 3,463 | 85.58% | 49.19% | 80.09% | 81.67% |

两行均使用验证集选出的 `best.pt`，最佳轮为第 16 轮。融合模型在 640 输入下为 2,389,879 个参数、5.4 GFLOPs，最佳权重约 5.5 MB。

- [训练及逐轮验证日志](results.csv) 和 [曲线](results.png)
- [作者测试集的精确指标](baseline_result.json)、[PR 曲线](test/BoxPR_curve.png) 和 [混淆矩阵](test/confusion_matrix.png)
- [运行配置](args.yaml)、[对应代码](../../code/baseline_100e/) 和 [数据集说明](../../datasets/README.md)

复现时进入仓库的 `code/baseline_100e/`。先按数据说明准备原始 ZIP，再运行：

```bash
python -m pip install -r requirements.txt
curl -fL -o yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
python prepare_llvip.py --modes rgbt --workers 8
cmp reference_split.csv data/LLVIP_prepared/split_manifest.csv
python train_yolo26.py --mode rgbt --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8
```

本实验使用 Ultralytics 8.4.215、PyTorch 2.12.1＋CUDA 13.0、OpenCV 5.1.0。历史脚本使用自动优化器选择，以上 MuSGD 配置已从权重元数据核实。
