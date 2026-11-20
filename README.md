# LLVIP RGB＋热红外 YOLO26n 基线

本记录对应 2026-11-20 的 100 轮训练。采用四通道早期融合，检测类别为 `person`。

| 评估集 | 图像对数 | mAP50 | mAP50–95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 项目验证集 | 1,589 | **93.0%** | **54.9%** | 92.8% | 89.1% |
| LLVIP 作者测试集 | 3,463 | 85.58% | 49.19% | 80.09% | 81.67% |

上表均使用验证集选出的 `best.pt`。训练集 10,436 对。输入 640 × 640，batch 32，seed 42，AMP，最终实际使用 MuSGD 和初始学习率 0.01。RTX 5090 单卡训练耗时 1.446 小时。按 Ultralytics 的融合模型摘要，参数量 **2,389,879**、计算量 **5.4 GFLOPs**，最佳权重约 **5.5 MB**。逐轮记录中验证集 mAP50–95 峰值在第 16 轮。

## 结果文件

- [实验进度及 5 轮试跑对照](多模态实验进度.md)
- [100 轮训练及逐轮验证日志](results/results.csv) 和 [训练曲线](results/results.png)
- [测试集精确指标](results/baseline_result.json)、[PR 曲线](results/test/BoxPR_curve.png) 和 [混淆矩阵](results/test/confusion_matrix.png)
- [运行配置](results/args.yaml) 和 [复现代码](reproduce/)

## 数据下载

- [LLVIP 作者项目](https://github.com/bupt-ai-cz/LLVIP) 和 [作者下载入口](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md)
- [本次训练所用镜像的直接下载链接](https://huggingface.co/datasets/McCheng/LLVIP/resolve/main/LLVIP.zip)
- [LLVIP 使用条款](https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md)

镜像文件为 4,003,858,172 字节。下载后保存为 `LLVIP_HF_McCheng.zip`，放入 `reproduce/`。本次使用的镜像 SHA256 为 `b3b55475c093ac54663c467859eb31a82b840b53a7196ad1887cb547e4518b45`。数据准备程序直接读取 ZIP。

## 复现

在已配好 CUDA PyTorch 的 Linux 主机上进入 `reproduce/`，准备约 40 GB 以上可用空间来转换四通道数据。

```bash
python -m pip install -r requirements.txt
curl -fL -o yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
python prepare_llvip.py --modes rgbt --workers 8
cmp reference_split.csv data/LLVIP_prepared/split_manifest.csv
python train_yolo26.py --mode rgbt --device 0 --epochs 100 --imgsz 640 --batch 32 --workers 8
```

本实验使用 Ultralytics 8.4.215、PyTorch 2.12.1＋CUDA 13.0、OpenCV 5.1.0。运行配置保留自动优化器设置，100 轮这次实际选择的 MuSGD 与初始学习率已从权重元数据核实。
