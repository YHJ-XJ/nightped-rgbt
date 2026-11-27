# 10 轮三种子协议复现

在本目录执行命令。安装 CUDA 版 PyTorch 后，安装项目依赖并获取指定版本的预训练权重。

```bash
python -m pip install -r requirements.txt
curl -fL -o yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
printf '%s  %s\n' \
  9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef \
  yolo26n.pt | sha256sum -c
```

从 [数据说明](../../datasets/README.md) 获取 LLVIP ZIP，修改变量为实际存放位置。数据准备会复用现有四通道 TIFF 和热红外图。两套模态都已准备时可以省略 `--archive`。

```bash
DATA="$PWD/data/LLVIP_prepared"
ARCHIVE="$PWD/LLVIP_HF_McCheng.zip"

python prepare_data.py setup --data-root "$DATA" --archive "$ARCHIVE" --workers 8
python run_protocol.py train --data-root "$DATA" --run-root runs/protocol_10e \
  --epochs 10 --device 0 --workers 8
python analyze_protocol.py --data-root "$DATA" --run-root runs/protocol_10e \
  --out runs/protocol_10e_analysis
```

默认跑三组、三个种子和一次 S3 同种子重复。预检单组可在训练命令后加 `--arms s2 --seeds 42 --no-repeat`。随后执行默认命令会验证并跳过已完成的同配置运行。若训练中断导致同名目录只有部分产物，可先把该次目录移到其他位置再重跑。要复评 GitHub 已归档的预测，将分析命令的 `--run-root` 设为 `../../results/protocol_10e`，选择一个新的 `--out`。

后续研究可用下列命令生成冷场景探针清单。它从正式训练划分中留出前缀 25 的 423 对图像（845 框），剩余 10,013 对用于训练。

```bash
python prepare_data.py probe --data-root "$DATA" --exclude-prefix 25 \
  --out data_protocol/prefix25
```

[本批次结果与解读](../../results/protocol_10e/README.md)。
