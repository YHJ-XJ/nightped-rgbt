# NightPed RGBT

在 LLVIP 数据集上研究低照度行人检测，以 YOLO26n 四通道早期融合作为 RGB＋热红外基线。

| 目录 | 内容 |
| --- | --- |
| [代码](code/) | 100 轮基线、30 轮对照、10 轮三种子实验的数据准备、训练脚本、依赖和固定划分。 |
| [实验结果](results/) | 实验进度、指标、训练曲线、逐轮日志和验证预测。按实验批次归档。 |
| [数据集说明](datasets/) | LLVIP 作者下载入口、镜像地址、校验值及数据准备命令。 |

## 实验入口

- [10 轮三种子协议实验](results/protocol_10e/README.md) 比较短训练日程、热通道初始化和低对比度表现，以第 10 轮权重为主。
- [30 轮模态对照](results/modality_comparison_30e/README.md) 记录 RGB-only、Thermal-only 和两种 RGBT 首层初始化。
- [100 轮 RGBT 基线](results/baseline_100e/README.md) 提供历史基线、验证指标及作者测试集结果。
- [实验记录](results/多模态实验进度.md) 记录各次训练的配置和主要观察。

在仓库根目录进入对应实验的 `code/` 子目录。数据下载及校验见 [LLVIP 数据说明](datasets/README.md)，训练命令在各自的实验 README 中。
