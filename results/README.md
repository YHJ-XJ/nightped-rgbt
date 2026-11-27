# 实验结果

| 实验 | 结果文件 | 对应代码 |
| --- | --- | --- |
| TCC 首轮留出实验 | [P0/P1/P2、前缀 25 评估和筛选结果](tcc_prefix25_10e/) | [TCC 实验代码](../code/tcc_prefix25_10e/) |
| 10 轮三种子协议 | [三组实验、重复运行、逐轮曲线与分层分析](protocol_10e/) | [协议代码](../code/protocol_10e/) |
| 100 轮 RGBT 基线 | [日志、验证图表和作者测试集结果](baseline_100e/) | [基线代码](../code/baseline_100e/) |
| 30 轮模态对照 | [S1、S2、S3 和 S3′ 的日志、指标和验证预测](modality_comparison_30e/) | [对照代码](../code/modality_comparison_30e/) |

实验配置和主要观察见 [多模态实验进度](多模态实验进度.md)。逐轮指标在各次训练的 `results.csv`。30 轮对照的 `baseline_result.json` 中以 `val_` 开头的指标来自验证集，100 轮基线同名的 JSON 记录作者测试集结果。
