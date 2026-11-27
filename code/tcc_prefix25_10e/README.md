# TCC 首轮留出实验复现

使用已安装 CUDA 版 PyTorch 的 Linux 环境，在仓库根目录进入 `code/tcc_prefix25_10e`。下面以先前 AutoDL 基线的数据路径为例。

```bash
cd code/tcc_prefix25_10e
export OLD_PROJECT=/root/autodl-tmp/LLVIP_YOLO26_cloud/LLVIP_YOLO26_cloud
bash start_tcc.sh prepare
bash start_tcc.sh train
```

如果数据在其他位置，可设置 `DATA_ROOT` 指向 `LLVIP_prepared`，通过 `LLVIP_ARCHIVE` 指向原始 LLVIP ZIP。[数据下载与校验说明](../../datasets/README.md)。

`prepare` 安装依赖、校验或下载固定的 YOLO26n 预训练权重、生成前缀 25 留出划分，并用 200 张训练图检查增强。增强质检输出位于 `runs/tcc_prefix25_10e/quality/`。若 CUDA 设备编号非 0，可设置 `TCC_DEVICE`。

`train` 顺序运行 P0 无增强、P1 整图热退化、P2 带符号局部热编辑。三组均为四通道 S3′ 均值初始化、种子 42、10 轮、640 × 640、batch 32、AdamW。每组使用第 10 轮 EMA 权重，在同一项目验证集和前缀 25 留出探针上评估并导出预测。若只需重做汇总，执行 `bash start_tcc.sh report`。

P1 和 P2 使用相同的候选图像与行人。匹配的是被选行人相对邻域的热对比度幅值降幅。P1 保留明暗方向并压缩整张热图的对比度，P2 在单个行人框内做羽化编辑并允许翻负。[实现](tcc.py) · [实测结果](../../results/tcc_prefix25_10e/README.md)。
