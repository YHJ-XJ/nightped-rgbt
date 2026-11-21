# LLVIP 数据下载与准备

LLVIP 是已配对的可见光与热红外低照度图像数据集。本项目检测一类行人。固定划分为训练 10,436 对、验证 1,589 对、作者测试 3,463 对。验证集从作者训练集中按文件名前缀分组划出。

- [LLVIP 作者项目](https://github.com/bupt-ai-cz/LLVIP)
- [作者数据下载入口](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md)
- [本项目所用的第三方镜像直链](https://huggingface.co/datasets/McCheng/LLVIP/resolve/main/LLVIP.zip)
- [LLVIP 使用条款](https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md)

镜像 ZIP 为 4,003,858,172 字节。下载后命名为 `LLVIP_HF_McCheng.zip`，放到所选实验的代码目录，和 `prepare_llvip.py` 同级。数据准备程序直接读取压缩包。Linux 下可在该代码目录执行：

```bash
echo 'b3b55475c093ac54663c467859eb31a82b840b53a7196ad1887cb547e4518b45  LLVIP_HF_McCheng.zip' | sha256sum -c
python prepare_llvip.py --modes rgb thermal rgbt --workers 8
cmp reference_split.csv data/LLVIP_prepared/split_manifest.csv
```

只跑四通道基线时，可在数据准备命令中指定 `--modes rgbt`。两套代码的 `reference_split.csv` 使用同一划分及校验值 `088b02a8a23b574a5a5f5900a83a8782080f20b53d48bfef42f2362b318de572`。
