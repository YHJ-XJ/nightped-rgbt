"""Train the three LLVIP YOLO26n baselines and evaluate each on the test split."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHANNELS = {"rgb": 3, "thermal": 3, "rgbt": 4}


def train(mode, args, split_hash):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    yaml_path = args.data_root / mode / "data.yaml"
    model = YOLO(str(ROOT / "yolo26n.pt"))

    def verify_model(trainer):
        first_conv = next(layer for layer in trainer.model.modules() if isinstance(layer, torch.nn.Conv2d))
        if first_conv.in_channels != CHANNELS[mode] or getattr(trainer.model.model[-1], "one2one_cv2", None) is None:
            raise RuntimeError(f"{mode}: expected {CHANNELS[mode]} input channels and the YOLO26 end-to-end head")
        print(f"Verified {mode}: {first_conv.in_channels} channels, end-to-end head", flush=True)

    model.add_callback("on_train_start", verify_model)
    model.train(
        data=str(yaml_path), project=str(args.run_root), name=mode, exist_ok=False,
        imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
        seed=args.seed, deterministic=True, workers=args.workers, cache=False,
        amp=args.device != "cpu", plots=True, nms=False,
        mosaic=0.0, mixup=0.0, cutmix=0.0, bgr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5, flipud=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    metrics = YOLO(str(best)).val(
        data=str(yaml_path), split="test", imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=str(run_dir), name="test",
        plots=True, exist_ok=False, nms=False,
    )
    summary = {
        "model": "yolo26n.pt",
        "mode": mode,
        "input_channels": CHANNELS[mode],
        "ultralytics": ultralytics.__version__,
        "torch": str(torch.__version__),
        "opencv": cv2.__version__,
        "split_manifest_sha256": split_hash,
        "epochs_completed": model.trainer.epoch + 1,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "device": args.device,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "test_speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "weights": str(best),
    }
    (run_dir / "baseline_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *CHANNELS), default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "LLVIP_prepared")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "LLVIP_YOLO26n")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0", help="Single GPU index, or cpu for a smoke test")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.run_root = args.run_root.resolve()

    import torch

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Activate the cloud environment with CUDA-enabled PyTorch.")
    manifest = args.data_root / "split_manifest.csv"
    split_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected = hashlib.sha256((ROOT / "reference_split.csv").read_bytes()).hexdigest()
    if split_hash != expected:
        if args.data_root == ROOT / "data" / "LLVIP_prepared":
            raise SystemExit("Prepared split differs from the packaged reference. Check data preparation.")
        print("Using a custom split. Metrics apply only to this split.", flush=True)
    modes = tuple(CHANNELS) if args.mode == "all" else (args.mode,)
    for mode in modes:
        if not (args.data_root / mode / "data.yaml").is_file():
            raise FileNotFoundError(args.data_root / mode / "data.yaml")
    for mode in modes:
        train(mode, args, split_hash)


if __name__ == "__main__":
    main()
