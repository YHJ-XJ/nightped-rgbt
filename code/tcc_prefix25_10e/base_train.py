"""Train LLVIP YOLO26n baselines and evaluate them on the val split.

T01 changes (2026-11-21):
  - --optimizer is required: the framework "auto" policy switches between
    MuSGD (lr 0.01) and AdamW (fitted lr) based on the total iteration count,
    which would silently change the recipe between the 100-epoch baseline and
    the 30-epoch screening runs.
  - Only the val split is evaluated by default; pass --final-test to also
    evaluate the author test split.
  - --t-init {auto,random,rgbmean} controls the 4th (thermal) first-conv filter.
    `random` re-initialises that filter explicitly instead of trusting the
    framework default, so a 4-channel checkpoint cannot smuggle a trained
    thermal filter into the "random" arm.
  - A first-conv snapshot plus assertions is written before training starts:
    the weight hash captured right after initialisation must still match at
    train start, which proves nothing overwrote it.
  - patience defaults to 100 on purpose (> the 30 screening epochs) so runs are
    never stopped early; comparisons use the final epoch, per the project log.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHANNELS = {"rgb": 3, "thermal": 3, "rgbt": 4}
OPTIMIZERS = ("AdamW", "SGD", "MuSGD", "Adam", "Adamax", "NAdam", "RAdam", "RMSprop")
T_INITS = ("auto", "random", "rgbmean")


def tensor_sha(tensor):
    """Short content hash, used to prove a weight tensor was not rewritten."""
    import torch

    arr = tensor.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def init_thermal_filter(model, t_init, num_channels):
    """Initialise the 4th first-conv filter *after* weights are loaded.

    Returns the applied mode plus before/after hashes so the verifier can prove
    the initialisation survived until training starts.
    """
    import torch

    w = model.model[0].conv.weight
    if num_channels != 4 or t_init == "auto":
        return {"applied": t_init, "sha_after": tensor_sha(w)}

    with torch.no_grad():
        before = tensor_sha(w)
        if t_init == "rgbmean":
            w[:, 3] = w[:, :3].mean(1)
        elif t_init == "random":
            # Explicit re-init: "keep whatever was loaded" is not the same thing
            # as "random", and a 4-channel checkpoint would keep trained filters.
            torch.nn.init.kaiming_uniform_(w[:, 3:4], a=math.sqrt(5))
        after = tensor_sha(w)
    return {"applied": t_init, "sha_before": before, "sha_after": after}


def build_trainer(t_init, num_channels, pretrained_channels, thermal_edit=None):
    """DetectionTrainer subclass that can initialise the thermal filter."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class LLVIPTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train" and thermal_edit is not None:
                dataset.transforms.insert(0, thermal_edit)
            return dataset

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            # Weight loading happens inside super(); any assignment must happen
            # after it returns, otherwise it is overwritten.
            self.t01_init = init_thermal_filter(model, t_init, num_channels)
            self.t01_t_init = t_init
            self.t01_pretrained_channels = pretrained_channels
            return model

    return LLVIPTrainer


def verify_and_snapshot(trainer, mode, t_init):
    """Assert the real first-conv state, then persist a snapshot."""
    import torch

    # Multi-GPU: every rank fires on_train_start, but only rank 0 may write the
    # snapshot or raise; other ranks would race on the same file.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None

    expected = CHANNELS[mode]
    first = next(layer for layer in trainer.model.modules() if isinstance(layer, torch.nn.Conv2d))
    head = getattr(trainer.model.model[-1], "one2one_cv2", None)
    if first.in_channels != expected or head is None:
        raise RuntimeError(f"{mode}: expected {expected} input channels and the YOLO26 end-to-end head")

    weight = first.weight.detach()
    state = getattr(trainer, "t01_init", None) or {}
    snap = {
        "mode": mode,
        "t_init": t_init,
        "in_channels": int(first.in_channels),
        "pretrained_channels": getattr(trainer, "t01_pretrained_channels", None),
        "init_state": state,
    }

    if "sha_after" in state:
        snap["weights_unchanged_since_init"] = tensor_sha(weight) == state["sha_after"]
        if not snap["weights_unchanged_since_init"]:
            raise RuntimeError("first-conv weights changed after t_init; the initialisation was overwritten")

    if weight.shape[1] == 4:
        thermal = weight[:, 3]
        rgb_mean = weight[:, :3].mean(1)
        same = bool(torch.allclose(thermal, rgb_mean))
        snap.update(
            fourth_equals_rgb_mean=same,
            fourth_vs_mean_max_abs=float((thermal - rgb_mean).abs().max()),
            fourth_l2=float(thermal.norm()),
            rgb_mean_l2=float(rgb_mean.norm()),
        )
        if t_init == "rgbmean":
            if not same:
                raise RuntimeError("t_init=rgbmean but the 4th filter is not the RGB mean")
            if state.get("sha_before") == state.get("sha_after"):
                raise RuntimeError("t_init=rgbmean did not change any weight")
        if t_init == "random":
            if same:
                raise RuntimeError("t_init=random but the 4th filter equals the RGB mean; branches swapped?")
            if state.get("sha_before") == state.get("sha_after"):
                raise RuntimeError("t_init=random did not re-initialise the 4th filter")

    path = Path(trainer.save_dir) / "first_conv_snapshot.json"
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    print("SNAP " + json.dumps(snap), flush=True)
    return snap


def train(mode, args, split_hash):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    yaml_path = args.data_root / mode / "data.yaml"
    channels = CHANNELS[mode]

    t_init = args.t_init
    if channels != 4 and t_init != "auto":
        print(f"[warn] {mode} has {channels} input channels; --t-init is ignored", flush=True)
        t_init = "auto"
    if channels == 4 and t_init == "auto":
        print("[warn] rgbt with --t-init auto keeps the framework default for the 4th filter; "
              "pass --t-init random|rgbmean to make it explicit", flush=True)

    checkpoint = YOLO(str(ROOT / "yolo26n.pt"))
    pretrained_channels = int(checkpoint.model.model[0].conv.in_channels)
    del checkpoint

    model = YOLO(str(ROOT / "yolo26n.pt"))
    model.add_callback("on_train_start", lambda t: verify_and_snapshot(t, mode, t_init))
    model.train(
        trainer=build_trainer(t_init, channels, pretrained_channels),
        data=str(yaml_path), project=str(args.run_root), name=mode, exist_ok=False,
        imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
        seed=args.seed, deterministic=True, workers=args.workers, cache=False,
        amp=args.device != "cpu", plots=True, nms=False,
        optimizer=args.optimizer, lr0=args.lr0, momentum=args.momentum,
        weight_decay=args.weight_decay, lrf=args.lrf, warmup_epochs=args.warmup_epochs,
        patience=args.patience, cos_lr=False, close_mosaic=0,
        mosaic=0.0, mixup=0.0, cutmix=0.0, bgr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5, flipud=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)

    summary = {
        "model": "yolo26n.pt",
        "mode": mode,
        "t_init": t_init,
        "input_channels": channels,
        "pretrained_channels": pretrained_channels,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "epochs_requested": args.epochs,
        "epochs_completed": model.trainer.epoch + 1,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "ultralytics": ultralytics.__version__,
        "torch": str(torch.__version__),
        "opencv": cv2.__version__,
        "split_manifest_sha256": split_hash,
        "weights": str(best),
    }

    exported_json = True
    try:
        val_metrics = YOLO(str(best)).val(
            data=str(yaml_path), split="val", imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, project=str(run_dir), name="val",
            plots=True, exist_ok=False, nms=False, save_json=True,
        )
    except Exception as exc:  # pycocotools may be missing on the cloud image
        exported_json = False
        print(f"[warn] val with save_json failed ({exc}); retrying without json", flush=True)
        val_metrics = YOLO(str(best)).val(
            data=str(yaml_path), split="val", imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, project=str(run_dir), name="val",
            plots=True, exist_ok=False, nms=False,
        )

    # Read the directory the validator actually used: a failed json attempt can
    # leave val/ taken and force the retry into val-2/. Prefer the validator
    # instance over the returned metrics object.
    validator = getattr(model, "validator", None)
    val_dir = Path(getattr(validator, "save_dir", getattr(val_metrics, "save_dir", run_dir / "val")))
    summary.update(
        val_map50=float(val_metrics.box.map50),
        val_map50_95=float(val_metrics.box.map),
        val_precision=float(val_metrics.box.mp),
        val_recall=float(val_metrics.box.mr),
        val_out_dir=str(val_dir),
        val_predictions_json=str(val_dir / "predictions.json") if exported_json else None,
    )
    print(f"VAL mAP50={summary['val_map50']:.4f}; mAP50-95={summary['val_map50_95']:.4f}", flush=True)

    if args.final_test:
        test_metrics = YOLO(str(best)).val(
            data=str(yaml_path), split="test", imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, project=str(run_dir), name="test",
            plots=True, exist_ok=False, nms=False,
        )
        summary.update(
            test_map50=float(test_metrics.box.map50),
            test_map50_95=float(test_metrics.box.map),
            test_precision=float(test_metrics.box.mp),
            test_recall=float(test_metrics.box.mr),
        )
        print(f"TEST mAP50={summary['test_map50']:.4f}; mAP50-95={summary['test_map50_95']:.4f}", flush=True)

    (run_dir / "baseline_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *CHANNELS), default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "LLVIP_prepared")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs" / "LLVIP_YOLO26n")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0", help="Single GPU index, or cpu for a smoke test")
    parser.add_argument("--optimizer", required=True, choices=OPTIMIZERS,
                        help="Required on purpose: the framework auto policy changes with the epoch budget")
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=100,
                        help="Kept above the screening epoch budget so runs are never stopped early")
    parser.add_argument("--t-init", dest="t_init", choices=T_INITS, default="auto",
                        help="4th first-conv filter: auto(untouched), random(explicit re-init), rgbmean(RGB filter mean)")
    parser.add_argument("--final-test", dest="final_test", action="store_true",
                        help="Also evaluate the author test split (final runs only)")
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
