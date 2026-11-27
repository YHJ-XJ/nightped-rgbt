"""Train the LLVIP protocol matrix or audit the three historical runs."""

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

from base_train import build_trainer, verify_and_snapshot


ROOT = Path(__file__).resolve().parent
ARMS = {"s2": ("thermal", "auto"), "s3": ("rgbt", "random"), "s3p": ("rgbt", "rgbmean")}
SEEDS = (42, 43, 44)
MANIFEST_SHA = "088b02a8a23b574a5a5f5900a83a8782080f20b53d48bfef42f2362b318de572"
WEIGHTS_SHA = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
OPTIMIZER = dict(optimizer="AdamW", lr0=0.002, lrf=0.01, momentum=0.9,
                 weight_decay=0.0005, warmup_epochs=3.0, warmup_bias_lr=0.1,
                 cos_lr=False, nbs=64)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def image_index(folder):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    index = {p.stem: p for p in paths}
    if len(index) != len(paths):
        raise ValueError(f"Duplicate image IDs: {folder}")
    return index


def check_data(root, modes, smoke=False):
    from ultralytics.utils import YAML

    manifest = root / "split_manifest.csv"
    digest = sha256(manifest)
    if not smoke and digest != MANIFEST_SHA:
        raise ValueError(f"Unexpected split manifest: {manifest} ({digest})")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if smoke and (digest == MANIFEST_SHA or len(rows) > 512):
        raise ValueError("Use a small, separately prepared dataset for --smoke.")
    for mode in sorted(modes):
        mode_root = root / mode
        config = YAML.load(mode_root / "data.yaml")
        if (Path(config["path"]).resolve() != mode_root
                or config.get("channels", 3) != (4 if mode == "rgbt" else 3)
                or config["names"] not in ({0: "person"}, ["person"])):
            raise ValueError(f"Wrong path or channels in {mode_root / 'data.yaml'}; run prepare_data.py setup")
        for split in ("train", "val"):
            ids = {r["id"] for r in rows if r["split"] == split}
            labels = {p.stem for p in (mode_root / "labels" / split).glob("*.txt")}
            if not ids or config[split] != f"images/{split}" or set(image_index(mode_root / "images" / split)) != ids or labels != ids:
                raise ValueError(f"{mode}/{split} does not match the requested split")
    if {"thermal", "rgbt"} <= modes:
        for row in rows:
            if row["split"] not in {"train", "val"}:
                continue
            relative = Path("labels") / row["split"] / f"{row['id']}.txt"
            if (root / "thermal" / relative).read_bytes() != (root / "rgbt" / relative).read_bytes():
                raise ValueError(f"Labels differ between thermal and RGBT: {relative}")
    return digest


def checkpoint_info(path):
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def gpu_label(device):
    import torch

    if device == "cpu":
        return "CPU"
    if not re.fullmatch(r"\d+", device) or not torch.cuda.is_available():
        raise SystemExit("Use one CUDA device index, or --smoke --device cpu.")
    return torch.cuda.get_device_name(int(device))


def evaluate(path, yaml_path, out, device, imgsz, batch, workers):
    from ultralytics import YOLO, __version__

    # A separate instance ensures best.pt or last.pt is used exactly as requested.
    if __version__ != "8.4.215":
        raise SystemExit("Install the pinned evaluation environment from requirements.txt.")
    if out.exists():
        raise FileExistsError(out)
    detector = YOLO(str(path))
    exported = []
    detector.add_callback("on_val_end", lambda validator: exported.append(len(validator.jdict)))
    metrics = detector.val(
        data=str(yaml_path), split="val", imgsz=imgsz, batch=batch, device=device, workers=workers,
        project=str(out.parent), name=out.name, exist_ok=False, plots=True, nms=False,
        half=False, conf=0.001, max_det=300, save_json=True,
    )
    saved = Path(metrics.save_dir).resolve()
    if saved != out:
        raise RuntimeError(f"Predictions were written to unexpected directory {saved}")
    pred = saved / "predictions.json"
    if not pred.exists():
        # Ultralytics may omit the file when no detections survive export.
        if exported != [0]:
            raise FileNotFoundError(pred)
        write_json(pred, [])
    return dict(
        map50=float(metrics.box.map50), map50_95=float(metrics.box.map), ap75=float(metrics.box.map75),
        precision=float(metrics.box.mp), recall=float(metrics.box.mr),
        speed_ms_per_image={k: float(v) for k, v in metrics.speed.items()},
        predictions=str(pred), predictions_sha256=sha256(pred), checkpoint=checkpoint_info(path),
    )


def run_one(args, arm, seed, repeat, split_hash, hardware):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    mode, t_init = ARMS[arm]
    name = f"{mode}_{t_init}_s{seed}_e{args.epochs}" + ("_repeat" if repeat else "")
    run_dir = args.run_root / name
    summary_path = run_dir / "protocol_result.json"
    pretrained = ROOT / "yolo26n.pt"
    if sha256(pretrained) != WEIGHTS_SHA:
        raise ValueError("Starting checkpoint differs from the pinned yolo26n.pt")
    config = dict(arm=arm, mode=mode, t_init=t_init, seed=seed, repeat=repeat, epochs=args.epochs,
                  imgsz=args.imgsz, batch=args.batch, workers=args.workers, split_manifest_sha256=split_hash,
                  pretrained_sha256=WEIGHTS_SHA, device=args.device, hardware=hardware, scope="smoke" if args.smoke else "protocol",
                  ultralytics=ultralytics.__version__, torch=str(torch.__version__), opencv=cv2.__version__)
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old["config"] != config or old["training_recipe"] != OPTIMIZER or any(
            sha256(Path(old[key]["checkpoint"]["path"])) != old[key]["checkpoint"]["sha256"]
            or sha256(Path(old[key]["predictions"])) != old[key]["predictions_sha256"]
            for key in ("last", "best")
        ):
            raise ValueError(f"Existing run does not match current inputs or artifacts: {run_dir}")
        print(f"Skip verified complete run: {run_dir}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    yaml_path = args.data_root / mode / "data.yaml"
    channels = 4 if mode == "rgbt" else 3
    model = YOLO(str(pretrained))
    model.add_callback("on_train_start", lambda t: verify_and_snapshot(t, mode, t_init))
    started = time.perf_counter()
    model.train(
        trainer=build_trainer(t_init, channels, 3), data=str(yaml_path),
        project=str(args.run_root), name=name, exist_ok=True,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers, seed=seed,
        device=args.device, amp=args.device != "cpu", cache=False, deterministic=True,
        patience=0, close_mosaic=0, save=True, plots=True, nms=False, **OPTIMIZER,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, bgr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5, flipud=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    )
    wall = time.perf_counter() - started
    if Path(model.trainer.save_dir).resolve() != run_dir:
        raise RuntimeError("Training run directory differed from requested run")
    with (run_dir / "results.csv").open(encoding="utf-8", newline="") as stream:
        trace = list(csv.DictReader(stream))
    if [int(r["epoch"]) for r in trace] != list(range(1, args.epochs + 1)):
        raise RuntimeError(f"Training did not finish all {args.epochs} rounds")
    peak = max(trace, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    stop = model.trainer.stopper
    _, params, _, flops = model.trainer.model.info(imgsz=args.imgsz)
    summary = {
        "config": config, "training_recipe": OPTIMIZER, "primary_checkpoint": "last",
        "completed_epochs": len(trace), "best_fitness_epoch": stop.best_epoch,
        "curve_peak_map50_95_epoch": int(peak["epoch"]),
        "curve_peak_map50_95": float(peak["metrics/mAP50-95(B)"]),
        "curve_last_map50_95": float(trace[-1]["metrics/mAP50-95(B)"]),
        "train_seconds_csv": float(trace[-1]["time"]), "train_call_wall_seconds": wall,
        "complexity": {"params": params, "gflops": flops, "imgsz": args.imgsz, "graph": "unfused training model"},
    }
    del model
    if args.device != "cpu":
        torch.cuda.empty_cache()
    for key in ("last", "best"):
        summary[key] = evaluate(run_dir / "weights" / f"{key}.pt", yaml_path, run_dir / f"val_{key}",
                                args.device, args.imgsz, args.batch, args.workers)
    write_json(summary_path, summary)
    print(f"Completed {name}: last mAP50-95={summary['last']['map50_95']:.4f}", flush=True)


def train(args):
    import ultralytics

    if ultralytics.__version__ != "8.4.215":
        raise SystemExit("Install the pinned environment from requirements.txt.")
    if args.smoke:
        args.epochs, args.batch, args.imgsz, args.workers = 1, 4, 128, 0
    else:
        args.batch, args.imgsz = 32, 640
        if args.device == "cpu":
            raise SystemExit("For a small CPU check, specify --smoke and a sample data root.")
    hardware = gpu_label(args.device)
    plan = [(arm, seed, False) for seed in args.seeds for arm in args.arms]
    if args.repeat and "s3" in args.arms and 42 in args.seeds:
        plan.append(("s3", 42, True))
    digest = check_data(args.data_root, {ARMS[arm][0] for arm in args.arms}, args.smoke)
    for arm, seed, repeat in plan:
        run_one(args, arm, seed, repeat, digest, hardware)


def audit(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    sources = dict(zip(ARMS, args.runs, strict=True))
    if len(set(sources.values())) != len(sources):
        raise ValueError("Legacy arms point to the same run directory. Resolve historical paths before evaluation.")
    records = {}
    for arm, run in sources.items():
        files = {}
        for key in ("best", "last"):
            path = run / "weights" / f"{key}.pt"
            files[key] = checkpoint_info(path) if path.is_file() else None
        snapshot = run / "first_conv_snapshot.json"
        records[arm] = {"run": str(run), "files": files,
                        "initialization_snapshot": json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.is_file() else None}
    owners = {}
    for arm, record in records.items():
        for info in record["files"].values():
            if info:
                owners.setdefault(info["sha256"], set()).add(arm)
    ambiguous = {arm for group in owners.values() if len(group) > 1 for arm in group}
    for arm, item in records.items():
        snapshot = item["initialization_snapshot"]
        if snapshot and (snapshot.get("mode"), snapshot.get("t_init")) != ARMS[arm]:
            ambiguous.add(arm)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"runs": records, "ambiguous_arms": sorted(ambiguous), "evaluations": {}}
    write_json(args.out / "weight_audit.json", report)
    if args.evaluate:
        check_data(args.data_root, {"rgbt", "thermal"})
        gpu_label(args.device)
        for arm, run in sources.items():
            if arm not in ambiguous and records[arm]["files"]["last"]:
                mode, _ = ARMS[arm]
                report["evaluations"][arm] = evaluate(
                    run / "weights" / "last.pt", args.data_root / mode / "data.yaml", args.out / f"{arm}_val_last",
                    args.device, 640, 32, args.workers,
                )
                write_json(args.out / "weight_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run-root", type=Path, required=True)
    t.add_argument("--epochs", type=int, choices=(10, 30), default=10)
    t.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    t.add_argument("--seeds", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    t.add_argument("--no-repeat", dest="repeat", action="store_false")
    t.add_argument("--smoke", action="store_true")
    h = sub.add_parser("audit")
    h.add_argument("--runs", type=Path, nargs=3, required=True, metavar=("S2_DIR", "S3_DIR", "S3P_DIR"))
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--evaluate", action="store_true")
    for command in (t, h):
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--device", default="0")
        command.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    args.data_root = args.data_root.resolve()
    if args.command == "train":
        args.run_root = args.run_root.resolve()
        args.arms, args.seeds = list(dict.fromkeys(args.arms)), list(dict.fromkeys(args.seeds))
        train(args)
    else:
        args.runs = [r.resolve() for r in args.runs]
        args.out = args.out.resolve()
        audit(args)


if __name__ == "__main__":
    main()
