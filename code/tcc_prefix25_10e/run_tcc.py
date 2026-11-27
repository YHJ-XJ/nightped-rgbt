"""Run the LLVIP prefix-25 TCC comparison with three paired arms."""

import argparse
import csv
import json
from pathlib import Path
import time

from analyze_protocol import evaluate_predictions, prepare_truth, write_csv
from base_train import build_trainer, verify_and_snapshot
from qc_tcc import check
from run_protocol import (MANIFEST_SHA, OPTIMIZER, ROOT, WEIGHTS_SHA, check_data,
                          checkpoint_info, evaluate, gpu_label, image_index, sha256, write_json)
from tcc import PROBABILITY, ThermalCounterfactual


ARMS = ("p0", "p1", "p2")
EDIT_VERSION = 1


def split_inputs(data, split, smoke):
    from ultralytics.utils import YAML

    digest = check_data(data, {"rgbt", "thermal"}, smoke)
    with (data / "split_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "train"]
    record = json.loads((split / "holdout_manifest.json").read_text(encoding="utf-8"))
    held = set(record["probe_ids"])
    all_train = {r["id"] for r in rows}
    if smoke:
        assert held and held < all_train
    elif (record["base_manifest_sha256"] != MANIFEST_SHA or record["excluded_train_prefix"] != "25"
          or held != {r["id"] for r in rows if r["prefix_group"] == "25"}
          or (len(all_train - held), len(held), sum(int(r["boxes"]) for r in rows if r["id"] in held))
          != (10013, 423, 845)):
        raise ValueError("Prefix 25 manifest does not match the fixed LLVIP protocol.")
    mode = split / "rgbt"
    sources = image_index(data / "rgbt" / "images" / "train")
    for name, selected in (("train", all_train - held), ("probe", held)):
        listed = [Path(p).resolve() for p in (mode / f"{name}.txt").read_text(encoding="utf-8").splitlines()]
        if len(listed) != len(selected) or set(listed) != {sources[k].resolve() for k in selected}:
            raise ValueError(f"Unexpected images in {mode / (name + '.txt')}.")
    for name, val in (("train", "images/val"), ("probe", (mode / "probe.txt").as_posix())):
        config = YAML.load(mode / f"{name}.yaml")
        if config != {"path": (data / "rgbt").as_posix(), "train": (mode / "train.txt").as_posix(),
                      "val": val, "channels": 4, "names": {0: "person"}}:
            raise ValueError(f"Incorrect split YAML: {mode / (name + '.yaml')}.")
    return digest, held, [sources[k] for k in sorted(all_train - held)]


def train_one(a, arm, digest, hardware, quality_sha):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.215":
        raise SystemExit("Install the pinned environment from requirements.txt.")
    weights = ROOT / "yolo26n.pt"
    if sha256(weights) != WEIGHTS_SHA:
        raise ValueError("Starting checkpoint differs from the pinned yolo26n.pt.")
    name = f"rgbt_rgbmean_{arm}_s{a.seed}_e{a.epochs}"
    run = a.run_root / name
    result = run / "tcc_result.json"
    config = dict(arm=arm, seed=a.seed, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                  hardware=hardware, device=a.device, workers=a.workers, thermal_init="rgbmean",
                  split_manifest_sha256=digest, holdout_sha256=sha256(a.split_dir / "holdout_manifest.json"),
                  starting_weights_sha256=WEIGHTS_SHA, quality_sha256=quality_sha, edit_version=EDIT_VERSION,
                  edit_probability=PROBABILITY, ultralytics=ultralytics.__version__,
                  torch=str(torch.__version__), opencv=cv2.__version__)
    if result.exists():
        old = json.loads(result.read_text(encoding="utf-8"))
        if old["config"] != config or any(sha256(Path(item["path"])) != item["sha256"]
                                          for item in old["weights"].values()) or any(
            sha256(run / f"{role}_last" / "predictions.json") != old[role]["predictions_sha256"]
            for role in ("val", "probe")
        ):
            raise ValueError(f"Existing run does not match current inputs or artifacts: {run}")
        print(f"Skip verified complete run: {run}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=False)
    yaml_path = a.split_dir / "rgbt" / "train.yaml"
    edit = None if arm == "p0" else ThermalCounterfactual(arm, a.seed)
    model = YOLO(str(weights))
    model.add_callback("on_train_start", lambda trainer: verify_and_snapshot(trainer, "rgbt", "rgbmean"))
    start = time.perf_counter()
    model.train(
        trainer=build_trainer("rgbmean", 4, 3, edit), data=str(yaml_path),
        project=str(a.run_root), name=name, exist_ok=True,
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, seed=a.seed,
        device=a.device, amp=a.device != "cpu", cache=False, deterministic=True,
        patience=0, close_mosaic=0, save=True, plots=True, nms=False, **OPTIMIZER,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, bgr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5, flipud=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    )
    wall = time.perf_counter() - start
    if Path(model.trainer.save_dir).resolve() != run:
        raise RuntimeError("Training run directory differed from requested run.")
    with (run / "results.csv").open(encoding="utf-8", newline="") as stream:
        trace = list(csv.DictReader(stream))
    if [int(r["epoch"]) for r in trace] != list(range(1, a.epochs + 1)):
        raise RuntimeError("Training did not complete the fixed schedule.")
    _, params, _, flops = model.trainer.model.info(imgsz=a.imgsz)
    summary = dict(config=config, training_recipe=OPTIMIZER, completed_epochs=len(trace),
                   primary_checkpoint="last", best_fitness_epoch=model.trainer.stopper.best_epoch,
                   train_seconds_csv=float(trace[-1]["time"]), train_call_wall_seconds=wall,
                   complexity=dict(params=params, gflops=flops, imgsz=a.imgsz, graph="unfused training model"),
                   weights={key: checkpoint_info(run / "weights" / f"{key}.pt") for key in ("last", "best")})
    del model
    if a.device != "cpu":
        torch.cuda.empty_cache()
    checkpoint = run / "weights" / "last.pt"
    for role, spec in (("val", yaml_path), ("probe", a.split_dir / "rgbt" / "probe.yaml")):
        summary[role] = evaluate(checkpoint, spec, run / f"{role}_last", a.device, a.imgsz, a.batch, a.workers)
    write_json(result, summary)
    print(f"Completed {name}. Probe mAP50-95: {summary['probe']['map50_95']:.4f}", flush=True)


def report(a, held):
    truth_val = prepare_truth(a.data_root)
    thermal = a.data_root / "thermal"
    files = image_index(thermal / "images" / "train")
    truth_probe = [(stem, files[stem], thermal / "labels" / "train" / f"{stem}.txt") for stem in sorted(held)]
    ap, fppi, totals = [], [], []
    for arm in a.arms:
        run = a.run_root / f"rgbt_rgbmean_{arm}_s{a.seed}_e{a.epochs}"
        result = json.loads((run / "tcc_result.json").read_text(encoding="utf-8"))
        thresholds = None
        for role, truth in (("val", truth_val), ("probe", truth_probe)):
            metrics = result[role]
            predictions = run / f"{role}_last" / "predictions.json"
            if sha256(predictions) != metrics["predictions_sha256"]:
                raise ValueError(f"Predictions differ from the run record: {predictions}")
            tags = dict(arm=arm, seed=a.seed, evaluation=role)
            print(f"Analyzing {arm} on {role} ...", flush=True)
            rows, fp, *_ = evaluate_predictions(predictions, truth, tags, thresholds)
            if role == "val":
                thresholds = {line["budget"]: line["threshold"] for line in fp if line["stratum"] == "all"}
            ap.extend(rows)
            fppi.extend(fp)
            totals.append(dict(tags, **{k: metrics[k] for k in ("map50", "map50_95", "ap75", "precision", "recall")},
                               params=result["complexity"]["params"], gflops=result["complexity"]["gflops"],
                               train_wall_s=result["train_call_wall_seconds"],
                               inference_ms=metrics["speed_ms_per_image"]["inference"]))
    out = a.run_root / "analysis"
    out.mkdir(exist_ok=True)
    write_csv(out / "stratified_ap.csv", ap)
    write_csv(out / "fppi_calibrated_on_val.csv", fppi)
    write_csv(out / "ultralytics_metrics.csv", totals)
    if set(a.arms) == set(ARMS) and not a.smoke:
        scores = {(x["arm"], x["evaluation"], x["stratum"]): x["map50_95"] for x in ap if x["prefix"] == "all"}

        def delta(other, evaluation, stratum):
            return float(100 * (scores["p2", evaluation, stratum] - scores[other, evaluation, stratum]))

        p0, p1 = (delta(other, "probe", "negative_or_near") for other in ("p0", "p1"))
        val_all, val_pos, probe_pos = (delta("p0", e, s)
                                      for e, s in (("val", "all"), ("val", "positive"), ("probe", "positive")))
        passed = p0 >= 3 and p1 >= 2 and val_all >= -0.5 and val_pos >= -1.0 and probe_pos >= -1.0
        write_json(out / "screening.json", dict(passed=passed, p2_minus_p0_pp=p0, p2_minus_p1_pp=p1,
                   val_all_delta_pp=val_all, val_positive_delta_pp=val_pos, probe_positive_delta_pp=probe_pos,
                   transition="advance" if passed else ("mechanism_not_supported" if p0 < 1 or p1 <= 0 else "inconclusive")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("quality", "train", "report"))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    a.data_root, a.split_dir, a.run_root = (path.resolve() for path in (a.data_root, a.split_dir, a.run_root))
    a.seed = 42
    a.epochs, a.imgsz, a.batch = (1, 128, 4) if a.smoke else (10, 640, 32)
    a.arms = list(dict.fromkeys(a.arms))
    if a.smoke:
        a.workers = 0
    digest, held, training_images = split_inputs(a.data_root, a.split_dir, a.smoke)
    qc = a.run_root / "quality"
    if a.action == "quality":
        check(training_images, qc, a.seed, a.imgsz)
    elif a.action == "report":
        report(a, held)
    else:
        if a.device == "cpu" and not a.smoke:
            raise SystemExit("Select a CUDA device for the full comparison.")
        if a.smoke:
            quality_sha = None
        else:
            path = qc / "quality.json"
            if not path.exists() or not json.loads(path.read_text(encoding="utf-8"))["passed"]:
                raise SystemExit("Run the augmentation quality check before starting training.")
            quality_sha = sha256(path)
        hardware = gpu_label(a.device)
        for arm in a.arms:
            train_one(a, arm, digest, hardware, quality_sha)


if __name__ == "__main__":
    main()
