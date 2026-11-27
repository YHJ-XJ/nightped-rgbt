"""Compare last and best validation predictions under fixed LLVIP strata and FPPI budgets."""

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from run_protocol import ARMS, MANIFEST_SHA, SEEDS, image_index, sha256

IOUS = np.arange(50, 100, 5) / 100
BUDGETS = (0.05, 0.10, 0.25)
STRATA = {
    "all": lambda c: np.isfinite(c),
    "negative": lambda c: c < -9.1,
    "near_zero": lambda c: np.abs(c) <= 9.1,
    "positive": lambda c: c > 9.1,
    "negative_or_near": lambda c: c <= 9.1,
}


def contrast(ii, boxes):
    """8-bit thermal mean inside a box minus the mean in a local ring."""
    if not len(boxes):
        return np.empty(0)
    h, w = (v - 1 for v in ii.shape)

    def box_sum(rect):
        x1, y1, x2, y2 = np.rint(rect).astype(int).T
        x1, x2 = x1.clip(0, w), x2.clip(0, w)
        y1, y2 = y1.clip(0, h), y2.clip(0, h)
        return ii[y2, x2] - ii[y2, x1] - ii[y1, x2] + ii[y1, x1], (x2 - x1) * (y2 - y1)

    pad = np.max(boxes[:, 2:] - boxes[:, :2], axis=1)[:, None] * 0.5
    inside, area = box_sum(boxes)
    outer, outer_area = box_sum(boxes + pad * np.array([-1, -1, 1, 1]))
    valid = (area > 0) & (outer_area > area)
    value = np.full(len(boxes), np.nan)
    value[valid] = inside[valid] / area[valid] - (outer[valid] - inside[valid]) / (outer_area[valid] - area[valid])
    return value


def overlaps(a, b):
    delta = np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    common = np.maximum(delta, 0).prod(axis=2)
    union = (a[:, 2:] - a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:] - b[:, :2]).prod(axis=1) - common
    return common / np.maximum(union, 1e-12)


def matching(matrix, scores, valid_gt, valid_det):
    """COCO-style ignored strata, matching in descending confidence order."""
    result = np.full((len(scores), len(IOUS)), -1, np.int8)
    hits = np.full((len(valid_gt), 2), -np.inf)
    for k, th in enumerate(IOUS):
        free = np.ones(len(valid_gt), bool)
        for d, line in enumerate(matrix):
            candidates = free & (line >= th)
            accepted = candidates & valid_gt
            if accepted.any():
                i = np.where(accepted, line, -1).argmax()
                free[i] = False
                result[d, k] = 1
                if k in (0, 5):
                    hits[i, int(k == 5)] = scores[d]
            elif candidates.any():
                free[np.where(candidates, line, -1).argmax()] = False
            elif valid_det[d]:
                result[d, k] = 0
    return result, hits


def ap101(scores, outcomes, n_gt):
    if not n_gt:
        return np.full(len(IOUS), np.nan)
    order = np.argsort(-scores, kind="stable")
    result = np.zeros(len(IOUS))
    for k, status in enumerate(outcomes[order].T):
        status = status[status >= 0]
        if not len(status):
            continue
        tp, fp = np.cumsum(status == 1), np.cumsum(status == 0)
        precision = np.maximum.accumulate((tp / (tp + fp))[::-1])[::-1]
        at = np.searchsorted(tp / n_gt, np.linspace(0, 1, 101))
        # Any recall level past the last true positive contributes zero.
        result[k] = np.r_[precision, 0][at].mean()
    return result


def prepare_truth(root):
    thermal = root / "thermal"
    images = image_index(thermal / "images" / "val")
    with (root / "split_manifest.csv").open(encoding="utf-8", newline="") as stream:
        ids = {r["id"] for r in csv.DictReader(stream) if r["split"] == "val"}
    if set(images) != ids:
        raise ValueError("Thermal validation images do not match the manifest.")
    return [(stem, images[stem], thermal / "labels" / "val" / f"{stem}.txt") for stem in sorted(ids)]


def evaluate_predictions(pred, truth, tags):
    data = defaultdict(list)
    for d in json.loads(pred.read_text(encoding="utf-8")):
        x, y, w, h = d["bbox"]
        data[str(d["image_id"]).zfill(6)].append((d["score"], x, y, x + w, y + h))
    extra = set(data) - {row[0] for row in truth}
    if extra:
        raise ValueError(f"Unexpected image ID in {pred}: {min(extra)}")
    by_group = defaultdict(list)
    hit_records, fp_scores = [], []
    for stem, image, label in truth:
        plane = cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if plane is None:
            raise ValueError(f"Cannot read {image}")
        h, w = plane.shape
        raw = label.read_text(encoding="utf-8").split()
        labels = np.asarray(raw, dtype=float).reshape(-1, 5)
        xy, size = labels[:, 1:3], labels[:, 3:5]
        gt = np.hstack((xy - size / 2, xy + size / 2)) * [w, h, w, h]
        det = np.array(sorted(data[stem], key=lambda r: -r[0]), dtype=float).reshape(-1, 5)
        score, boxes = det[:, 0], det[:, 1:]
        ii = cv2.integral(plane, sdepth=cv2.CV_64F)
        cgt, cdet = contrast(ii, gt), contrast(ii, boxes)
        iou = overlaps(boxes, gt)
        for stratum, select in STRATA.items():
            gin, din = select(cgt), select(cdet)
            if stratum == "all":
                gin, din = np.ones(len(cgt), bool), np.ones(len(cdet), bool)
            result, hits = matching(iou, score, gin, din)
            by_group[stratum].append((stem[:2], score, result, int(gin.sum())))
            if stratum == "all":
                hit_records.append((cgt, hits))
                fp_scores.extend(score[result[:, 0] == 0])

    prefixes = ["all", *sorted({r[0][:2] for r in truth})]
    ap_rows = []
    for stratum, records in by_group.items():
        for prefix in prefixes:
            rows = [r for r in records if prefix == "all" or r[0] == prefix]
            n = sum(r[3] for r in rows)
            ap = ap101(np.concatenate([r[1] for r in rows]), np.concatenate([r[2] for r in rows]), n)
            ap_rows.append(dict(tags, prefix=prefix, stratum=stratum, n_gt=n,
                                ap50=ap[0], ap75=ap[5], map50_95=ap.mean()))
    cgt = np.concatenate([r[0] for r in hit_records])
    hit_scores = np.concatenate([r[1] for r in hit_records])
    fp_scores = np.sort(np.asarray(fp_scores))[::-1]
    fppi_rows, main_hits = [], None
    for budget in BUDGETS:
        limit = math.floor(budget * len(truth))
        threshold = float(np.nextafter(fp_scores[limit], np.inf)) if len(fp_scores) > limit else 0.001
        fp = int((fp_scores >= threshold).sum())
        hits = hit_scores >= threshold
        if budget == 0.10:
            main_hits = hits[:, 1]
        for stratum, select in STRATA.items():
            mask = select(cgt) if stratum != "all" else np.ones(len(cgt), bool)
            n = int(mask.sum())
            fppi_rows.append(dict(tags, budget=budget, threshold=threshold, actual_fppi=fp / len(truth),
                                  stratum=stratum, n_gt=n, recall50=hits[mask, 0].mean() if n else np.nan,
                                  recall75=hits[mask, 1].mean() if n else np.nan))
    return ap_rows, fppi_rows, main_hits, cgt


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(out, ap_rows, hits, cohort, cgt):
    last = {(r["arm"], r["seed"], r["repeat"], r["stratum"]): r["map50_95"] for r in ap_rows
            if r["checkpoint"] == "last" and r["prefix"] == "all"}
    agg, paired = [], []
    for arm, stratum in itertools.product(ARMS, STRATA):
        values = [last.get((arm, seed, False, stratum)) for seed in SEEDS]
        available = [v for v in values if v is not None and np.isfinite(v)]
        if available:
            agg.append(dict(arm=arm, stratum=stratum, n_seeds=len(available), mean=statistics.mean(available),
                            std=statistics.stdev(available) if len(available) > 1 else "",
                            spread=max(available) - min(available)))
    for arm, seed, stratum in itertools.product(("s3", "s3p"), SEEDS, STRATA):
        a, b = last.get((arm, seed, False, stratum)), last.get(("s2", seed, False, stratum))
        if a is not None and b is not None:
            paired.append(dict(arm=arm, versus="s2", seed=seed, stratum=stratum, delta_map50_95_pp=(a - b) * 100))
    write_csv(out / "seed_summary_last.csv", agg)
    write_csv(out / "paired_seed_deltas_last.csv", paired)
    comparisons = []
    mask = STRATA["negative_or_near"](cgt)
    keys = [key for key in hits if key[-1] == "last"]
    for a, b in itertools.combinations(sorted(keys), 2):
        same_arm = a[0] == b[0]
        same_seed_cross_arm = a[1] == b[1] and not a[2] and not b[2] and "s2" in {a[0], b[0]}
        if not same_arm and not same_seed_cross_arm:
            continue
        ha, hb = hits[a][mask], hits[b][mask]
        comparisons.append(dict(a=repr(a[:3]), b=repr(b[:3]), n_gt=int(mask.sum()), a_hits=int(ha.sum()),
                                b_hits=int(hb.sum()), a_only=int((ha & ~hb).sum()), b_only=int((hb & ~ha).sum()),
                                union=int((ha | hb).sum())))
    write_csv(out / "negative_or_near_box_churn_last_fppi010.csv", comparisons)
    complete = all((arm, seed, False, "all") in last for arm in ARMS for seed in SEEDS)
    (out / "analysis_scope.json").write_text(json.dumps(
        dict(cohort=cohort, matrix_complete=complete, repeat_present=("s3", 42, True, "all") in last,
             subgroup_ap="COCO-style, separate from Ultralytics validation metrics",
             churn="negative or near-zero; FPPI 0.10 at IoU 0.5, hit at IoU 0.75"), indent=2
    ) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--audit-report", type=Path, help="Phase 0 weight_audit.json, instead of matrix results")
    a = p.parse_args()
    a.data_root, a.out = a.data_root.resolve(), a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    digest = sha256(a.data_root / "split_manifest.csv")
    truth = prepare_truth(a.data_root)
    ap_rows, fppi_rows, overview, hits = [], [], [], {}
    cohort = None
    inputs = []
    if a.audit_report:
        if digest != MANIFEST_SHA:
            raise ValueError("Audit analysis requires the original split.")
        report = json.loads(a.audit_report.read_text(encoding="utf-8"))
        inputs = [(dict(arm=arm, seed=42, repeat=False, checkpoint="last"), item)
                  for arm, item in report["evaluations"].items()]
        cohort = {"scope": "historical 30-epoch last checkpoints"}
    else:
        for f in sorted(a.run_root.glob("*/protocol_result.json")):
            run = json.loads(f.read_text(encoding="utf-8"))
            c = run["config"]
            fields = ("scope", "epochs", "imgsz", "batch", "split_manifest_sha256", "pretrained_sha256",
                      "device", "hardware", "ultralytics", "torch", "opencv", "workers")
            sig = {key: c[key] for key in fields} | {"recipe": run["training_recipe"]}
            if c["split_manifest_sha256"] != digest or (cohort is not None and cohort != sig):
                raise ValueError(f"Nonmatching data or recipe: {f}")
            cohort = sig
            for ckpt in ("last", "best"):
                item = run[ckpt]
                tags = dict(arm=c["arm"], seed=c["seed"], repeat=c["repeat"], checkpoint=ckpt)
                overview.append(dict(tags, epochs=c["epochs"], hardware=c["hardware"],
                                     train_seconds=run["train_seconds_csv"], **run["complexity"],
                                     **{key: item[key] for key in ("map50", "ap75", "map50_95", "precision", "recall")},
                                     inference_ms_per_image=item["speed_ms_per_image"]["inference"],
                                     checkpoint_sha256=item["checkpoint"]["sha256"],
                                     checkpoint_size_bytes=item["checkpoint"]["size_bytes"],
                                     checkpoint_path=item["checkpoint"]["path"]))
                resolved = f.parent / f"val_{ckpt}" / "predictions.json"
                inputs.append((tags, item | {"predictions": str(resolved)}))
    if not inputs:
        raise SystemExit("No completed training or audit results to analyze.")
    for tags, item in inputs:
        pred = Path(item["predictions"])
        if sha256(pred) != item["predictions_sha256"]:
            raise ValueError(f"Prediction hash changed: {pred}")
        key = tuple(tags.values())
        if key in hits:
            raise ValueError(f"Duplicate run: {key}")
        print("Analyzing", key, flush=True)
        ap, fp, hits[key], cgt = evaluate_predictions(pred, truth, tags)
        ap_rows.extend(ap)
        fppi_rows.extend(fp)
    write_csv(a.out / "ultralytics_validation.csv", overview)
    write_csv(a.out / "subgroup_ap.csv", ap_rows)
    write_csv(a.out / "recall_at_fp_budget.csv", fppi_rows)
    summarize(a.out, ap_rows, hits, cohort, cgt)
    print(a.out)


if __name__ == "__main__":
    main()
