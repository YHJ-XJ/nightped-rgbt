"""Check the paired P1/P2 images before training."""

import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.utils import img2label_paths
from ultralytics.utils.patches import imread

from run_protocol import write_json
from tcc import ThermalCounterfactual, region, ring_mean


def boxes_in_image(label, shape):
    labels = np.asarray(Path(label).read_text(encoding="utf-8").split(), dtype=float).reshape(-1, 5)
    xy, wh = labels[:, 1:3], labels[:, 3:5]
    h, w = shape
    return np.hstack((xy - wh / 2, xy + wh / 2)) * [w, h, w, h]


def check(paths, out, seed, size, count=200):
    rng = np.random.default_rng(seed)
    paths = sorted(str(p) for p in paths)
    images = rng.choice(paths, min(count, len(paths)), replace=False).tolist()
    global_edit = ThermalCounterfactual("p1", seed)
    local_edit = ThermalCounterfactual("p2", seed)
    rows, preview = [], []
    for image, label in zip(images, img2label_paths(images)):
        source = imread(image, flags=cv2.IMREAD_UNCHANGED)
        if source is None or source.ndim != 3 or source.shape[2] != 4 or source.dtype != np.uint8:
            raise ValueError(f"Expected an RGB+thermal uint8 TIFF: {image}")
        h, w = source.shape[:2]
        scale = size / max(h, w)
        if scale != 1:
            source = cv2.resize(source, (min(math.ceil(w * scale), size),
                                         min(math.ceil(h * scale), size)), interpolation=cv2.INTER_LINEAR)
        thermal = source[:, :, 3]
        boxes = boxes_in_image(label, thermal.shape)
        plan = local_edit.plan(thermal, boxes, Path(image).stem)
        if plan is None:
            continue
        p1 = global_edit.edit_plane(thermal, plan)
        p2 = local_edit.edit_plane(thermal, plan)
        restored = p2.copy()
        region(restored, plan.box)[:] = region(thermal, plan.box)
        if not np.array_equal(restored, thermal):
            raise RuntimeError(f"Local edit changed pixels outside the target box: {image}")
        measured = [float(region(p, plan.box).mean()) - ring_mean(p, plan.box, plan.outer)
                    for p in (p1, p2)]
        rows.append(dict(image=Path(image).stem, source=plan.source, target=plan.target,
                         p1=measured[0], p2=measured[1], new_saturation=plan.new_saturation))
        if len(preview) < 30:
            panels = [cv2.resize(region(p, plan.outer), (120, 108), interpolation=cv2.INTER_AREA)
                      for p in (thermal, p1, p2)]
            preview.append(np.hstack(panels))
    out.mkdir(parents=True, exist_ok=True)
    if preview:
        ok, jpg = cv2.imencode(".jpg", np.vstack(preview))
        if not ok:
            raise RuntimeError("Could not write TCC preview.")
        (out / "source_p1_p2.jpg").write_bytes(jpg.tobytes())
    required = max(1, math.ceil(len(images) * 0.04))
    worst_target = max((abs(r["p2"] - r["target"]) for r in rows), default=None)
    worst_match = max((abs(abs(r["p1"]) - abs(r["p2"])) for r in rows), default=None)
    flips = sum(r["p2"] < 0 for r in rows)
    passed = (len(rows) >= required and flips > 0 and all(r["p1"] >= -0.01 for r in rows)
              and worst_target <= 3 and worst_match <= 2)
    result = dict(passed=passed, seed=seed, imgsz=size, sampled_images=len(images),
                  accepted_edits=len(rows), negative_targets=flips,
                  max_target_error=worst_target, max_strength_mismatch=worst_match, edits=rows)
    write_json(out / "quality.json", result)
    print(f"TCC quality: {len(rows)}/{len(images)} edited, {flips} became negative "
          f"(target error {worst_target}, strength mismatch {worst_match}).", flush=True)
    if not passed:
        raise RuntimeError(f"TCC quality check did not pass. Inspect {out / 'quality.json'}.")
    return result
