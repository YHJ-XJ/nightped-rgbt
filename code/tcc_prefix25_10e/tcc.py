"""Paired thermal-only edits for P1 and P2."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np


PROBABILITY = 0.5
MIN_SOURCE = 20.0
TARGET_RANGE = (-25.0, 5.0)


def region(image, rect):
    x1, y1, x2, y2 = rect
    return image[y1:y2, x1:x2]


def rounded(boxes, shape):
    h, w = shape
    return np.rint(boxes).astype(int).clip([0, 0, 0, 0], [w, h, w, h])


def outer_boxes(boxes):
    pad = np.max(boxes[:, 2:] - boxes[:, :2], axis=1)[:, None] * 0.5
    return boxes + pad * [-1, -1, 1, 1]


def intersects(box, others):
    return np.any(np.all(np.minimum(box[2:], others[:, 2:]) >
                         np.maximum(box[:2], others[:, :2]), axis=1))


def ring_mean(thermal, box, outer):
    inside, window = region(thermal, box), region(thermal, outer)
    return float((window.sum() - inside.sum()) / (window.size - inside.size))


def uint8(values):
    return np.rint(values.clip(0, 255)).astype(np.uint8)


@dataclass(frozen=True)
class Edit:
    index: int
    box: np.ndarray
    outer: np.ndarray
    target: float
    source: float
    local: np.ndarray
    scale: float
    new_saturation: float


class ThermalCounterfactual:
    def __init__(self, mode, seed):
        self.mode = mode
        self.seed = seed

    def plan(self, thermal, boxes, stem):
        # Stable per-image draws keep exposure paired across arms and workers.
        key = hashlib.blake2b(f"{self.seed}:{stem}".encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(key, "little"))
        if not len(boxes) or rng.random() >= PROBABILITY:
            return None
        inner = rounded(boxes, thermal.shape)
        raw_outer = np.rint(outer_boxes(boxes)).astype(int)
        outer = rounded(raw_outer, thermal.shape)
        for i in rng.permutation(len(inner)):
            box = inner[i]
            if np.any(box[2:] - box[:2] < 6) or np.any(raw_outer[i] != outer[i]):
                continue
            if intersects(box, np.delete(outer, i, axis=0)):
                continue
            patch = region(thermal, box)
            ring = ring_mean(thermal, box, outer[i])
            source = float(patch.mean()) - ring
            if source < MIN_SOURCE:
                continue
            # Keep both arms in the contrast-magnitude reduction regime.
            target = float(rng.uniform(max(TARGET_RANGE[0], -0.9 * source), TARGET_RANGE[1]))
            _, foreground = cv2.threshold(patch, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(closed, kernel).astype(np.float32)
            feather = float(np.clip(round(min(mask.shape) * 0.08), 2, 5))
            mask = cv2.GaussianBlur(mask, (0, 0), feather / 1.5)
            y, x = np.indices(mask.shape)
            distance = np.minimum(np.minimum(y, mask.shape[0] - 1 - y),
                                  np.minimum(x, mask.shape[1] - 1 - x))
            mask *= (distance / feather).clip(0, 1).astype(np.float32)
            values = patch.astype(np.float32)
            low, high = 0.0, 255.0
            if float((values - high * mask).clip(0, 255).mean()) - ring > target:
                continue
            for _ in range(12):
                shift = (low + high) / 2
                if float((values - shift * mask).clip(0, 255).mean()) - ring > target:
                    low = shift
                else:
                    high = shift
            edited = uint8(values - high * mask)
            actual = float(edited.mean()) - ring
            saturation = float(((edited == 0) & (patch != 0)).mean())
            if abs(actual - target) > 3 or saturation >= 0.01:
                continue
            return Edit(i, box, outer[i], target, source, edited, abs(actual) / source, saturation)
        return None

    def edit_plane(self, thermal, plan):
        if plan is None:
            return thermal
        if self.mode == "p1":
            mean = float(thermal.mean())
            return uint8((thermal.astype(np.float32) - mean) * plan.scale + mean)
        edited = thermal.copy()
        region(edited, plan.box)[:] = plan.local
        return edited

    def __call__(self, labels):
        image = labels["img"]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("TCC expects uint8 RGB+thermal TIFF data.")
        instances = deepcopy(labels["instances"])
        instances.convert_bbox("xyxy")
        h, w = image.shape[:2]
        instances.denormalize(w, h)
        thermal = image[:, :, 3]
        plan = self.plan(thermal, instances.bboxes, Path(labels["im_file"]).stem)
        if plan is not None:
            # Ultralytics may buffer source images. Copy before changing channel 4.
            edited = image.copy()
            edited[:, :, 3] = self.edit_plane(thermal, plan)
            labels["img"] = edited
        return labels
