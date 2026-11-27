"""Convert paired LLVIP images and VOC boxes into comparable YOLO datasets.

The author's train/test split is retained. Validation is a deterministic set of
whole filename-prefix groups carved from the author's training set.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np


BUNDLE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = BUNDLE / "LLVIP_HF_McCheng.zip"
DEFAULT_OUT = BUNDLE / "data" / "LLVIP_prepared"
MODES = ("rgb", "thermal", "rgbt")


def index_archive(zf: zipfile.ZipFile) -> dict[str, dict[str, dict[str, str]]]:
    index: dict[str, dict[str, dict[str, str]]] = {
        kind: {split: {} for split in ("train", "test")}
        for kind in ("visible", "infrared")
    }
    index["annotations"] = {"all": {}}
    for name in zf.namelist():
        parts = Path(name).parts
        low = [part.lower() for part in parts]
        if len(parts) < 3:
            continue
        suffix = Path(name).suffix.lower()
        for kind in ("visible", "infrared"):
            if kind in low:
                at = low.index(kind)
                if at + 2 < len(low) and low[at + 1] in ("train", "test") and suffix in (".jpg", ".jpeg", ".png"):
                    split = low[at + 1]
                    stem = Path(name).stem
                    if stem in index[kind][split]:
                        raise ValueError(f"Duplicate {kind}/{split} ID: {stem}")
                    index[kind][split][stem] = name
        if "annotations" in low and suffix == ".xml":
            stem = Path(name).stem
            if stem in index["annotations"]["all"]:
                raise ValueError(f"Duplicate annotation ID: {stem}")
            index["annotations"]["all"][stem] = name
    return index


def validate_pairs(index: dict[str, dict[str, dict[str, str]]]) -> dict[str, list[str]]:
    pairs = {}
    labels = set(index["annotations"]["all"])
    for split in ("train", "test"):
        visible = set(index["visible"][split])
        infrared = set(index["infrared"][split])
        if visible != infrared or not visible <= labels:
            raise ValueError(
                f"{split} pairing/annotation mismatch: visible={len(visible)}, "
                f"infrared={len(infrared)}, labels={len(visible & labels)}"
            )
        pairs[split] = sorted(visible)
    if set(pairs["train"]) & set(pairs["test"]):
        raise ValueError("The author's train and test sets overlap")
    if not pairs["train"] or not pairs["test"]:
        raise ValueError("Could not find LLVIP visible/infrared train/test image pairs")
    return pairs


def split_train(stems: list[str], seed: int, fraction: float) -> dict[str, str]:
    """Keep likely scene sequences together using the first two ID digits.

    This is a filename heuristic, not an assertion about the collection metadata.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        groups[stem[:2]].append(stem)
    if len(groups) < 2:
        raise ValueError("Need at least two filename-prefix groups for held-out validation")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = round(len(stems) * fraction)
    # Keep the seeded group order, but stop before a large group makes the
    # validation set much larger than requested.
    selected = []
    count = 0
    for key in keys:
        size = len(groups[key])
        if selected and count + size >= target:
            if abs(count + size - target) < abs(count - target):
                selected.append(key)
            break
        selected.append(key)
        count += size
        if count >= target:
            break
    val_groups = set(selected)
    return {stem: "val" if stem[:2] in val_groups else "train" for stem in stems}


def read_image(raw: bytes, grayscale: bool, stem: str) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"Could not decode image {stem}")
    return image


def parse_boxes(raw: bytes, width: int, height: int, stem: str) -> tuple[list[str], list[tuple[float, float, float, float]]]:
    root = ET.fromstring(raw)
    size = root.find("size")
    if size is not None:
        xml_w = size.findtext("width")
        xml_h = size.findtext("height")
        if xml_w and xml_h and (int(xml_w), int(xml_h)) != (width, height):
            raise ValueError(f"{stem}: XML size differs from paired images")
    output = []
    dropped = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").lower().strip()
        if name != "person":
            raise ValueError(f"{stem}: unexpected class {name!r}; expected person")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"{stem}: object without bndbox")
        x1, y1, x2, y2 = (float(box.findtext(tag, "nan")) for tag in ("xmin", "ymin", "xmax", "ymax"))
        if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
            raise ValueError(f"{stem}: invalid/out-of-bounds box {(x1, y1, x2, y2)}")
        if x1 == x2 or y1 == y2:
            dropped.append((x1, y1, x2, y2))
            continue
        output.append(
            f"0 {(x1 + x2) / (2 * width):.6f} {(y1 + y2) / (2 * height):.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return output, dropped


def write_yaml(root: Path, channels: int) -> None:
    lines = [
        f"path: {json.dumps(root.as_posix(), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"channels: {channels}",
        "names:",
        "  0: person",
    ]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(archive: Path, output: Path, modes: tuple[str, ...], seed: int, val_fraction: float, limit: int | None, resume: bool = False, workers: int = 4) -> None:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    output = output.resolve()
    if not output.as_posix().isascii():
        raise ValueError(f"Use an ASCII-only output path for OpenCV TIFF reading: {output}")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output is not empty: {output}")
    if not 0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    if workers < 1:
        raise ValueError("--workers must be positive")
    with zipfile.ZipFile(archive) as zf:
        index = index_archive(zf)
        pairs = validate_pairs(index)
        assign = split_train(pairs["train"], seed, val_fraction)
        assign.update({stem: "test" for stem in pairs["test"]})
        source_split = {stem: split for split, names in pairs.items() for stem in names}
        stems = pairs["train"] + pairs["test"]
        if limit is not None:
            if limit < 4:
                raise ValueError("--limit must be at least 4")
            train_ids = [stem for stem in pairs["train"] if assign[stem] == "train"]
            val_ids = [stem for stem in pairs["train"] if assign[stem] == "val"]
            subset = max(1, limit // 4)
            stems = train_ids[:limit] + val_ids[:subset] + pairs["test"][:subset]

        output.mkdir(parents=True, exist_ok=True)
        for mode in modes:
            for split in ("train", "val", "test"):
                for kind in ("images", "labels"):
                    (output / mode / kind / split).mkdir(parents=True, exist_ok=True)

        def process_stem(stem: str) -> tuple[str, str, int, list[dict]]:
            split = assign[stem]
            original = source_split[stem]
            rgb_raw = zf.read(index["visible"][original][stem])
            thermal_raw = zf.read(index["infrared"][original][stem])
            rgb = read_image(rgb_raw, False, stem)
            thermal = read_image(thermal_raw, True, stem)
            if rgb.shape[:2] != thermal.shape[:2]:
                raise ValueError(f"{stem}: visible and infrared dimensions differ")
            height, width = thermal.shape
            label, dropped = parse_boxes(zf.read(index["annotations"]["all"][stem]), width, height, stem)
            label_raw = ("\n".join(label) + ("\n" if label else "")).encode("utf-8")
            for mode in modes:
                image_dir = output / mode / "images" / split
                label_dir = output / mode / "labels" / split
                if mode == "rgbt":
                    b, g, r = cv2.split(rgb)
                    ok, encoded = cv2.imencodemulti(".tiff", [r, g, b, thermal], [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE])
                    if not ok:
                        raise RuntimeError(f"{stem}: TIFF encoding failed")
                    image_raw, suffix = encoded.tobytes(), ".tiff"
                elif mode == "rgb":
                    image_raw, suffix = rgb_raw, Path(index["visible"][original][stem]).suffix.lower()
                else:
                    image_raw, suffix = thermal_raw, Path(index["infrared"][original][stem]).suffix.lower()
                (image_dir / f"{stem}{suffix}").write_bytes(image_raw)
                (label_dir / f"{stem}.txt").write_bytes(label_raw)
            return stem, split, len(label), [{"id": stem, "box": coords} for coords in dropped]

        counts = Counter()
        boxes = Counter()
        dropped_boxes = []
        with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("id", "source_split", "prefix_group", "split", "boxes"))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = iter(stems)
                number = 0
                while batch := list(islice(pending, workers * 2)):
                    for stem, split, nboxes, dropped in pool.map(process_stem, batch):
                        number += 1
                        writer.writerow((stem, source_split[stem], stem[:2], split, nboxes))
                        counts[split] += 1
                        boxes[split] += nboxes
                        dropped_boxes.extend(dropped)
                        if number % 500 == 0 or number == len(stems):
                            print(f"Prepared {number}/{len(stems)} pairs", flush=True)

    for mode in modes:
        write_yaml(output / mode, 4 if mode == "rgbt" else 3)
    summary = {
        "pairs": dict(counts), "boxes": dict(boxes), "dropped_degenerate_boxes": dropped_boxes,
        "modes": list(modes), "seed": seed, "val_fraction": val_fraction,
        "actual_val_fraction_of_author_train": counts["val"] / len(pairs["train"]),
        "split_method": "seeded whole two-digit filename-prefix groups; stop before large overshoot",
        "archive": str(archive),
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Overwrite files in an interrupted output using the same archive and settings")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.zip, args.out, tuple(args.modes), args.seed, args.val_fraction, args.limit, args.resume, args.workers)
