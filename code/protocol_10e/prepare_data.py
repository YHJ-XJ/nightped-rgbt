"""Reuse the full prepared split and produce an optional prefix holdout."""

import argparse
import csv
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from prepare_llvip import index_archive, prepare, validate_pairs, write_yaml
from run_protocol import MANIFEST_SHA, check_data, image_index, sha256, write_json


SOURCE_SHA = "b3b55475c093ac54663c467859eb31a82b840b53a7196ad1887cb547e4518b45"
MODES = ("thermal", "rgbt")


def manifest_rows(root):
    path = root / "split_manifest.csv"
    if sha256(path) != MANIFEST_SHA:
        raise ValueError(f"Prepared data do not have the pinned full split: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_archive(path):
    if path is None:
        raise FileNotFoundError("Provide --archive with the original LLVIP ZIP.")
    if sha256(path) != SOURCE_SHA:
        raise ValueError(f"ZIP hash differs from the one used for the original experiments: {path}")


def setup(root, archive, workers):
    if not (root / "split_manifest.csv").is_file():
        verify_archive(archive)
        prepare(archive, root, MODES, 42, 0.15, None, workers=workers)
    rows = manifest_rows(root)
    if not (root / "rgbt" / "data.yaml").is_file():
        raise FileNotFoundError("The existing split has no complete RGBT mode. Prepare a fresh root with setup.")
    if not (root / "thermal").exists():
        verify_archive(archive)
        with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix=".protocol_thermal_", dir=root) as tmp:
            stage = Path(tmp) / "thermal"
            index = index_archive(z)
            validate_pairs(index)
            for split in ("train", "val", "test"):
                (stage / "images" / split).mkdir(parents=True)
                (stage / "labels" / split).mkdir(parents=True)
            for row in rows:
                stem, split = row["id"], row["split"]
                entry = index["infrared"][row["source_split"]][stem]
                (stage / "images" / split / f"{stem}{Path(entry).suffix.lower()}").write_bytes(z.read(entry))
                shutil.copyfile(root / "rgbt" / "labels" / split / f"{stem}.txt",
                                stage / "labels" / split / f"{stem}.txt")
            write_yaml(stage, 3)
            target = root / "thermal"
            if not target.resolve().is_relative_to(root) or target.exists():
                raise ValueError(f"Thermal target must be a new directory inside {root}")
            stage.rename(target)
    for mode in MODES:
        write_yaml(root / mode, 4 if mode == "rgbt" else 3)
    print("Ready. Split SHA256:", check_data(root, set(MODES)))


def probe(root, prefix, out):
    from ultralytics.utils import YAML

    rows = manifest_rows(root)
    if not re.fullmatch(r"\d{2}", prefix):
        raise ValueError("--exclude-prefix expects a two-digit training prefix")
    train = [r for r in rows if r["split"] == "train" and r["prefix_group"] != prefix]
    held = [r for r in rows if r["split"] == "train" and r["prefix_group"] == prefix]
    if not train or not held:
        raise ValueError(f"No valid holdout for training prefix {prefix}")
    check_data(root, set(MODES))
    out.mkdir(parents=True, exist_ok=False)
    for mode in MODES:
        d = out / mode
        d.mkdir()
        mode_root = root / mode
        images = image_index(mode_root / "images" / "train")
        files = {}
        for key, group in (("train", train), ("probe", held)):
            path = d / f"{key}.txt"
            path.write_text("".join(f"{images[r['id']].as_posix()}\n" for r in group), encoding="utf-8")
            files[key] = path.as_posix()
        for name, val in (("train.yaml", "images/val"), ("probe.yaml", files["probe"])):
            cfg = {"path": mode_root.as_posix(), "train": files["train"], "val": val,
                   "channels": 4 if mode == "rgbt" else 3, "names": {0: "person"}}
            YAML.save(d / name, cfg)
    report = {
        "base_manifest_sha256": MANIFEST_SHA, "excluded_train_prefix": prefix,
        "counts": {"train_images": len(train), "probe_images": len(held),
                   "probe_boxes": sum(int(r["boxes"]) for r in held),
                   "val_images": Counter(r["split"] for r in rows)["val"]},
        "probe_ids": [r["id"] for r in held],
    }
    write_json(out / "holdout_manifest.json", report)
    print(json.dumps(report["counts"], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--archive", type=Path)
    s.add_argument("--workers", type=int, default=8)
    h = sub.add_parser("probe")
    h.add_argument("--exclude-prefix", default="25")
    h.add_argument("--out", type=Path, required=True)
    for action in (s, h):
        action.add_argument("--data-root", type=Path, required=True)
    a = p.parse_args()
    if a.command == "setup":
        setup(a.data_root.resolve(), a.archive, a.workers)
    else:
        probe(a.data_root.resolve(), a.exclude_prefix, a.out.resolve())


if __name__ == "__main__":
    main()
