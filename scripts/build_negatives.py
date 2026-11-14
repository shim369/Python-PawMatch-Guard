"""Build a small set of *no-animal* negative photos from Open Images (validation split).

Selection (deterministic, seed 42):
  * license is exactly CC BY 2.0 (per-image metadata, not the dataset-wide label);
  * the image has >= MIN_LABELS human-verified positive labels (so it is reasonably well
    labelled) and NONE of them is a class in the "Animal" subtree of the label hierarchy.
Caveat: a missing label is not proof that no animal is visible, so spot-check the result.

Downloads the 300k-pixel thumbnail of each image, re-encodes it as JPEG (drops EXIF and any
embedded data) and writes data/negatives/<ImageID>.jpg plus a manifest with the attribution
CC BY requires (data/negatives_manifest.csv, committed). Images themselves are not committed.
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://storage.googleapis.com/openimages"
FILES = {
    "images.csv": f"{BASE}/2018_04/validation/validation-images-with-rotation.csv",
    "hierarchy.json": f"{BASE}/2018_04/bbox_labels_600_hierarchy.json",
    "labels.csv": f"{BASE}/v5/validation-annotations-human-imagelabels-boxable.csv",
    "classes.csv": f"{BASE}/v7/oidv7-class-descriptions-boxable.csv",
}
LICENSE = "https://creativecommons.org/licenses/by/2.0/"
MIN_LABELS, TARGET, SEED, MAX_BYTES = 3, 300, 42, 10_000_000
RAW, OUT = Path("data/openimages"), Path("data/negatives")
MANIFEST = Path("data/negatives_manifest.csv")


def fetch_metadata() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        if not (RAW / name).exists():
            urllib.request.urlretrieve(url, RAW / name)


def descendants(node: dict, target: str, inside: bool = False) -> set[str]:
    """All label MIDs below (and including) the node named ``target``."""
    here = inside or node.get("LabelName") == target
    out = {node["LabelName"]} if here and "LabelName" in node else set()
    for key in ("Subcategory", "Part"):
        for child in node.get(key, []):
            out |= descendants(child, target, here)
    return out


def animal_labels() -> set[str]:
    animal_mid = next(r[0] for r in csv.reader((RAW / "classes.csv").open()) if r[1] == "Animal")
    found = descendants(json.loads((RAW / "hierarchy.json").read_text()), animal_mid)
    if len(found) < 10:
        raise RuntimeError("animal subtree not found in the hierarchy")
    return found


def candidates() -> list[dict]:
    animals = animal_labels()
    positives: dict[str, set[str]] = {}
    for row in csv.DictReader((RAW / "labels.csv").open()):
        if row["Confidence"] == "1":
            positives.setdefault(row["ImageID"], set()).add(row["LabelName"])
    ok = {i for i, labs in positives.items() if len(labs) >= MIN_LABELS and not labs & animals}
    rows = [r for r in csv.DictReader((RAW / "images.csv").open())
            if r["License"] == LICENSE and r["ImageID"] in ok]
    rows.sort(key=lambda r: r["ImageID"])
    random.Random(SEED).shuffle(rows)
    return rows


def download(row: dict) -> bool:
    try:
        req = urllib.request.Request(
            row["Thumbnail300KURL"], headers={"User-Agent": "dataset-builder"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            return False
        img = Image.open(io.BytesIO(data))
        img.load()
        img.convert("RGB").save(OUT / f"{row['ImageID']}.jpg", "JPEG", quality=90)
        return True
    except Exception:  # noqa: BLE001 - dead links are expected; just skip
        return False


def main() -> None:
    fetch_metadata()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = candidates()
    print(f"{len(rows)} candidate images (CC BY 2.0, labelled, no animal label)")
    kept: list[dict] = []
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(rows), 64):
            if len(kept) >= TARGET:
                break
            chunk = rows[i:i + 64]
            for row, ok in zip(chunk, pool.map(download, chunk), strict=True):
                if ok and len(kept) < TARGET:
                    kept.append(row)
    for p in OUT.glob("*.jpg"):  # drop surplus files from the last chunk
        if p.stem not in {r["ImageID"] for r in kept}:
            p.unlink()
    with MANIFEST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "license", "author", "title", "landing_url", "source_url"])
        for r in sorted(kept, key=lambda r: r["ImageID"]):
            w.writerow([r["ImageID"], r["License"], r["Author"], r["Title"],
                        r["OriginalLandingURL"], r["Thumbnail300KURL"]])
    print(f"saved {len(kept)} images to {OUT} and {MANIFEST}")


if __name__ == "__main__":
    main()
