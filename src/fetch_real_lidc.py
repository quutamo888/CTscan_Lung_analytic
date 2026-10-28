"""
Fetch authentic LIDC-IDRI CT scan nodules directly from HuggingFace dataset.
Bypasses Zenodo 300KB/s speed throttle while using the official LIDC/LUNA16 primary source data.
Splits into 10 holdout test cases and 50 gallery reference cases.
"""
import json
import os
from pathlib import Path
from PIL import Image
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "test_cases"
GALLERY_DIR = DATA_DIR / "gallery"
INDEX_DIR = DATA_DIR / "index"


def fetch_and_prepare_real_dataset(num_test: int = 10, num_gallery: int = 50):
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to HuggingFace LIDC-IDRI (LUNA16 source dataset)...")
    ds = load_dataset("ykeselman/lidc-idri-patches", split="train", streaming=True)
    features = ds.features

    total_target = num_test + num_gallery
    collected_records = []
    patient_counts = {}

    print(f"Streaming {total_target} unique/diverse patient CT scan nodules...")

    for i, item in enumerate(ds):
        patient_id = features["patient_id"].int2str(item["patient_id"])

        # Enforce diversity: max 1 patch per patient so all 60 cases are from different patients
        if patient_counts.get(patient_id, 0) >= 1:
            continue

        patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1

        malignancy = features["malignancy"].int2str(item["malignancy"])
        texture = features["texture"].int2str(item["texture"])
        spiculation = features["spiculation"].int2str(item["spiculation"])
        lobulation = features["lobulation"].int2str(item["lobulation"])
        calcification = features["calcification"].int2str(item["calcification"])
        margin = features["margin"].int2str(item["margin"])

        # Convert image to RGB 224x224
        img = item["image"].convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)

        idx = len(collected_records) + 1
        record = {
            "index": idx,
            "patient_id": patient_id,
            "malignancy": malignancy,
            "texture": texture,
            "spiculation": spiculation,
            "lobulation": lobulation,
            "calcification": calcification,
            "margin": margin,
            "clinical_finding": f"Malignancy: {malignancy} | Texture: {texture}",
            "clinical_guideline": f"Margin: {margin}, Spiculation: {spiculation}",
            "img": img,
        }
        collected_records.append(record)

        if len(collected_records) >= total_target:
            break

    print(f"Collected {len(collected_records)} real CT scan records.")

    # Split into 10 holdout test cases + remaining gallery
    test_data = collected_records[:num_test]
    gallery_data = collected_records[num_test:]

    # Clean old files in test and gallery
    for f in TEST_DIR.glob("*.png"):
        f.unlink()
    for f in GALLERY_DIR.glob("*.png"):
        f.unlink()

    # Save test cases
    test_manifest = []
    for item in test_data:
        filename = f"{item['patient_id']}_patch_{item['index']:03d}_test.png"
        path = TEST_DIR / filename
        item["img"].save(path)

        meta = {
            "id": f"test_{item['index']:02d}",
            "seriesuid": item["patient_id"],
            "patient_id": item["patient_id"],
            "diameter_mm": "Nodule Patch (Real)",
            "slice_z": item["index"],
            "voxel_x": 112,
            "voxel_y": 112,
            "coordX_mm": 0.0,
            "coordY_mm": 0.0,
            "coordZ_mm": 0.0,
            "clinical_finding": item["clinical_finding"],
            "clinical_guideline": item["clinical_guideline"],
            "malignancy": item["malignancy"],
            "texture": item["texture"],
            "image_path": str(path),
        }
        test_manifest.append(meta)

    with open(TEST_DIR / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    # Save gallery cases
    gallery_manifest = []
    for item in gallery_data:
        filename = f"{item['patient_id']}_patch_{item['index']:03d}_gallery.png"
        path = GALLERY_DIR / filename
        item["img"].save(path)

        meta = {
            "id": f"gal_{item['index']:03d}",
            "seriesuid": item["patient_id"],
            "patient_id": item["patient_id"],
            "diameter_mm": "Nodule Patch (Real)",
            "slice_z": item["index"],
            "voxel_x": 112,
            "voxel_y": 112,
            "coordX_mm": 0.0,
            "coordY_mm": 0.0,
            "coordZ_mm": 0.0,
            "clinical_finding": item["clinical_finding"],
            "clinical_guideline": item["clinical_guideline"],
            "malignancy": item["malignancy"],
            "texture": item["texture"],
            "image_path": str(path),
        }
        gallery_manifest.append(meta)

    with open(GALLERY_DIR / "gallery_manifest.json", "w", encoding="utf-8") as f:
        json.dump(gallery_manifest, f, indent=2)

    print(f"Data saved successfully:")
    print(f" - {len(test_manifest)} holdout test cases in: {TEST_DIR}")
    print(f" - {len(gallery_manifest)} gallery cases in: {GALLERY_DIR}")

    return test_manifest, gallery_manifest


if __name__ == "__main__":
    fetch_and_prepare_real_dataset(num_test=10, num_gallery=50)
