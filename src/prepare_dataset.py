"""
Dataset Preparation: LUNA16 slice extraction and 10-case holdout test split.
Handles both real LUNA16 (.mhd/.raw) and auto-generates sample demo data if raw files absent.
"""
import csv
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from src.mhd_reader import read_mhd, world_to_voxel, extract_axial_slice, apply_lung_window


def generate_synthetic_luna_sample(output_dir: Path, num_scans: int = 35) -> tuple[Path, Path]:
    """
    Generate synthetic LUNA16-compatible scans (.mhd/.raw + annotations.csv)
    when full 130GB dataset is not yet downloaded.
    Simulates chest wall, lung parenchyma (-600 HU), spine (+700 HU), and lung nodules.
    """
    raw_dir = output_dir / "luna16_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotations_file = raw_dir / "annotations.csv"

    annotations = []

    print(f"Creating {num_scans} synthetic LUNA16-compatible CT volumes...")
    for i in range(num_scans):
        seriesuid = f"1.3.6.1.4.1.14519.5.2.1.6279.6001.{100000 + i}"
        mhd_path = raw_dir / f"{seriesuid}.mhd"
        raw_path = raw_dir / f"{seriesuid}.raw"

        # Dimensions: 32 slices, 256x256 pixels
        dim_z, dim_y, dim_x = 32, 256, 256
        spacing_z, spacing_y, spacing_x = 2.5, 0.78, 0.78
        origin_x, origin_y, origin_z = -100.0, -100.0, -40.0

        # Background air: -1000 HU
        vol = np.full((dim_z, dim_y, dim_x), -1000, dtype=np.int16)

        # Body contour (soft tissue: +40 HU)
        yy, xx = np.ogrid[:dim_y, :dim_x]
        body_mask = ((xx - 128) ** 2 / (100 ** 2)) + ((yy - 128) ** 2 / (80 ** 2)) <= 1.0
        vol[:, body_mask] = 40

        # Spine (+700 HU)
        spine_mask = ((xx - 128) ** 2 + (yy - 180) ** 2) <= 12 ** 2
        vol[:, spine_mask] = 700

        # Left and Right Lungs (-700 HU)
        left_lung = ((xx - 85) ** 2 / (35 ** 2)) + ((yy - 120) ** 2 / (50 ** 2)) <= 1.0
        right_lung = ((xx - 171) ** 2 / (35 ** 2)) + ((yy - 120) ** 2 / (50 ** 2)) <= 1.0
        vol[:, left_lung] = -700
        vol[:, right_lung] = -700

        # Add Gaussian noise
        noise = np.random.normal(0, 15, vol.shape).astype(np.int16)
        vol += noise

        # Inject simulated nodule with random diameter and location
        side = "left" if (i % 2 == 0) else "right"
        target_z = 10 + (i % 12)
        target_y = 100 + (i * 3) % 40
        target_x = (85 if side == "left" else 171) + ((i * 5) % 20 - 10)
        diameter_mm = round(5.0 + (i * 1.3) % 22.0, 1)
        radius_vox = max(3, int(diameter_mm / (2 * spacing_x)))

        # Nodule tissue density (-50 to +100 HU)
        for dz in range(-2, 3):
            curr_z = target_z + dz
            if 0 <= curr_z < dim_z:
                nodule_mask = ((xx - target_x) ** 2 + (yy - target_y) ** 2) <= radius_vox ** 2
                vol[curr_z, nodule_mask] = 50 + (i % 50)

        # Write .raw
        vol.tofile(str(raw_path))

        # Write .mhd
        with open(mhd_path, "w", encoding="utf-8") as f:
            f.write("ObjectType = Image\n")
            f.write("NDims = 3\n")
            f.write("BinaryData = True\n")
            f.write("BinaryDataByteOrderMSB = False\n")
            f.write("CompressedData = False\n")
            f.write("TransformMatrix = 1 0 0 0 1 0 0 0 1\n")
            f.write(f"Offset = {origin_x} {origin_y} {origin_z}\n")
            f.write("CenterOfRotation = 0 0 0\n")
            f.write("AnatomicalOrientation = ???\n")
            f.write(f"ElementSpacing = {spacing_x} {spacing_y} {spacing_z}\n")
            f.write(f"DimSize = {dim_x} {dim_y} {dim_z}\n")
            f.write("ElementType = MET_SHORT\n")
            f.write(f"ElementDataFile = {raw_path.name}\n")

        # World coordinates (mm)
        world_x = round(origin_x + target_x * spacing_x, 2)
        world_y = round(origin_y + target_y * spacing_y, 2)
        world_z = round(origin_z + target_z * spacing_z, 2)

        annotations.append({
            "seriesuid": seriesuid,
            "coordX": world_x,
            "coordY": world_y,
            "coordZ": world_z,
            "diameter_mm": diameter_mm,
        })

    # Save annotations.csv
    with open(annotations_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
        writer.writeheader()
        writer.writerows(annotations)

    print(f"Generated {num_scans} scans with annotations at {annotations_file}")
    return raw_dir, annotations_file


def process_dataset(data_dir: Path, num_test_holdout: int = 10):
    """
    Extract slices for nodules, apply lung windowing, and split into:
    - 10 holdout cases for UI testing
    - Remaining cases for gallery database
    """
    raw_dir = data_dir / "luna16_raw"
    annotations_file = raw_dir / "annotations.csv"

    # If raw files don't exist, create synthetic set
    mhd_files = list(raw_dir.glob("*.mhd")) if raw_dir.exists() else []
    if not mhd_files or not annotations_file.exists():
        raw_dir, annotations_file = generate_synthetic_luna_sample(data_dir, num_scans=35)

    gallery_dir = data_dir / "gallery"
    test_dir = data_dir / "test_cases"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(annotations_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "seriesuid": row["seriesuid"],
                "coordX": float(row["coordX"]),
                "coordY": float(row["coordY"]),
                "coordZ": float(row["coordZ"]),
                "diameter_mm": float(row["diameter_mm"]),
            })

    print(f"Total annotated nodules found: {len(records)}")

    # Extract slices
    extracted_items = []
    for idx, item in enumerate(records):
        seriesuid = item["seriesuid"]
        mhd_candidates = list(raw_dir.glob(f"*{seriesuid}*.mhd"))
        if not mhd_candidates:
            continue

        mhd_file = mhd_candidates[0]
        try:
            vol, origin, spacing = read_mhd(str(mhd_file))
        except Exception as e:
            print(f"Warning: Failed reading {mhd_file.name}: {e}")
            continue

        world_coord = np.array([item["coordX"], item["coordY"], item["coordZ"]])
        voxel_coord = world_to_voxel(world_coord, origin, spacing)
        vx, vy, vz = voxel_coord[0], voxel_coord[1], voxel_coord[2]

        # Extract 2D axial slice
        img = extract_axial_slice(vol, vz)

        # Risk stratification label based on Fleischner Society pulmonary nodule guidelines
        diameter = item["diameter_mm"]
        if diameter < 6.0:
            finding = "Small Nodule (<6mm, Low Risk)"
            recommendation = "Optional CT at 12 months"
        elif diameter < 8.0:
            finding = "Intermediate Nodule (6-8mm, Medium Risk)"
            recommendation = "Follow-up CT at 6-12 months"
        else:
            finding = "Large Nodule (>8mm, High Suspicion)"
            recommendation = "Consider contrast-enhanced CT, PET/CT, or Biopsy"

        meta = {
            "id": f"luna_{idx:03d}_{seriesuid[:8]}",
            "seriesuid": seriesuid,
            "slice_z": int(vz),
            "voxel_x": int(vx),
            "voxel_y": int(vy),
            "coordX_mm": item["coordX"],
            "coordY_mm": item["coordY"],
            "coordZ_mm": item["coordZ"],
            "diameter_mm": diameter,
            "clinical_finding": finding,
            "clinical_guideline": recommendation,
        }
        extracted_items.append((img, meta))

    print(f"Successfully extracted {len(extracted_items)} 2D CT nodule slices.")

    # Split: exactly num_test_holdout into test_cases, rest into gallery
    assert len(extracted_items) >= num_test_holdout, "Not enough slices extracted to hold out 10 cases!"

    test_items = extracted_items[:num_test_holdout]
    gallery_items = extracted_items[num_test_holdout:]

    # Save holdout test cases
    test_manifest = []
    for img, meta in test_items:
        img_filename = f"{meta['id']}_test.png"
        img_path = test_dir / img_filename
        img.save(img_path)
        meta["image_path"] = str(img_path)
        test_manifest.append(meta)

    with open(test_dir / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    # Save gallery cases
    gallery_manifest = []
    for img, meta in gallery_items:
        img_filename = f"{meta['id']}_gallery.png"
        img_path = gallery_dir / img_filename
        img.save(img_path)
        meta["image_path"] = str(img_path)
        gallery_manifest.append(meta)

    with open(gallery_dir / "gallery_manifest.json", "w", encoding="utf-8") as f:
        json.dump(gallery_manifest, f, indent=2)

    print(f"Split complete:")
    print(f" - Reserved {len(test_manifest)} holdout test cases in: {test_dir}")
    print(f" - Indexed {len(gallery_manifest)} gallery cases in: {gallery_dir}")

    return test_manifest, gallery_manifest


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "data"
    process_dataset(base_dir, num_test_holdout=10)
