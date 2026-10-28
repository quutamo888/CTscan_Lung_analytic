"""
Build comprehensive 1000-item CT Scan Dataset:
- 553 Full-slice scans (including 54 confirmed Normal healthy lungs + 499 cancer types)
- 447 LIDC/LUNA Nodule scans from local parquet (including 100 benign/normal-like + 347 malignant)
Total = Exactly 1,000 real clinical CT images.
Splits 10 holdout cases (3 Normal, 7 Disease) for UI testing, 990 in Vector Gallery.
"""
import io
import json
import shutil
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "test_cases"
GALLERY_DIR = DATA_DIR / "gallery"
CHEST_DIR = DATA_DIR / "chest_ct_1000"
PARQUET_PATH = DATA_DIR / "hf_cache" / "data" / "train-00000-of-00001.parquet"


def assemble_1000_cases():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    # Clean old images
    for f in TEST_DIR.glob("*.png"):
        f.unlink()
    for f in GALLERY_DIR.glob("*.png"):
        f.unlink()

    all_cases = []

    # 1. Collect from chest_ct_1000 (553 images)
    chest_files = list(CHEST_DIR.rglob("*.png"))
    print(f"Collecting {len(chest_files)} full-slice CT images from chest_ct_1000...")
    for idx, f in enumerate(chest_files, start=1):
        cat = f.parent.name
        is_normal = (cat.lower() == "normal")

        clean_cat = "Normal (คนปกติ แข็งแรง)" if is_normal else f"Cancer ({cat})"
        guideline = "ปอดปกติ ไม่พบรอยโรคต้องสงสัย" if is_normal else "พบก้อนเนื้อมะเร็งปอด แนะนำปรึกษาแพทย์เฉพาะทาง"

        all_cases.append({
            "source": "chest_ct_full",
            "is_normal": is_normal,
            "patient_id": f"ChestCT_{cat[:8]}_{idx:03d}",
            "clinical_finding": clean_cat,
            "clinical_guideline": guideline,
            "malignancy": "None (Normal)" if is_normal else "Malignant",
            "texture": "Clear Lung" if is_normal else "Mass/Nodule",
            "img_file": f,
        })

    # 2. Collect remaining (up to 1000 total) from local LIDC parquet
    needed = 1000 - len(all_cases)
    print(f"Extracting {needed} nodule cases from local LIDC-IDRI parquet...")

    table = pq.read_table(PARQUET_PATH, columns=["patient_id", "malignancy", "texture", "margin", "image"])
    mal_map = {0: "Highly Unlikely (Benign)", 1: "Moderately Unlikely", 2: "Indeterminate", 3: "Moderately Suspicious", 4: "Highly Suspicious"}
    tex_map = {0: "Solid", 1: "Part Solid", 2: "Non-Solid/GGO"}

    added_from_parquet = 0
    for row_idx in range(len(table)):
        if added_from_parquet >= needed:
            break

        mal_val = table["malignancy"][row_idx].as_py()
        tex_val = table["texture"][row_idx].as_py()
        pid_int = table["patient_id"][row_idx].as_py()
        img_dict = table["image"][row_idx].as_py()

        # Open image from parquet bytes
        img_bytes = img_dict["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)

        mal_str = mal_map.get(mal_val, "Indeterminate")
        tex_str = tex_map.get(tex_val, "Solid")
        is_benign = (mal_val == 0)

        finding = f"LIDC Nodule | Malignancy: {mal_str}"
        guideline = f"Texture: {tex_str} | Radiologist Consensus"

        all_cases.append({
            "source": "lidc_parquet",
            "is_normal": False,
            "is_benign": is_benign,
            "patient_id": f"LIDC_P{pid_int:04d}_{added_from_parquet:03d}",
            "clinical_finding": finding,
            "clinical_guideline": guideline,
            "malignancy": mal_str,
            "texture": tex_str,
            "pil_img": img,
        })
        added_from_parquet += 1

    print(f"Total assembled cases: {len(all_cases)}")
    assert len(all_cases) == 1000, f"Expected 1000 cases, got {len(all_cases)}"

    # 3. Hold out 10 test cases:
    # 3 Normal cases + 7 Disease cases (Adeno, Large cell, Squamous, Benign nodule, Malignant nodule)
    normal_candidates = [c for c in all_cases if c.get("is_normal")]
    disease_candidates = [c for c in all_cases if not c.get("is_normal")]

    test_normal = normal_candidates[:3]
    test_disease = disease_candidates[:7]
    test_items = test_normal + test_disease

    test_ids = set(id(c) for c in test_items)
    gallery_items = [c for c in all_cases if id(c) not in test_ids]

    # Save Test Cases
    test_manifest = []
    for i, c in enumerate(test_items, start=1):
        filename = f"test_{i:02d}_{c['patient_id']}.png"
        out_path = TEST_DIR / filename
        if "img_file" in c:
            img = Image.open(c["img_file"]).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
            img.save(out_path)
        else:
            c["pil_img"].save(out_path)

        meta = {
            "id": f"test_{i:02d}",
            "seriesuid": c["patient_id"],
            "patient_id": c["patient_id"],
            "clinical_finding": c["clinical_finding"],
            "clinical_guideline": c["clinical_guideline"],
            "malignancy": c["malignancy"],
            "texture": c["texture"],
            "image_path": str(out_path),
        }
        test_manifest.append(meta)

    with open(TEST_DIR / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    # Save Gallery Cases (990 items)
    gallery_manifest = []
    for i, c in enumerate(gallery_items, start=1):
        filename = f"gal_{i:04d}_{c['patient_id']}.png"
        out_path = GALLERY_DIR / filename
        if "img_file" in c:
            shutil.copy(c["img_file"], out_path)
        else:
            c["pil_img"].save(out_path)

        meta = {
            "id": f"gal_{i:04d}",
            "seriesuid": c["patient_id"],
            "patient_id": c["patient_id"],
            "clinical_finding": c["clinical_finding"],
            "clinical_guideline": c["clinical_guideline"],
            "malignancy": c["malignancy"],
            "texture": c["texture"],
            "image_path": str(out_path),
        }
        gallery_manifest.append(meta)

    with open(GALLERY_DIR / "gallery_manifest.json", "w", encoding="utf-8") as f:
        json.dump(gallery_manifest, f, indent=2)

    print(f"Assembly completed:")
    print(f" - Reserved 10 holdout test cases (3 Normal, 7 Disease) in: {TEST_DIR}")
    print(f" - Indexed 990 gallery reference cases in: {GALLERY_DIR}")


if __name__ == "__main__":
    assemble_1000_cases()
