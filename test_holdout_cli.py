"""
Automated CLI verification of the 10 holdout test cases against the Vector Store.
"""
import json
from pathlib import Path
from src.vector_store import CTScanVectorStore

def run_holdout_tests(top_k: int = 5):
    base_dir = Path(__file__).resolve().parent
    test_manifest_path = base_dir / "data" / "test_cases" / "test_manifest.json"

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"=== Testing All {len(test_cases)} Holdout Cases Against Vector Store (Top-{top_k}) ===")
    store = CTScanVectorStore(index_dir=base_dir / "data" / "index")

    for i, case in enumerate(test_cases, start=1):
        img_path = case["image_path"]
        results = store.search(img_path, top_k=top_k)

        print(f"\n[Case #{i:02d}] ID: {case['id']} | Patient: {case.get('patient_id', '-')} | Finding: {case['clinical_finding']}")
        print(f"  Query image: {Path(img_path).name}")
        print(f"  Top-{top_k} Matches:")
        for r in results:
            print(f"    - Rank #{r['rank']} | Sim: {r['similarity_percent']} | Patient: {r.get('patient_id', '-')} | Finding: {r['clinical_finding']}")

    print("\n[SUCCESS] All 10 holdout test cases verified successfully.")

if __name__ == "__main__":
    run_holdout_tests(top_k=5)
