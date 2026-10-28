"""
Gradio Web Interface for CT Scan Vector Search & Dataset Reviewer.
Tabs:
1. 🔍 Search & Compare K-Top 5
2. 📚 Dataset Collection Reviewer + 🧬 Vector Point & Collector Inspector
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import gradio as gr
from PIL import Image

from src.vector_store import CTScanVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "test_cases"
TEST_MANIFEST_PATH = TEST_DIR / "test_manifest.json"
GALLERY_MANIFEST_PATH = BASE_DIR / "data" / "gallery" / "gallery_manifest.json"

# Load manifests
if TEST_MANIFEST_PATH.exists():
    with open(TEST_MANIFEST_PATH, "r", encoding="utf-8") as f:
        TEST_CASES = json.load(f)
else:
    TEST_CASES = []

if GALLERY_MANIFEST_PATH.exists():
    with open(GALLERY_MANIFEST_PATH, "r", encoding="utf-8") as f:
        GALLERY_ITEMS = json.load(f)
else:
    GALLERY_ITEMS = []

# Initialize Vector Store
store = CTScanVectorStore(index_dir=BASE_DIR / "data" / "index")


def get_test_case_options():
    options = {}
    for i, case in enumerate(TEST_CASES, start=1):
        patient = case.get("patient_id", case.get("seriesuid", f"Case {i}"))
        finding = case.get("clinical_finding", "")
        label = f"Case #{i:02d}: {patient} | {finding}"
        options[label] = case
    return options


TEST_OPTIONS = get_test_case_options()


def on_select_test_case(case_label: str):
    if not case_label or case_label not in TEST_OPTIONS:
        return None, "กรุณาเลือกเคสทดสอบ"
    case = TEST_OPTIONS[case_label]
    img = Image.open(case["image_path"])
    patient = case.get("patient_id", case.get("seriesuid", "-"))
    info = (
        f"**เคสทดสอบ:** `{case['id']}`\n\n"
        f"- **Patient / Series ID:** `{patient}`\n"
        f"- **ลักษณะทางคลินิก (Finding):** {case.get('clinical_finding', '-')}\n"
        f"- **รายละเอียดรอยโรค (Pathology/Guideline):** {case.get('clinical_guideline', '-')}\n"
        f"- **Malignancy (ระดับความเสี่ยงมะเร็ง):** `{case.get('malignancy', 'N/A')}`\n"
        f"- **Texture (เนื้อเยื่อก้อน):** `{case.get('texture', 'N/A')}`"
    )
    return img, info


def search_similar_scans(query_image, top_k: int = 5):
    if query_image is None:
        return [], "กรุณาเลือกหรืออัปโหลดภาพ CT Scan ก่อนค้นหา"

    temp_path = BASE_DIR / "data" / "temp_query.png"
    query_image.save(temp_path)

    results = store.search(temp_path, top_k=top_k)

    gallery_items = []
    details_md = f"### ผลการค้นหาภาพที่ใกล้เคียงที่สุด K-Top {top_k}\n\n"

    for r in results:
        patient = r.get("patient_id", r.get("seriesuid", "-"))
        caption = (
            f"อันดับ #{r['rank']} ({r['similarity_percent']})\n"
            f"{patient}\n"
            f"{r.get('clinical_finding', '')}"
        )
        gallery_items.append((r["image_path"], caption))

        details_md += (
            f"#### อันดับ #{r['rank']} | ความคล้ายคลึง: **{r['similarity_percent']}**\n"
            f"- **Patient ID:** `{patient}`\n"
            f"- **ลักษณะทางคลินิก:** {r.get('clinical_finding', '-')}\n"
            f"- **รายละเอียด:** {r.get('clinical_guideline', '-')}\n"
            f"- **ระดับความเสี่ยง:** `{r.get('malignancy', 'N/A')}` | **Texture:** `{r.get('texture', 'N/A')}`\n"
            f"---\n"
        )

    return gallery_items, details_md


# --- Tab 2: Gallery Reviewer Functions ---
def get_category_name(item: dict) -> str:
    finding = item.get("clinical_finding", "").lower()
    pid = item.get("patient_id", "").lower()
    if "normal" in finding or "normal" in pid:
        return "Normal (คนปกติ)"
    elif "adeno" in finding or "adeno" in pid:
        return "Adenocarcinoma (มะเร็งต่อม)"
    elif "large.cell" in finding or "large.ce" in pid:
        return "Large Cell Carcinoma (มะเร็งเซลล์ใหญ่)"
    elif "squamous" in finding or "squamous" in pid:
        return "Squamous Cell Carcinoma (มะเร็งสความัส)"
    elif "lidc" in finding or "lidc" in pid:
        return "LIDC Pulmonary Nodules (ก้อนเนื้อในปอด)"
    return "Other (อื่นๆ)"


CATEGORY_CHOICES = [
    "ทั้งหมด (All)",
    "Normal (คนปกติ)",
    "Adenocarcinoma (มะเร็งต่อม)",
    "Large Cell Carcinoma (มะเร็งเซลล์ใหญ่)",
    "Squamous Cell Carcinoma (มะเร็งสความัส)",
    "LIDC Pulmonary Nodules (ก้อนเนื้อในปอด)",
]


def filter_gallery(selected_category: str, page: int = 1, page_size: int = 18):
    filtered = []
    for item in GALLERY_ITEMS:
        cat = get_category_name(item)
        if selected_category == "ทั้งหมด (All)" or selected_category == cat:
            filtered.append(item)

    total_count = len(filtered)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = filtered[start_idx:end_idx]

    gallery_display = []
    for item in page_items:
        patient = item.get("patient_id", "-")
        caption = f"{patient}\n{item.get('clinical_finding', '')}\nTexture: {item.get('texture', '-')}"
        gallery_display.append((item["image_path"], caption))

    summary_text = (
        f"**จำนวนภาพในหมวดนี้:** `{total_count}` ภาพ | "
        f"**หน้า:** `{page} / {total_pages}` | "
        f"**แสดงภาพที่:** `{start_idx + 1} - {min(end_idx, total_count)}`"
    )

    return gallery_display, summary_text, total_pages


# --- Vector Point Inspector ---
INSPECT_OPTIONS = {
    f"#{i+1:04d} | {item['id']} | {item.get('patient_id', '-')} | {item.get('clinical_finding', '-')[:30]}": i
    for i, item in enumerate(GALLERY_ITEMS)
}


def inspect_vector_point(choice_label: str):
    if not choice_label or choice_label not in INSPECT_OPTIONS:
        return None, "{}", "กรุณาเลือกรายการภาพ", "[]", "[]"

    idx = INSPECT_OPTIONS[choice_label]
    item = GALLERY_ITEMS[idx]
    vec = store.embeddings[idx]

    img = Image.open(item["image_path"])

    # Vector Payload document format
    payload = {
        "point_id": item["id"],
        "collection": "lung_ct_scans",
        "vector_size": len(vec),
        "distance": "Cosine",
        "payload": {
            "patient_id": item.get("patient_id", "-"),
            "clinical_finding": item.get("clinical_finding", "-"),
            "clinical_guideline": item.get("clinical_guideline", "-"),
            "malignancy": item.get("malignancy", "-"),
            "texture": item.get("texture", "-"),
            "source_image": Path(item["image_path"]).name,
        },
    }

    norm = float(np.linalg.norm(vec))
    v_min = float(np.min(vec))
    v_max = float(np.max(vec))
    v_mean = float(np.mean(vec))
    non_zeros = int(np.count_nonzero(vec))

    stats_md = (
        f"#### สถิติข้อมูล Vector Point #{idx + 1}\n\n"
        f"- **Vector Index ID:** `{item['id']}`\n"
        f"- **จำนวนมิติ (Dimensions):** `2,048` มิติ\n"
        f"- **ชนิดข้อมูล (Dtype):** `float32`\n"
        f"- **เวกเตอร์ขนาด (L2 Norm):** `{norm:.4f}` *(Unit Sphere / Cosine Ready)*\n"
        f"- **ช่วงค่าตัวเลข (Range):** `[{v_min:.5f}, {v_max:.5f}]`\n"
        f"- **ค่าเฉลี่ย (Mean):** `{v_mean:.6f}`\n"
        f"- **มิติที่มีค่าไม่เป็นศูนย์ (Non-zeros):** `{non_zeros} / 2,048` ({non_zeros/2048*100:.1f}%)"
    )

    # First 30 values sample preview
    preview_30 = [round(float(x), 5) for x in vec[:30]]
    preview_str = f"ตัวอย่าง 30 มิติแรก (จากทั้งหมด 2,048 มิติ):\n{json.dumps(preview_30, indent=2)}\n\n... อีก 2,018 มิติด้านล่าง"

    # Full vector JSON
    full_vector_str = json.dumps([round(float(x), 6) for x in vec.tolist()], indent=2)

    return img, json.dumps(payload, indent=2, ensure_ascii=False), stats_md, preview_str, full_vector_str


def create_app():
    with gr.Blocks(title="Lung CT Scan Vector Search & Reviewer") as demo:
        gr.Markdown("# 🫁 ระบบค้นหา เปรียบเทียบ และสำรวจคลังภาพ CT Scan ปอด (1,000 Scans)")

        with gr.Tabs():
            # --- Tab 1: Search & Compare ---
            with gr.Tab("🔍 1. ค้นหาและเปรียบเทียบภาพ (Vector Search K-Top)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### เลือกภาพ Input (เคสทดสอบ หรือ อัปโหลดเอง)")
                        test_dropdown = gr.Dropdown(
                            choices=list(TEST_OPTIONS.keys()),
                            label="ชุดทดสอบ 10 เคส (มีเคสคนปกติ และเคสโรค)",
                            value=list(TEST_OPTIONS.keys())[0] if TEST_OPTIONS else None,
                        )

                        query_img_display = gr.Image(
                            type="pil",
                            label="ภาพ CT Scan ที่นำเข้า",
                            height=280,
                        )

                        query_info_display = gr.Markdown(label="ข้อมูลประกอบภาพ (Metadata)")

                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1, label="จำนวนภาพเปรียบเทียบ (K-Top)"
                        )
                        search_btn = gr.Button("🔍 ค้นหาภาพเปรียบเทียบในคลังข้อมูล", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### ผลลัพธ์ K-Top จาก Vector Database")
                        result_gallery = gr.Gallery(
                            label="ภาพ CT Scan ที่ใกล้เคียงที่สุด",
                            columns=3,
                            rows=2,
                            height=360,
                            preview=True,
                        )
                        result_details = gr.Markdown()

                test_dropdown.change(
                    fn=on_select_test_case,
                    inputs=[test_dropdown],
                    outputs=[query_img_display, query_info_display],
                )

                search_btn.click(
                    fn=search_similar_scans,
                    inputs=[query_img_display, top_k_slider],
                    outputs=[result_gallery, result_details],
                )

                if TEST_OPTIONS:
                    demo.load(
                        fn=on_select_test_case,
                        inputs=[test_dropdown],
                        outputs=[query_img_display, query_info_display],
                    )

            # --- Tab 2: Gallery Reviewer & Vector Inspector ---
            with gr.Tab("📚 2. สำรวจคลังข้อมูลในระบบ (Dataset Collection Reviewer)"):
                gr.Markdown(
                    "ตรวจสอบภาพทั้งหมด **990 ภาพ** ในคลัง Vector Database "
                    "สามารถกรองดูเฉพาะภาพ **คนปกติ (Normal)** หรือภาพ **โรคมะเร็งแต่ละชนิด** ได้"
                )

                # Section A: Gallery Grid
                with gr.Row():
                    category_filter = gr.Dropdown(
                        choices=CATEGORY_CHOICES,
                        value="ทั้งหมด (All)",
                        label="กรองตามกลุ่มข้อมูล / โรค (Category Filter)",
                        scale=2,
                    )
                    page_slider = gr.Slider(
                        minimum=1,
                        maximum=55,
                        value=1,
                        step=1,
                        label="เลือกหน้า (Page)",
                        scale=2,
                    )
                    refresh_btn = gr.Button("🔄 โหลดข้อมูล", variant="secondary", scale=1)

                gallery_stats_md = gr.Markdown()

                browse_gallery = gr.Gallery(
                    label="คลังภาพ CT Scan ในระบบ",
                    columns=6,
                    rows=3,
                    height=500,
                    preview=True,
                )

                # Interactions for reviewer
                def update_page_max(selected_cat):
                    display_items, summary, max_p = filter_gallery(selected_cat, page=1)
                    return display_items, summary, gr.update(maximum=max_p, value=1)

                category_filter.change(
                    fn=update_page_max,
                    inputs=[category_filter],
                    outputs=[browse_gallery, gallery_stats_md, page_slider],
                )

                page_slider.change(
                    fn=lambda cat, p: filter_gallery(cat, page=p)[:2],
                    inputs=[category_filter, page_slider],
                    outputs=[browse_gallery, gallery_stats_md],
                )

                refresh_btn.click(
                    fn=lambda cat, p: filter_gallery(cat, page=p)[:2],
                    inputs=[category_filter, page_slider],
                    outputs=[browse_gallery, gallery_stats_md],
                )

                demo.load(
                    fn=lambda: filter_gallery("ทั้งหมด (All)", page=1)[:2],
                    outputs=[browse_gallery, gallery_stats_md],
                )

                # Section B: Deep Vector Point Inspector
                gr.Markdown("---")
                gr.Markdown("## 🧬 ส่องข้อมูล Vector & โครงสร้าง Collector (Vector Inspector)")
                gr.Markdown(
                    "ดูโครงสร้างจริงของข้อมูลใน Vector Database: แต่ละภาพถูกจัดเก็บเป็น **Point (ID + 2,048 Float Vector + JSON Payload)** "
                    "เทียบเท่ามาตรฐานการเก็บของ Qdrant / ChromaDB / Milvus"
                )

                inspect_dropdown = gr.Dropdown(
                    choices=list(INSPECT_OPTIONS.keys()),
                    value=list(INSPECT_OPTIONS.keys())[0] if INSPECT_OPTIONS else None,
                    label="เลือกภาพที่ต้องการตรวจสอบโครงสร้าง Vector Point",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        inspect_img = gr.Image(type="pil", label="ภาพ CT Scan ต้นฉบับ", height=240)
                        inspect_stats = gr.Markdown()

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📄 Vector Payload Document (Metadata ที่ผูกกับเวกเตอร์)")
                        inspect_payload_json = gr.Code(language="json", label="Document Payload")

                    with gr.Column(scale=2):
                        gr.Markdown("#### 🔢 ตัวเลข Vector ตัวจริง (2,048 Dimensions - Float32)")
                        inspect_vec_preview = gr.Code(language="json", label="Vector Preview (First 30 Dims)")

                with gr.Accordion("🔍 ดูตัวเลขเวกเตอร์จริงครบทั้ง 2,048 มิติ (Full Dense Vector Array)", open=False):
                    inspect_full_vector = gr.Code(language="json", label="Full 2,048 Float Array")

                # Wire inspector callbacks
                inspect_dropdown.change(
                    fn=inspect_vector_point,
                    inputs=[inspect_dropdown],
                    outputs=[
                        inspect_img,
                        inspect_payload_json,
                        inspect_stats,
                        inspect_vec_preview,
                        inspect_full_vector,
                    ],
                )

                if INSPECT_OPTIONS:
                    demo.load(
                        fn=lambda: inspect_vector_point(list(INSPECT_OPTIONS.keys())[0]),
                        outputs=[
                            inspect_img,
                            inspect_payload_json,
                            inspect_stats,
                            inspect_vec_preview,
                            inspect_full_vector,
                        ],
                    )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
