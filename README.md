# 🫁 ระบบค้นหาและเปรียบเทียบภาพ CT Scan โรคปอด (K-Top 5 Vector Database)

ระบบค้นหาและเปรียบเทียบภาพเอกซเรย์คอมพิวเตอร์ทรวงอก (Chest CT Scan) ที่มีลักษณะทางกายวิภาคและรอยโรคใกล้เคียงกัน โดยใช้ **AI Vision Feature Extractor** ร่วมกับ **Vector Database Engine (Cosine Distance)** พร้อมคลังภาพทางการแพทย์ 1,000 เคส และชุดทดสอบจริง 10 เคส

---

## 🚀 วิธีเปิดใช้งานโปรแกรม

### วิธีที่ 1: ดับเบิลคลิกไฟล์ `run.bat` (แนะนำสุด - UV Mode)
* ดับเบิลคลิกที่ไฟล์ **`run.bat`** ในโฟลเดอร์นี้
* ระบบจะตรวจจับ `uv` และเปิดเซิร์ฟเวอร์ให้อัตโนมัติทันที

### วิธีที่ 2: รันผ่าน Terminal (Command Line)
```bash
# รันผ่าน uv (เร็วและจัดการ environment อัตโนมัติ)
uv run python -m src.ui

# หรือรันผ่าน python ปกติ
python -m src.ui
```
เปิดใช้งานผ่านเบราว์เซอร์ได้ที่: **`http://127.0.0.1:7860`**

![หน้าต่าง Terminal ขณะรันโปรแกรมด้วย UV Mode](ss/Screenshot%202026-10-28%20154751.png)

---

## 📂 แหล่งที่มาของ Dataset (URLs & Credibility)

ข้อมูลภาพ CT Scan ทั้งหมด **1,000 รายการ** ในระบบนี้ มาจากแหล่งข้อมูลทางการแพทย์ระดับสากลที่มีการตรวจสอบความถูกต้อง สามารถตรวจสอบและดาวน์โหลดได้จากลิงก์ทางการดังนี้:

### 1. LIDC-IDRI (The Lung Image Database Consortium)
* **URL ทางการ (TCIA):** [https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
* **Collection Portal:** [https://www.cancerimagingarchive.net/collection/lidc-idri/](https://www.cancerimagingarchive.net/collection/lidc-idri/)
* **DOI:** [https://doi.org/10.7937/K9/TCIA.2015.LO9UZMDG](https://doi.org/10.7937/K9/TCIA.2015.LO9UZMDG)
* **Hugging Face Mirror (Parquet Patches):** [https://huggingface.co/datasets/ykeselman/lidc-idri-patches](https://huggingface.co/datasets/ykeselman/lidc-idri-patches)
* **ผู้จัดทำและสนับสนุน:** National Cancer Institute (NCI), National Institutes of Health (NIH) สหรัฐอเมริกา
* **ความน่าเชื่อถือ:** **มาตรฐานระดับโลก (Gold Standard)** ในงานวิจัย CT ปอดทั่วโลก มีการอ้างอิงในงานวิจัยวิชาการกว่า 2,000+ ฉบับ
* **ผู้กำกับข้อมูล (Ground Truth Label):** ผ่านการตรวจและลงคะแนนโดย **รังสีแพทย์ทรวงอกผู้เชี่ยวชาญ 4 ท่าน (4 Academic Thoracic Radiologists)** ในกระบวนการอ่านภาพ 2 ขั้นตอน (Two-phase consensus):
  * **Malignancy Score:** ระดับความเสี่ยงมะเร็ง (1 = Highly Unlikely ถึง 5 = Highly Suspicious)
  * **Texture:** ความหนาแน่นเนื้อเยื่อ (Solid, Part-Solid, Non-Solid / Ground-Glass Opacity)
  * **Spiculation & Margin:** ความคมชัดและลักษณะขอบก้อนเนื้อ

### 2. LUNA16 Challenge (Subset ของ LIDC-IDRI)
* **URL ทางการ:** [https://luna16.grand-challenge.org/](https://luna16.grand-challenge.org/)
* **หน้าดาวน์โหลด Zenodo:** [https://zenodo.org/records/2595813](https://zenodo.org/records/2595813) (Part 1) และ [https://zenodo.org/records/2596478](https://zenodo.org/records/2596478) (Part 2)
* **คำอธิบาย:** ชุดข้อมูลย่อยของ LIDC-IDRI ที่คัดเฉพาะก้อนเนื้อ (Nodule >= 3mm) ที่รังสีแพทย์อย่างน้อย 3 ใน 4 ท่านลงคะแนนตรงกัน เพื่อใช้แข่งขันการตรวจจับมะเร็งปอดระดับนานาชาติ

### 3. Chest CT-Scan Dataset (คลาสคนปกติ และ ชนิดมะเร็งปอด)
* **URL ทางการบน Hugging Face:** [https://huggingface.co/datasets/Mahadih534/Chest_CT-Scan_images-Dataset](https://huggingface.co/datasets/Mahadih534/Chest_CT-Scan_images-Dataset)
* **URL ต้นฉบับบน Kaggle:** [https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
* **ความน่าเชื่อถือ:** รวบรวมจากโรงพยาบาลและสถาบันวิจัยมะเร็ง โดยมีภาพที่ยืนยันผลตรวจชิ้นเนื้อพยาธิวิทยา (Biopsy / Histopathological Confirmation)
* **กลุ่มข้อมูลคนปกติ (Normal Healthy Lungs):** มีภาพ CT ทรวงอกของผู้ตรวจสุขภาพปกติที่ไม่มีรอยโรค (True Negative) จำนวน 54 เคส ช่วยให้ระบบสามารถทดสอบเปรียบเทียบระหว่าง **"คนปกติ"** กับ **"คนมีโรค"** ได้จริง

---

## 📊 การแบ่งชุดข้อมูล (Dataset Split)

| ส่วนข้อมูล | จำนวน | วัตถุประสงค์ |
| :--- | :---: | :--- |
| **คลังภาพอ้างอิง (Gallery)** | **990 เคส** | แปลงเป็นเวกเตอร์ 2,048 มิติ บันทึกใน Vector DB สำหรับค้นหา |
| **ชุดทดสอบจริง (Holdout Tests)** | **10 เคส** | แยกออกมาเด็ดขาด ไม่นำเข้าคลัง เพื่อใช้จำลองเป็นภาพคนไข้ใหม่ |

### รายละเอียด 10 เคสทดสอบ (Holdout Test Cases)
1. **Case #01 - #03:** ภาพ CT Scan ของ **คนปกติ (Normal)** ไม่พบก้อนเนื้อ
2. **Case #04 - #07:** ภาพ CT Scan ของ **มะเร็งปอดชนิดต่อม (Adenocarcinoma)**
3. **Case #08 - #10:** ภาพ CT Scan ของ **ก้อนเนื้อในปอด (LIDC Pulmonary Nodules)**

---

## ⚙️ กระบวนการทำงานของระบบ (System Pipeline)

```text
[ ภาพ CT Scan ใหม่ ] 
        │
        ▼
[ 1. Image Preprocessing ] 
  - ปรับความเปรียบต่างเนื้อเยื่อปอด (Lung Window: Center -600 HU, Width 1500 HU)
  - ปรับขนาดเข้าสู่มาตรฐานโมเดล (224 × 224 pixels RGB)
        │
        ▼
[ 2. AI Feature Extraction (Embedding Model) ]
  - ประมวลผลผ่านโมเดล Qdrant ResNet-50 ONNX Vision Encoder
  - สกัดออกมาเป็นเวกเตอร์ตัวเลขทศนิยม 2,048 มิติ (Vector Feature)
  - ทำ L2-Normalization ให้อยู่บน Unit Sphere
        │
        ▼
[ 3. Vector Database Search (Cosine Similarity) ]
  - คำนวณ Dot Product กับ 990 เวกเตอร์ในคลังข้อมูลพร้อมกัน
  - ใช้เวลาค้นหาเพียง ~1 millisecond (0.001 วินาที)
        │
        ▼
[ 4. ผลลัพธ์ K-Top 5 ]
  - ดึง 5 ภาพที่มีองศาเวกเตอร์ใกล้เคียงที่สุด
  - แสดงภาพคู่ขนาน, ค่าความคล้ายคลึง (%), ชนิดของโรค, และคำแนะนำทางคลินิก
```

---

## 🗄️ การจัดเก็บข้อมูลใน Vector Database

ข้อมูล Vector Index ถูกบันทึกไว้ในโฟลเดอร์ `data/index/`:
1. **`embeddings.npy` (7.8 MB):** เมทริกซ์เวกเตอร์ขนาด `(990, 2048)` ชนิดข้อมูล `float32`
2. **`metadata.json` (515 KB):** เอกสาร Payload กำกับแต่ละเวกเตอร์ ตามมาตรฐานการจัดเก็บแบบ Point Document:
```json
{
  "point_id": "gal_0001",
  "collection": "lung_ct_scans",
  "vector_size": 2048,
  "distance": "Cosine",
  "payload": {
    "patient_id": "ChestCT_normal_175",
    "clinical_finding": "Normal (คนปกติ แข็งแรง)",
    "malignancy": "None (Normal)",
    "texture": "Clear Lung",
    "source_image": "gal_0001_ChestCT_normal_175.png"
  }
}
```

---

## 🖥️ ฟังก์ชันในหน้า Web Interface

### 🔍 แท็บที่ 1: ค้นหาและเปรียบเทียบภาพ (Vector Search K-Top)
* เลือกทดสอบจาก 10 เคสที่กันไว้ (มีทั้งคนปกติและผู้ป่วย) หรืออัปโหลดรูปภาพ CT เอง
* ปรับเลือกจำนวนภาพเปรียบเทียบ K-Top (1 ถึง 10 ภาพ)
* แสดงผลการจับคู่ พร้อมเปอร์เซ็นต์ความคล้ายคลึงและผลวินิจฉัยของแพทย์

#### ตัวอย่างที่ 1: ทดสอบค้นหาด้วยภาพปอดคนปกติ (Normal Case #03)
> เมื่อนำภาพคนปกติเข้าสู่ระบบ ระบบค้นพบภาพคนปกติในคลัง (K-Top 5) ด้วยคะแนนความคล้ายคลึงสูงถึง **89.3%**
![ผลการค้นหาเปรียบเทียบภาพคนปกติ](ss/Screenshot%202026-10-28%20154823.png)

#### ตัวอย่างที่ 2: ทดสอบค้นหาด้วยภาพมะเร็งปอด (Adenocarcinoma Case #08)
> เมื่อนำภาพมะเร็งปอดเข้าสู่ระบบ ระบบค้นพบภาพมะเร็งชนิดเดียวกันในคลัง ด้วยคะแนนความคล้ายคลึง **82.4%**
![ผลการค้นหาเปรียบเทียบภาพมะเร็งปอด](ss/Screenshot%202026-10-28%20154837.png)

---

### 📚 แท็บที่ 2: สำรวจคลังข้อมูล & Vector Inspector
* **Dataset Reviewer:** กรองดูภาพตามประเภทโรค: Normal (คนปกติ), Adenocarcinoma, Large Cell, Squamous, หรือ LIDC Nodules พร้อม Pagination เลื่อนดูภาพทั้งหมด 990 ภาพ
* **Vector Inspector:** ส่องดูตัวเลขเวกเตอร์จริง 2,048 มิติ (Float32 Array) พร้อมสถิติ L2 Norm และโครงสร้างเอกสาร JSON Metadata Payload

![ส่องดูโครงสร้าง Vector Point 2048 มิติ และ Payload](ss/Screenshot%202026-10-28%20154846.png)

---

## 📁 โครงสร้างไฟล์ในโปรเจกต์

```text
D:\test\dataset_ctscan\
├── run.bat                  # สคริปต์คลิกเดียวรันโปรแกรมผ่าน uv
├── pyproject.toml           # การตั้งค่า Dependencies ของโปรเจกต์
├── README.md                # เอกสารคู่มือ แหล่งอ้างอิง และ URLs ของ Dataset
├── data/
│   ├── gallery/             # คลังภาพ CT Scan 990 ภาพ
│   ├── test_cases/          # ภาพทดสอบ 10 เคสที่แยกไว้
│   ├── index/               # Vector Database Index (embeddings.npy + metadata.json)
│   ├── chest_ct_1000/       # ข้อมูลดิบ Chest CT (Normal + Cancer)
│   └── hf_cache/            # ข้อมูล LIDC-IDRI Parquet
├── src/
│   ├── build_1000_dataset.py# สคริปต์จัดสรรและรวบรวม 1,000 เคส
│   ├── vector_store.py      # เอนจินสกัดเวกเตอร์และค้นหา Cosine Similarity
│   ├── mhd_reader.py        # Parser อ่านไฟล์ 3D CT MHD/RAW
│   └── ui.py                # Gradio Web Interface ทั้งระบบ
└── test_holdout_cli.py      # สคริปต์รันเทสต์ 10 เคสผ่าน Command Line
```
