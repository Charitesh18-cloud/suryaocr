import io
import os
import tempfile
import shutil

import pypdfium2
import streamlit as st
from PIL import Image, ImageDraw

from surya.models import load_predictors
from surya.common.surya.schema import TaskNames
from surya.debug.draw import draw_polys_on_image, draw_bboxes_on_image
from surya.debug.text import draw_text_on_image
from surya.settings import settings
from surya.scripts import pipeline


# === Load Surya predictors ===
@st.cache_resource()
def load_predictors_cached():
    return load_predictors()

predictors = load_predictors_cached()
recognition = predictors["recognition"]
detection = predictors["detection"]

st.set_page_config(layout="wide")

# === Custom CSS ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.title-box {
    background-color: #000000;
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1.5rem;
}

.sidebar-title {
    background-color: #000000 !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 1.5rem !important;
    padding: 0.8rem 1rem !important;
    border-radius: 8px !important;
    text-align: center !important;
    margin-bottom: 1rem !important;
}

.main-box {
    background-color: #f0f0f0;
    padding: 1.75rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}

.main-box ul {
    padding-left: 1.8rem;
    font-size: 1.4rem;
}

.main-box li {
    margin: 0.9rem 0;
    font-weight: 600;
}

hr.sidebar-line {
    border: none;
    border-top: 2px solid #666;
    width: 60%;
    margin: 20px auto;
}

.sidebar-buttons-box {
    background: white;
    border-radius: 12px;
    padding: 1.5rem 1rem;
    margin: 1rem auto;
    width: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border: 1px solid #ddd;
}

.sidebar-buttons-box button {
    display: block !important;
    width: 90% !important;
    margin: 10px auto !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    background: #f9f9f9 !important;
    color: #333 !important;
}

.sidebar-buttons-box button:hover {
    background: #eee !important;
}

section[data-testid="stSidebar"] {
    background-color: #f0f0f0 !important;
    padding: 2rem 1rem !important;
    border-radius: 0 20px 20px 0 !important;
    border-right: 5px double #000 !important;
    box-shadow: 6px 0 16px rgba(0,0,0,0.1);
}

section[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #003366 !important;
    color: white !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    padding: 1rem 1.5rem !important;
    width: 100% !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] .stFileUploader {
    display: flex !important;
    flex-direction: column;
    align-items: center;
    width: 95% !important;
    margin: 0 auto !important;
}

section[data-testid="stSidebar"] input[type="file"] + div button {
    background-color: #003366 !important;
    color: white !important;
    width: 100% !important;
    font-size: 1.3rem !important;
    font-weight: 400 !important;
}

section[data-testid="stSidebar"] button:hover {
    opacity: 0.95;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

textarea {
    font-size: 1.4rem !important;
}
</style>
""", unsafe_allow_html=True)

# === Title ===
st.markdown("""
<div class="title-box">
    <h1>📝 OCR DIGITIZATION PLATFORM 🚀</h1>
</div>
""", unsafe_allow_html=True)

# === Instructions ===
st.markdown("""
<div class="main-box">
    <ul>
        <li>🔍 Run Text Detection — Detects and highlights text regions.</li>
        <li>🗂️ Run Layout Analysis — Finds sections like paragraphs, tables, headings.</li>
        <li>✏️ Run OCR — Extracts text from the detected regions.</li>
        <li>📊 Run Table Recognition — Recognizes table structures.</li>
        <li>✅ Check PDF Text Quality — Checks if PDF already has good text.</li>
        <li>🚀 <strong style="color: #003366;">Run Full Pipeline — Runs all steps in sequence.</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.markdown('<div class="sidebar-title">📂 SIDEBAR</div>', unsafe_allow_html=True)

# File uploader
in_files = st.sidebar.file_uploader(
    "**📄 Upload PDF(s) or image(s)**\n_Limit: 200MB per file_",
    type=["pdf", "png", "jpg", "jpeg", "gif", "webp"],
    accept_multiple_files=True
)

# First line: between uploader and Run Full Pipeline
st.sidebar.markdown('<hr class="sidebar-line">', unsafe_allow_html=True)

# Run Full Pipeline button
run_full_pipeline = st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

# Second line: between Full Pipeline and 6 buttons
st.sidebar.markdown('<hr class="sidebar-line">', unsafe_allow_html=True)

# Only render white box if needed
run_text_det = run_layout_det = run_text_rec = run_table_rec = run_ocr_errors = help_btn = False

with st.sidebar:
    with st.container():
        run_text_det = st.button("🔍 Run Text Detection", use_container_width=True)
        run_layout_det = st.button("🗂️ Run Layout Analysis", use_container_width=True)
        run_text_rec = st.button("✏️ Run OCR", use_container_width=True)
        run_table_rec = st.button("📊 Run Table Recognition", use_container_width=True)
        run_ocr_errors = st.button("✅ Check PDF Text Quality", use_container_width=True)
        help_btn = st.button("ℹ️ Help", use_container_width=True)

# === Help info ===
if help_btn:
    st.sidebar.info("""
**ℹ️ OCR Digitization Help**  
• 📄 Upload PDFs or images  
• 🔘 Use buttons to run detection/recognition  
• 📑 Outputs appear next to your image  
• 📥 Download extracted text easily!
""")

# === Options ===
skip_table_detection = st.sidebar.checkbox("⏭️ Skip Table Detection")
skip_text_detection = st.sidebar.checkbox("⏭️ Skip Text Detection")
recognize_math = st.sidebar.checkbox("➗ Recognize Math", value=True)
ocr_with_boxes = st.sidebar.checkbox("🔲 OCR with Boxes", value=True)

if not in_files:
    st.stop()

filetype = in_files[0].type

def open_pdf(pdf_file):
    return pypdfium2.PdfDocument(io.BytesIO(pdf_file.getvalue()))

@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_file)
    renderer = doc.render(pypdfium2.PdfBitmap.to_pil, page_indices=[page_num - 1], scale=dpi / 72)
    img = list(renderer)[0].convert("RGB")
    doc.close()
    return img

@st.cache_data()
def page_counter(pdf_file):
    doc = open_pdf(pdf_file)
    count = len(doc)
    doc.close()
    return count

if "pdf" in filetype:
    page_count = page_counter(in_files[0])
    process_full_pdf = st.sidebar.checkbox("📚 Process entire PDF", value=False)
    if not process_full_pdf:
        page_number = st.sidebar.number_input("Page number", 1, page_count, 1)
        pil_images = [get_page_image(in_files[0], page_number)]
    else:
        pil_images = [get_page_image(in_files[0], p+1) for p in range(page_count)]
else:
    pil_images = [Image.open(f).convert("RGB") for f in in_files]

def text_detection(img):
    pred = predictors["detection"]([img])[0]
    return draw_polys_on_image([p.polygon for p in pred.bboxes], img.copy()), pred

def layout_detection(img):
    pred = predictors["layout"]([img])[0]
    return draw_polys_on_image(
        [p.polygon for p in pred.bboxes],
        img.copy(),
        labels=[f"{p.label}-{round(p.top_k[p.label], 2)}" for p in pred.bboxes],
        label_font_size=18
    ), pred

def ocr(img, skip_det, math, with_boxes):
    bbs = [[[0, 0, img.width, img.height]]] if skip_det else None
    tasks = [TaskNames.ocr_with_boxes] if with_boxes else [TaskNames.ocr_without_boxes]
    pred = predictors["recognition"]([img], task_names=tasks, bboxes=bbs, det_predictor=predictors["detection"], highres_images=[img], math_mode=math, return_words=True)[0]
    rec_img = draw_text_on_image([l.bbox for l in pred.text_lines], [l.text for l in pred.text_lines], img.size)
    word_img = img.copy()
    draw = ImageDraw.Draw(word_img)
    for line in pred.text_lines:
        for word in line.words:
            draw.rectangle(word.bbox, outline="red", width=2)
    return rec_img, pred, word_img

def ocr_errors(pdf_file, page_count, sample_len=512, max_samples=10, max_pages=15):
    from pdftext.extraction import plain_text_output
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        f.seek(0)
        page_middle = page_count // 2
        page_range = range(max(page_middle - max_pages, 0), min(page_middle + max_pages, page_count))
        text = plain_text_output(f.name, page_range=page_range)
    gap = len(text) // max_samples if text else 0
    if not text or gap == 0:
        return "PDF has no or very little text", ["no text"]
    gap = max(gap, sample_len)
    samples = [text[i:i+sample_len] for i in range(0, len(text), gap)]
    results = predictors["ocr_error"](samples)
    label = "PDF has good text."
    if results.labels.count("bad") / len(results.labels) > 0.2:
        label = "PDF may have garbled OCR text."
    return label, results.labels

for idx, img in enumerate(pil_images):
    with st.expander(f"📄 Page {idx+1} Output", expanded=True):
        col1, col2 = st.columns([0.5, 0.5])

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            if run_full_pipeline:
                status_placeholder = st.empty()
                with status_placeholder.status(f"🔄 Running Full Pipeline on Page {idx+1}...", expanded=True) as status:
                    tmp_dir = tempfile.gettempdir()
                    output_dir = os.path.join(tmp_dir, f"surya_image_pipeline_{idx}")
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)
                    os.makedirs(output_dir, exist_ok=True)

                    status.write(f"🔍 Running Layout + OCR...")
                    pipeline.run_layout_on_image_file(img, output_dir, recognition, detection)

                    output_file = os.path.join(output_dir, "final_ocr_output.txt")
                    with open(output_file, "r", encoding="utf-8") as f:
                        ocr_text = f.read()

                    status.update(label="✅ Full Pipeline Done!", state="complete")

                st.text_area("Final OCR Output", value=ocr_text, height=800, label_visibility="collapsed")
                st.download_button(
                    "⬇️ Download Final OCR Text",
                    data=ocr_text,
                    file_name=f"surya_pipeline_result_{idx+1}.txt",
                    mime="text/plain"
                )

            if run_text_det:
                result_img, pred = text_detection(img)
                st.image(result_img, caption="Detected Text", use_container_width=True)
                st.json(pred.model_dump())

            if run_layout_det:
                result_img, pred = layout_detection(img)
                st.image(result_img, caption="Layout Detection", use_container_width=True)
                st.json(pred.model_dump())

            if run_text_rec:
                result_img, pred, word_img = ocr(img, skip_text_detection, recognize_math, ocr_with_boxes)
                st.image(result_img, caption="OCR Result", use_container_width=True)
                st.json(pred.model_dump())
                st.image(word_img, caption="Word Boxes", use_container_width=True)

            if run_table_rec:
                result_img, preds = layout_detection(img)
                st.image(result_img, caption="Table Recognition", use_container_width=True)
                st.json([p.model_dump() for p in preds])

if run_ocr_errors:
    if "pdf" not in filetype:
        st.error("❌ This is only for PDFs.")
    else:
        label, results = ocr_errors(in_files[0], page_counter(in_files[0]))
        st.write(label)
        st.json(results)
