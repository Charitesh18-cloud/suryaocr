import os
import subprocess
import json
import re
from PIL import Image

from pdf2image import convert_from_path

from surya.layout import LayoutPredictor


def run_surya_layout_on_pdf(pdf_path, output_folder):
    """
    Runs Surya CLI layout tool on a PDF to produce annotated images and JSON.
    """
    os.makedirs(output_folder, exist_ok=True)
    cmd = [
        "surya_layout",
        pdf_path,
        "--images",
        "--output_dir", output_folder
    ]

    print(f"📄 Running Surya layout on: {os.path.basename(pdf_path)}")
    subprocess.run(cmd, check=True)
    print(f"✅ Layout results saved in: {output_folder}")


def crop_clean_segments_with_margin(
    pdf_path,
    results_json_path,
    layout_image_dir,
    output_dir,
    dpi=300,
    margin_x_ratio=0.05,
    margin_y_ratio=0.05
):
    """
    Crop layout segments from PDF pages using CLI results.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(results_json_path, "r") as f:
        layout_data = json.load(f)

    if len(layout_data) != 1:
        raise ValueError("Expected one document in JSON.")

    doc_key = list(layout_data.keys())[0]
    page_data_list = layout_data[doc_key]

    original_images = convert_from_path(pdf_path, dpi=dpi)

    layout_images = [f for f in os.listdir(layout_image_dir) if f.endswith("_layout.png")]

    for page_data in page_data_list:
        page_num = page_data["page"]
        original_img = original_images[page_num - 1].convert("RGB")
        orig_w, orig_h = original_img.size

        matched_layout_file = None
        for fname in layout_images:
            if f"_{page_num - 1}_" in fname or fname.endswith(f"_{page_num - 1}_layout.png"):
                matched_layout_file = fname
                break

        if not matched_layout_file:
            print(f"❌ Layout image for page {page_num - 1} not found.")
            continue

        layout_img = Image.open(os.path.join(layout_image_dir, matched_layout_file))
        layout_w, layout_h = layout_img.size

        scale_x = orig_w / layout_w
        scale_y = orig_h / layout_h

        for idx, box in enumerate(page_data["bboxes"]):
            x1, y1, x2, y2 = map(float, box["bbox"])

            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            width = x2_scaled - x1_scaled
            height = y2_scaled - y1_scaled

            margin_x = margin_x_ratio * width
            margin_y = margin_y_ratio * height

            new_x1 = int(max(0, x1_scaled - margin_x))
            new_y1 = int(max(0, y1_scaled - margin_y))
            new_x2 = int(min(orig_w, x2_scaled + margin_x))
            new_y2 = int(min(orig_h, y2_scaled + margin_y))

            if new_x2 <= new_x1 or new_y2 <= new_y1:
                continue

            cropped = original_img.crop((new_x1, new_y1, new_x2, new_y2))
            out_name = f"{doc_key}_page_{page_num}_box_{idx + 1}.png"
            out_path = os.path.join(output_dir, out_name)
            cropped.save(out_path)
            print(f"✅ Saved: {out_path}")


def run_layout_on_image(image: Image.Image):
    """
    Runs Surya's LayoutPredictor directly on a PIL image.
    """
    predictor = LayoutPredictor()
    result = predictor([image])[0]
    return result


def crop_segments_from_image(image: Image.Image, layout_result, output_dir, margin_ratio=0.05):
    """
    Crop segments from an image using LayoutPredictor result.
    """
    os.makedirs(output_dir, exist_ok=True)
    orig_w, orig_h = image.size

    for idx, bbox in enumerate(layout_result.bboxes):
        x1, y1, x2, y2 = bbox.bbox
        width = x2 - x1
        height = y2 - y1

        margin_x = margin_ratio * width
        margin_y = margin_ratio * height

        new_x1 = int(max(0, x1 - margin_x))
        new_y1 = int(max(0, y1 - margin_y))
        new_x2 = int(min(orig_w, x2 + margin_x))
        new_y2 = int(min(orig_h, y2 + margin_y))

        if new_x2 <= new_x1 or new_y2 <= new_y1:
            continue

        cropped = image.crop((new_x1, new_y1, new_x2, new_y2))
        out_name = f"image_box_{idx + 1}.png"
        out_path = os.path.join(output_dir, out_name)
        cropped.save(out_path)
        print(f"✅ Saved: {out_path}")


def ocr_segments(segment_folders, output_txt_path, recognition, detection):
    """
    Runs OCR on all PNGs inside segment folders.
    Uses pre-loaded predictors.
    """
    full_text = ""

    for folder_path in segment_folders:
        page_id = os.path.basename(folder_path.strip("/"))

        files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

        def extract_box_number(fname):
            match = re.search(r"box_(\d+)", fname)
            return int(match.group(1)) if match else float('inf')

        files.sort(key=extract_box_number)

        full_text += f"\n=== SEGMENTS: {page_id} ===\n\n"

        for fname in files:
            image_path = os.path.join(folder_path, fname)
            img = Image.open(image_path)

            preds = recognition([img], det_predictor=detection)
            segment_text = ""
            for pred in preds:
                for line in pred.text_lines:
                    segment_text += line.text + "\n"

            if not segment_text.strip():
                segment_text = "<No Text Detected>"

            full_text += f"[{fname}]\n{segment_text.strip()}\n\n"

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ Combined OCR saved to: {output_txt_path}")


def run_layout_on_image_file(image: Image.Image, output_dir: str, recognition, detection):
    """
    Tie it all together: layout → crop → OCR for a single image file.
    Uses pre-loaded predictors.
    """
    print(f"🖼️ Running layout & OCR pipeline on image")
    layout_result = run_layout_on_image(image)

    crops_dir = os.path.join(output_dir, "crops")
    crop_segments_from_image(image, layout_result, crops_dir)

    output_txt_path = os.path.join(output_dir, "final_ocr_output.txt")
    ocr_segments([crops_dir], output_txt_path, recognition, detection)

    print(f"✅ Image pipeline done: {output_txt_path}")
