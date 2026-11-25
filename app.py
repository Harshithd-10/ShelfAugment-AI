
import io
import os
import zipfile
import tempfile
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

CLASS_NAMES = [
    "Coca Cola", "Sprite", "Pepsi",
    "Mountain Dew", "7UP", "Fanta"
]

st.set_page_config(
    page_title="ShelfAugment AI",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {
  padding: 1.4rem 1.6rem; border-radius: 16px;
  background: linear-gradient(120deg,#102a43,#176b87);
  color: white; margin-bottom: 1rem;
}
.hero h1 {color:white; margin:0;}
.hero p {color:#e4f1f7; margin:.4rem 0 0 0;}
.small-note {color: #64748b; font-size: .9rem;}
</style>
<div class="hero">
  <h1>🛒 ShelfAugment AI</h1>
  <p>Retail shelf-image augmentation with YOLO bounding-box preservation.</p>
</div>
""", unsafe_allow_html=True)

def decode_image(upload):
    raw = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a supported image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def parse_yolo(text, class_count=len(CLASS_NAMES)):
    boxes, labels = [], []
    if not text or not text.strip():
        return boxes, labels
    for line_no, line in enumerate(text.strip().splitlines(), 1):
        vals = line.split()
        if len(vals) != 5:
            raise ValueError(f"Annotation line {line_no} must contain 5 values.")
        cls = int(vals[0])
        box = [float(x) for x in vals[1:]]
        if not 0 <= cls < class_count:
            raise ValueError(f"Invalid class ID {cls} on line {line_no}.")
        xc, yc, w, h = box
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized YOLO box on line {line_no}.")
        boxes.append(box)
        labels.append(cls)
    return boxes, labels

def encode_yolo(boxes, labels):
    rows = []
    for cls, box in zip(labels, boxes):
        rows.append(str(int(cls)) + " " + " ".join(f"{float(v):.6f}" for v in box))
    return "\n".join(rows) + ("\n" if rows else "")

def apply_augmentation(image, boxes, labels, mode, brightness, contrast, angle, scale, perspective):
    h, w = image.shape[:2]
    bbox_params = A.BboxParams(
        format="yolo", label_fields=["class_labels"],
        min_visibility=0.30, clip=True, filter_invalid_bboxes=True
    )

    if mode == "Lighting":
        aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-brightness, brightness),
                contrast_limit=(-contrast, contrast), p=1.0
            ),
            A.RandomGamma(gamma_limit=(75, 125), p=0.5)
        ])
        result = aug(image=image)
        out_boxes, out_labels = boxes, labels

    elif mode == "Camera angle":
        aug = A.Compose([
            A.Affine(
                rotate=(-angle, angle),
                scale=(max(0.5, 1-scale), 1+scale),
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114)
            )
        ], bbox_params=bbox_params)
        result = aug(image=image, bboxes=boxes, class_labels=labels)
        out_boxes, out_labels = result["bboxes"], result["class_labels"]

    elif mode == "Perspective":
        aug = A.Compose([
            A.Perspective(
                scale=(0.02, perspective),
                keep_size=True,
                fit_output=False,
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114)
            )
        ], bbox_params=bbox_params)
        result = aug(image=image, bboxes=boxes, class_labels=labels)
        out_boxes, out_labels = result["bboxes"], result["class_labels"]

    else:  # Combined lighting + camera geometry
        aug = A.Compose([
            A.Affine(
                rotate=(-angle, angle),
                scale=(max(0.5, 1-scale), 1+scale),
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114)
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-brightness, brightness),
                contrast_limit=(-contrast, contrast), p=1.0
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.5)
        ], bbox_params=bbox_params)
        result = aug(image=image, bboxes=boxes, class_labels=labels)
        out_boxes, out_labels = result["bboxes"], result["class_labels"]

    return result["image"], list(out_boxes), list(out_labels)

def draw_boxes(image, boxes, labels):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for box, cls in zip(boxes, labels):
        xc, yc, bw, bh = box
        x1, y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
        x2, y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), max(2, w//900))
        name = CLASS_NAMES[int(cls)] if 0 <= int(cls) < len(CLASS_NAMES) else str(cls)
        cv2.putText(canvas, name, (x1, max(18,y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(.45,w/2500), (255,0,0), 2)
    return canvas

def to_png_bytes(image):
    return Image.fromarray(image).convert("RGB").save_to_bytes() if hasattr(Image.Image, "save_to_bytes") else pil_png(image)

def pil_png(image):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()

tabs = st.tabs(["Dashboard", "Image Studio", "Batch Dataset", "About"])

with tabs[0]:
    st.subheader("Project dashboard")
    c1,c2,c3 = st.columns(3)
    c1.metric("Augmentation modes", "4")
    c2.metric("Annotation format", "YOLO")
    c3.metric("Product classes", "6")
    st.markdown("#### Available transformations")
    st.write("- **Lighting:** brightness, contrast and gamma variation.")
    st.write("- **Camera angle:** small affine rotation and scale changes.")
    st.write("- **Perspective:** simulated viewpoint variation.")
    st.write("- **Combined:** geometric and lighting transformations together.")
    st.info("This application prepares augmented images and annotations. It does not train or evaluate a planogram-compliance detector.")

with tabs[1]:
    st.subheader("Single-image augmentation")
    left,right = st.columns([1,1])
    with left:
        image_file = st.file_uploader("Upload a shelf image", type=["jpg","jpeg","png"], key="single_img")
        label_file = st.file_uploader("Optional YOLO annotation (.txt)", type=["txt"], key="single_lbl")
        mode = st.selectbox("Augmentation method", ["Lighting","Camera angle","Perspective","Combined"])
        with st.expander("Augmentation controls", expanded=True):
            brightness = st.slider("Brightness strength", 0.0, 0.5, 0.20, 0.01)
            contrast = st.slider("Contrast strength", 0.0, 0.5, 0.20, 0.01)
            angle = st.slider("Maximum rotation (degrees)", 0, 15, 5)
            scale = st.slider("Scale variation", 0.0, 0.25, 0.05, 0.01)
            perspective = st.slider("Maximum perspective strength", 0.02, 0.10, 0.05, 0.01)
        run = st.button("Generate augmented image", type="primary", use_container_width=True)
    if image_file:
        try:
            original = decode_image(image_file)
            annotation_text = label_file.getvalue().decode("utf-8") if label_file else ""
            boxes, labels = parse_yolo(annotation_text)
            with right:
                st.image(draw_boxes(original, boxes, labels), caption="Original image", use_container_width=True)
                st.caption(f"Image size: {original.shape[1]} × {original.shape[0]} | YOLO boxes: {len(boxes)}")
            if run:
                augmented, new_boxes, new_labels = apply_augmentation(
                    original, boxes, labels, mode, brightness, contrast, angle, scale, perspective
                )
                st.success("Augmentation completed.")
                a,b = st.columns(2)
                a.image(draw_boxes(augmented,new_boxes,new_labels), caption=f"{mode} result", use_container_width=True)
                b.metric("Original boxes", len(boxes))
                b.metric("Retained boxes", len(new_boxes))
                b.metric("Boxes removed by visibility filtering", max(0,len(boxes)-len(new_boxes)))
                st.download_button("Download augmented image", pil_png(augmented),
                    file_name=f"shelfaugment_{mode.lower().replace(' ','_')}.png", mime="image/png")
                st.download_button("Download updated YOLO labels", encode_yolo(new_boxes,new_labels),
                    file_name="augmented_labels.txt", mime="text/plain")
        except Exception as e:
            st.error(f"Could not process this upload: {e}")
    else:
        st.info("Upload an image to begin. YOLO labels are optional; without them, only the image is transformed.")

with tabs[2]:
    st.subheader("Batch dataset augmentation")
    st.write("Upload a ZIP containing an `images/` folder and an optional `labels/` folder. Label files must share image filenames.")
    batch_zip = st.file_uploader("Dataset ZIP", type=["zip"], key="batch_zip")
    batch_modes = st.multiselect("Methods to generate", ["lighting","camera_angle","perspective","combined"],
                                 default=["lighting","camera_angle","combined"])
    batch_brightness = st.slider("Batch brightness/contrast strength", 0.0, 0.5, 0.15, 0.01)
    batch_angle = st.slider("Batch maximum rotation", 0, 15, 5)
    process = st.button("Process batch and create ZIP", type="primary", disabled=batch_zip is None)
    if process and batch_zip:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with zipfile.ZipFile(batch_zip) as zf:
                    for member in zf.infolist():
                        target = (root / member.filename).resolve()
                        if not str(target).startswith(str(root.resolve()) + os.sep):
                            raise ValueError("Unsafe path detected in uploaded ZIP.")
                    zf.extractall(root)
                image_dir = root / "images"
                label_dir = root / "labels"
                if not image_dir.exists():
                    raise ValueError("ZIP must contain an images/ directory at its root.")
                image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
                if not image_paths:
                    raise ValueError("No JPG, JPEG or PNG images found.")
                output = root / "shelfaugment_output"
                rows = []
                progress = st.progress(0)
                for idx, img_path in enumerate(image_paths):
                    image = np.array(Image.open(img_path).convert("RGB"))
                    label_path = label_dir / (img_path.stem + ".txt")
                    label_text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
                    boxes, labels = parse_yolo(label_text)
                    for method in batch_modes:
                        method_name = {"lighting":"Lighting","camera_angle":"Camera angle",
                                       "perspective":"Perspective","combined":"Combined"}[method]
                        aug_img, aug_boxes, aug_labels = apply_augmentation(
                            image, boxes, labels, method_name,
                            batch_brightness, batch_brightness, batch_angle, 0.05, 0.04
                        )
                        out_img_dir = output / method / "images"
                        out_lbl_dir = output / method / "labels"
                        out_img_dir.mkdir(parents=True, exist_ok=True)
                        out_lbl_dir.mkdir(parents=True, exist_ok=True)
                        out_name = f"{img_path.stem}_{method}"
                        Image.fromarray(aug_img).save(out_img_dir / f"{out_name}.jpg", quality=95)
                        (out_lbl_dir / f"{out_name}.txt").write_text(
                            encode_yolo(aug_boxes,aug_labels), encoding="utf-8"
                        )
                        rows.append({"source_image":img_path.name,"augmentation":method,
                                     "original_boxes":len(boxes),"generated_boxes":len(aug_boxes),
                                     "image":f"{method}/images/{out_name}.jpg",
                                     "label":f"{method}/labels/{out_name}.txt"})
                    progress.progress((idx+1)/len(image_paths))
                pd.DataFrame(rows).to_csv(output / "augmentation_metadata.csv", index=False)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer,"w",zipfile.ZIP_DEFLATED) as outzip:
                    for file in output.rglob("*"):
                        if file.is_file():
                            outzip.write(file, file.relative_to(output))
                st.success(f"Processed {len(image_paths)} images and generated {len(rows)} augmented samples.")
                st.download_button("Download augmented dataset ZIP", zip_buffer.getvalue(),
                    file_name="ShelfAugment_AI_batch_output.zip", mime="application/zip")
                st.dataframe(pd.DataFrame(rows).head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Batch processing failed: {e}")

with tabs[3]:
    st.subheader("About ShelfAugment AI")
    st.markdown("""
**Purpose:** create controlled retail shelf-image variations to help build more robust
planogram-compliance training datasets.

**Dataset classes used in the notebook**
""")
    st.table(pd.DataFrame({"Class ID":range(len(CLASS_NAMES)), "Product":CLASS_NAMES}))
    st.markdown("""
**Annotation behavior**
- YOLO labels use `class_id x_center y_center width height`, normalized to 0–1.
- Lighting-only transforms preserve the original boxes.
- Geometric transforms update boxes and filter boxes with insufficient visibility.
- If no annotation file is supplied, the app still transforms the image but cannot preserve or export object labels.
""")
    st.caption("ShelfAugment AI · Streamlit prototype")
