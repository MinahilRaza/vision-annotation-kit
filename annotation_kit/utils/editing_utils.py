import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from annotation_kit.annotators.bbox_annotator import BboxAnnotator
from annotation_kit.utils.labeling_utils import convert_annotations_to_yolo

def edit_labels(image_file, annotation_file, classes_json="dataset/classes.json"):
    """
    Interactive bounding box editor.

    Controls:
    - Click twice to draw a box
    - Press number key to select class
    - Press 'd' then click inside box to delete
    - Press 'z' to save
    """

    # ---- Load class labels ----
    with open(classes_json) as f:
        data = json.load(f)
        text_labels = [cls["name"] for cls in data.values()]

    if not text_labels:
        text_labels = ["object"]

    # ---- Build the text block ----
    instructions = """
    Interactive Bounding Box Annotator
    ----------------------------------
    Instructions:
    - Press number key to select class.
    - Click twice to draw a bounding box.
    - Press 'z' to save annotations.
    - Press 'd' and click inside box to delete.
    """

    class_text = "\nAvailable classes:\n"
    for i, cls in enumerate(text_labels):
        class_text += f"{i}: {cls}\n"

    full_text = instructions + class_text

    print(full_text)
    # Load image
    image = Image.open(image_file)

    # Load classes
    with open(classes_json) as f:
        data = json.load(f)
        text_labels = [cls["name"] for cls in data.values()]

    if not text_labels:
        text_labels = ["object"]

    print("\nAvailable classes:")
    for i, cls in enumerate(text_labels):
        print(f"{i}: {cls}")

    current_label = text_labels[0]

    # Colors
    colors = [
        "#440154", "#570707", "#6057BD", "#3817CB",
        "#3f8888", "#034d11", "#07290d", "#7ac78d",
        "#23888e", "#82981f"
    ]
    label_colors = {
        text_labels[i]: colors[i % len(colors)]
        for i in range(len(text_labels))
    }

    # Load existing annotations
    boxes = []

    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            for line in f.readlines():
                x1, y1, x2, y2, label, conf = line.strip().split()
                boxes.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "label": label,
                    "conf": conf
                })
    else:
        print("No existing annotations found. Starting with an empty annotation set.")
               

    # Setup figure
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(4, 1)
    ax = fig.add_subplot(gs[:3, 0])
    ax.imshow(image)
    plt.axis("off")

    ax_text = fig.add_subplot(gs[3, 0])
    ax_text.axis("off")
    # Left column (Instructions)
    ax_text.text(
        0.02, 0.95,
        instructions,
        ha="left",
        va="top",
        family="monospace",
        fontsize=10
    )

    # Right column (Classes)
    ax_text.text(
        0.55, 0.95,
        class_text,
        ha="left",
        va="top",
        family="monospace",
        fontsize=10
    )

    drawing = False
    start_point = None
    delete_mode = False

    rect_patches = []
    label_patches = []

    def redraw_boxes():
        nonlocal rect_patches, label_patches

        for p in rect_patches + label_patches:
            p.remove()

        rect_patches.clear()
        label_patches.clear()

        for b in boxes:
            color = label_colors.get(b["label"], "red")

            rect = patches.Rectangle(
                (b["x1"], b["y1"]),
                b["x2"] - b["x1"],
                b["y2"] - b["y1"],
                linewidth=2,
                edgecolor=color,
                facecolor="none"
            )
            ax.add_patch(rect)
            rect_patches.append(rect)

            label_y = b["y1"] - 20 if b["y1"] > 25 else b["y1"] + 5

            label_bg = patches.Rectangle(
                (b["x1"], label_y),
                60,
                18,
                linewidth=0,
                edgecolor=color,
                facecolor=color
            )
            ax.add_patch(label_bg)
            label_patches.append(label_bg)

            ax.text(
                b["x1"] + 2,
                label_y + 14,
                b["label"],
                color="white",
                fontsize=8
            )

        ax.set_title(f"Current class: {current_label}")
        fig.canvas.draw()

    def on_click(event):
        nonlocal drawing, start_point, delete_mode

        if event.inaxes != ax:
            return

        x, y = event.xdata, event.ydata

        if delete_mode:
            for b in boxes[:]:
                if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
                    boxes.remove(b)
            delete_mode = False
            redraw_boxes()
            return

        if not drawing:
            drawing = True
            start_point = (x, y)
            return

        # Finish box
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y

        boxes.append({
            "x1": round(min(x1, x2), 2),
            "y1": round(min(y1, y2), 2),
            "x2": round(max(x1, x2), 2),
            "y2": round(max(y1, y2), 2),
            "label": current_label,
            "conf": "1.0"
        })

        redraw_boxes()

    def on_key(event):
        nonlocal current_label, delete_mode

        if event.key == "z":
            with open(annotation_file, "w") as f:
                for b in boxes:
                    f.write(
                        f'{b["x1"]} {b["y1"]} {b["x2"]} {b["y2"]} '
                        f'{b["label"]} {b["conf"]}\n'
                    )
            print("Saved annotations to:", annotation_file)

        elif event.key == "d":
            delete_mode = True
            print("Delete mode activated. Click inside box to remove.")
            ax.set_title("Delete mode: Click inside box to delete.")
            fig.canvas.draw()

        elif event.key and event.key.isdigit():
            idx = int(event.key)
            if idx < len(text_labels):
                current_label = text_labels[idx]
                print("Selected class:", current_label)
                redraw_boxes()
                fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw_boxes()
    plt.show()
    
def generate_and_edit_labels(classes_json, dataset_dir, batch_size=4, allow_editing=True):
    annotator = BboxAnnotator(classes_json=classes_json)

    # read text labels from classes.json
    with open(classes_json) as f:
        data = json.load(f)
        text_labels = [q for cls in data.values() for q in cls["queries"]]

    image_dir = dataset_dir + "/images"
    yolo_annotations_dir = dataset_dir + "/labels"
    coco_annotation_dir = dataset_dir + "/coco_annotations"
    # get image type from 1st image in directory
    img_extension = os.path.splitext(os.listdir(image_dir)[0])[1]

    annotator.generate_annotations(image_dir, coco_annotation_dir, batch_size, text_labels, threshold=0.21)

    if allow_editing:
        # for each annotation file, open the corresponding image and annotation in the interactive editor
        annotation_files = [f for f in os.listdir(coco_annotation_dir) if f.endswith('_nms.txt')]
        for ann_file in annotation_files:
            image_file = os.path.join(image_dir, ann_file[:-8] + img_extension)
            annotation_file = os.path.join(coco_annotation_dir, ann_file)
            edit_labels(image_file, annotation_file, classes_json)

    # Convert annotations to YOLO format
    convert_annotations_to_yolo(coco_annotation_dir, image_dir, yolo_annotations_dir, classes_json)