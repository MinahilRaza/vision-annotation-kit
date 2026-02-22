from collections import defaultdict
import json
import os
import random
import shutil
from PIL import Image

def filter_enclosed_annotations(annotations, overlap_threshold=0.9):
    """
    Groups annotations by label and removes those that are ≥ `overlap_threshold`
    enclosed by a larger one with higher confidence (or larger area if confidences tie).

    Parameters
    ----------
    annotations : list[dict]
        Each annotation is:
        {
          "label": str,
          "box": (x_min, y_min, x_max, y_max),
          "confidence": float
        }
    overlap_threshold : float, optional
        Fraction of the smaller box's area that must lie inside the larger one.
        Default is 0.9 (90%).
    """

    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def intersection_area(b1, b2):
        x_left   = max(b1[0], b2[0])
        y_top    = max(b1[1], b2[1])
        x_right  = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        return (x_right - x_left) * (y_bottom - y_top)

    def mostly_enclosed(bigger, smaller):
        """Return True if >= overlap_threshold of `smaller` is inside `bigger`."""
        inter = intersection_area(bigger, smaller)
        return inter / area(smaller) >= overlap_threshold

    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann["label"]].append(ann)

    filtered = []

    for label, anns in grouped.items():
        keep = anns.copy()
        for i, a in enumerate(anns):
            for j, b in enumerate(anns):
                if i == j:
                    continue

                # decide which one is considered bigger based on area
                if area(a["box"]) >= area(b["box"]):
                    big, small = a, b
                else:
                    big, small = b, a

                if mostly_enclosed(big["box"], small["box"]):
                    # Remove the weaker one
                    if big["confidence"] > small["confidence"]:
                        if small in keep:
                            keep.remove(small)
                    elif big["confidence"] < small["confidence"]:
                        if big in keep:
                            keep.remove(big)
                    else:  # confidences equal -> keep larger area
                        if area(big["box"]) >= area(small["box"]):
                            if small in keep:
                                keep.remove(small)
                        else:
                            if big in keep:
                                keep.remove(big)

        filtered.extend(keep)

    return filtered

def convert_box_to_yolo_format(box, image_width, image_height):
    """
    Convert bounding box from (x_min, y_min, x_max, y_max) format to YOLO format
    (x_center, y_center, width, height) normalized by image dimensions.

    Parameters
    ----------
    box : tuple
        A tuple (x_min, y_min, x_max, y_max) representing the bounding box.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple
        A tuple (x_center, y_center, width, height) in YOLO format.
    """
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return (x_center, y_center, width, height)

def train_validation_split(image_dir, annotation_dir, split_ratio=0.8):
    """
    Splits the dataset into training and validation sets based on the given split ratio.
    Ensures that both images and their corresponding annotations are included in the split.
    This creates folders 'train' and 'val' in both image_dir and annotation_dir.

    Parameters
    ----------
    image_dir : str
        Directory containing the images.
    annotation_dir : str
        Directory containing the annotations.
    split_ratio : float, optional
        The ratio of the dataset to be used for training. Default is 0.8 (80% training, 20% validation).

    Returns
    -------
    tuple
        Two lists: (train_image_names, val_image_names)
    """

    # Create train and val directories if they don't exist. Delete if they do to avoid mixing old and new data
    if os.path.exists(os.path.join(image_dir, 'train')):
        shutil.rmtree(os.path.join(image_dir, 'train'))
    if os.path.exists(os.path.join(image_dir, 'val')):
        shutil.rmtree(os.path.join(image_dir, 'val'))
    if os.path.exists(os.path.join(annotation_dir, 'train')):
        shutil.rmtree(os.path.join(annotation_dir, 'train'))
    if os.path.exists(os.path.join(annotation_dir, 'val')):
        shutil.rmtree(os.path.join(annotation_dir, 'val'))

    image_names = [f for f in os.listdir(image_dir)]
    annotation_names = [f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f))]
    
    random.shuffle(image_names)
    split_index = int(len(image_names) * split_ratio)

    train_image_names = image_names[:split_index]
    val_image_names = image_names[split_index:]

    train_image_dir = os.path.join(image_dir, 'train')
    val_image_dir = os.path.join(image_dir, 'val')
    train_annotation_dir = os.path.join(annotation_dir, 'train')
    val_annotation_dir = os.path.join(annotation_dir, 'val')

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_annotation_dir, exist_ok=True)
    os.makedirs(val_annotation_dir, exist_ok=True)

    for image_name in train_image_names:
        shutil.copy(os.path.join(image_dir, image_name), train_image_dir)
        shutil.copy(os.path.join(annotation_dir, image_name[:-4] + '.txt'), train_annotation_dir)

    for image_name in val_image_names:
        shutil.copy(os.path.join(image_dir, image_name), val_image_dir)
        shutil.copy(os.path.join(annotation_dir, image_name[:-4] + '.txt'), val_annotation_dir)

def map_queries_to_classes(classes_json_file):
    """
    Maps each query to its corresponding class based on the provided classes JSON structure.

    Parameters
    ----------
    queries : list of str
        List of query strings to be mapped to classes.
    classes_json : dict
        Dictionary representing the classes and their associated queries.

    Returns
    -------
    dict
        A dictionary mapping each query to its corresponding class label.
    """
    
    with open(classes_json_file) as f:
        data = json.load(f)
        query_to_class_dict = {q : cls["name"] for cls in data.values() for q in cls["queries"]}
    return query_to_class_dict

def label_map(classes_json_file="../dataset/classes.json"):
    """
    Creates a mapping from class names to integer labels based on the classes.json file.

    Returns
    -------
    dict
        A dictionary mapping class names to integer labels.
    """
    with open(classes_json_file) as f:
        data = json.load(f)
        class_names = [cls["name"] for cls in data.values()]
        class_names.sort()
        if "misc" in class_names:
            class_names.remove("misc")
    return {name: idx for idx, name in enumerate(class_names)}

def convert_annotations_to_yolo(annotation_dir, image_dir, output_dir, classes_json_file="../dataset/classes.json"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('_nms.txt')]

    print(f"Converting {len(annotation_files)} annotation files to YOLO format...")
    
    # Get image extension from the first image in the directory
    image_extension = os.path.splitext(os.listdir(image_dir)[0])[1]
    # Get image dimensions from the first image (assuming all images have the same dimensions)
    first_image_path = os.path.join(image_dir, annotation_files[0][:-8] + image_extension)
    with Image.open(first_image_path) as image:
        image_width, image_height = image.size

    for ann_file in annotation_files:
        image_name = ann_file[:-8] + image_extension 
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(annotation_dir, ann_file)
        output_path = os.path.join(output_dir, image_name[:-4] + '.txt')
        print(f"Processing annotation: {output_path}")
        print(f"Corresponding image: {image_path}")

        if not os.path.exists(image_path):
            continue  # Skip if corresponding image does not exist

        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        with open(output_path, 'w') as f_out:
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue  # Skip malformed lines

                xmin, ymin, xmax, ymax, label, _ = parts
                xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])

                yolo_box = convert_box_to_yolo_format((xmin, ymin, xmax, ymax), image_width, image_height)

                label_mapping = label_map(classes_json_file)
                # Map label to index
                if label in label_mapping:
                    label = label_mapping[label]
                else:
                    continue  # Skip unknown labels
                yolo_line = f"{label} {' '.join(map(str, yolo_box))}\n"
                f_out.write(yolo_line)