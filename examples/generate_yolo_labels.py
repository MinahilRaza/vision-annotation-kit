"""
generate_yolo_labels.py

This script generates bounding box annotations for images using the OWLv2 model and converts them to YOLO format. 
It processes images in batches, applies Non-Maximum Suppression (NMS) to filter out redundant annotations, and 
saves the final annotations in a specified directory. The script also creates a dataset.yaml file for use with 
YOLO training. 

Usage:
    python generate_yolo_labels.py --batch_size 4 --classes_json ../dataset/classes.json --dataset_dir ../dataset/train --validation_dir ../dataset/val
"""
import argparse
import json
import os

from annotation_kit.annotators.bbox_annotator import BboxAnnotator
from annotation_kit.utils.labeling_utils import label_map
from annotation_kit.utils.editing_utils import generate_and_edit_labels

# take batch size, classes_json, input directory, output directory as arguments
parser = argparse.ArgumentParser(description="Generate YOLO labels using OWLv2")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing images")
parser.add_argument("--classes_json", type=str, default='../dataset/classes.json', help="Path to classes JSON file")
parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing training images. Must contain images in 'images' subdirectory.")
parser.add_argument("--validation_dir", type=str, default=None, help="Directory containing validation images. If not provided, will use the same directory as training images.")
parser.add_argument("--edit_labels", action='store_true', default=False, help="Whether to edit the generated labels using an interactive tool")

if __name__ == "__main__":
    args = parser.parse_args()
    
    # generate and edit labels for training data
    generate_and_edit_labels(args.classes_json, args.dataset_dir, args.batch_size, args.edit_labels)
    annotator = BboxAnnotator(classes_json=args.classes_json)

    if args.validation_dir:
        # generate and edit labels for validation data
        generate_and_edit_labels(args.classes_json, args.validation_dir, args.batch_size, args.edit_labels)
    else:
        # If validation directory is not provided, use the same directory as training images
        val_image_dir = args.dataset_dir + "/images"
        val_coco_annotation_dir = args.dataset_dir + "/coco_annotations"
        val_yolo_annotation_dir = args.dataset_dir + "/labels"

    # Create dataset.yaml file
    dataset_yaml_path = args.dataset_dir + "/dataset.yaml"
    with open(dataset_yaml_path, 'w') as f:
        f.write(f"path: {args.dataset_dir}\n")
        f.write(f"train: images/\n")
        if args.validation_dir:
            f.write(f"val: {val_image_dir}\n")
        else:
            f.write("val: images/\n")

        # Add number of classes to dataset.yaml
        with open(args.classes_json) as d:
            data = json.load(d)
            num_classes = len([cls["name"] for cls in data.values()])
            f.write(f"nc: {num_classes}\n")
    # Add label map to dataset.yaml if the file exists
    dataset_yaml_path = args.dataset_dir + "/dataset.yaml"
    if os.path.exists(dataset_yaml_path):
        label_mapping = label_map(args.classes_json)
        with open(dataset_yaml_path, 'a') as f:
            f.write("names:\n")
            for name, idx in label_mapping.items():
                f.write(f"  {idx}: {name}\n")