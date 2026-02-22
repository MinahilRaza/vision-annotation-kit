"""
This module defines the BboxAnnotator class, which uses the OWLv2 model to generate bounding box annotations for images based on specified text labels. The class provides methods for annotating individual images as well as processing batches of images from a directory. The generated annotations are saved in a specified directory in a text format that includes the bounding box coordinates, label, and confidence score.
"""

import time
import os
import torch
from PIL import Image

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from annotation_kit.utils.labeling_utils import filter_enclosed_annotations, map_queries_to_classes   

class BboxAnnotator:
    def __init__(self, 
                 model_name="google/owlv2-base-patch16-ensemble",
                 classes_json:str=None):
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)

        assert classes_json is not None, "classes_json must be provided"
        
        self.query_to_class = map_queries_to_classes(classes_json)

    def _annotate(self, image, text_labels, threshold=0.1):

        inputs = self.processor(text=text_labels, images=image, return_tensors="pt")

        start = time.time()
        outputs = self.model(**inputs)
        print(f"Inference time: {time.time() - start} seconds")

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        image_size = (image.height, image.width) if isinstance(image, Image.Image) else (image[0].height, image[0].width)
        target_sizes = torch.tensor([image_size]*len(image)) if isinstance(image, list) else torch.tensor([image_size])

        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        start = time.time()
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=text_labels
        )
        print(f"Post processing time: {time.time() - start} seconds")
        
        all_annotations = []
        for result in results: 
            boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
            annotations = []

            for box, score, text_label in zip(boxes, scores, text_labels):
                box = [round(i, 2) for i in box.tolist()]
                # replace negative values with 0
                box = [max(0, i) for i in box]
                annotations.append({
                    "label": text_label,
                    "confidence": round(score.item(), 3),
                    "box": box
                })
            all_annotations.append(annotations)
        return all_annotations
    
    def _save_annotations(self, all_annotations, annotations_dir, image_names):
        if not os.path.exists(annotations_dir):
            os.makedirs(annotations_dir)

        for image_name, annotations in zip(image_names, all_annotations):
            
            annotation_file = os.path.join(annotations_dir, f"{image_name[:-4]}.txt")
            with open(annotation_file, "w") as f:
                for annotation in annotations:
                    text_label = annotation["label"].replace(" ", "_")
                    score = annotation["confidence"]
                    box = annotation["box"]
                    xmin, ymin, xmax, ymax = box
                    f.write(f"{xmin} {ymin} {xmax} {ymax} {text_label} {score}\n")
            
            # Map queries to classes
            for annotation in annotations:
                if annotation["label"] in self.query_to_class:
                    annotation["label"] = self.query_to_class[annotation["label"]]
            # Apply Non-Maximum Suppression (NMS) to the annotations
            annotations_nms = filter_enclosed_annotations(annotations)
            # save annotations with nms and post processing
            annotation_file_nms = os.path.join(annotations_dir, f"{image_name[:-4]}_nms.txt")
            with open(annotation_file_nms, "w") as f:
                for annotation in annotations_nms:
                    text_label = annotation["label"].replace(" ", "_")
                    score = annotation["confidence"]
                    box = annotation["box"]
                    xmin, ymin, xmax, ymax = box
                    f.write(f"{xmin} {ymin} {xmax} {ymax} {text_label} {score}\n")
        print(f"Annotations saved to {annotations_dir}")

    def generate_annotations(self, image_dir, annotations_dir, batch_size, text_labels, threshold=0.1):
        """Generate bounding box annotations for images in a directory.
        
        Args:
            image_dir (str): Directory containing input images.
            annotations_dir (str): Directory to save output annotation files.
            batch_size (int): Number of images to process in a batch.
            text_labels (list): List of text labels for object detection.
            threshold (float): Confidence threshold for detections.
        """
        image_names = os.listdir(image_dir)
        num_batches = (len(image_names) + batch_size - 1) // batch_size

        for i in range(num_batches):
            images = []
            all_annotations = []
            print(f"Processing batch {i+1}/{num_batches}")
            for image_name in image_names[i * batch_size:(i + 1) * batch_size]:
                image_path = os.path.join(image_dir, image_name)
                images.append(Image.open(image_path))

            all_annotations = self._annotate(images, [text_labels]*len(images), threshold)

            self._save_annotations(all_annotations, annotations_dir, image_names[i * batch_size:(i + 1) * batch_size])
