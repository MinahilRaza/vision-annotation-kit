
# Object Detection Dataset Generation with OWLv2

This repository provides a pipeline for:

* Generating object detection labels using **OWLv2**
* Editing annotations interactively
* Converting annotations to YOLO format
* Preparing a dataset for training with **Ultralytics YOLO**

## Overview

The workflow is as follows:

1. Collect images
2. Define detection classes and queries in `classes.json`. See [`dataset/classes.json`](dataset/classes.json) for a sample.
3. Generate labels using OWLv2
4. Edit labels with an interactive tool
5. Convert to YOLO format
6. Split into train/validation and prepare dataset
7. Train with Ultralytics


## Installation

Install the package using:

```bash
pip install -e .
```

Alternatively, you can install the dependencies from the requirements file when not using this as a package:

```bash
pip install -r requirements.txt
```
---

## Usage

### Step 1 — Define Classes

All detection classes are defined in:

```
dataset/classes.json
```

Each class contains **multiple descriptive queries**.

Example:

```json
{
  "car": {
    "queries": [
      "a red car",
      "a blue car",
      "a parked car",
      "a vehicle on the road"
    ]
  }
}
```

#### Why do we use multiple queries?

We use **open-vocabulary detection** with OWLv2.
Providing multiple natural-language descriptions improves detection robustness.

### Step 2 — Generate and Edit Labels with OWLv2

After collecting images, generate bounding boxes using:

```bash
cd examples
python generate_yolo_labels.py --batch_size 4 --classes_json ../dataset/classes.json --dataset_dir ../dataset/train --edit_labels
```
 The directory `../dataset/train` should contain an `images` subfolder.

## Data Collection
The data has been collected in the form of `rosbags`. Each rosbag contains an odom topic (for getting the pose) and an image topic. The image data is collected at 15 Hz. We downsample the data to collect 3 images per second. See [`rosbag_utils.py`](annotation_kit/utils/rosbag_utils.py)
