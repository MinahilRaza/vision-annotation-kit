
import csv
import math
import os
import sys
import cv2
import numpy as np

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def sharpness_score(cv_image):
    """
    Compute a sharpness score for the given image using the variance of the Laplacian.
    Higher scores indicate sharper images.
    """
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def quaternion_to_yaw(q):
    """Convert a quaternion to a yaw angle (in radians).
    Args:        q: A geometry_msgs.msg.Quaternion (w,x,y,z) object.
    Returns:        The yaw angle in radians.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def extract_images_and_odom(bag_path, image_topic, odom_topic, output_dir, sample_rate=15):
    """
    Extracts images from the specified ROS bag, computes their sharpness scores, and saves the sharpest image every `sample_rate` frames along with the corresponding odometry data.
    The extracted images are saved in the output directory, and a CSV file containing the image names and their corresponding poses is also generated.

    Args:
        bag_path (str): Path to the ROS bag file.
        image_topic (str): Name of the image topic to extract.
        odom_topic (str): Name of the odometry topic to extract.
        output_dir (str): Directory where the extracted images and CSV file will be saved.
        sample_rate (int): Number of frames to sample before saving the sharpest image. Default is 15.
    """
    os.makedirs(output_dir, exist_ok=True)

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id='mcap'
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    if image_topic not in type_map or odom_topic not in type_map:
        print("Required topic not found in bag.")
        return

    image_msg_type = get_message(type_map[image_topic])
    odom_msg_type = get_message(type_map[odom_topic])

    odom_buffer = []  # (timestamp, odom_msg)

    buffer_images = []
    buffer_scores = []
    buffer_times = []

    dataset_rows = []  # For CSV

    frame_count = 0
    group_index = 0

    while reader.has_next():
        topic, data, t = reader.read_next()

        # --- Store odometry ---
        if topic == odom_topic:
            odom_msg = deserialize_message(data, odom_msg_type)
            odom_buffer.append((t, odom_msg))
            odom_buffer = odom_buffer[-200:]

        # --- Process images ---
        elif topic == image_topic:
            msg = deserialize_message(data, image_msg_type)

            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                score = sharpness_score(cv_image)

                buffer_images.append(cv_image)
                buffer_scores.append(score)
                buffer_times.append(t)

                frame_count += 1

                if len(buffer_images) == sample_rate:

                    best_idx = int(np.argmax(buffer_scores))
                    best_image = buffer_images[best_idx]
                    best_time = buffer_times[best_idx]

                    # Find closest odometry
                    closest_odom = min(
                        odom_buffer,
                        key=lambda x: abs(x[0] - best_time)
                    )[1]

                    # Extract position
                    x = closest_odom.pose.pose.position.x
                    y = closest_odom.pose.pose.position.y
                    z = closest_odom.pose.pose.position.z

                    # Extract yaw
                    q = closest_odom.pose.pose.orientation
                    yaw = quaternion_to_yaw(q)

                    # Save image
                    image_name = f"{group_index:05d}.png"
                    image_path = os.path.join(output_dir, image_name)
                    cv2.imwrite(image_path, best_image)

                    # Store row for CSV
                    dataset_rows.append([image_name, x, y, z, yaw])

                    print(f"Saved {image_name}")

                    group_index += 1
                    buffer_images.clear()
                    buffer_scores.clear()
                    buffer_times.clear()

            except Exception as e:
                print(f"Failed frame {frame_count}: {e}")

    # --- Write CSV ---
    csv_path = os.path.join(output_dir, "poses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "x", "y", "z", "yaw"])
        writer.writerows(dataset_rows)

    print(f"\nDone. Processed {frame_count} frames.")
    print(f"Saved {group_index} images.")
    print(f"CSV saved to {csv_path}")

