#!/usr/bin/env python3

import argparse
import rclpy
from annotation_kit.utils.rosbag_utils import extract_images_and_odom


def main():
    parser = argparse.ArgumentParser(
        description="Extract images and odometry from a ROS2 bag file."
    )

    parser.add_argument(
        "bag_path",
        type=str,
        help="Path to the ROS2 bag file"
    )

    parser.add_argument(
        "image_topic",
        type=str,
        help="Image topic name"
    )

    parser.add_argument(
        "odom_topic",
        type=str,
        help="Odometry topic name"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where extracted data will be saved"
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=15,
        help="Frame sampling rate (default: 5)"
    )

    args = parser.parse_args()

    rclpy.init()
    extract_images_and_odom(
        args.bag_path,
        args.image_topic,
        args.odom_topic,
        args.output_dir,
        args.sample_rate,
    )
    rclpy.shutdown()


if __name__ == "__main__":
    main()