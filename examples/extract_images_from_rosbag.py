#!/usr/bin/env python3

import sys
import rclpy
from utils.rosbag_utils import extract_images_and_odom


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage:")
        print(" python extract.py <bag_path> <image_topic> <odom_topic> <output_dir> <sample_rate>")
        sys.exit(1)

    bag_path = sys.argv[1]
    image_topic = sys.argv[2]
    odom_topic = sys.argv[3]
    output_dir = sys.argv[4]
    sample_rate = int(sys.argv[5]) if len(sys.argv) > 5 else 15

    rclpy.init()
    extract_images_and_odom(bag_path, image_topic, odom_topic, output_dir, sample_rate)
    rclpy.shutdown()


