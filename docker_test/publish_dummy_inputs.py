#!/usr/bin/env python3
import time
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Header

from drowsiness_detection_msg.msg import EarMarValue


class DummyInputs(Node):
    def __init__(self):
        super().__init__("dummy_inputs")
        self.img_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.metrics_pub = self.create_publisher(EarMarValue, "/ear_mar", 10)
        self.timer = self.create_timer(0.25, self.tick)  # 4 Hz
        self.i = 0

    def tick(self):
        h, w = 360, 480
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Simple pattern to avoid totally uniform frames
        img[:, :, 0] = (self.i * 5) % 255
        img[:, :, 1] = 40
        img[:, :, 2] = 40

        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = w * 3
        msg.data = img.tobytes()

        m = EarMarValue()
        m.header = msg.header

        # First 12 frames (3 seconds): extreme low EAR (drowsy/occluded)
        # After 12 frames: normal EAR (awake/clear)
        if self.i < 12:
            m.ear_value = 0.05
            m.mar_value = 0.70
        else:
            m.ear_value = 0.35
            m.mar_value = 0.10

        self.img_pub.publish(msg)
        self.metrics_pub.publish(m)
        self.i += 1
        
        if self.i > 20:
            # Reset loop so we can trigger it again if we want
            self.i = 0


def main():
    rclpy.init()
    node = DummyInputs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
