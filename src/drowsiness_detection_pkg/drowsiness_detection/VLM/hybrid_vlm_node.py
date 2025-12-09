#!/usr/bin/env python3
"""
ROS2 Node: Hybrid VLM Subscriber
Subscribes to:
  - /camera/image_raw (from PySpin IR camera node)
  - /ear_mar (EAR/MAR metrics from IR camera node)

Publishes:
  - NOTHING (runs Hybrid VLM analysis on received frames)

When an occlusion event is detected:
  1. OcclusionEventTracker captures 8 frames
  2. VLMRequestHandler submits to Qwen2.5-VL (async, non-blocking)
  3. Results saved to disk with event metadata
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from drowsiness_detection_msg.msg import EarMarValue
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time
from collections import deque

# Import core VLM logic (no ROS, no camera)
from drowsiness_detection.VLM.hybrid_vlm_core import (
    CONFIG,
    detect_ambiguity_flags,
    OcclusionEventTracker,
    VLMRequestHandler,
)



class HybridVlmNode(Node):
    """
    Subscribes to IR camera topics and runs Hybrid drowsiness + VLM occlusion logic.
    
    Flow:
    1. Receive /camera/image_raw (rgb8) + /ear_mar (EAR, MAR values)
    2. Convert image to BGR (for ambiguity detection)
    3. Detect ambiguity flags + check for prolonged eye closure + yawning
    4. Track occlusion events with OcclusionEventTracker
    5. When event ends, submit 8-frame clip to VLM (async, non-blocking)
    """

    def __init__(self):
        super().__init__("hybrid_vlm_node")

        # Parameters
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value

        self.declare_parameter("output_dir", "./vlm_triggers")
        output_dir = self.get_parameter("output_dir").value

        # Bridge for image conversion
        self.bridge = CvBridge()

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.metrics_sub = self.create_subscription(
            EarMarValue,
            "/ear_mar",
            self.metrics_callback,
            10,
        )

        # Latest EAR/MAR from /ear_mar topic
        self.latest_ear = None
        self.latest_mar = None
        self.latest_metrics_stamp = None

        # Eye closure tracking
        self._closure_start = None

        # Counters
        self.frame_count = 0
        self.event_count = 0
        self.submitted_events = []

        # Core VLM components
        CONFIG["output_dir"] = output_dir
        self.event_tracker = OcclusionEventTracker()
        self.vlm_handler = VLMRequestHandler(
            max_workers=CONFIG["max_concurrent_vlm"],
            output_dir=CONFIG["output_dir"],
        )

        # Lock for shared state
        self.lock = threading.Lock()

        self.get_logger().info(
            f"[HybridVLM] Node initialized (driver_id={self.driver_id})"
        )
        self.get_logger().info(
            f"[HybridVLM] Subscribing to /camera/image_raw and /ear_mar"
        )
        self.get_logger().info(f"[HybridVLM] Output dir: {output_dir}")
        self.get_logger().info(
            f"[HybridVLM] Triggers: Drowsiness + Yawning + ANY Ambiguity Flags"
        )

    # ---------------------- Metrics callback ----------------------
    def metrics_callback(self, msg: EarMarValue):
        """Receive EAR/MAR metrics from camera node"""
        with self.lock:
            self.latest_ear = float(msg.ear_value)
            self.latest_mar = float(msg.mar_value)
            self.latest_metrics_stamp = msg.header.stamp

    # ---------------------- Image callback -----------------------
    def image_callback(self, msg: Image):
        """Receive image from camera node and run Hybrid analysis"""
        try:
            # Camera node publishes rgb8, convert to BGR for ambiguity detection
            cv_image_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            frame_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f"[HybridVLM] CV bridge error: {e}")
            return

        # Get latest metrics (may be slightly delayed)
        with self.lock:
            ear = self.latest_ear
            mar = self.latest_mar

        # Process frame with hybrid logic
        self.process_frame(frame_bgr, ear, mar)

    # ---------------------- Core Hybrid Logic ----------------------
    def process_frame(self, frame_bgr: np.ndarray, ear: float, mar: float):
        """
        Main hybrid processing pipeline:
        1. Extract brightness
        2. Detect ambiguity flags
        3. Check prolonged closure + yawning
        4. Update event tracker
        5. Submit VLM request if event ends
        """
        self.frame_count += 1

        # Brightness (grayscale mean)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        # Treat "ear > 0" as proxy for face detected (IR camera produces landmarks)
        face_detected = ear is not None and ear > 0.0
        face_conf = 1.0 if face_detected else 0.0

        # Classical metrics
        now = self.get_clock().now().nanoseconds * 1e-9

        eyes_closed = ear is not None and ear < CONFIG["ear_low_threshold"]
        if eyes_closed:
            if self._closure_start is None:
                self._closure_start = now
            closure_duration = now - self._closure_start
            prolonged_closure = closure_duration >= CONFIG["drowsiness_duration_min"]
        else:
            self._closure_start = None
            prolonged_closure = False

        yawning = mar is not None and mar > CONFIG["mar_yawn_threshold"]

        # Detect ambiguity (hands, occlusion, no_face, lighting, etc.)
        ambiguity_flags = detect_ambiguity_flags(
            frame_bgr,
            face_detected,
            ear,
            mar,
            brightness,
            face_conf,
        )

        # TRIGGER: Has flags? (any of: prolonged closure, yawning+ambiguity, or ambiguity alone)
        has_flags = bool(
            prolonged_closure or (yawning and ambiguity_flags) or bool(ambiguity_flags)
        )

        metrics = {
            "ear": ear,
            "mar": mar,
            "brightness": brightness,
            "face_conf": face_conf,
        }

        # Update event tracker
        frames_to_submit, should_submit = self.event_tracker.update(
            frame_bgr,
            has_flags,
            ambiguity_flags if has_flags else [],
            metrics,
        )

        # If event ended, submit to VLM
        if should_submit and frames_to_submit:
            clip_id = f"{self.driver_id}_event_{self.frame_count:06d}"
            self.get_logger().info(
                f"[HybridVLM] 🔔 Submitting occlusion event: {clip_id}"
            )
            self.get_logger().info(
                f"[HybridVLM]    Frames: {len(frames_to_submit)} "
                f"| Flags: {','.join(self.event_tracker.last_flags)}"
            )

            self.vlm_handler.submit_request(
                frames_to_submit,
                clip_id,
                self.event_tracker.last_flags,
                metrics,
            )
            self.submitted_events.append(clip_id)
            self.event_count += 1

            # Log VLM status
            pending = self.vlm_handler.pending_requests
            completed = self.vlm_handler.completed_requests
            self.get_logger().info(
                f"[HybridVLM]    VLM status: {completed} completed, {pending} pending"
            )

    # ---------------------- Cleanup -----------------------
    def destroy_node(self):
        """Graceful shutdown: wait for pending VLM requests, then exit"""
        self.get_logger().info("[HybridVLM] Shutting down HybridVlmNode...")
        self.get_logger().info(
            f"[HybridVLM] Total frames: {self.frame_count} | Events: {self.event_count}"
        )

        if hasattr(self, "vlm_handler"):
            self.vlm_handler.shutdown()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HybridVlmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[HybridVLM] Keyboard interrupt — shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
