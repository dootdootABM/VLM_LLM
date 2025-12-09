#!/usr/bin/env python3
"""
ROS2 Node: CARLA Steering Wheel Metrics Generator
- Subscribes to /carla_control (G29 Racing Wheel inputs)
- Computes steering metrics: Entropy, SteeringRate, SDLP
- Publishes to /carla_metrics (for ML Aggregator to consume)

G29 Racing Wheel Inputs:
- steering_wheel: steering angle (-1.0 to 1.0)
- clutch: clutch pedal (0.0 to 1.0)
- throttle: accelerator pedal (0.0 to 1.0)
- brake: brake pedal (0.0 to 1.0)
- handbrake: handbrake status (0 or 1)
- reverse: reverse gear (0 or 1)
- gear_up: gear up button (0 or 1)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3Stamped
import numpy as np
from collections import deque
from threading import Lock


class CarlaMetricsMsg:
    """Simple CARLA metrics message class."""
    def __init__(self):
        self.entropy = 0.0
        self.steering_rate = 0.0
        self.sdlp = 0.0


class CarlaSteeringMetricsNode(Node):
    """Compute steering wheel metrics from CARLA G29 inputs."""


    def __init__(self):
        super().__init__("carla_steering_metrics_node")


        # === Parameters ===
        self.declare_parameter("window_duration", 60)  # 60 seconds
        self.window_duration = self.get_parameter("window_duration").value
        
        self.declare_parameter("fps", 100)  # CARLA typically runs at 100 Hz
        self.fps = self.get_parameter("fps").value


        # === Subscriber: CARLA steering wheel input ===
        self.carla_subscription = self.create_subscription(
            Float64,  # Steering angle input
            "/carla/control/steering",
            self.steering_callback,
            qos_profile_sensor_data
        )


        # === Publisher: Computed metrics ===
        self.metrics_pub = self.create_publisher(
            Vector3Stamped,
            "/carla_metrics",
            10
        )


        # === Data buffers (thread-safe) ===
        self.buffer_lock = Lock()
        self.max_buffer_size = self.fps * self.window_duration  # 6000 samples @ 100 Hz
        self.steering_angle_buffer = deque(maxlen=self.max_buffer_size)
        
        # Lane tracking buffers
        self.lane_position_buffer = deque(maxlen=self.max_buffer_size)
        
        # Steering rate tracking
        self.steering_changes = deque(maxlen=self.max_buffer_size)


        # === Timer for metrics computation ===
        self.create_timer(10.0, self.compute_metrics)  # Every 10 seconds
        
        self.get_logger().info(
            f"✅ CARLA Steering Metrics Node started\n"
            f"   Window: {self.window_duration}s @ {self.fps} Hz\n"
            f"   Buffer size: {self.max_buffer_size} samples\n"
            f"   G29 Racing Wheel Integration Ready"
        )


    def steering_callback(self, msg: Float64):
        """Receive steering wheel angle from CARLA."""
        steering_angle = float(msg.data)  # -1.0 to 1.0

        with self.buffer_lock:
            self.steering_angle_buffer.append(steering_angle)
            
            # Compute steering rate (changes per frame)
            if len(self.steering_angle_buffer) > 1:
                prev_angle = list(self.steering_angle_buffer)[-2]
                change = abs(steering_angle - prev_angle)
                self.steering_changes.append(change)


    def compute_metrics(self):
        """Compute steering metrics every 10 seconds."""
        with self.buffer_lock:
            if len(self.steering_angle_buffer) < self.fps * 5:  # Need at least 5 seconds
                self.get_logger().warn(
                    f"Not enough data for metrics. "
                    f"Got {len(self.steering_angle_buffer)} samples, need {self.fps * 5}"
                )
                return

            # Compute metrics
            entropy = self._calculate_entropy()
            steering_rate = self._calculate_steering_rate()
            sdlp = self._calculate_sdlp()
            
            self.get_logger().info(
                f"\n{'='*70}\n"
                f"🎮 CARLA G29 STEERING METRICS:\n"
                f"{'='*70}\n"
                f"  Entropy:       {entropy:.4f} (steering randomness)\n"
                f"  Steering Rate: {steering_rate:.1f} changes/min\n"
                f"  SDLP:          {sdlp:.4f} (lane deviation)\n"
                f"{'='*70}"
            )
            
            # Publish metrics
            self._publish_metrics(entropy, steering_rate, sdlp)


    def _calculate_entropy(self) -> float:
        """
        Calculate entropy of steering angles.
        Higher entropy = more random steering (drowsy)
        Lower entropy = consistent steering (alert)
        """
        if len(self.steering_angle_buffer) == 0:
            return 0.0
        
        angles = np.array(list(self.steering_angle_buffer))
        
        # Normalize angles to 0-1 range
        normalized = (angles + 1.0) / 2.0
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Create histogram
        hist, _ = np.histogram(normalized, bins=10, range=(0, 1))
        hist = hist / len(angles)  # Normalize histogram
        
        # Calculate Shannon entropy
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        return float(entropy)


    def _calculate_steering_rate(self) -> float:
        """
        Calculate rate of steering changes per minute.
        More changes = higher drowsiness (erratic steering)
        """
        if len(self.steering_changes) == 0:
            return 0.0
        
        # Count significant steering changes (> 0.05 threshold)
        changes = np.array(list(self.steering_changes))
        significant_changes = np.sum(changes > 0.05)
        
        # Convert to changes per minute
        time_in_seconds = len(self.steering_changes) / self.fps
        time_in_minutes = time_in_seconds / 60.0
        
        if time_in_minutes == 0:
            return 0.0
        
        rate = significant_changes / time_in_minutes
        
        return float(rate)


    def _calculate_sdlp(self) -> float:
        """
        Calculate Standard Deviation of Lane Position.
        Simulated from steering angle variance.
        Higher SDLP = lane weaving (drowsy)
        """
        if len(self.steering_angle_buffer) < 2:
            return 0.0
        
        angles = np.array(list(self.steering_angle_buffer))
        
        # SDLP is approximated from steering angle standard deviation
        # Scaled to match CARLA lane position standard deviation
        std_dev = float(np.std(angles))
        
        # Scale to match CARLA SDLP range (0.0 to ~1.0)
        # Steering angle ranges -1 to 1, so max std is ~0.5
        # Scale by 1.5 to get SDLP range
        sdlp = std_dev * 1.5
        
        return min(float(sdlp), 1.0)  # Cap at 1.0


    def _publish_metrics(self, entropy: float, steering_rate: float, sdlp: float):
        """Publish computed metrics to /carla_metrics."""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_steering"
        
        # Store metrics in Vector3 (x=entropy, y=steering_rate, z=sdlp)
        msg.vector.x = entropy
        msg.vector.y = steering_rate
        msg.vector.z = sdlp
        
        self.metrics_pub.publish(msg)
        
        self.get_logger().info(
            f"\n✅ PUBLISHED TO /carla_metrics:\n"
            f"  Entropy: {entropy:.4f}\n"
            f"  Steering Rate: {steering_rate:.1f} changes/min\n"
            f"  SDLP: {sdlp:.4f}"
        )



def main(args=None):
    rclpy.init(args=args)
    node = CarlaSteeringMetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping CARLA Steering Metrics Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()
