#!/usr/bin/env python3
"""
ROS2 Node: ML Aggregator (Camera Only) - Step 1
- Subscribes to /ear_mar (60 FPS EAR/MAR values)
- Buffers for 60 seconds
- Computes minute-level metrics using utils.py functions
- Runs Camera ML model
- Publishes metrics + ML predictions to /ml_predictions
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from drowsiness_detection_msg.msg import EarMarValue
from std_msgs.msg import Float64MultiArray
import numpy as np
from collections import deque
import joblib
from threading import Lock

# Import utility functions from your utils.py
from drowsiness_detection.utils import (
    calculate_perclos,
    calculate_blink_frequency,
    calculate_yawn_frequency,
)


class MLAggregatorNode(Node):
    """Aggregates EAR/MAR data and runs Camera ML model every minute."""

    def __init__(self):
        super().__init__("ml_aggregator_node")

        # === Parameters ===
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value
        
        self.declare_parameter("window_duration", 60)  # 60 seconds
        self.window_duration = self.get_parameter("window_duration").value
        
        self.declare_parameter("fps", 60)
        self.fps = self.get_parameter("fps").value

        # === EAR/MAR thresholds ===
        self.ear_threshold = 0.26  # EAR below this = eye closed
        self.mar_threshold = 0.6   # MAR above this = mouth open (yawning)
        self.min_consec_frames = 2  # Min consecutive frames for PERCLOS
        
        # === Subscriber: Camera metrics ===
        self.subscription = self.create_subscription(
            EarMarValue,
            "/ear_mar",
            self.ear_mar_callback,
            qos_profile_sensor_data
        )

        # === Publisher: ML predictions ===
        self.ml_predictions_pub = self.create_publisher(
            Float64MultiArray,
            "/ml_predictions",
            10
        )

        # === Data buffers (thread-safe) ===
        self.buffer_lock = Lock()
        self.max_buffer_size = self.fps * self.window_duration  # 3600 samples @ 60 FPS
        self.ear_buffer = deque(maxlen=self.max_buffer_size)
        self.mar_buffer = deque(maxlen=self.max_buffer_size)

        # === ML models ===
        self._load_ml_models()

        # === Timer for minute-level inference ===
        self.create_timer(self.window_duration, self.run_ml_inference)
        
        self.get_logger().info(
            f"✅ ML Aggregator Node started\n"
            f"   Driver ID: {self.driver_id}\n"
            f"   Window: {self.window_duration}s @ {self.fps} FPS\n"
            f"   Buffer size: {self.max_buffer_size} samples"
        )

    def _load_ml_models(self):
        """Load pre-trained ML model for camera."""
        try:
            self.camera_model = joblib.load('models/model_camera_rf.pkl')
            self.camera_scaler = joblib.load('models/model_camera_rf_scaler.pkl')
            self.get_logger().info("✅ Camera model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load camera model: {e}")
            self.get_logger().error(
                "Ensure trained models exist at:\n"
                "  - models/model_camera_rf.pkl\n"
                "  - models/model_camera_rf_scaler.pkl"
            )
            self.camera_model = None

    def ear_mar_callback(self, msg: EarMarValue):
        """Receive EAR/MAR values at 60 FPS from camera node."""
        ear = float(msg.ear_value)
        mar = float(msg.mar_value)

        with self.buffer_lock:
            self.ear_buffer.append(ear)
            self.mar_buffer.append(mar)

    def run_ml_inference(self):
        """Run ML inference every 60 seconds."""
        with self.buffer_lock:
            if len(self.ear_buffer) < self.fps * 10:  # Need at least 10 seconds of data
                self.get_logger().warn(
                    f"Not enough data for inference. "
                    f"Got {len(self.ear_buffer)} samples, need {self.fps * 10}"
                )
                return

            # === CAMERA METRICS ===
            camera_metrics = self._compute_camera_metrics()

            self.get_logger().info(
                f"\n{'='*70}\n"
                f"📊 MINUTE {self.get_clock().now().seconds % 3600}s METRICS:\n"
                f"{'='*70}\n"
                f"  PERCLOS:     {camera_metrics['perclos']:>6.1f}%\n"
                f"  BlinkRate:   {camera_metrics['blink_rate']:>6.1f} blinks/min\n"
                f"  YawnRate:    {camera_metrics['yawn_rate']:>6.2f} yawns/min\n"
                f"{'='*70}"
            )

            # === Run ML model ===
            camera_result = self._run_camera_model(
                camera_metrics['perclos'],
                camera_metrics['blink_rate'],
                camera_metrics['yawn_rate']
            )

            # === Publish results (metrics + predictions) ===
            self._publish_ml_predictions(camera_result, camera_metrics)

    def _compute_camera_metrics(self) -> dict:
        """
        Compute camera-based metrics using utility functions.
        Uses: EAR, MAR data from the last 60 seconds.
        """
        ear_array = np.array(list(self.ear_buffer))
        mar_array = np.array(list(self.mar_buffer))

        # PERCLOS: Percentage of Eye Closure
        perclos = calculate_perclos(
            ear_values=ear_array,
            ear_threshold=self.ear_threshold,
            min_consec_frames=self.min_consec_frames
        )

        # BlinkRate: Blinks per minute
        blink_rate = calculate_blink_frequency(
            ear_values=ear_array,
            ear_threshold=self.ear_threshold,
            fps=self.fps
        )

        # YawnRate: Yawns per minute
        yawn_rate = calculate_yawn_frequency(
            mar_values=mar_array,
            mar_threshold=self.mar_threshold,
            min_consec_frames=6,  # ~100ms at 60 FPS
            fps=self.fps
        )

        return {
            'perclos': perclos,
            'blink_rate': blink_rate,
            'yawn_rate': yawn_rate
        }

    def _run_camera_model(self, perclos, blink_rate, yawn_rate):
        """Run Camera Model (Random Forest on facial metrics)."""
        if self.camera_model is None:
            self.get_logger().warn("❌ Camera model not loaded - skipping inference")
            return None

        try:
            # Prepare features: [PERCLOS, BlinkRate, YawnRate]
            features = np.array([[perclos, blink_rate, yawn_rate]])
            features_scaled = self.camera_scaler.transform(features)
            
            # Get probabilities
            proba = self.camera_model.predict_proba(features_scaled)[0]
            
            result = {
                'name': 'camera',
                'probability': float(proba[1]),  # P(drowsy)
                'confidence': float(max(proba)),
                'prediction': 'DROWSY' if proba[1] > 0.5 else 'ALERT',
                'alert_prob': float(proba[0]),
                'drowsy_prob': float(proba[1])
            }
            
            self.get_logger().info(
                f"\n{'='*70}\n"
                f"🎥 CAMERA MODEL PREDICTION:\n"
                f"{'='*70}\n"
                f"  Status:              {result['prediction']}\n"
                f"  Drowsiness Prob:     {result['probability']:.1%}\n"
                f"  Model Confidence:    {result['confidence']:.1%}\n"
                f"{'='*70}"
            )
            
            return result
        except Exception as e:
            self.get_logger().error(f"❌ Camera model error: {e}")
            return None

    def _publish_ml_predictions(self, camera_result, camera_metrics):
        """
        Publish ML predictions with metrics to /ml_predictions topic.
        
        Message format (Float64MultiArray):
          data[0] = PERCLOS (%)
          data[1] = BlinkRate (blinks/min)
          data[2] = YawnRate (yawns/min)
          data[3] = Drowsiness Probability (0-1)
          data[4] = Model Confidence (0-1)
        
        Consumed by: Reasoning LLM Node, Control Node (alerts)
        """
        msg = Float64MultiArray()
        
        if camera_result:
            # Publish: metrics + prediction + confidence
            msg.data.extend([
                camera_metrics['perclos'],           # data[0]
                camera_metrics['blink_rate'],        # data[1]
                camera_metrics['yawn_rate'],         # data[2]
                camera_result['probability'],        # data[3] - P(drowsy)
                camera_result['confidence']          # data[4] - Model confidence
            ])
            
            self.get_logger().info(
                f"\n{'='*70}\n"
                f"✅ PUBLISHED TO /ml_predictions:\n"
                f"{'='*70}\n"
                f"  PERCLOS:         {camera_metrics['perclos']:.1f}%\n"
                f"  BlinkRate:       {camera_metrics['blink_rate']:.1f} blinks/min\n"
                f"  YawnRate:        {camera_metrics['yawn_rate']:.2f} yawns/min\n"
                f"  Drowsy Prob:     {camera_result['probability']:.1%}\n"
                f"  Confidence:      {camera_result['confidence']:.1%}\n"
                f"  Prediction:      {camera_result['prediction']}\n"
                f"{'='*70}"
            )
        else:
            # Default values if model failed
            msg.data.extend([
                camera_metrics['perclos'],
                camera_metrics['blink_rate'],
                camera_metrics['yawn_rate'],
                0.5,  # Default drowsy probability
                0.5   # Default confidence
            ])
            self.get_logger().warn(
                f"⚠️  Model failed - published metrics with default probabilities:\n"
                f"  PERCLOS: {camera_metrics['perclos']:.1f}%\n"
                f"  BlinkRate: {camera_metrics['blink_rate']:.1f} blinks/min\n"
                f"  YawnRate: {camera_metrics['yawn_rate']:.2f} yawns/min"
            )
        
        self.ml_predictions_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MLAggregatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping ML Aggregator...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
