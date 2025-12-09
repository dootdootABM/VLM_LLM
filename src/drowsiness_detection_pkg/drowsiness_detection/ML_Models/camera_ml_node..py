#!/usr/bin/env python3
"""
ROS2 Node: Camera ML Model Inference
- Subscribes to /ear_mar (60 FPS EAR/MAR values)
- Buffers for 60 seconds
- Computes minute-level metrics
- Runs Camera ML model (3 features: PERCLOS, BlinkRate, blink_duration_mean)
- Publishes to /camera_predictions
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



class CameraMLNode(Node):
    """Camera ML Model - Facial Metrics Inference."""


    def __init__(self):
        super().__init__("camera_ml_node")


        # === Parameters ===
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value
        
        self.declare_parameter("window_duration", 60)  # 60 seconds
        self.window_duration = self.get_parameter("window_duration").value
        
        self.declare_parameter("fps", 60)
        self.fps = self.get_parameter("fps").value


        # === EAR/MAR thresholds ===
        self.ear_threshold = 0.26
        self.mar_threshold = 0.6
        self.min_consec_frames = 2
        
        # === Subscriber: Camera metrics ===
        self.subscription = self.create_subscription(
            EarMarValue,
            "/ear_mar",
            self.ear_mar_callback,
            qos_profile_sensor_data
        )


        # === Publisher: Camera ML predictions ===
        self.predictions_pub = self.create_publisher(
            Float64MultiArray,
            "/camera_predictions",
            10
        )


        # === Data buffers (thread-safe) ===
        self.buffer_lock = Lock()
        self.max_buffer_size = self.fps * self.window_duration
        self.ear_buffer = deque(maxlen=self.max_buffer_size)
        self.mar_buffer = deque(maxlen=self.max_buffer_size)


        # === ML models ===
        self._load_ml_models()


        # === Timer for minute-level inference ===
        self.create_timer(self.window_duration, self.run_ml_inference)
        
        self.get_logger().info(
            f"✅ Camera ML Node started\n"
            f"   Driver ID: {self.driver_id}\n"
            f"   Window: {self.window_duration}s @ {self.fps} FPS\n"
            f"   Model: Camera (PERCLOS, BlinkRate, blink_duration_mean)"
        )


    def _load_ml_models(self):
        """Load pre-trained ML model for camera."""
        try:
            self.camera_model = joblib.load('models/model_camera_rf.pkl')
            self.camera_scaler = joblib.load('models/model_camera_rf_scaler.pkl')
            self.get_logger().info("✅ Camera model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load camera model: {e}")
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
            if len(self.ear_buffer) < self.fps * 10:
                self.get_logger().warn(
                    f"Not enough data for inference. "
                    f"Got {len(self.ear_buffer)} samples, need {self.fps * 10}"
                )
                return

            # === CAMERA METRICS ===
            camera_metrics = self._compute_camera_metrics()

            self.get_logger().info(
                f"\n{'='*70}\n"
                f"🎥 CAMERA METRICS (60s window):\n"
                f"{'='*70}\n"
                f"  PERCLOS:              {camera_metrics['perclos']:>6.1f}%\n"
                f"  BlinkRate:            {camera_metrics['blink_rate']:>6.1f} blinks/min\n"
                f"  Blink Duration Mean:  {camera_metrics['blink_duration_mean']:>6.4f}s\n"
                f"{'='*70}"
            )

            # === Run ML model ===
            camera_result = self._run_camera_model(
                camera_metrics['perclos'],
                camera_metrics['blink_rate'],
                camera_metrics['blink_duration_mean']
            )

            # === Publish results ===
            self._publish_predictions(camera_result, camera_metrics)


    def _compute_camera_metrics(self) -> dict:
        """Compute camera-based metrics."""
        ear_array = np.array(list(self.ear_buffer))
        mar_array = np.array(list(self.mar_buffer))

        perclos = calculate_perclos(
            ear_values=ear_array,
            ear_threshold=self.ear_threshold,
            min_consec_frames=self.min_consec_frames
        )

        blink_rate = calculate_blink_frequency(
            ear_values=ear_array,
            ear_threshold=self.ear_threshold,
            fps=self.fps
        )

        blink_duration_mean = self._calculate_blink_duration_mean(ear_array)

        return {
            'perclos': perclos,
            'blink_rate': blink_rate,
            'blink_duration_mean': blink_duration_mean
        }


    def _calculate_blink_duration_mean(self, ear_values: np.ndarray) -> float:
        """Calculate mean blink duration from EAR values."""
        if len(ear_values) == 0:
            return 0.0
        
        below_threshold = ear_values < self.ear_threshold
        blink_starts = np.where(np.diff(below_threshold.astype(int)) == 1)[0]
        blink_ends = np.where(np.diff(below_threshold.astype(int)) == -1)[0]
        
        if len(blink_starts) == 0 or len(blink_ends) == 0:
            return 0.0
        
        blink_durations_frames = []
        for start, end in zip(blink_starts, blink_ends):
            if end > start:
                blink_durations_frames.append(end - start)
        
        if len(blink_durations_frames) == 0:
            return 0.0
        
        mean_duration_frames = np.mean(blink_durations_frames)
        mean_duration_seconds = mean_duration_frames / self.fps
        
        return max(0.0, mean_duration_seconds)


    def _run_camera_model(self, perclos, blink_rate, blink_duration_mean):
        """Run Camera Model (Random Forest on facial metrics)."""
        if self.camera_model is None:
            self.get_logger().warn("❌ Camera model not loaded - skipping inference")
            return None

        try:
            features = np.array([[perclos, blink_rate, blink_duration_mean]])
            features_scaled = self.camera_scaler.transform(features)
            proba = self.camera_model.predict_proba(features_scaled)[0]
            
            result = {
                'probability': float(proba[1]),
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


    def _publish_predictions(self, camera_result, camera_metrics):
        """
        Publish camera ML predictions to /camera_predictions topic.
        
        Message format (Float64MultiArray):
          data[0] = PERCLOS (%)
          data[1] = BlinkRate (blinks/min)
          data[2] = Blink Duration Mean (seconds)
          data[3] = Drowsiness Probability (0-1)
          data[4] = Model Confidence (0-1)
        """
        msg = Float64MultiArray()
        
        if camera_result:
            msg.data.extend([
                camera_metrics['perclos'],
                camera_metrics['blink_rate'],
                camera_metrics['blink_duration_mean'],
                camera_result['probability'],
                camera_result['confidence']
            ])
            
            self.get_logger().info(
                f"\n{'='*70}\n"
                f"✅ PUBLISHED TO /camera_predictions:\n"
                f"{'='*70}\n"
                f"  PERCLOS:              {camera_metrics['perclos']:.1f}%\n"
                f"  BlinkRate:            {camera_metrics['blink_rate']:.1f} blinks/min\n"
                f"  Blink Duration Mean:  {camera_metrics['blink_duration_mean']:.4f}s\n"
                f"  Drowsy Prob:          {camera_result['probability']:.1%}\n"
                f"  Confidence:           {camera_result['confidence']:.1%}\n"
                f"  Prediction:           {camera_result['prediction']}\n"
                f"{'='*70}"
            )
        else:
            msg.data.extend([
                camera_metrics['perclos'],
                camera_metrics['blink_rate'],
                camera_metrics['blink_duration_mean'],
                0.5,
                0.5
            ])
            self.get_logger().warn(
                f"⚠️  Model failed - published metrics with default probabilities"
            )
        
        self.predictions_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraMLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping Camera ML Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()