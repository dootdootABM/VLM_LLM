#!/usr/bin/env python3
"""
ROS2 Node: CARLA Steering ML Model Inference
- Subscribes to /carla_metrics (steering metrics from CARLA node)
- Buffers for 60 seconds
- Runs CARLA ML model (3 features: Entropy, SteeringRate, SDLP)
- Publishes to /carla_predictions (3-CLASS: 0=Alert, 1=Drowsy, 2=Very Drowsy)
"""


import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64MultiArray
import numpy as np
from collections import deque
import joblib
from threading import Lock




class CarlaMLNode(Node):
    """CARLA Steering ML Model - G29 Wheel Metrics Inference (3-CLASS)."""



    def _init_(self):
        super()._init_("carla_ml_node")



        # === Parameters ===
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value
        
        self.declare_parameter("window_duration", 60)
        self.window_duration = self.get_parameter("window_duration").value
        
        self.declare_parameter("fps", 100)
        self.fps = self.get_parameter("fps").value



        # === Class names ===
        self.class_names = {0: "Alert", 1: "Drowsy", 2: "Very Drowsy"}

        # === Subscriber: CARLA steering metrics ===
        self.carla_subscription = self.create_subscription(
            Vector3Stamped,
            "/carla_metrics",
            self.carla_metrics_callback,
            qos_profile_sensor_data
        )



        # === Publisher: CARLA ML predictions ===
        self.predictions_pub = self.create_publisher(
            Float64MultiArray,
            "/carla_predictions",
            10
        )



        # === Data buffers (thread-safe) ===
        self.buffer_lock = Lock()
        self.max_buffer_size = self.fps * self.window_duration
        
        self.entropy_buffer = deque(maxlen=self.max_buffer_size)
        self.steering_rate_buffer = deque(maxlen=self.max_buffer_size)
        self.sdlp_buffer = deque(maxlen=self.max_buffer_size)



        # === ML models ===
        self._load_ml_models()



        # === Timer for minute-level inference ===
        self.create_timer(self.window_duration, self.run_ml_inference)
        
        self.get_logger().info(
            f"CARLA ML Node started (3-CLASS)\n"
            f"   Driver ID: {self.driver_id}\n"
            f"   Window: {self.window_duration}s @ {self.fps} Hz\n"
            f"   Model: CARLA Steering (Entropy, SteeringRate, SDLP)\n"
            f"   Classes: 0=Alert, 1=Drowsy, 2=Very Drowsy"
        )



    def _load_ml_models(self):
        """Load pre-trained ML model for CARLA steering."""
        try:
            self.carla_model = joblib.load('models/model_carla_steering_rf.pkl')
            self.carla_scaler = joblib.load('models/model_carla_steering_rf_scaler.pkl')
            self.get_logger().info("CARLA steering model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load CARLA model: {e}")
            self.carla_model = None



    def carla_metrics_callback(self, msg: Vector3Stamped):
        """Receive CARLA steering metrics."""
        entropy = float(msg.vector.x)
        steering_rate = float(msg.vector.y)
        sdlp = float(msg.vector.z)


        with self.buffer_lock:
            self.entropy_buffer.append(entropy)
            self.steering_rate_buffer.append(steering_rate)
            self.sdlp_buffer.append(sdlp)



    def run_ml_inference(self):
        """Run ML inference every 60 seconds."""
        with self.buffer_lock:
            if len(self.entropy_buffer) < self.fps * 10:
                self.get_logger().warn(
                    f"Not enough data for inference. "
                    f"Got {len(self.entropy_buffer)} samples, need {self.fps * 10}"
                )
                return


            # === CARLA METRICS ===
            carla_metrics = self._compute_carla_metrics()


            self.get_logger().info(
                f"\nCARLA METRICS (60s window):\n"
                f"  Entropy:              {carla_metrics['entropy']:>6.4f}\n"
                f"  Steering Rate:        {carla_metrics['steering_rate']:>6.1f} changes/min\n"
                f"  SDLP:                 {carla_metrics['sdlp']:>6.4f}"
            )


            # === Run ML model ===
            carla_result = self._run_carla_model(
                carla_metrics['entropy'],
                carla_metrics['steering_rate'],
                carla_metrics['sdlp']
            )


            # === Publish results ===
            self._publish_predictions(carla_result, carla_metrics)



    def _compute_carla_metrics(self) -> dict:
        """Compute CARLA steering metrics."""
        entropy = np.mean(list(self.entropy_buffer)) if len(self.entropy_buffer) > 0 else 0.0
        steering_rate = np.mean(list(self.steering_rate_buffer)) if len(self.steering_rate_buffer) > 0 else 0.0
        sdlp = np.mean(list(self.sdlp_buffer)) if len(self.sdlp_buffer) > 0 else 0.0


        return {
            'entropy': entropy,
            'steering_rate': steering_rate,
            'sdlp': sdlp
        }



    def _run_carla_model(self, entropy, steering_rate, sdlp):
        """Run CARLA Model (Random Forest on steering metrics) - 3-CLASS."""
        if self.carla_model is None:
            self.get_logger().warn("CARLA model not loaded - skipping inference")
            return None


        try:
            features = np.array([[entropy, steering_rate, sdlp]])
            features_scaled = self.carla_scaler.transform(features)
            
            # Get prediction and probabilities for 3 classes
            prediction = self.carla_model.predict(features_scaled)[0]
            proba = self.carla_model.predict_proba(features_scaled)[0]
            
            # proba = [P(Alert), P(Drowsy), P(Very Drowsy)]
            result = {
                'prediction': int(prediction),
                'class_name': self.class_names[prediction],
                'alert_prob': float(proba[0]),
                'drowsy_prob': float(proba[1]),
                'very_drowsy_prob': float(proba[2]),
                'confidence': float(max(proba))
            }
            
            self.get_logger().info(
                f"\nCARLA MODEL PREDICTION (3-CLASS):\n"
                f"  Predicted Class:     {result['class_name']} ({result['prediction']})\n"
                f"  Alert Prob:          {result['alert_prob']:.1%}\n"
                f"  Drowsy Prob:         {result['drowsy_prob']:.1%}\n"
                f"  Very Drowsy Prob:    {result['very_drowsy_prob']:.1%}\n"
                f"  Model Confidence:    {result['confidence']:.1%}"
            )
            
            return result
        except Exception as e:
            self.get_logger().error(f"CARLA model error: {e}")
            return None



    def _publish_predictions(self, carla_result, carla_metrics):
        """
        Publish CARLA ML predictions to /carla_predictions topic.
        
        Message format (Float64MultiArray):
          data[0] = Entropy
          data[1] = Steering Rate (changes/min)
          data[2] = SDLP
          data[3] = Predicted Class (0=Alert, 1=Drowsy, 2=Very Drowsy)
          data[4] = Alert Probability (0-1)
          data[5] = Drowsy Probability (0-1)
          data[6] = Very Drowsy Probability (0-1)
          data[7] = Model Confidence (0-1)
        """
        msg = Float64MultiArray()
        
        if carla_result:
            msg.data.extend([
                carla_metrics['entropy'],
                carla_metrics['steering_rate'],
                carla_metrics['sdlp'],
                float(carla_result['prediction']),
                carla_result['alert_prob'],
                carla_result['drowsy_prob'],
                carla_result['very_drowsy_prob'],
                carla_result['confidence']
            ])
            
            self.get_logger().info(
                f"\nPUBLISHED TO /carla_predictions:\n"
                f"  Entropy:              {carla_metrics['entropy']:.4f}\n"
                f"  Steering Rate:        {carla_metrics['steering_rate']:.1f} changes/min\n"
                f"  SDLP:                 {carla_metrics['sdlp']:.4f}\n"
                f"  Prediction:           {carla_result['class_name']} ({carla_result['prediction']})\n"
                f"  Alert Prob:           {carla_result['alert_prob']:.1%}\n"
                f"  Drowsy Prob:          {carla_result['drowsy_prob']:.1%}\n"
                f"  Very Drowsy Prob:     {carla_result['very_drowsy_prob']:.1%}\n"
                f"  Confidence:           {carla_result['confidence']:.1%}"
            )
        else:
            msg.data.extend([
                carla_metrics['entropy'],
                carla_metrics['steering_rate'],
                carla_metrics['sdlp'],
                1.0,  # Default to Drowsy (middle class)
                0.33, 0.33, 0.33, 0.33
            ])
            self.get_logger().warn(
                "Model failed - published metrics with default probabilities"
            )
        
        self.predictions_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = CarlaMLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping CARLA ML Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()



if _name_ == "_main_":
    main()