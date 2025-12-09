#!/usr/bin/env python3
"""
ROS2 Node: Integrated Safety-Critical LLM Reasoning
Subscribes to:
  - /camera_predictions (Camera ML: EAR, MAR, drowsy_prob, confidence) - every 60s
  - /carla_predictions (CARLA ML: entropy, steering_rate, SDLP, drowsy_prob, confidence) - every 60s
  - /vlm_occlusion_results (VLM JSON: async, when occlusion flags detected)

LLM Trigger:
  - Timer-based: Every 60 seconds (triggered by Camera + CARLA ML window)
  - VLM provides CONTEXT ONLY (not a trigger)
  - When 1-minute window completes:
    1. Gather camera + carla data
    2. Include latest VLM context (if available)
    3. Send to Ollama for safety-critical reasoning
    4. Publish final decision
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, String
import json
import time
from threading import Lock
import requests
import re


class IntegratedSafetyCriticalLLMNode(Node):
    """Integrated LLM reasoning: Camera ML + CARLA ML (every 1 min) + VLM context (async)."""


    def __init__(self):
        super().__init__("integrated_safety_critical_llm_node")


        # === Parameters ===
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value
        
        self.declare_parameter("ollama_endpoint", "http://localhost:11434")
        self.ollama_endpoint = self.get_parameter("ollama_endpoint").value
        
        self.declare_parameter("ollama_model", "llama3.1:8b")
        self.ollama_model = self.get_parameter("ollama_model").value
        
        self.declare_parameter("confidence_threshold", 0.6)
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        
        self.declare_parameter("drowsy_threshold", 0.5)
        self.drowsy_threshold = self.get_parameter("drowsy_threshold").value


        # === Subscribers ===
        self.camera_sub = self.create_subscription(
            Float64MultiArray,
            "/camera_predictions",
            self.camera_callback,
            qos_profile_sensor_data
        )

        self.carla_sub = self.create_subscription(
            Float64MultiArray,
            "/carla_predictions",
            self.carla_callback,
            qos_profile_sensor_data
        )
        
        # VLM provides CONTEXT ONLY (async, when occlusion flags detected)
        self.vlm_sub = self.create_subscription(
            String,
            "/vlm_occlusion_results",
            self.vlm_callback,
            qos_profile_sensor_data
        )


        # === Publisher ===
        self.alert_pub = self.create_publisher(
            String,
            "/drowsiness_alert",
            10
        )


        # === Data storage (thread-safe) ===
        self.data_lock = Lock()
        self.camera_data = None
        self.carla_data = None
        self.vlm_context = None  # Latest VLM context (may be from previous event)
        self.window_start_time = time.time()


        # === Initialize Ollama ===
        self._init_ollama()
        
        # === Timer for 1-minute LLM reasoning (triggered by 60s window) ===
        self.create_timer(60.0, self.reason_on_timer)


        self.get_logger().info(
            f"✅ Integrated Safety-Critical LLM Node started\n"
            f"   Driver ID: {self.driver_id}\n"
            f"   Ollama: {self.ollama_endpoint}/{self.ollama_model}\n"
            f"   Confidence Threshold: {self.confidence_threshold:.1%}\n"
            f"   Drowsy Threshold: {self.drowsy_threshold:.1%}"
        )
        self.get_logger().info(
            f"   Data Sources:\n"
            f"     - /camera_predictions (Camera ML - every 60s)\n"
            f"     - /carla_predictions (CARLA ML - every 60s)\n"
            f"     - /vlm_occlusion_results (VLM - async, context only)"
        )
        self.get_logger().info(
            f"   LLM Trigger: TIMER-BASED (every 60 seconds)\n"
            f"   VLM Role: Provides reliability context, does NOT trigger LLM"
        )


    def _init_ollama(self):
        """Initialize Ollama connection."""
        try:
            response = requests.get(f"{self.ollama_endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                self.get_logger().info(f"✅ Ollama connected. Models: {model_names}")
                
                if any(self.ollama_model in m for m in model_names):
                    self.get_logger().info(f"✅ {self.ollama_model} available")
                else:
                    self.get_logger().warn(f"⚠️  {self.ollama_model} not found")
            else:
                self.get_logger().error(f"❌ Ollama API error: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            self.get_logger().error(
                f"❌ Cannot connect to Ollama at {self.ollama_endpoint}\n"
                f"   Run: ollama serve"
            )
        except Exception as e:
            self.get_logger().error(f"❌ Ollama init error: {e}")


    def camera_callback(self, msg: Float64MultiArray):
        """Receive camera ML predictions (every 60s from camera_ml_node)."""
        if len(msg.data) < 5:
            return
        
        with self.data_lock:
            self.camera_data = {
                'perclos': float(msg.data[0]),
                'blink_rate': float(msg.data[1]),
                'blink_duration_mean': float(msg.data[2]),
                'drowsy_prob': float(msg.data[3]),
                'confidence': float(msg.data[4]),
                'timestamp': time.time()
            }
        
        self.get_logger().info(
            f"📷 Camera ML received: drowsy={self.camera_data['drowsy_prob']:.1%}, "
            f"conf={self.camera_data['confidence']:.1%}"
        )


    def carla_callback(self, msg: Float64MultiArray):
        """Receive CARLA ML predictions (every 60s from carla_ml_node)."""
        if len(msg.data) < 5:
            return
        
        with self.data_lock:
            self.carla_data = {
                'entropy': float(msg.data[0]),
                'steering_rate': float(msg.data[1]),
                'sdlp': float(msg.data[2]),
                'drowsy_prob': float(msg.data[3]),
                'confidence': float(msg.data[4]),
                'timestamp': time.time()
            }
        
        self.get_logger().info(
            f"🎮 CARLA ML received: drowsy={self.carla_data['drowsy_prob']:.1%}, "
            f"conf={self.carla_data['confidence']:.1%}"
        )


    def vlm_callback(self, msg: String):
        """
        Receive VLM occlusion analysis results (async, when flags detected).
        VLM provides CONTEXT ONLY - does NOT trigger LLM reasoning.
        """
        try:
            vlm_json = json.loads(msg.data)
            with self.data_lock:
                self.vlm_context = vlm_json
            
            self.get_logger().info(
                f"📸 VLM Context received (ASYNC): "
                f"occlusion={vlm_json.get('occlusion', 'none')}, "
                f"lighting={vlm_json.get('lighting', 'normal')}, "
                f"eyes_visible={vlm_json.get('eyes_visible', True)}"
            )
            self.get_logger().info(
                f"   (VLM does NOT trigger LLM - waits for next 60s timer)"
            )
        
        except json.JSONDecodeError as e:
            self.get_logger().error(f"❌ VLM JSON parse error: {e}")


    def reason_on_timer(self):
        """
        Timer callback: Every 60 seconds, gather ML data and invoke LLM reasoning.
        This is the MAIN TRIGGER for LLM - not VLM flags.
        """
        with self.data_lock:
            if self.camera_data is None or self.carla_data is None:
                self.get_logger().warn(
                    f"⏱️  60s timer fired, but missing ML data: "
                    f"camera={'ready' if self.camera_data else 'pending'}, "
                    f"carla={'ready' if self.carla_data else 'pending'}"
                )
                return
            
            # Make copies to avoid lock during LLM call
            camera_copy = self.camera_data.copy()
            carla_copy = self.carla_data.copy()
            vlm_copy = self.vlm_context.copy() if self.vlm_context else None
        
        # Invoke integrated LLM reasoning (MAIN TRIGGER)
        self._invoke_integrated_reasoning(camera_copy, carla_copy, vlm_copy)


    def _invoke_integrated_reasoning(self, camera_data, carla_data, vlm_context):
        """Invoke Ollama with integrated sensor data."""
        
        context = self._build_integrated_prompt(camera_data, carla_data, vlm_context)
        
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"🧠 60-SECOND WINDOW: INTEGRATED LLM REASONING\n"
            f"{'='*70}\n"
            f"Camera ML:  {camera_data['drowsy_prob']:.1%} (conf: {camera_data['confidence']:.1%})\n"
            f"CARLA ML:   {carla_data['drowsy_prob']:.1%} (conf: {carla_data['confidence']:.1%})\n"
            f"VLM Context: {'Available (from occlusion event)' if vlm_context else 'Not yet (no events)'}\n"
            f"{'='*70}"
        )
        
        try:
            response = self._query_ollama(context)
            if response:
                self._publish_alert(response)
        
        except Exception as e:
            self.get_logger().error(f"❌ LLM error: {e}")


    def _build_integrated_prompt(self, camera_data, carla_data, vlm_context) -> str:
        """Build integrated prompt with all sensor data."""
        
        system_prompt = """You are a safety-critical driver monitoring assistant.
You act as a judge that resolves conflicts between sensors.

RELIABILITY RULES (follow strictly):
- If eye metrics show very low confidence (<0.6), treat them as UNRELIABLE.
- If steering metrics show very low confidence (<0.6), treat them as UNRELIABLE.
- If VLM reports eyes_visible=false or lighting in ["very_dark","dark"], eye metrics are UNRELIABLE.
- If VLM reports occlusion in ["sunglasses","hands","turned"], eye metrics are UNRELIABLE.
- When eye metrics are unreliable, rely MORE on steering/vehicle metrics.
- When vehicle metrics are unreliable, rely MORE on eye/facial metrics.
- If BOTH modalities are unreliable/conflicting, output final_state="unknown" with "soft_alert".

LABEL RULES:
- "severely_drowsy": consistent strong evidence from ≥2 reliable sources.
- "mildly_drowsy": mixed evidence but ≥1 reliable source suggests drowsiness.
- "alert": no significant drowsiness detected.
- "unknown": unreliable data or strong conflicts between sensors.

INTERVENTION RULES:
- "no_alert": normal alertness detected.
- "soft_alert": mild drowsiness or uncertain conditions.
- "strong_alert": moderate drowsiness from reliable sources.
- "takeover_request": severe drowsiness from multiple reliable sources.
"""
        
        # Build VLM context section
        vlm_section = ""
        if vlm_context:
            vlm_section = f"""
VLM CONTEXT (from recent occlusion event - async analysis):
- Occlusion Type: {vlm_context.get('occlusion', 'none')}
- Lighting Condition: {vlm_context.get('lighting', 'normal')}
- Eyes Visible: {vlm_context.get('eyes_visible', True)}
- Mouth Visible: {vlm_context.get('mouth_visible', True)}
- VLM Reliability: {vlm_context.get('reliability', 0.7):.1%}
- Posture: {vlm_context.get('posture', 'normal')}
- VLM Notes: {vlm_context.get('notes', 'N/A')}
"""
        else:
            vlm_section = "\nVLM CONTEXT: No occlusion events detected yet (normal conditions assumed)"
        
        user_prompt = f"""Analyze this integrated 1-minute driver monitoring window:

CAMERA/EYE METRICS (Facial Analysis - from camera_ml_node):
- PERCLOS (% eye closure): {camera_data['perclos']:.1f}%
- Blink Rate: {camera_data['blink_rate']:.1f} blinks/min
- Avg Blink Duration: {camera_data['blink_duration_mean']:.4f}s
- Camera Model Drowsiness Probability: {camera_data['drowsy_prob']:.1%}
- Camera Model Confidence: {camera_data['confidence']:.1%}
- Camera Reliability: {"RELIABLE" if camera_data['confidence'] > 0.6 else "UNRELIABLE"}

VEHICLE/STEERING METRICS (Vehicle Control - from carla_ml_node):
- Steering Entropy (randomness): {carla_data['entropy']:.4f}
- Steering Rate (corrections/min): {carla_data['steering_rate']:.1f}
- Lane Position Deviation (SDLP): {carla_data['sdlp']:.4f}
- Steering Model Drowsiness Probability: {carla_data['drowsy_prob']:.1%}
- Steering Model Confidence: {carla_data['confidence']:.1%}
- Steering Reliability: {"RELIABLE" if carla_data['confidence'] > 0.6 else "UNRELIABLE"}
{vlm_section}

DECISION TASK:
Output ONLY a valid JSON object with these exact fields:
{{
  "final_state": "alert|mildly_drowsy|severely_drowsy|unknown",
  "intervention_action": "no_alert|soft_alert|strong_alert|takeover_request",
  "llm_confidence": 0.X (0.0 to 1.0),
  "reasoning": "1-3 sentences explaining the decision and which sensors were most informative"
}}

Consider:
1. Which sensors are reliable (camera/steering confidence > 0.6)?
2. Does VLM indicate conditions that affect eye metric reliability?
3. Do reliable sensors agree or conflict?
4. What is the most conservative safe decision?

RESPOND WITH JSON ONLY:"""
        
        return system_prompt + "\n\n" + user_prompt


    def _query_ollama(self, prompt: str) -> dict:
        """Query Ollama with integrated prompt."""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3  # Lower temp for safety-critical decisions
            }
            
            response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                text_output = result.get("response", "")
                self.get_logger().debug(f"Raw response: {text_output[:300]}")
                return self._parse_json_response(text_output)
            else:
                self.get_logger().error(f"Ollama error: {response.status_code}")
                return None
        
        except requests.exceptions.Timeout:
            self.get_logger().error("Ollama request timed out (90s)")
            return None
        except Exception as e:
            self.get_logger().error(f"Ollama query error: {e}")
            return None


    def _parse_json_response(self, text_output: str) -> dict:
        """Extract and parse JSON from Ollama response."""
        try:
            # Clean markdown
            text_output = text_output.replace("```json", "").replace("```", "")
            
            # Find JSON
            json_match = re.search(
                r'\{[^{}]*"final_state"[^{}]*"intervention_action"[^{}]*\}',
                text_output,
                re.DOTALL
            )
            
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate required fields
                required = ['final_state', 'intervention_action', 'llm_confidence', 'reasoning']
                if all(field in parsed for field in required):
                    return parsed
                else:
                    self.get_logger().warn(f"Missing fields: {parsed}")
                    return None
            else:
                self.get_logger().warn(f"No JSON found: {text_output[:200]}")
                return None
        
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON parse error: {e}")
            return None


    def _publish_alert(self, alert_data: dict):
        """Publish integrated safety-critical alert (every 60s)."""
        try:
            alert_json = {
                'driver_id': self.driver_id,
                'final_state': alert_data.get('final_state', 'unknown'),
                'intervention_action': alert_data.get('intervention_action', 'soft_alert'),
                'llm_confidence': float(alert_data.get('llm_confidence', 0.5)),
                'reasoning': str(alert_data.get('reasoning', 'Evaluation required')),
                'timestamp': time.time(),
                'llm_model': self.ollama_model,
                'window_duration_sec': 60,
                'sensors': ['camera_ml', 'carla_ml', 'vlm_context']
            }
            
            msg = String()
            msg.data = json.dumps(alert_json, indent=2)
            
            self.alert_pub.publish(msg)
            
            self.get_logger().info(
                f"\n{'='*70}\n"
                f"✅ 60-SECOND ALERT PUBLISHED:\n"
                f"{'='*70}\n"
                f"State: {alert_json['final_state']}\n"
                f"Action: {alert_json['intervention_action']}\n"
                f"LLM Confidence: {alert_json['llm_confidence']:.1%}\n"
                f"Reasoning: {alert_json['reasoning']}\n"
                f"{'='*70}"
            )
        
        except Exception as e:
            self.get_logger().error(f"❌ Failed to publish alert: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedSafetyCriticalLLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping Integrated Safety-Critical LLM Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()