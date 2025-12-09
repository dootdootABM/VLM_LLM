#!/usr/bin/env python3
"""
ROS2 Node: Drowsiness Alert Dispatcher
Subscribes to:
  - /drowsiness_alert (JSON from integrated_safety_critical_llm_node)

Publishes to (based on final_state and intervention_action):
  - /fan_speed (Int32) → Fan Controller
  - /audio_file (String) → Audio Player
  - /wheel_vibration (Vibration) → Steering Wheel Vibration

Decision Mapping:
  - alert (normal) → No action
  - mildly_drowsy + soft_alert → Fan (level 1) + Audio warning
  - severely_drowsy + strong_alert → Fan (level 2) + Audio alert + Wheel vibration (medium)
  - severely_drowsy + takeover_request → Fan (level 3) + Audio urgent + Wheel vibration (high)
  - unknown + soft_alert → Fan (level 1) + Audio caution
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
from drowsiness_detection_msg.msg import Vibration
import json
import os
from pathlib import Path


class DrowsinessAlertDispatcher(Node):
    """Dispatcher node that triggers actuators based on LLM drowsiness decisions."""

    def __init__(self):
        super().__init__("drowsiness_alert_dispatcher")

        # === Parameters ===
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value
        
        self.declare_parameter("audio_dir", "./audio_alerts")
        self.audio_dir = self.get_parameter("audio_dir").value

        # === Publishers ===
        self.fan_pub = self.create_publisher(Int32, "/fan_speed", 10)
        self.audio_pub = self.create_publisher(String, "/audio_file", 10)
        self.vibration_pub = self.create_publisher(Vibration, "/wheel_vibration", 10)

        # === Subscriber ===
        self.alert_sub = self.create_subscription(
            String,
            "/drowsiness_alert",
            self.alert_callback,
            10
        )

        # === Audio file mapping ===
        self.audio_files = {
            'soft_warning': os.path.join(self.audio_dir, 'soft_warning.wav'),
            'mild_alert': os.path.join(self.audio_dir, 'mild_alert.wav'),
            'strong_alert': os.path.join(self.audio_dir, 'strong_alert.wav'),
            'urgent_alert': os.path.join(self.audio_dir, 'urgent_alert.wav'),
        }

        # === State tracking ===
        self.last_state = None
        self.last_action = None

        self.get_logger().info(
            f"✅ Drowsiness Alert Dispatcher started\n"
            f"   Driver ID: {self.driver_id}\n"
            f"   Audio directory: {self.audio_dir}\n"
            f"   Subscribing to /drowsiness_alert"
        )
        self.get_logger().info(
            f"   Publishing to:\n"
            f"     - /fan_speed (Fan Controller)\n"
            f"     - /audio_file (Audio Player)\n"
            f"     - /wheel_vibration (Steering Wheel)"
        )

    def alert_callback(self, msg: String):
        """Receive LLM drowsiness alert and dispatch to actuators."""
        try:
            alert_data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"❌ Failed to parse alert JSON: {e}")
            return

        final_state = alert_data.get('final_state', 'unknown')
        intervention_action = alert_data.get('intervention_action', 'no_alert')
        reasoning = alert_data.get('reasoning', '')
        llm_confidence = alert_data.get('llm_confidence', 0.5)

        self.get_logger().info(
            f"\n{'='*70}\n"
            f"📢 DROWSINESS ALERT RECEIVED\n"
            f"{'='*70}\n"
            f"State: {final_state}\n"
            f"Action: {intervention_action}\n"
            f"Confidence: {llm_confidence:.1%}\n"
            f"Reasoning: {reasoning}\n"
            f"{'='*70}"
        )

        # Dispatch based on state + action
        self._dispatch_alert(final_state, intervention_action, llm_confidence)

    def _dispatch_alert(self, final_state: str, intervention_action: str, confidence: float):
        """Dispatch alert to appropriate actuators."""

        # ========== ALERT (Normal, no drowsiness) ==========
        if final_state == "alert":
            self.get_logger().info("✅ Driver is ALERT - No action needed")
            self._deactivate_all()
            self.last_state = "alert"

        # ========== MILDLY DROWSY + SOFT ALERT ==========
        elif final_state == "mildly_drowsy" and intervention_action == "soft_alert":
            self.get_logger().warn("⚠️  MILDLY DROWSY - Soft alert activated")
            self._activate_fan(level=1)  # Low fan
            self._play_audio('soft_warning')  # Warning beep
            self.last_state = "mildly_drowsy"
            self.last_action = "soft_alert"

        # ========== UNKNOWN + SOFT ALERT ==========
        elif final_state == "unknown" and intervention_action == "soft_alert":
            self.get_logger().warn("❓ UNKNOWN CONDITIONS - Caution alert")
            self._activate_fan(level=1)  # Low fan
            self._play_audio('soft_warning')  # Caution beep
            self.last_state = "unknown"
            self.last_action = "soft_alert"

        # ========== SEVERELY DROWSY + STRONG ALERT ==========
        elif final_state == "severely_drowsy" and intervention_action == "strong_alert":
            self.get_logger().error("🔴 SEVERELY DROWSY - Strong alert activated")
            self._activate_fan(level=2)  # Medium fan
            self._play_audio('strong_alert')  # Loud alert
            self._activate_vibration(duration=0.3, intensity=40)  # Medium vibration
            self.last_state = "severely_drowsy"
            self.last_action = "strong_alert"

        # ========== SEVERELY DROWSY + TAKEOVER REQUEST ==========
        elif final_state == "severely_drowsy" and intervention_action == "takeover_request":
            self.get_logger().error("🚨 SEVERE DROWSINESS - TAKEOVER REQUEST ACTIVATED")
            self._activate_fan(level=3)  # Maximum fan
            self._play_audio('urgent_alert')  # Urgent alarm
            self._activate_vibration(duration=0.5, intensity=60)  # Max vibration
            self.last_state = "severely_drowsy"
            self.last_action = "takeover_request"

        # ========== MILDLY DROWSY + STRONG ALERT (Escalation) ==========
        elif final_state == "mildly_drowsy" and intervention_action == "strong_alert":
            self.get_logger().warn("🟠 MILDLY DROWSY (escalated) - Strong alert")
            self._activate_fan(level=2)  # Medium fan
            self._play_audio('strong_alert')  # Loud alert
            self._activate_vibration(duration=0.2, intensity=30)  # Light vibration
            self.last_state = "mildly_drowsy"
            self.last_action = "strong_alert"

        # ========== DEFAULT / UNKNOWN ==========
        else:
            self.get_logger().info(f"ℹ️  State: {final_state}, Action: {intervention_action}")
            self._deactivate_all()
            self.last_state = final_state
            self.last_action = intervention_action

    def _activate_fan(self, level: int):
        """Activate fan at specified level (0-3)."""
        msg = Int32()
        msg.data = level
        self.fan_pub.publish(msg)
        self.get_logger().info(f"📤 Fan activated: Level {level}")

    def _play_audio(self, audio_type: str):
        """Play audio alert."""
        audio_file = self.audio_files.get(audio_type)

        if not audio_file:
            self.get_logger().warn(f"Unknown audio type: {audio_type}")
            return

        if not os.path.isfile(audio_file):
            self.get_logger().warn(f"Audio file not found: {audio_file}")
            return

        msg = String()
        msg.data = audio_file
        self.audio_pub.publish(msg)
        self.get_logger().info(f"📤 Audio playing: {audio_type} ({audio_file})")

    def _activate_vibration(self, duration: float, intensity: int):
        """Activate steering wheel vibration."""
        msg = Vibration()
        msg.duration = duration
        msg.intensity = intensity
        self.vibration_pub.publish(msg)
        self.get_logger().info(
            f"📤 Steering wheel vibration activated: {duration}s @ {intensity}%"
        )

    def _deactivate_all(self):
        """Deactivate all actuators."""
        self.get_logger().info("✅ All actuators deactivated")
        # Fan off
        msg_fan = Int32()
        msg_fan.data = 0
        self.fan_pub.publish(msg_fan)


def main(args=None):
    rclpy.init(args=args)
    node = DrowsinessAlertDispatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Drowsiness Alert Dispatcher...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()