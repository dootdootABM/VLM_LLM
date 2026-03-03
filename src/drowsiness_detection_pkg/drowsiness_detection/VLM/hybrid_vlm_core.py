#!/usr/bin/env python3
"""
Hybrid Drowsiness Detection - Core VLM Logic (No ROS, No Camera)
Extracted from webcam script for use in ROS2 subscriber node.

Classes:
- OcclusionEventTracker: tracks events from start to end
- VLMRequestHandler: manages async Qwen2.5-VL calls
- detect_ambiguity_flags(): detects occlusion/suspicious conditions
- encode_frame_b64(): encodes frames for VLM
"""

import os
import cv2
import numpy as np
import threading
import base64
from collections import deque
from datetime import datetime
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# SYSTEM PROMPT (strict JSON schema)
# ----------------------------
SYSTEM_PROMPT = """
SYSTEM: You are Qwen2.5-VL (qwen2.5vl:3b-q8_0) for automated driver behavior evaluation.
You will receive 8 consecutive frames from a driver's face/upper torso.
Produce JSON-only output following exactly this schema:


{
  "clip_id": "<string>",
  "drowsy": {"label": "yes"|"no","confidence": 0.0},
  "behaviors": {
      "yawn": {"detected": false,"confidence":0.0},
      "cover_mouth": {"detected": false,"confidence":0.0},
      "eat": {"detected": false,"confidence":0.0},
      "drink": {"detected": false,"confidence":0.0},
      "sneeze": {"detected": false,"confidence":0.0},
      "cry": {"detected": false,"confidence":0.0},
      "micro_sleep": {"detected": false,"confidence":0.0},
      "talking": {"detected": false,"confidence":0.0},
      "hands_on_face": {"detected": false,"confidence":0.0},
      "glasses": {"detected": false,"confidence":0.0},
      "sunglasses": {"detected": false,"confidence":0.0},
      "mask": {"detected": false,"confidence":0.0},
      "other_occlusion": {"detected": false,"confidence":0.0}
  },
  "eye_visibility": {"left_eye":"open","right_eye":"open"},
  "mouth_visibility": "open"|"closed"|"occluded",
  "head_posture": {"pitch":"neutral","yaw":"center","roll":"neutral"},
  "occlusion": {"face_occluded": true|false, "reason":"mask"|"hand"|"sun_glare"|"other"|null},
  "lighting": {"level":"normal","issue":null},
  "evidence_frames": [0,1,2,3,4,5,6,7],
  "notes": ""
}


- Always fill all fields, even if uncertain.
- Use numeric confidence scores 0.0-1.0.
- Do not output per-frame arrays, PERCLOS, blink rates, or markdown. Return only the JSON object.
"""

# ----------------------------
# USER PROMPT TEMPLATE
# ----------------------------
USER_PROMPT_TEMPLATE = """
You will analyze the following 8 consecutive frames from a driver's cabin using Qwen2.5VL:3b-q8_0.
Treat all 8 frames together as a single 2-second scene. Do NOT evaluate frames independently or produce per-frame outputs.
Aggregate all observed behaviors over the scene: yawning, covering mouth, eating, drinking, sneezing, crying, micro-sleep, eyes closed, looking away, talking, hands on face, wearing glasses/sunglasses, mask occlusion, or any other gestures.
Also report mouth visibility as "open", "closed", or "occluded".
Report face occlusion in "occlusion" field with reason (mask, hand, sun_glare, other, or null).
Provide a short textual summary of the scene in the "notes" field.


Clip id: {clip_id}
Frames: images[0..7] in order (0 = earliest, 7 = latest)


Answer only JSON that strictly conforms to the SYSTEM_PROMPT schema.
Always fill all fields, even if uncertain. Do NOT include per-frame outputs, metrics like PERCLOS or blink rates, or markdown.
"""

# ----------------------------
# Ollama HTTP Config
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b-q8_0"

# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    # Classical thresholds (from literature)
    "ear_low_threshold": 0.26,          # Soukupová & Čech 2016
    "ear_extreme_threshold": 0.10,      # Very low (potential occlusion)
    "mar_yawn_threshold": 0.60,         # Abtahi 2014
    
    # Blink filtering logic
    "blink_duration_max": 0.3,          # Normal blink: 100-300ms
    "drowsiness_duration_min": 2.0,     # Drowsiness: 2+ seconds
    
    # Brightness heuristics (empirical)
    "brightness_very_dark": 45,
    "brightness_dark": 80,
    "brightness_normal_min": 60,
    "brightness_normal_max": 180,
    "brightness_bright": 200,
    "brightness_overexposed": 230,
    
    # Edge density (for detecting occlusion/sunglasses/hands)
    "edge_density_threshold": 0.18,
    
    # Sampling & buffering
    "sampling_hz": 4.0,                 # 4 frames per second
    "buffer_duration": 2.0,             # 2 seconds = 8 frames
    
    # Frame encoding
    "frame_width": 480,
    "frame_height": 360,
    "jpeg_quality": 70,
    
    # VLM triggers
    "vlm_trigger_on_ambiguity": True,
    
    # HTTP & threading
    "http_timeout": 300,                # 5 minutes
    "max_concurrent_vlm": 2,
    
    # Output
    "output_dir": "./vlm_triggers",
}

# ----------------------------
# Helper: Ambiguity Detection
# ----------------------------
def detect_ambiguity_flags(frame_bgr, face_detected, ear, mar, brightness, face_conf):
    """
    Detect situations where classical metrics may be unreliable.
    Returns list of flags: "no_face", "extreme_low_ear", "conflicting_eye_mouth", 
                           "very_dark", "overexposed", "busy_edges"
    """
    flags = []

    if not face_detected or face_conf < 0.7:
        flags.append("no_face")

    if ear is not None and ear < CONFIG["ear_extreme_threshold"]:
        flags.append("extreme_low_ear")

    if ear is not None and mar is not None:
        eyes_closed = ear < CONFIG["ear_low_threshold"]
        mouth_open = mar > CONFIG["mar_yawn_threshold"]
        if eyes_closed and mouth_open:
            flags.append("conflicting_eye_mouth")

    if brightness is not None:
        if brightness < CONFIG["brightness_very_dark"]:
            flags.append("very_dark")
        elif brightness > CONFIG["brightness_overexposed"]:
            flags.append("overexposed")

    if face_detected:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.mean(edges > 0)
        if edge_density > CONFIG["edge_density_threshold"]:
            flags.append("busy_edges")  # hands, sunglasses, occlusion

    return flags

def encode_frame_b64(frame, quality=70):
    """Encode frame to base64 JPEG"""
    success, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if success:
        return base64.b64encode(buf.tobytes()).decode("ascii")
    return None

# ----------------------------
# Event Tracker
# ----------------------------
class OcclusionEventTracker:
    """Tracks occlusion events from start to end, captures 8-frame progression"""

    def __init__(self):
        self.event_active = False
        self.event_frames = deque(maxlen=8)
        self.event_start_time = None
        self.last_flags = None
        self.event_metadata = {}

    def update(self, frame, has_flags, current_flags, metrics):
        """
        Update event state with new frame.
        
        Returns:
            (frames_to_submit, should_submit)
            - frames_to_submit: list of 8 frames if event ended, else None
            - should_submit: True if event ended and ready to submit, else False
        """
        should_submit = False
        frames_to_submit = None

        if has_flags and not self.event_active:
            # FLAG JUST TRIGGERED - START CAPTURING EVENT
            self.event_active = True
            self.event_start_time = time.time()
            self.event_frames.clear()
            self.event_frames.append(frame.copy())
            self.last_flags = current_flags
            self.event_metadata = metrics.copy()
            print("\n📍 EVENT STARTED: capturing frames until flags clear...")
            return None, False

        elif has_flags and self.event_active:
            # FLAG STILL ACTIVE - KEEP RECORDING
            self.event_frames.append(frame.copy())
            self.last_flags = current_flags
            print(f"   📍 Recording frame {len(self.event_frames)}/8...")
            return None, False

        elif not has_flags and self.event_active:
            # FLAG ENDED - SUBMIT FULL EVENT SEQUENCE
            self.event_active = False

            # Pad to 8 frames if needed
            while len(self.event_frames) < 8:
                self.event_frames.append(frame.copy())

            frames_to_submit = list(self.event_frames)
            should_submit = True

            event_duration = time.time() - self.event_start_time
            print(f"\n✅ EVENT ENDED: Submitting {len(frames_to_submit)} frames")
            print(f"   Duration: {event_duration:.1f}s | Flags: {','.join(self.last_flags)}")
            return frames_to_submit, should_submit

        return None, False

# ----------------------------
# VLM Request Handler
# ----------------------------
class VLMRequestHandler:
    """Handles all VLM requests in a separate thread pool"""

    def __init__(self, max_workers=2, output_dir="./vlm_triggers"):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VLM")
        self.semaphore = threading.Semaphore(max_workers)
        self.output_dir = output_dir
        self.pending_requests = 0
        self.completed_requests = 0
        self.lock = threading.Lock()

    def submit_request(self, frames, clip_id, flags, metrics):
        """
        INSTANTLY submit VLM request to thread pool and return.
        Does not block the caller.
        """
        with self.lock:
            self.pending_requests += 1

        try:
            self.semaphore.acquire(blocking=False)
            future = self.executor.submit(
                self._vlm_request_blocking,
                frames, clip_id, flags, metrics
            )
            future.add_done_callback(lambda f: self._on_request_complete())
            return future
        except Exception as e:
            print(f"[VLM] ❌ Failed to submit request: {e}")
            with self.lock:
                self.pending_requests -= 1
            self.semaphore.release()
            return None

    def _on_request_complete(self):
        """Called when a VLM request completes"""
        with self.lock:
            self.pending_requests -= 1
            self.completed_requests += 1
        self.semaphore.release()

    def _vlm_request_blocking(self, frames, clip_id, flags, metrics):
        """
        Process VLM request in background thread.
        Takes 1-3 minutes but doesn't block the main node.
        """
        imgs_b64 = [encode_frame_b64(f, CONFIG["jpeg_quality"]) for f in frames]
        if not all(imgs_b64):
            print(f"[VLM] ❌ Frame encoding failed for {clip_id}")
            return {"error": "frame encoding failed"}

        payload = {
            "model": MODEL_NAME,
            "system_prompt": SYSTEM_PROMPT,
            "prompt": USER_PROMPT_TEMPLATE.format(clip_id=clip_id),
            "images": imgs_b64,
            "stream": False,
        }

        try:
            print(f"[VLM] 🔄 Analyzing event {clip_id} ({len(frames)} frames)...")
            print(f"     Flags: {','.join(flags)}")
            start = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=CONFIG["http_timeout"])
            elapsed = time.time() - start

            print(f"[VLM] ✅ Response received ({elapsed:.1f}s)")
            resp.raise_for_status()

            vlm_json = resp.json()
            if "response" in vlm_json:
                response_text = vlm_json["response"]
                try:
                    vlm_analysis = json.loads(response_text)
                    print(f"[VLM] ✅ Parsed VLM analysis")
                    drowsy_label = vlm_analysis.get("drowsy", {}).get("label", "unknown")
                    drowsy_conf = vlm_analysis.get("drowsy", {}).get("confidence", 0)
                    print(f"     Drowsy: {drowsy_label} (conf: {drowsy_conf:.2f})")
                    notes = vlm_analysis.get("notes", "N/A")[:100]
                    print(f"     Notes: {notes}")
                    self._save_clip(clip_id, frames, vlm_analysis, flags)
                    return vlm_analysis
                except json.JSONDecodeError as je:
                    print(f"[VLM] ❌ Response not valid JSON: {je}")
                    self._save_clip(clip_id, frames, {"error": "invalid json"}, flags)
                    return {"error": "invalid json response"}
            else:
                self._save_clip(clip_id, frames, vlm_json, flags)
                return vlm_json

        except requests.Timeout:
            print(f"[VLM] ❌ TIMEOUT after {CONFIG['http_timeout']}s")
            return {"error": "timeout"}
        except Exception as e:
            print(f"[VLM] ❌ HTTP error: {e}")
            return {"error": str(e)}

    def _save_clip(self, clip_id, frames, vlm_analysis, flags):
        """Save frames and analysis to disk"""
        out_dir = os.path.join(self.output_dir, "events", clip_id)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"[SAVE] ❌ Failed to create directory: {e}")
            return

        frames_saved = 0
        for i, f in enumerate(frames):
            try:
                frame_path = os.path.join(out_dir, f"frame_{i:02d}.jpg")
                if cv2.imwrite(frame_path, f):
                    frames_saved += 1
            except Exception as e:
                print(f"[SAVE] ❌ Error saving frame {i}: {e}")

        try:
            analysis = {
                "clip_id": clip_id,
                "timestamp": datetime.now().isoformat(),
                "frames_count": frames_saved,
                "flags_detected": flags,
                "vlm_analysis": vlm_analysis,
            }
            json_path = os.path.join(out_dir, "analysis.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(analysis, fh, indent=2, ensure_ascii=False)

            print(f"✅ SAVED: {frames_saved}/{len(frames)} frames")
            print(f"   Directory: {out_dir}")
        except Exception as e:
            print(f"❌ SAVE error: {e}")

    def shutdown(self):
        """Gracefully shutdown VLM thread pool"""
        print("\n⏳ Shutting down VLM thread pool...")
        print(f"   Pending: {self.pending_requests} | Completed: {self.completed_requests}")
        self.executor.shutdown(wait=True)
        print("✅ VLM thread pool shutdown complete")
