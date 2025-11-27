import ollama
import json
from datetime import datetime

# ============================================
# Dummy Input Data Structure
# ============================================

# 1. Sensor Features (from your drowsiness detection pipeline)
sensor_features = {
    "timestamp": datetime.now().isoformat(),
    "EAR": 0.18,  # Eye Aspect Ratio (normal ~0.25-0.30, drowsy <0.20)
    "MAR": 0.45,  # Mouth Aspect Ratio (normal <0.5, yawning >0.6)
    "PERCLOS": 0.35,  # Percentage Eye Closure (drowsy >0.15)
    "blink_rate": 28,  # blinks per minute (normal 15-20, drowsy >25)
    "yawn_count": 3,  # yawns in last minute
    "head_pose_pitch": -12.5,  # degrees (negative = head dropping)
    "head_pose_yaw": 2.3,
    "heart_rate": 68,  # bpm
    "heart_rate_variability": 42,  # ms (lower = more drowsy)
}

# 2. Model Probabilities (from your ML classifier)
model_probabilities = {
    "alert": 0.15,
    "mildly_drowsy": 0.52,
    "severely_drowsy": 0.33,
    "predicted_class": "mildly_drowsy",
    "confidence": 0.52
}

# 3. Qwen VLM Context (extracted from camera image)
qwen_context = {
    "scene_description": "Highway driving at night. Driver's eyes are partially closed with visible drooping eyelids. The cabin lighting is dim. No phone or distraction detected.",
    "environmental_factors": ["nighttime", "highway", "low_cabin_light", "monotonous_road"],
    "detected_behaviors": ["eyes_partially_closed", "head_tilting_forward", "reduced_facial_expression"],
    "risk_indicators": ["prolonged_eye_closure", "head_nodding"],
    "ambiguity_flag": False,
    "occlusion_detected": False
}

# 4. Recent History Window (temporal context from last 30 seconds)
recent_history = [
    {"time": "-30s", "EAR": 0.28, "class": "alert"},
    {"time": "-25s", "EAR": 0.26, "class": "alert"},
    {"time": "-20s", "EAR": 0.22, "class": "alert"},
    {"time": "-15s", "EAR": 0.19, "class": "mildly_drowsy"},
    {"time": "-10s", "EAR": 0.17, "class": "mildly_drowsy"},
    {"time": "-5s", "EAR": 0.18, "class": "mildly_drowsy"},
    {"time": "now", "EAR": 0.18, "class": "mildly_drowsy"}
]

# ============================================
# Build Prompt for Llama 3.1 Reasoning
# ============================================

def build_reasoning_prompt(sensor_features, model_probabilities, qwen_context, recent_history):
    prompt = f"""You are an expert driver drowsiness detection system. Analyze the following multimodal inputs and provide a final drowsiness assessment with reasoning.

## SENSOR FEATURES:
{json.dumps(sensor_features, indent=2)}

## MODEL PROBABILITIES:
{json.dumps(model_probabilities, indent=2)}

## VISUAL CONTEXT (from VLM):
{json.dumps(qwen_context, indent=2)}

## RECENT HISTORY (last 30 seconds):
{json.dumps(recent_history, indent=2)}

## YOUR TASK:
Based on ALL the above information:
1. Determine the final drowsiness state: ALERT, MILDLY_DROWSY, or SEVERELY_DROWSY
2. Calculate a risk score (0-100, where 100 = critical drowsiness)
3. Provide a clear explanation justifying your decision
4. Suggest immediate action (if needed)

## OUTPUT FORMAT (JSON):
{{
  "final_drowsiness_state": "MILDLY_DROWSY or SEVERELY_DROWSY or ALERT",
  "risk_score": 75,
  "confidence": 0.85,
  "reasoning": "Detailed explanation considering all inputs...",
  "key_factors": ["factor1", "factor2", "factor3"],
  "recommended_action": "Alert driver with audio warning",
  "temporal_trend": "deteriorating or stable or improving"
}}

Provide ONLY the JSON output, no additional text."""
    
    return prompt

# ============================================
# Call Llama 3.1 for Reasoning
# ============================================

def get_llama_reasoning(prompt, model_name="llama3.1:8b"):
    """
    Call Llama 3.1 via Ollama for drowsiness reasoning
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are an expert AI system for driver safety analysis. Always respond in valid JSON format.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.2,  # Low temperature for consistent reasoning
                'top_p': 0.9,
                'num_predict': 500  # Max tokens for response
            }
        )
        
        return response['message']['content']
    
    except Exception as e:
        print(f"Error calling Llama 3.1: {e}")
        return None

# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("DRIVER DROWSINESS REASONING WITH LLAMA 3.1")
    print("=" * 60)
    
    # Build the prompt
    prompt = build_reasoning_prompt(
        sensor_features,
        model_probabilities,
        qwen_context,
        recent_history
    )
    
    print("\n[INFO] Sending data to Llama 3.1 for reasoning...\n")
    
    # Get reasoning from Llama 3.1
    llama_response = get_llama_reasoning(prompt)
    
    if llama_response:
        print("=" * 60)
        print("LLAMA 3.1 REASONING OUTPUT:")
        print("=" * 60)
        print(llama_response)
        print("=" * 60)
        
        # Parse and save the result
        try:
            reasoning_result = json.loads(llama_response)
            
            # Save to file for logging/evaluation
            output_file = f"reasoning_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "inputs": {
                        "sensor_features": sensor_features,
                        "model_probabilities": model_probabilities,
                        "qwen_context": qwen_context,
                        "recent_history": recent_history
                    },
                    "llama_reasoning": reasoning_result
                }, f, indent=2)
            
            print(f"\n[SUCCESS] Reasoning saved to: {output_file}")
            
        except json.JSONDecodeError:
            print("\n[WARNING] Response is not valid JSON, saving as raw text...")
            with open(f"reasoning_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
                f.write(llama_response)
    
    else:
        print("[ERROR] Failed to get response from Llama 3.1")
