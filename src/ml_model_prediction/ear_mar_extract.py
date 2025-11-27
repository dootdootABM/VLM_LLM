import cv2, csv, numpy as np, mediapipe as mp

# --- Inputs ---
video_path = "C:/Users/Downloads/10.MOV/10.mov"  # path to video ( enter correct format .mov or .mp4 )
model_asset_path = "C:/Users/Downloads/face_landmarker.task"  # path to MediaPipe Face Landmarker model

# --- MediaPipe setup ---
from mediapipe import solutions as mp_solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_asset_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Landmark indices (MediaPipe FaceMesh) ---
right_eye = [33, 159, 158, 133, 153, 145]      # 6 points
left_eye  = [362, 380, 374, 263, 386, 385]     # 6 points
mouth_idx = [78, 81, 13, 311, 308, 402, 14, 178]  # 8 outer-lip points

def ear_from_landmarks(pts, eye_idx):
    a = np.linalg.norm(pts[eye_idx[1]] - pts[eye_idx[5]])
    b = np.linalg.norm(pts[eye_idx[2]] - pts[eye_idx[4]])
    c = np.linalg.norm(pts[eye_idx[0]] - pts[eye_idx[3]])
    return (a + b) / (2.0 * c) if c > 0 else np.nan

def avg_ear(pts):
    return 0.5*(ear_from_landmarks(pts, left_eye) + ear_from_landmarks(pts, right_eye))

def mar_from_landmarks(pts):
    a = np.linalg.norm(pts[mouth_idx[1]] - pts[mouth_idx[7]])
    b = np.linalg.norm(pts[mouth_idx[2]] - pts[mouth_idx[6]])
    c = np.linalg.norm(pts[mouth_idx[3]] - pts[mouth_idx[5]])
    d = np.linalg.norm(pts[mouth_idx[0]] - pts[mouth_idx[4]])
    denom = 2.0 * d
    return (a + b + c) / denom if denom > 0 else np.nan

def mp_landmarks_to_numpy(image_rgb, mp_lms):
    h, w = image_rgb.shape[:2]
    # mp_lms are normalized [0..1]; convert to pixel coordinates
    pts = np.array([[lm.x * w, lm.y * h] for lm in mp_lms], dtype=np.float32)
    return pts

# --- Video loop ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0  # fallback

rows = []
frame_idx = 0

while True:
    ok, bgr = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    timestamp_s = frame_idx / fps
    if len(result.face_landmarks) == 0:
        rows.append((frame_idx, timestamp_s, np.nan, np.nan))
    else:
        lms = result.face_landmarks[0]
        pts = mp_landmarks_to_numpy(rgb, lms)
        ear = float(avg_ear(pts))
        mar = float(mar_from_landmarks(pts))
        rows.append((frame_idx, timestamp_s, ear, mar))
    frame_idx += 1

cap.release()

with open("ear_mar_per_frame.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time_s", "EAR", "MAR"])
    writer.writerows(rows)

print("Saved ear_mar_per_frame.csv with", len(rows), "rows")