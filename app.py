from flask import Flask, Response, send_from_directory, render_template, jsonify
import os
import cv2
import numpy as np
import torch
import json
import time

# Import modules for detection, tracking, and speed estimation
from calibration.calibration_utils import calibrate_camera, undistort_image
from detection.yolo_utils import load_yolo_model, detect_vehicles
from tracking.sort import Sort
from speed_estimation.speed_utils import estimate_speed

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load YOLO Model
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = load_yolo_model("detection/yolov8x.pt", device=device)

# Load camera calibration data
try:
    calib_data = np.load("calibration/calibration_data.npz")
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
except FileNotFoundError:
    raise Exception("Calibration data not found!")

# Compute Perspective Transform
def compute_perspective_transform():
    image_points = np.float32([[800, 410], [1125, 410], [1920, 850], [0, 850]])
    real_points = np.float32([[0, 0], [32, 0], [32, 140], [0, 140]])
    return cv2.getPerspectiveTransform(image_points, real_points)

perspective_matrix = compute_perspective_transform()

# Initialize SORT Tracker
tracker = Sort()

# Path to video file
video_path = "data/videos/traffic_video.mp4"

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield "data: " + json.dumps({"error": "Could not open video"}) + "\n\n"
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    time_interval = 1.0 / fps
    
    vehicle_count = 0
    vehicle_types = {"car": 0, "bike": 0, "truck": 0, "bus": 0}
    high_speed_violations = 0
    speed_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = undistort_image(frame, camera_matrix, dist_coeffs)
        detections = detect_vehicles(yolo_model, undistorted_frame)

        dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections]) if len(detections) > 0 else np.empty((0, 5))
        tracked_objects = tracker.update(dets)

        vehicle_data = []
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            speed = estimate_speed(obj, perspective_matrix, time_interval)
            vehicle_data.append({"id": int(track_id), "x": int(x1), "y": int(y1), "speed": round(speed, 1)})
            
            if speed > 80:
                high_speed_violations += 1
            
            speed_data.append(speed)
            vehicle_count += 1
            vehicle_types["car"] += 1  # Placeholder, replace with proper classification

        yield f"data: {json.dumps({'vehicles': vehicle_data, 'vehicle_count': vehicle_count, 'vehicle_types': vehicle_types, 'high_speed_violations': high_speed_violations, 'speed_data': speed_data})}\n\n"

        
        time.sleep(time_interval)
    
    cap.release()

@app.route('/api/detect')
def detect():
    return Response(generate_frames(), mimetype="text/event-stream")

@app.route("/")
def serve_index():
    return render_template("analysis.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)