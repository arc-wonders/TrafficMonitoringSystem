import os
import cv2
import numpy as np
import torch
from typing import Any
import supervision as sv  # Import Supervision for polygon zone masking

# Workaround for OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure PyTorch uses GPU if available
torch.set_num_threads(1)
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Import all required modules from your project
from calibration.calibration_utils import calibrate_camera, undistort_image
from detection.yolo_utils import load_yolo_model, detect_vehicles
from tracking.tracker import SORTTracker
from speed_estimation.speed_utils import estimate_speed

def compute_perspective_transform() -> np.ndarray:
    """ 
    Computes a perspective transform matrix mapping image coordinates to real-world coordinates.
    """
    image_points = np.float32([
        [800, 410], [1125, 410], [1920, 850], [0, 850]
    ])
    real_points = np.float32([
        [0, 0], [32, 0], [32, 140], [0, 140]
    ])
    return cv2.getPerspectiveTransform(image_points, real_points)

def main() -> None:
    try:
        calib_data = np.load("calibration/calibration_data.npz")
        camera_matrix: np.ndarray = calib_data["camera_matrix"]
        dist_coeffs: np.ndarray = calib_data["dist_coeffs"]
    except FileNotFoundError:
        ret, camera_matrix, dist_coeffs, _, _ = calibrate_camera("calibration/calibration_images", display=False)
        np.savez("calibration/calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    
    print("Loading YOLO model...")
    yolo_model = load_yolo_model("detection/yolov8x.pt", device=device)
    perspective_matrix = compute_perspective_transform()
    tracker = SORTTracker()

    # Define Polygonal Zone using user-defined points
    image_points = np.float32([
        [800, 410], [1125, 410], [1920, 850], [0, 850]
    ])
    zone = sv.PolygonZone(image_points, (sv.Position.TOP_CENTER, sv.Position.BOTTOM_CENTER))

    video_path: str = "data/videos/traffic_video.mp4"
    cap: Any = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return
    
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 1000:
        fps = 25.0
    time_interval: float = 1.0 / fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = undistort_image(frame, camera_matrix, dist_coeffs)
        detections = detect_vehicles(yolo_model, undistorted_frame)
        tracked_objects = tracker.update(detections, frame=undistorted_frame)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Check if object is within the user-defined polygonal zone
            if not zone.contains(np.array([[center_x, center_y]])):
                continue
            
            speed = estimate_speed(obj, perspective_matrix, time_interval)
            cv2.rectangle(undistorted_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                undistorted_frame,
                f"ID:{int(track_id)} {speed:.1f} km/h",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        cv2.imshow("Speed Detection", undistorted_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
