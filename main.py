import os
import cv2
import numpy as np
import torch
from typing import Any

# Workaround for OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure PyTorch uses GPU if availableq
torch.set_num_threads(1)
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Import all required modules from your project
from calibration.calibration_utils import calibrate_camera, undistort_image
from detection.yolo_utils import load_yolo_model, detect_vehicles
from tracking.sort import Sort  # Using SORT instead of DeepSORT
from speed_estimation.speed_utils import estimate_speed

def compute_perspective_transform() -> np.ndarray:
    """ 
    Computes a perspective transform matrix mapping image coordinates to real-world coordinates.
    
    Returns:
        A 3x3 perspective transform matrix.
    """
    # Define placeholder points from your image (update these with actual calibration points)
    image_points = np.float32([
        [800, 410],   # Top-left
        [1125, 410],  # Top-right
        [1920, 850],  # Bottom-right
        [0, 850]   # Bottom-left
    ])
    
    # Define corresponding real-world coordinates (in meters)
    real_points = np.float32([
        [0,   0],   # Top-left  (real-world point)
        [32,  0],   # Top-right
        [32, 140],  # Bottom-right
        [0,  140]   # Bottom-left
    ])
    
    return cv2.getPerspectiveTransform(image_points, real_points)

def main() -> None:
    # --- Camera Calibration ---
    try:
        calib_data = np.load("calibration/calibration_data.npz")
        camera_matrix: np.ndarray = calib_data["camera_matrix"]
        dist_coeffs: np.ndarray = calib_data["dist_coeffs"]
        print("Loaded calibration data.")
    except FileNotFoundError:
        print("Calibration data not found. Running calibration...")
        ret, camera_matrix, dist_coeffs, _, _ = calibrate_camera("calibration/calibration_images", display=False)
        np.savez("calibration/calibration_data.npz",
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs)
        print("Calibration completed and data saved.")
    
    # --- Load YOLO Model ---
    print("Loading YOLO model...")
    yolo_model = load_yolo_model("detection/yolov8x.pt", device=device)
    print("YOLO model loaded successfully.")

    # --- Compute Perspective Transform ---
    perspective_matrix = compute_perspective_transform()
    print("Perspective Transform Matrix:\n", perspective_matrix)

    # --- Initialize SORT Tracker ---
    tracker = Sort()

    # --- Open Video Capture ---
    video_path: str = "data/videos/traffic_video.mp4"  # Ensure the path is correct.
    cap: Any = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Get FPS (fallback to 25 if FPS is invalid)
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 1000:
        fps = 25.0
    print("Video FPS:", fps)
    time_interval: float = 1.0 / fps

    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame using calibration data
        undistorted_frame = undistort_image(frame, camera_matrix, dist_coeffs)

        # Detect vehicles in the frame using YOLO
        detections = detect_vehicles(yolo_model, undistorted_frame)

        # Convert detections to the format required by SORT [x1, y1, x2, y2, score]
        dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections]) if len(detections) > 0 else np.empty((0, 5))

        # Track detected vehicles using SORT
        tracked_objects = tracker.update(dets)

        # Process each tracked object
        for obj in tracked_objects:
            # Unpack object bounding box and track ID: [x1, y1, x2, y2, track_id]
            x1, y1, x2, y2, track_id = obj

            # Estimate speed using the perspective transform
            speed = estimate_speed(obj, perspective_matrix, time_interval)

            # Draw bounding box and speed info on the frame
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

        # Display the processed frame
        cv2.imshow("Speed Detection", undistorted_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
