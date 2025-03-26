from ultralytics import YOLO
from typing import List
import numpy as np
import cv2

def load_yolo_model(model_path: str, device: str = "cuda") -> YOLO:
    """
    Loads the YOLOv8 model from the specified path and moves it to the given device.
    
    Parameters:
        model_path (str): Path to the YOLOv8 model file.
        device (str): Device to run the model on, e.g., "cuda" or "cpu".
    
    Returns:
        YOLO: The loaded YOLOv8 model.
    """
    model = YOLO(model_path)
    model.to(device)
    return model

def detect_vehicles(model: YOLO, frame: np.ndarray) -> List[List[float]]:
    """
    Runs YOLOv8 on a frame and extracts vehicle detections.
    
    Parameters:
        model (YOLO): The loaded YOLOv8 model.
        frame (np.ndarray): Input image frame (BGR format expected).
    
    Returns:
        List[List[float]]: A list of detections in the format [x1, y1, x2, y2, confidence].
                           Bounding box coordinates are integers, confidence is a float.
    """
    results = model(frame)
    detections: List[List[float]] = []
    
    # Iterate over each result from the model.
    for r in results:
        # Iterate through each detected box.
        for box in r.boxes:
            # Ensure bounding box and confidence values are available.
            if box.xyxy is None or len(box.xyxy) == 0:
                continue
            if box.conf is None or len(box.conf) == 0:
                continue
            
            # Convert bounding box coordinates to integers.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])
    
    return detections

# Example usage (for testing purposes):
if __name__ == "__main__":
    # Load model and test detection on an example frame.
    model = load_yolo_model("best.pt", device="cuda")
    
    # Create a dummy frame for testing (normally, you'll use cv2.imread or a video stream).
    dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    
    detected_objects = detect_vehicles(model, dummy_frame)
    print("Detections:", detected_objects)
