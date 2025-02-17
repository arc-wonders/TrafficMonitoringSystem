from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 Nano version (fast and lightweight)
reader = easyocr.Reader(['en'])  # Initialize OCR reader

# Open video file
video_path = 0 # Replace with 0 for webcam54
cap = cv2.VideoCapture(video_path)

car_count = 0  # Initialize car count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 detection
    results = model.predict(frame, conf=0.25, show=False)
    detections = results[0].boxes.data  # Get bounding boxes
    
    car_count = 0  # Reset car count per frame
    violating_cars = []
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) == 2:  # Class 2 is "car"
            car_count += 1
            plate_region = frame[int(y1):int(y2), int(x1):int(x2)]  # Extract license plate area
            plate_text = reader.readtext(plate_region, detail=0)  # Read plate number
            
            # Assume a simple rule: If car is in a restricted zone, it's violating
            if y2 > frame.shape[0] * 0.75:  # Example rule (modify as needed)
                violating_cars.append(plate_text)
    
    # Display information
    cv2.putText(frame, f"Cars: {car_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if violating_cars:
        cv2.putText(frame, f"Violations: {', '.join(str(p) for p in violating_cars)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
