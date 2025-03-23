# Traffic Monitoring System 🚦

## Overview
This **Traffic Monitoring System** is designed for **high-precision vehicle detection and tracking** using computer vision techniques. It can analyze traffic footage to detect:
- **Vehicle Speed** estimation
- **Traffic Light Violations**
- **Seat Belt Usage**
- **Zebra Crossing Violations**
- **Vehicle Counting & Tracking**

This system is built using **Python**, leveraging **YOLOv8, OpenCV, and Optical Flow** techniques for accuracy.

## Features ✨
- **Vehicle Detection** using YOLOv8
- **Speed Estimation** with Optical Flow (Lucas-Kanade / Farneback)
- **Multi-Point Calibration & Perspective Correction**
- **SORT Object Tracking** (custom implementation)
- **High-Precision Speed Measurement** with Multi-Frame Averaging
- **Modular Codebase** for easy scalability

## Technologies Used 🛠
- **Python**
- **YOLOv8** (Object Detection)
- **OpenCV** (Computer Vision & Image Processing)
- **SORT Tracker** (Custom implementation)
- **Homography & Camera Calibration**
- **Optical Flow** (Speed Estimation)
- **Flask** (for Web Integration, future integration planned)

## Installation 🔧
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/traffic-monitoring-system.git
cd traffic-monitoring-system
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the System
```bash
python main.py --video input_video.mp4
```

## Usage 📌
- Ensure the input video is placed in the `/videos` folder.
- Modify configuration settings in `config.py` to adjust tracking thresholds.
- Output analytics and processed videos will be stored in `/output`.

## Future Improvements 🚀
- **Integration with Live CCTV Feeds**
- **LiDAR/RADAR Sensor Support for Enhanced Accuracy**
- **Web Dashboard for Real-Time Traffic Analytics**
- **Deep Learning-Based Speed Estimation Model**

## Contributing 🤝
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## Contact 📬
For any queries or collaboration opportunities, reach out via:
- **Email:** arkinkansra@gmail.com
- **LinkedIn:** [Arkin Kansra](https://www.linkedin.com/in/arkin-kansra-8271b7233/)
