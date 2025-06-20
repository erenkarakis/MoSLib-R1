# MoSLib-R1

MoSLib-R1 (Monocular-Stereo Library - Release 1) is a Python-based SLAM (Simultaneous Localization and Mapping) helper library that works with both monocular and stereo camera systems. This project is designed to perform fundamental visual tasks such as camera calibration, image capturing, and depth estimation.

---

## 🚀 Features

- Monocular camera calibration  
- Stereo camera calibration  
- Real-time image capturing  
- Depth estimation (stereo matching)  
- Modular structure based on OpenCV
- Pixel depth map (under development)
- SLAM features (under development)

---

## 📁 File Structure

| File/Module               | Description                        |
|--------------------------|----------------------------------|
| `CameraCalibration.py`    | Monocular camera calibration     |
| `StereoCamCalibration.py` | Stereo camera calibration        |
| `CameraCapture.py`        | Monocular image capturing and display |
| `StereoCamCapture.py`     | Capturing images from stereo cameras |
| `DepthEstimation.py`      | Generating depth maps             |

---

## 🧰 Requirements

- Python 3.7+  
- OpenCV (`opencv-python`)  
- NumPy  

This library is still under development.
