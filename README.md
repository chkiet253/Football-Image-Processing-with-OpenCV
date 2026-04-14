# ⚽ Football Image Processing with OpenCV

Welcome to the **Football Image Processing with OpenCV** project – a modular and extensible system for detecting, tracking, and analyzing football (soccer) players and ball movement from raw video footage.

A comprehensive Python toolkit for detecting, segmenting, tracking, and measuring football players and ball movements in video footage using OpenCV and computer vision techniques.

## 🧠 Project Overview

This project provides end‑to‑end image processing pipelines for football (soccer) match analysis, including:

* **Fine-Tunning YOLOv8** to bouding box
* **Player & Ball Detection** using classical computer vision (color thresholding, background/background subtraction) or optional integration with pre‑trained object detectors (YOLO, etc.).
* **Team Kit Segmentation** by clustering jersey colors (e.g. via K‑Means) to assign players to teams automatically.
* **Player Tracking** across frames using OpenCV trackers like **CSRT**, **KCF** or **MIL**.
* **Camera Motion Estimation** with optical flow to compensate for panning or zooming.
* **Perspective Transformation (Homography)** to map pixel positions into real-world coordinates (meters).
* **Speed, Distance & Activity Metrics**: compute per-player distance traveled, instantaneous speed, and activity count (frame-to-frame histogram changes).

## 🚀 Quick Start

### 1. Clone the Repository
```bash
cd Football-Image-Processing-with-OpenCV
```

### 2. Install Dependencies

Make sure you have Python 3.8+ and install required libraries:

```bash
pip install -r requirements.txt
```

### 3. Prepare Input Video

Place your input `.mp4` file in the `input_videos/` directory.

### 4. Run the Pipeline

```bash
python main.py
```
## 🖼️ Sample Input & Output

| Original Input| Processed Output|
|----------------|------------------|
| ![](sample_input.gif) | ![](sample_output.gif) |

## License

This project is licensed under the MIT License.
