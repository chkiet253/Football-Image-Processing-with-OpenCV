# ⚽ Football Image Processing with OpenCV

Welcome to the **Football Image Processing with OpenCV** project – a modular and extensible system for detecting, tracking, and analyzing football (soccer) players and ball movement from raw video footage.

A comprehensive Python toolkit for detecting, segmenting, tracking, and measuring football players and ball movements in video footage using OpenCV and computer vision techniques.

## 🧠 Project Overview

This project provides end‑to‑end image processing pipelines for football (soccer) match analysis, including:

* **Player & Ball Detection** using classical computer vision (color thresholding, background/background subtraction) or optional integration with pre‑trained object detectors (YOLO, etc.).
* **Team Kit Segmentation** by clustering jersey colors (e.g. via K‑Means) to assign players to teams automatically.
* **Player Tracking** across frames using OpenCV trackers like **CSRT**, **KCF** or **MIL**.
* **Camera Motion Estimation** with optical flow to compensate for panning or zooming.
* **Perspective Transformation (Homography)** to map pixel positions into real-world coordinates (meters).
* **Speed, Distance & Activity Metrics**: compute per-player distance traveled, instantaneous speed, and activity count (frame-to-frame histogram changes).


## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/lawrence253/Football-Image-Processing-with-OpenCV.git
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

This will:

* Detect players and the ball
* Assign teams based on color
* Track player positions
* Estimate camera movement
* Compute distance and speed
* Output annotated video to `output_videos/output_video.avi`

---

## 🖼️ Sample Input & Output

| Original Input| Processed Output|
|----------------|------------------|
| ![](sample_input.gif) | ![](sample_output.gif) |

## 🧠 Credits

* Developed by [lawrence253](https://github.com/lawrence253)
* YOLO detection via [Ultralytics](https://github.com/ultralytics/ultralytics)

## 💬 Reference

[1]: https://github.com/AnshChoudhary/Football-Tracking?utm_source=chatgpt.com "Football Tracking using YOLOv8 and OpenCV - GitHub"
[2]: https://www.iieta.org/download/file/fid/118759?utm_source=chatgpt.com "[PDF] Football Player Tracking and Performance Analysis Using ... - IIETA"
[3]: https://forum.opencv.org/t/detecting-football-color-kits-in-python/15020?utm_source=chatgpt.com "Detecting Football Color kits in Python - OpenCV Forum"
[4]: https://www.youtube.com/watch?pp=0gcJCfwAo7VqN5tD&v=neBZ6huolkg&utm_source=chatgpt.com "Build an AI/ML Football Analysis system with YOLO, OpenCV, and ..."
[5]: https://arxiv.org/abs/2204.02573?utm_source=chatgpt.com "Detecting key Soccer match events to create highlights using Computer Vision"
[6]: https://arxiv.org/abs/2402.00163?utm_source=chatgpt.com "Improving Object Detection Quality in Football Through Super-Resolution Techniques"

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
