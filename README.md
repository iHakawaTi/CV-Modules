# Computer Vision Modules with MediaPipe

A curated set of computer vision modules leveraging [MediaPipe](https://mediapipe.dev/) to detect **hand landmarks**, **pose estimation**, and **facial features**. Designed for modular integration, rapid prototyping, and deployment in computer vision applications.

---

## Features

This repository currently includes the following modules:

### Hand Detection Module

* Detects and tracks 21 hand landmarks in real-time.
* Supports both single and multi-hand detection.
* Useful for gesture recognition, human-computer interaction, and sign language interpretation.

### Pose Estimation Module

* Provides full-body pose detection using 33 landmarks.
* Captures key joint angles and body posture for motion tracking, fitness analysis, and AR applications.

### Facial Feature Module

* Detects facial landmarks including eyes, lips, nose, and eyebrows.
* Enables facial expression analysis, eye tracking, and emotion detection.

### Face Mesh Module
* Predicts over 460 high-fidelity facial landmarks.
* Supports 3D face modeling, detailed facial tracking, and AR overlays.
  
All modules are built on **MediaPipe**, ensuring efficient performance and high accuracy.

---

## Tech Stack

* **MediaPipe** – Core framework for ML pipelines and landmark detection
* **OpenCV** – For image capture, display, and frame processing
* **Python** – Primary language for scripting and modular logic

---

## Getting Started

### Installation

Make sure you have Python 3.7 or later installed.

```bash
pip install mediapipe opencv-python
```

### Clone the Repository

```bash
git clone https://github.com/iHakawaTi/CV-Modules.git
cd CV-Modules
```

---

## Project Structure

```
.
├── hand_module.py        # Hand detection and tracking
├── pose_module.py        # Full-body pose estimation
├── face_module.py        # Basic facial landmarks
├── face_mesh_module.py   # Detailed face mesh detection
└── examples/             # Example scripts for testing modules
```

---

## Use Cases

* Fitness and sports motion tracking
* Gesture-controlled interfaces
* Real-time facial expression analysis
* AR/VR applications
* Sign language interpretation

---

## Contributing

Pull requests and feedback are welcome. If you find a bug or want to suggest a feature, feel free to open an issue.

---

## License

MIT License – you are free to use, modify, and distribute this project.

---

