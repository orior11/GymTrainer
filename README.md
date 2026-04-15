# GymTrainer: AI-Powered Real-Time Exercise Analysis 🏋️‍♂️🤖
Deep Learning Final Project (2026a) | HIT Authors: Amit Wagensberg & Ori Zarfaty

## 📌 Overview
GymTrainer is a real-time computer vision application designed to act as a personal AI fitness trainer. It utilizes MediaPipe Pose for skeletal tracking and a custom Gated Recurrent Unit (GRU) neural network, implemented in PyTorch, to recognize exercises and count repetitions with high accuracy. 

By analyzing the temporal dynamics of movement, GymTrainer distinguishes between similar exercises and provides immediate audio-visual feedback, ensuring a robust solution for home and gym environments.

## 📉 Baseline Comparison
To evaluate the effectiveness of our deep learning approach, we compared the GRU model against a Random Forest baseline:
* **The Challenge:** The Random Forest model processes frames independently, making it difficult to distinguish between static poses and active movement.
* **The Solution:** The GRU model analyzes a sliding window of 30 frames. This allows the system to understand the "flow" of motion and the temporal relationship between different phases of an exercise.

## 🚀 Features
* **Real-Time Action Recognition:** Classifies 4 distinct exercises:
    * Squat
    * Push-up
    * Shoulder Press
    * Barbell Bicep Curl
* **Repetition Counting:** Combines geometric triggers (2D angle calculation) with GRU classification to count valid repetitions accurately based on movement stages (up/down).
* **Audio-Visual Feedback:** Real-time TTS (Text-to-Speech) announcements for exercise changes and rep counts, alongside on-screen skeletal overlays and stability indicators.
* **Privacy First:** All processing is performed locally on the CPU/GPU using PyTorch and OpenCV; no video data is sent to the cloud.

## 🛠️ Architecture
The system operates on a 3-stage pipeline:
1.  **Data Acquisition:** Extracts 33 3D skeletal landmarks from webcam or video streams using MediaPipe Pose.
2.  **Feature Engineering:** Converts raw coordinates into 6 biomechanical 3D joint angles to ensure the model remains invariant to camera distance and user height.
3.  **Classification:** A GRU model receives a sequence of 30 frames and outputs a probability distribution across the exercise classes.

## 📂 Project Structure
```text
GymTrainer/
├── gym_gru_model.pth           # 🏆 Final GRU Model Weights (PyTorch)
├── gym_pose_classifier.pkl     # 📉 Baseline Random Forest Model
├── X_data.npy                  # Processed features (30-frame sequences)
├── y_data.npy                  # Labels for training
├── classes.npy                 # Class names mapping
├── preprocess.py               # Data extraction & angle calculation from videos
├── train_model.py              # GRU training and validation script
├── main.py                     # 🚀 Main Live Application (Inference)
├── Final_project_poster.pptx   # Project Poster Presentation
└── README.md                   # Project Documentation# GymTrainer: AI-Powered Real-Time Exercise Analysis 🏋️‍♂️🤖

**Deep Learning Final Project (2026a) | HIT** **Authors:** Amit Wagensberg & Ori Zarfaty

## 📌 Overview
GymTrainer is a real-time computer vision application designed to act as a personal AI fitness trainer. It utilizes **MediaPipe Pose** for skeletal tracking and a custom **Gated Recurrent Unit (GRU)** neural network, implemented in **PyTorch**, to recognize exercises and count repetitions with high accuracy.

By analyzing the temporal dynamics of movement, GymTrainer distinguishes between similar exercises and provides immediate audio-visual feedback, ensuring a robust solution for home and gym environments.

## 📉 Baseline Comparison
To evaluate the effectiveness of our deep learning approach, we compared the GRU model against a **Random Forest** baseline:
* **The Challenge:** The Random Forest model processes frames independently, making it difficult to distinguish between static poses and active movement.
* **The Solution:** The **GRU** model analyzes a **sliding window of 30 frames**. This allows the system to understand the "flow" of motion and the temporal relationship between different phases of an exercise.

## 🚀 Features
* **Real-Time Action Recognition:** Classifies 4 distinct exercises:
    * Squat
    * Push-up
    * Shoulder Press
    * Barbell Bicep Curl
* **Repetition Counting:** Combines geometric triggers with GRU classification to count valid repetitions accurately.
* **Calorie Estimation:** Estimates calories burned per rep based on exercise type and intensity.
* **Audio-Visual Feedback:** Real-time TTS (Text-to-Speech) announcements and on-screen skeletal overlays with stability indicators.
* **Privacy First:** All processing is performed locally on the CPU using PyTorch; no video data is sent to the cloud.

## 🛠️ Architecture
The system operates on a 3-stage pipeline:

1. **Data Acquisition:** Extracts 33 3D skeletal landmarks from the webcam stream using **MediaPipe Pose**.
2. **Feature Engineering:** Converts raw coordinates into **6 biomechanical joint angles** (knees, elbows, and shoulders) to ensure the model remains invariant to camera distance and user height.
3. **Classification:** A **GRU** model receives a sequence of 30 frames and outputs a probability distribution across the exercise classes.

## 📂 Project Structure

```text
GymTrainer/
├── GymTrainer/                 # 🆕 Updated New Project Folder
│   ├── data/                   # Raw video datasets
│   ├── main.py                 # Main live application
│   ├── preprocess.py           # Data extraction script
│   └── train_model.py          # GRU training script
│
├── GymTrainerPro/              # 🏛️ Old Project Folder
│   ├── data/
│   ├── classes.npy
│   ├── gym_gru_model.pth
│   ├── gym_pose_classifier.pkl
│   ├── main.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── X_data.npy
│   └── y_data.npy
│
├── classes.npy                 # Root class names mapping
├── gym_gru_model.pth           # Final GRU Model Weights
├── gym_pose_classifier.pkl     # Baseline Random Forest Model
├── X_data.npy                  # Processed feature sequences
└── y_data.npy                  # Labels for training
