# GymTrainer: AI-Powered Real-Time Exercise Analysis 🏋️‍♂️🤖

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
```bash
GymTrainer/
├── models/
│   ├── gym_gru_model.pt       # 🏆 Final GRU Model (PyTorch)
│   └── random_forest_base.pkl # 📉 Baseline Random Forest Model
│
├── data_processing/
│   ├── X_data.npy              # Processed features (30-frame sequences)
│   ├── y_data.npy              # Labels
│   └── classes.npy             # Class names
│
├── scripts/
│   ├── train_gru.py           # GRU training script (PyTorch)
│   ├── preprocess.py           # Data extraction & angle calculation
│   ├── rf_baseline.py          # Random Forest training script
│   └── main.py                 # 🚀 Main Application (Inference)
│
├── Final_project_poster.pptx   # Project Poster Presentation
└── README.md                   # Project Documentation
