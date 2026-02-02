# GymTrainer: AI-Powered Real-Time Exercise Analysis 🏋️‍♂️🤖

**Deep Learning Final Project (2026a) | HIT**
**Authors:** Amit Wagensberg & Ori Zarfaty

![GymTrainer Banner](path/to/your/image.png)
*(Optional: Place the AI image you generated here)*

## 📌 Overview
GymTrainer is a real-time computer vision application designed to act as a personal AI fitness trainer. It utilizes **MediaPipe Pose** for skeletal tracking and a custom **Long Short-Term Memory (LSTM)** neural network to recognize exercises and count repetitions with high accuracy.

Unlike simple geometric counters, GymTrainer uses deep learning to understand the *temporal dynamics* of movement, distinguishing between similar exercises (like Shoulder Press vs. Push-ups) and providing audio-visual feedback.

## 🚀 Features
* **Real-Time Action Recognition:** Classifies 4 distinct exercises:
    * Squat
    * Push-up
    * Shoulder Press
    * Bicep Curl
* **Repetition Counting:** Uses geometric logic combined with AI classification to count valid reps.
* **Calorie Estimation:** Estimates calories burned per rep based on exercise type.
* **Audio Feedback:** TTS (Text-to-Speech) announces rep counts and sets.
* **Visual Feedback:** On-screen skeleton overlay, progress bars, and stability indicators.
* **Privacy First:** All processing is done locally on the CPU (no video sent to the cloud).

## 🛠️ Architecture
The system operates on a 3-stage pipeline:

1.  **Data Acquisition:**
    * Input: Webcam video stream.
    * Tool: **MediaPipe Pose** extracts 33 3D skeletal landmarks.
2.  **Feature Engineering:**
    * Raw coordinates (x, y, z) are converted into **6 biomechanical joint angles**.
    * This makes the model invariant to camera distance and user height.
3.  **Classification (Deep Learning):**
    * **Model:** Custom LSTM (Recurrent Neural Network).
    * **Input:** A sliding window of **30 frames** (temporal sequence).
    * **Output:** Probability distribution across the 4 exercise classes.

## 📂 Project Structure
```bash
GymTrainer/
├── data/                  # Dataset folder (videos)
├── models/
│   └── gym_lstm_model.keras  # The trained LSTM model
├── utils/
│   ├── classes.npy        # Class labels
│   ├── X_data.npy         # Processed training features
│   └── y_data.npy         # Processed training labels
├── train_model.py         # Script to extract features and train the LSTM
├── main.py                # Main application (Webcam inference)
├── requirements.txt       # Dependencies
└── README.md
