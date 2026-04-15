import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import pyttsx3
import threading
import os

class GymGRUModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GymGRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(64, 128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'gym_gru_model.pth'

if os.path.exists('classes.npy'):
    actions = np.load('classes.npy')
else:
    actions = np.array(['barbell biceps curl', 'push-up', 'shoulder press', 'squat'])

num_classes = len(actions)
input_size = 6 
model = GymGRUModel(input_size, num_classes).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"SUCCESS: Loaded model with classes: {actions}")
else:
    print(f"ERROR: {model_path} not found!")

threshold = 0.5  
sequence_length = 30 
requiredFrames = 5 
sequence = []
stabilityCounter = 0
currentExercise = "Waiting"
stats = {action: {'count': 0, 'stage': None} for action in actions}

def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

def calculateAngle3d(a, b, c):
    a, b, c = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def getAngle2d(landmarks, a_idx, b_idx, c_idx):
    a = [landmarks[a_idx].x, landmarks[a_idx].y]
    b = [landmarks[b_idx].x, landmarks[b_idx].y]
    c = [landmarks[c_idx].x, landmarks[c_idx].y]
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def extract_features(landmarks):
    l = landmarks.landmark
    features = [
        calculateAngle3d(l[24], l[26], l[28]), 
        calculateAngle3d(l[23], l[25], l[27]),
        calculateAngle3d(l[12], l[14], l[16]),
        calculateAngle3d(l[11], l[13], l[15]), 
        calculateAngle3d(l[14], l[12], l[24]),
        calculateAngle3d(l[13], l[11], l[23]),
    ]
    return np.array(features).tolist()

def count_reps(action, landmarks):
    global stats
    angle = 0
    stage = stats[action]['stage']
    rep_done = False

    if 'squat' in action:
        angle = getAngle2d(landmarks, 23, 25, 27)
        if angle > 160: stage = "up"
        if angle < 100 and stage == "up": stage, rep_done = "down", True
    elif 'curl' in action or 'press' in action or 'push' in action:
        angle = getAngle2d(landmarks, 11, 13, 15)
        if angle > 150: stage = "down"
        if angle < 60 and stage == "down": stage, rep_done = "up", True

    if rep_done:
        stats[action]['count'] += 1
        speak(str(stats[action]['count']))
    stats[action]['stage'] = stage

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            feat = extract_features(results.pose_landmarks)
            sequence.append(feat)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                input_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.softmax(output, dim=1)[0]
                    confidence, pred = torch.max(prob, 0)
                
                debug_info = " | ".join([f"{actions[i]}: {prob[i]:.2f}" for i in range(len(actions))])
                print(f"Probabilities: {debug_info}", end='\r')

                if confidence.item() > threshold:
                    detected_name = actions[pred.item()]
                    if detected_name != currentExercise:
                        stabilityCounter += 1
                        if stabilityCounter > requiredFrames:
                            currentExercise = detected_name
                            stabilityCounter = 0
                            speak(f"Exercise changed to {currentExercise}")
                    else:
                        stabilityCounter = 0

            if currentExercise != "Waiting":
                count_reps(currentExercise, results.pose_landmarks.landmark)

        # תצוגה
        cv2.rectangle(image, (0,0), (640, 40), (40,40,40), -1)
        cv2.putText(image, f"DETECTED: {currentExercise.upper()}", (10, 25), 1, 1.5, (0,255,0), 2)
        if currentExercise in stats:
            cv2.putText(image, f"REPS: {stats[currentExercise]['count']}", (10, 80), 1, 3, (0,0,255), 3)

        cv2.imshow('Gym Trainer Pro - Debug Mode', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()