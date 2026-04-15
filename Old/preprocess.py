import cv2
import mediapipe as mp
import numpy as np
import os

dataPath = r'C:\Users\PinhasZ\gymTrainer\data\additional_video'
classes = ['barbell biceps curl', 'push-up', 'shoulder press', 'squat']

numOfFrames = 30
framesToSkip = 2


outputX = 'X_data.npy'
outputY = 'y_data.npy'
outputClasses = 'classes.npy'

meadiapipePose = mp.solutions.pose
pose = meadiapipePose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def calculateAngle3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosineAngle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosineAngle, -1.0, 1.0))
    return np.degrees(angle)

def extract_features(landmarks):
    l = landmarks.landmark
    return [
        calculateAngle3d(l[24], l[26], l[28]), # rightKnee
        calculateAngle3d(l[23], l[25], l[27]), # leftKnee
        calculateAngle3d(l[12], l[14], l[16]), # rightElbow
        calculateAngle3d(l[11], l[13], l[15]), # leftElbow
        calculateAngle3d(l[14], l[12], l[24]), # rightShoulder
        calculateAngle3d(l[13], l[11], l[23]), # leftShoulder
    ]

sequences = []
labels = []

print("Starting LSTM Preprocessing")

for classIndex, className in enumerate(classes):
    classFile = os.path.join(dataPath, className)
    if not os.path.exists(classFile):
        print(f"Skipping {className} (folder not found)")
        continue

    print(f"Processing {className}...")
    videoFiles = [f for f in os.listdir(classFile) if f.endswith('.mp4')]

    for videoName in videoFiles:
        videoPath = os.path.join(classFile, videoName)
        cap = cv2.VideoCapture(videoPath)

        framesBuffer = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            imageRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(imageRgb)

            if results.pose_landmarks:
                features = extract_features(results.pose_landmarks)
                framesBuffer.append(features)

            if len(framesBuffer) == numOfFrames:
                sequences.append(framesBuffer)
                labels.append(classIndex)

                framesBuffer = framesBuffer[framesToSkip:]

        cap.release()

X = np.array(sequences)
y = np.array(labels)

print(f"\nDone! Data shapes:")
print(f"X (Input): {X.shape}  -> (Samples, TimeSteps, Features)")
print(f"y (Labels): {y.shape}")

np.save(outputX, X)
np.save(outputY, y)
np.save(outputClasses, classes)

print("files saved successfully.")

