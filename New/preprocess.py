import cv2
import mediapipe as mp
import numpy as np
import os

dataPath = r'C:\Users\PinhasZ\gymTrainer\data'

datasets = [
    'youtube_vids', 
    'my_test_video_1', 
    'similar_dataset'
]

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
        calculateAngle3d(l[24], l[26], l[28]),
        calculateAngle3d(l[23], l[25], l[27]),
        calculateAngle3d(l[12], l[14], l[16]),
        calculateAngle3d(l[11], l[13], l[15]),
        calculateAngle3d(l[14], l[12], l[24]),
        calculateAngle3d(l[13], l[11], l[23]),
    ]

sequences = []
labels = []

print("Starting Preprocessing...")

if not os.path.exists(dataPath):
    print(f"\nError: Directory not found: {dataPath}")
    exit()

print("Main data folder found. Starting deep scan...\n")

valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')

for datasetName in datasets:
    datasetPath = os.path.join(dataPath, datasetName)
    
    if not os.path.exists(datasetPath):
        print(f"Skipping dataset '{datasetName}' - Directory not found!")
        continue
        
    print(f"Scanning dataset: {datasetName}...")

    for classIndex, className in enumerate(classes):
        classPath = os.path.join(datasetPath, className)
        
        if not os.path.exists(classPath):
            continue
            
        videoFiles = [f for f in os.listdir(classPath) if f.lower().endswith(valid_extensions)]
        
        if len(videoFiles) == 0:
            continue
            
        print(f"   -> Processing {len(videoFiles)} videos of exercise: {className}")

        for videoName in videoFiles:
            videoPath = os.path.join(classPath, videoName)
            cap = cv2.VideoCapture(videoPath)
            framesBuffer = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    break

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

y_indices = np.array(labels)
if len(y_indices) > 0:
    y = np.zeros((len(y_indices), len(classes)))
    y[np.arange(len(y_indices)), y_indices] = 1
else:
    y = np.array([])

print(f"\n" + "="*30)
print(f"Done! Data shapes:")
print(f"X (Input):  {X.shape}  -> (Samples, TimeSteps, Features)")
print(f"y (Labels): {y.shape}  -> (Samples, Classes)")

if X.shape[0] > 0:
    np.save(outputX, X)
    np.save(outputY, y)
    np.save(outputClasses, classes)
    print("Files saved successfully.")
else:
    print("No data was generated for saving. The model failed to detect skeletons in the videos.")