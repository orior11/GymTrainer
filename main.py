import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import threading

modelPath = 'gym_lstm_model.keras'
threshold = 0.7

requiredFrames = 15
stabilityCounter = 0
pendingExercise = None

actions = np.array(['barbell biceps curl', 'push-up', 'shoulder press', 'squat'])

caloriesPerRep = {
    'barbell biceps curl': 0.2,
    'push-up': 0.3,
    'shoulder press': 0.25,
    'squat': 0.45,
    'hammer_curl': 0.15
}

stats = {action: {'count': 0, 'stage': None, 'cals': 0.0} for action in actions}
currentExercise = "Waiting" 
lastSpokenExercise = ""


def speak(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=run).start()

print("Loading Model...")
try:
    model = tf.keras.models.load_model(modelPath)
    print("SUCCESS: Model Loaded!")
except Exception as e:
    print(f"ERROR: Could not load {modelPath}. {e}")
    exit()

meadiapipePose = mp.solutions.pose
meadiapipeDrawing = mp.solutions.drawing_utils


def calculateAngle3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


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
    return (np.array(features) / 180.0).tolist()


def getAngle2d(landmarks, a_idx, b_idx, c_idx):
    a = [landmarks[a_idx].x, landmarks[a_idx].y]
    b = [landmarks[b_idx].x, landmarks[b_idx].y]
    c = [landmarks[c_idx].x, landmarks[c_idx].y]
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


def count_reps(action, landmarks):
    global stats

    angle = 0
    repProgress = 0
    stage = stats[action]['stage']
    repCompleted = False

    if action == 'squat':
        angle = getAngle2d(landmarks, 23, 25, 27)
        repProgress = np.interp(angle, (90, 160), (100, 0))
        if angle > 160: stage = "up"
        if angle < 90 and stage == 'up':
            stage = "down"
            repCompleted = True

    elif action == 'push-up':
        angle = getAngle2d(landmarks, 11, 13, 15)
        repProgress = np.interp(angle, (90, 160), (100, 0))
        if angle > 160: stage = "up"
        if angle < 90 and stage == 'up':
            stage = "down"
            repCompleted = True

    elif action == 'shoulder press':
        angle = getAngle2d(landmarks, 11, 13, 15)
        repProgress = np.interp(angle, (80, 160), (0, 100))
        if angle < 75: stage = "down"
        if angle > 160 and stage == 'down':
            stage = "up"
            repCompleted = True

    elif 'curl' in action:
        angle = getAngle2d(landmarks, 11, 13, 15)
        repProgress = np.interp(angle, (30, 160), (100, 0))
        if angle > 160: stage = "down"
        if angle < 30 and stage == 'down':
            stage = "up"
            repCompleted = True

    if repCompleted:
        stats[action]['count'] += 1
        stats[action]['cals'] += caloriesPerRep.get(action, 0.1)
        speak(str(stats[action]['count']))

    stats[action]['stage'] = stage
    return int(repProgress)


print("Starting Camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("ERROR: Camera failed to open. Please restart terminal.")
    exit()

sequence = []
per = 0

print("Starting Main Loop. Press 'q' to exit.")

with meadiapipePose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("Failed to read frame")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            current_stage = stats[currentExercise]['stage'] if currentExercise in stats else None
            
            if current_stage == "down":
                skel_color = (0, 255, 0) 
            elif current_stage == "up":
                skel_color = (0, 0, 255) 
            else:
                skel_color = (255, 255, 255) 

            meadiapipeDrawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                meadiapipePose.POSE_CONNECTIONS,
                meadiapipeDrawing.DrawingSpec(color=skel_color, thickness=2, circle_radius=2),
                meadiapipeDrawing.DrawingSpec(color=skel_color, thickness=2, circle_radius=2)
            )

            features = extract_features(results.pose_landmarks)
            sequence.append(features)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                result = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                bestMatchIndex = np.argmax(result)

                if result[bestMatchIndex] > threshold:
                    detected_exercise = actions[bestMatchIndex]

                    if detected_exercise == currentExercise:
                        stabilityCounter = 0
                        pendingExercise = None
                    else:
                        if detected_exercise == pendingExercise:
                            stabilityCounter += 1
                        else:
                            pendingExercise = detected_exercise
                            stabilityCounter = 0

                        if stabilityCounter > requiredFrames:
                            if pendingExercise in stats:
                                stats[pendingExercise]['count'] = 0
                                stats[pendingExercise]['cals'] = 0
                                stats[pendingExercise]['stage'] = None

                            currentExercise = pendingExercise
                            stabilityCounter = 0

                            if currentExercise != lastSpokenExercise:
                                cleanName = currentExercise.replace("barbell biceps ", "").replace("_", " ")
                                speak(f"Starting set of {cleanName}")
                                lastSpokenExercise = currentExercise

                if currentExercise != "Waiting":
                    per = count_reps(currentExercise, results.pose_landmarks.landmark)

        cv2.rectangle(image, (0, 0), (640, 80), (245, 117, 16), -1)

        cv2.putText(image, 'EXERCISE', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, currentExercise.replace("_", " ").upper(), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        reps = stats[currentExercise]['count'] if currentExercise in stats else 0
        cals = stats[currentExercise]['cals'] if currentExercise in stats else 0

        cv2.putText(image, 'REPS', (300, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(reps), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'KCAL', (450, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{cals:.1f}", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if currentExercise != "Waiting":
            cv2.rectangle(image, (580, 100), (620, 350), (0, 255, 0), 1)
            bar_height = np.interp(per, (0, 100), (350, 100))
            cv2.rectangle(image, (580, int(bar_height)), (620, 350), (0, 255, 0), -1)
            cv2.putText(image, f'{per}%', (565, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if stabilityCounter > 0:
            stab_width = int((stabilityCounter / requiredFrames) * 100)
            cv2.rectangle(image, (220, 75), (220 + stab_width, 80), (0, 255, 255), -1)

        cv2.imshow('AI Gym Trainer Pro', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()