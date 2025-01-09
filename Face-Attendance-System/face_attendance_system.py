import cv2
import dlib
import numpy as np
import os
import pickle
from datetime import datetime

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

encodings_dir = 'trained_encodings_dlib'
os.makedirs(encodings_dir, exist_ok=True)

attendance_file = os.path.abspath("attendance_log.csv")
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time\n")
    print(f"Created attendance log file at {attendance_file}.")

def is_marked_present(name):
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(attendance_file, "r") as f:
            for line in f.readlines():
                if name in line and today in line:
                    print(f"{name} already marked present today.")  # Debugging
                    return True
    except Exception as e:
        print(f"Error reading attendance file: {e}")
    return False

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    try:
        with open(attendance_file, "a") as f:
            f.write(f"{name},{date},{time}\n")
        print(f"Attendance marked for {name} at {time} on {date}.")
    except Exception as e:
        print(f"Error writing to attendance file: {e}")

def eye_aspect_ratio(eye_points):
    a = np.linalg.norm(eye_points[1] - eye_points[5])
    b = np.linalg.norm(eye_points[2] - eye_points[4])
    c = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (a + b) / (2.0 * c)
    return ear

def detect_liveness(shape, prev_shape):
    try:
        # Blink Detection
        left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < 0.2:  
            print("Blink detected.") 
            blink_detected = True
        else:
            blink_detected = False

        head_movement_detected = False
        if prev_shape is not None:
            delta_x = abs(shape.part(30).x - prev_shape.part(30).x)
            delta_y = abs(shape.part(30).y - prev_shape.part(30).y)
            if delta_x > 2 or delta_y > 2:  
                print("Head movement detected.")  
                head_movement_detected = True

        gaze_detected = True  

        return blink_detected and head_movement_detected and gaze_detected
    except Exception as e:
        print(f"Error in liveness detection: {e}")
        return False

encodings = {}
try:
    for file in os.listdir(encodings_dir):
        if file.endswith('.pkl'):
            name = os.path.splitext(file)[0]
            with open(f"{encodings_dir}/{file}", "rb") as f:
                encodings[name] = pickle.load(f)
    print(f"Loaded {len(encodings)} encodings: {list(encodings.keys())}")  # Debugging
except Exception as e:
    print(f"Error loading encodings: {e}")

def recognize_faces(frame, prev_shape):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        print("No faces detected in frame.")
        return None

    for face in faces:
        try:
            shape = shape_predictor(gray, face)
            encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape, 1))

            name = "Unknown"
            distances = []
            for known_name, known_encoding in encodings.items():
                distance = np.linalg.norm(known_encoding - encoding)
                distances.append((distance, known_name))

            if distances:
                distances.sort()
                best_match_distance, best_match_name = distances[0]
                if best_match_distance < 0.6: 
                    name = best_match_name

            print(f"Detected: {name}, Distance: {best_match_distance if distances else 'N/A'}")

            if name != "Unknown" and detect_liveness(shape, prev_shape):
                if not is_marked_present(name):
                    mark_attendance(name)
                else:
                    print(f"{name} is already marked present today.")
            else:
                print("Liveness check failed, skipping attendance.") 

            left, top, right, bottom = (face.left(), face.top(), face.right(), face.bottom())
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return shape
        except Exception as e:
            print(f"Error recognizing face: {e}")
    return None

video_capture = cv2.VideoCapture(0)
prev_shape = None 

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error accessing the camera.")
        break

    prev_shape = recognize_faces(frame, prev_shape)
    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting attendance system.")
        break

video_capture.release()
cv2.destroyAllWindows()
