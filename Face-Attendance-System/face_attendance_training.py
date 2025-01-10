import cv2
import dlib
import numpy as np
import os
import pickle

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

encodings_dir = 'trained_encodings_dlib'
os.makedirs(encodings_dir, exist_ok=True)

def process_images_from_folder(path):
    resolved_path = os.path.abspath(path)
    print(f"Processing dataset folder: {resolved_path}")

    if not os.path.isdir(resolved_path):
        print(f"Invalid folder path: {resolved_path}")
        return

    person_folders = [f for f in os.listdir(resolved_path) if os.path.isdir(os.path.join(resolved_path, f))]

    if not person_folders:
        print(f"No subfolders found in dataset folder: {resolved_path}")
        return

    for person_name in person_folders:
        person_path = os.path.join(resolved_path, person_name)
        print(f"Processing person: {person_name}")
        encodings = []

        images = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
        if not images:
            print(f"No valid images found for {person_name}")
            continue

        for image_path in images:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            process_single_image(image, image_path, encodings)

        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            save_encoding(avg_encoding, person_name)

def process_images_from_file(path, name):
    encodings = []
    resolved_path = os.path.abspath(path)
    print(f"Resolved path: {resolved_path}")

    if os.path.isfile(resolved_path):
        # Process a single file
        image = cv2.imread(resolved_path)
        if image is None:
            print(f"Could not read image: {resolved_path}")
            return
        process_single_image(image, resolved_path, encodings)
    else:
        print(f"Invalid file path: {resolved_path}")
        return

    if encodings:
        avg_encoding = np.mean(encodings, axis=0)
        save_encoding(avg_encoding, name)

def process_single_image(image, image_path, encodings):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        print(f"No faces detected in image: {image_path}")
        return

    for face in faces:
        shape = shape_predictor(gray, face)
        encoding = np.array(face_rec_model.compute_face_descriptor(image, shape, 1))
        encodings.append(encoding)
        print(f"Captured encoding from image: {os.path.basename(image_path)}")

def capture_images_from_webcam(name):
    video_capture = cv2.VideoCapture(0)
    encodings = []

    print(f"Starting training for {name} using webcam. Please look at the camera.")

    while len(encodings) < 10:
        ret, frame = video_capture.read()
        if not ret:
            print("Error accessing the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = shape_predictor(gray, face)
            encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape, 1))
            encodings.append(encoding)
            print(f"Captured encoding {len(encodings)}/10 for {name}.")

        cv2.putText(frame, f"Captured {len(encodings)}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Training - Press 'q' to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Training interrupted by user.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if encodings:
        avg_encoding = np.mean(encodings, axis=0)
        save_encoding(avg_encoding, name)

def save_encoding(encoding, name):
    filepath = f"{encodings_dir}/{name}.pkl"
    if os.path.exists(filepath):
        print(f"Overwriting existing encoding for {name}.")
    with open(filepath, "wb") as f:
        pickle.dump(encoding, f)
    print(f"Training for {name} completed and saved at {filepath}.")

def main():
    while True:
        choice = input("Choose training method: (1) Webcam (2) File (3) Dataset Folder (or type 'exit' to quit): ")

        if choice.lower() == 'exit':
            print("Exiting training.")
            break

        if choice == '1':
            name = input("Enter the name of the person to train: ")
            capture_images_from_webcam(name)
        elif choice == '2':
            name = input("Enter the name of the person to train: ")
            path = input("Enter the file path: ")
            process_images_from_file(path, name)
        elif choice == '3':
            path = input("Enter the dataset folder path: ")
            process_images_from_folder(path)
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()

