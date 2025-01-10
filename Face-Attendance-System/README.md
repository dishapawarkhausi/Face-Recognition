# Face Attendance System

This project is a **Face Attendance System** that automates attendance tracking using facial recognition and liveness detection. The system ensures accurate attendance logging while preventing spoofing attempts with photos or videos. It is designed for workplaces, schools, and event management scenarios.

---

## Features

- **Automated Attendance**:
  - Recognizes individuals using facial recognition.
  - Logs name, date, and time in a CSV file (`attendance_log.csv`).

- **Liveness Detection**:
  - Detects blinks, head movements, and gaze patterns to ensure the person is live.

- **Training Options**:
  - Train using **webcam**, a **single image file**, or a **dataset folder** containing subfolders for multiple individuals.

- **Data Storage**:
  - Stores facial encodings as `.pkl` files.
  - Maintains attendance logs in a CSV file.

---

## Workflow

### Training Module (`face_attendance_training.py`):
1. Choose a training method:
   - **Webcam**: Capture live face data.
   - **File**: Use a single image file.
   - **Dataset Folder**: Use a folder containing subfolders for each person.
2. For each detected face:
   - Extract 68 facial landmarks using `dlib`.
   - Compute a 128-dimensional facial encoding.
3. Save the encoding for each person in a `.pkl` file.

### Attendance Module (`face_attendance_system.py`):
1. Load saved facial encodings.
2. Capture live video feed from the webcam.
3. Detect faces, compute encodings, and compare them with stored encodings.
4. Verify liveness using blink detection, head movement, and gaze patterns.
5. Log attendance in the CSV file.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dishapawarkhausi/face-attendance-system.git
   cd face-attendance-system
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required `dlib` models:
   - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

4. Extract the downloaded models and place them in the project directory.

---

## Usage

### Training the System
1. Run the training script:
   ```bash
   python face_attendance_training.py
   ```
2. Choose a training method:
   - **Webcam**: Input the person’s name and capture live data.
   - **File**: Input the person’s name and provide the image file path.
   - **Dataset Folder**: Provide the folder path containing subfolders for each person.

### Running the Attendance System
1. Start the attendance script:
   ```bash
   python face_attendance_system.py
   ```
2. The webcam captures faces in real-time and logs attendance after recognition and liveness verification.

---

## Libraries Used

- **OpenCV (`cv2`)**: For image and video processing.
- **dlib**: For face detection, landmark extraction, and recognition.
- **NumPy**: For numerical computations (e.g., encoding distances).
- **Pickle**: For saving and loading facial encodings.
- **os**: For file and directory management.
- **datetime**: For logging attendance timestamps.

---

## File Structure

```plaintext
face-attendance-system/
├── face_attendance_training.py  # Training module
├── face_attendance_system.py    # Attendance module
├── requirements.txt             # Python dependencies
├── attendance_log.csv           # Attendance log file
├── trained_encodings_dlib/      # Directory for storing encodings
├── shape_predictor_68_face_landmarks.dat  # Facial landmark model
├── dlib_face_recognition_resnet_model_v1.dat  # Face recognition model
```

---

## Challenges Solved

- **Accuracy**: Achieved precise recognition using a pre-trained deep learning model.
- **Spoof Prevention**: Implemented liveness detection to ensure only live individuals are marked present.
- **Scalability**: New individuals can easily be added by training their faces.

---

## Future Enhancements

- Add a **Graphical User Interface (GUI)** for user-friendly interactions.
- Implement **cloud storage** for attendance logs.
- Use advanced liveness detection techniques, such as voice recognition.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- `dlib` library and models: [dlib.net](http://dlib.net)
- OpenCV for video and image processing.
