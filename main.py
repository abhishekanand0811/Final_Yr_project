import json
import os
import cv2
import numpy as np
import torch
import face_recognition 
from ultralytics import YOLO
from utils import speak   # Shared text-to-speech functionality
from object_detection import perform_object_detection
from face_recognition_module import register_new_face, recognize_face

# File to store face data
faces_data_file = "faces_data.json"

# Load existing face data if available
if os.path.exists(faces_data_file):
    with open(faces_data_file, "r") as f:
        faces_data = json.load(f)
else:
    faces_data = {"embeddings": [], "names": []}

def save_faces_data():
    """Save face data to a JSON file."""
    with open(faces_data_file, "w") as f:
        json.dump(faces_data, f)

def main():
    # Load YOLO model for object detection
    yolo_model = YOLO("yolo11m.pt")  # Replace with your YOLO model file

    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera is live. Press 'ESC' to exit.")
    scaling_factor = 0.05  # Adjust based on your depth calculation needs

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform object detection and depth estimation
        objects, depth_map = perform_object_detection(frame, yolo_model, scaling_factor)

        # Check if a face is detected in the frame
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) > 0:
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]

                # Check if the face matches an existing one
                face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
                matches = face_recognition.compare_faces(
                    [np.array(embedding) for embedding in faces_data["embeddings"]],
                    face_encoding,
                    tolerance=0.6,
                )

                if True in matches:
                    match_index = matches.index(True)
                    person_name = faces_data["names"][match_index]
                    speak(f"Hello, you have seen this person before. It's {person_name}.")
                else:
                    speak("This is a new person. Registering face.")
                    new_name = f"Person_{len(faces_data['names']) + 1}"
                    register_new_face(frame, face_location, new_name, faces_data)
                    save_faces_data()
        
        # Display the video feed with bounding boxes
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
