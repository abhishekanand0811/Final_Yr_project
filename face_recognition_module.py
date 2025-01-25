import dlib
import face_recognition # type: ignore
import numpy as np

def register_new_face(frame, face_location, name, faces_data):
    """Register a new face in the local database."""
    face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
    faces_data["embeddings"].append(face_encoding.tolist())
    faces_data["names"].append(name)

def recognize_face(frame, faces_data):
    """Recognize a face from the database."""
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return None

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            [np.array(embedding) for embedding in faces_data["embeddings"]],
            face_encoding,
            tolerance=0.6,
        )
        if True in matches:
            match_index = matches.index(True)
            return faces_data["names"][match_index]
    return None
