import face_recognition
import cv2
import os
import numpy as np
from PIL import Image

# Directory containing known faces
KNOWN_FACES_DIR = "known_faces"

# Tolerance for face recognition (lower means more strict)
TOLERANCE = 0.4

# Initialize a list to store known face encodings and their names
known_face_encodings = []
known_face_names = []

def load_known_faces():
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

        # Check if it's a directory (to avoid non-folder files)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_folder, filename)

                    try:
                        # Open the image using PIL and convert it to RGB
                        with Image.open(image_path) as img:
                            img = img.convert("RGB")
                            # Save to a temporary NumPy array for face_recognition
                            image = np.array(img)

                        # Get the face encoding
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            known_face_encodings.append(encodings[0])

                            # Use the folder name as the person's name
                            known_face_names.append(person_name)
                        else:
                            print(f"Warning: No face found in {filename} of {person_name}")
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
# Load the known faces at the start
load_known_faces()


def recognize_faces(frame):
    """
    Recognizes faces in the given frame.

    Args:
        frame (numpy.ndarray): The input image/frame from the camera.

    Returns:
        list: A list of recognized faces with their names, bounding boxes, and confidence scores.
    """
    if frame is None:
        print("Error: Received an empty frame.")
        return [], frame

    recognized_faces = []

    # Convert the frame to RGB using cv2.cvtColor for robustness
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error converting frame to RGB: {e}")
        return [], frame

    # Detect all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encodings with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
        name = "Unknown"
        confidence = 0.0  # Initialize confidence as 0

        # Find the best match if any
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # Confidence is inversely related to the distance (lower distance = higher confidence)
                confidence = 100 - (face_distances[best_match_index] * 100)

        # Store the recognized face details
        recognized_faces.append({
            'name': name,
            'location': face_location,
            'confidence': confidence
        })

        # Draw a rectangle around the face and label it
        top, right, bottom, left = face_location
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} ({confidence:.2f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return recognized_faces, frame



if __name__ == "__main__":
    # For testing, capture video from your webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recognize faces in the frame
        recognized_faces, annotated_frame = recognize_faces(frame)

        # Display the annotated frame
        cv2.imshow('Face Recognition', annotated_frame)

        # Print recognized faces
        for face in recognized_faces:
            print(f"Recognized: {face['name']}")

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
