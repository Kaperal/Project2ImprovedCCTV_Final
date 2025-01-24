import cv2
import math
from ultralytics import YOLO

# Load YOLOv8 model (use 'yolov8n.pt' for a lightweight model or your custom model)
model = YOLO("Model/best.pt")  # Replace with 'your_custom_model.pt' if you have one

# Define the classes we're interested in (for the pre-trained COCO model)
# COCO class IDs: '0' is for 'person', and a custom ID should be used for 'gun' if trained
#TARGET_CLASSES = {'person': 0, 'gun': 1}  # You may need to adjust the IDs if using a custom model
TARGET_CLASSES = model.names
print(TARGET_CLASSES)

def detect_objects(frame):
    """
    Detects guns and humans in the given frame.

    Args:
        frame (numpy.ndarray): The input image/frame from the camera.

    Returns:
        list: A list of detected objects with their labels.
    """
    # Perform object detection
    results = model(frame, stream= True, imgsz = 640)

    # Extract the detections
    detected_objects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
            w = x2 - x1
            h = y2 - y1
            # put box in frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # class name
            cls = int(box.cls[0])
            # class name
            cls = int(box.cls[0])  # Get the class index of the detected object
            label = TARGET_CLASSES[cls]  # Use the class index to get the label from the model's names
            detected_objects.append({
                'label': label,
                'confidence': confidence,
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            })


            # put label on frame
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


    return detected_objects, frame


def main():
    # Open the camera feed
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect guns and humans
        detections, annotated_frame = detect_objects(frame)

        # Print detected objects to console
        for detection in detections:
            print(f"Detected: {detection['label']} with confidence {detection['confidence']:.2f}")

        # Display the annotated frame
        cv2.imshow('Gun and Human Detection', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
