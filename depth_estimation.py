import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO

# --- Pseudo Depth Estimation ---
def calculate_depth(image):
    """
    Generates a pseudo-depth map from an image by converting it to grayscale
    and applying a distance gradient.
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize intensity values to simulate depth
    depth_map = cv2.GaussianBlur(gray_image, (15, 15), 0)  # Apply Gaussian Blur for smoothness
    depth_map = 255 - depth_map  # Invert to simulate "closer = darker"
    depth_map = depth_map / 255.0  # Normalize between 0 and 1

    return depth_map

# --- Object Detection Setup ---
# Load the YOLO model
yolo_model = YOLO("yolo11m.pt")  # Replace with your YOLO model file

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 for default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press SPACE to capture an image or ESC to exit.")

captured_image = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break
    elif key == 32:  # SPACE key to capture
        captured_image = frame
        print("Image captured!")
        break

cap.release()
cv2.destroyAllWindows()

if captured_image is not None:
    # Perform pseudo-depth estimation
    depth_map = calculate_depth(captured_image)

    # Perform object detection
    results = yolo_model.predict(source=captured_image, save=False)

    # Extract detection details
    boxes = results[0].boxes.xyxy  # Bounding box coordinates
    scores = results[0].boxes.conf  # Confidence scores
    classes = results[0].boxes.cls  # Class IDs
    class_names = yolo_model.names  # Class labels

    # Scaling factor for converting pseudo-depth values to real-world distances
    scaling_factor = 100  # Arbitrary value for scaling to centimeters

    # Initialize an output image for visualization
    output_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

    # Print detected objects and their calculated distances
    print("Detected Objects and Distances:")
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(cls)]
        
        # Extract depth values within the bounding box
        object_depth_region = depth_map[y1:y2, x1:x2]
        non_zero_depth_values = object_depth_region[object_depth_region > 0]  # Ignore invalid depth values
        
        if len(non_zero_depth_values) > 0:
            # Calculate the median depth value
            depth_value = np.median(non_zero_depth_values)
            real_distance = depth_value * scaling_factor
        else:
            # No valid depth data in the bounding box
            real_distance = -1.0

        # Print object details
        print(f"{class_name} (Confidence: {score:.2f}, Distance: {real_distance:.2f} cm)")

        # Draw bounding boxes and labels on the image
        label = f"{class_name} ({score:.2f}, {real_distance:.2f} cm)"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes and distances
    plt.figure(figsize=(10, 8))
    plt.imshow(output_image)
    plt.axis("off")
    plt.title("Object Detection with Distances")
    plt.show()

    # Save the output image
    output_path = "output_with_distances.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Image with detections saved to {output_path}")
else:
    print("No image captured. Exiting.")
