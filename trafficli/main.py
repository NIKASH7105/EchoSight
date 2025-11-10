from ultralytics import YOLO
import cv2
import math 


cap = cv2.VideoCapture(0)

model = YOLO("weights/v9 - 64 epochs.pt")

# object classes
classNames = ["green", "red", "yellow"]

def Distance_finder(Focal_Length, real_face_width, obj_width_in_frame):
    distance = (real_face_width * Focal_Length)/obj_width_in_frame
    return distance

print("Traffic Light Detection Started")
print("Press 'q' to quit")
print("-" * 40)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break
        
    results = model(img, stream=True)

    # Process detections
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calculate distance
                obj_width = x2 - x1
                dist = Distance_finder(650, 7.1, obj_width)

                # Get confidence and class
                confidence = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                
                # Store detection info
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'distance': dist
                })

                # Color coding for bounding boxes
                if class_name == 'green':
                    color = (0, 255, 0)  # Green
                elif class_name == 'yellow':
                    color = (0, 255, 255)  # Yellow
                else:  # red
                    color = (0, 0, 255)  # Red

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Draw label with class name and confidence
                label = f"{class_name}: {confidence:.2f}"
                org = [x1, y1-10]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.8
                thickness = 2
                cv2.putText(img, label, org, font, fontScale, color, thickness)

    # Print detection summary
    if detections:
        print(f"Detected {len(detections)} traffic light(s):")
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.2f} (distance: {det['distance']:.1f}cm)")
    else:
        print("No traffic lights detected")

    # Add status text on image
    status_text = f"Detections: {len(detections)}"
    cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Traffic Light Detection', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped")