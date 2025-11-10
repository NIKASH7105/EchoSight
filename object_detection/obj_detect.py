import cv2
import numpy as np
import time
import os
from gtts import gTTS
import pygame
import threading
import queue
from ultralytics import YOLO

# Load YOLOv8 model (much better compatibility than YOLOv3)
try:
    # YOLOv8 is the latest and most compatible version
    model = YOLO("yolov8n.pt")  # nano version for speed
    print("✓ Loaded YOLOv8n model (Ultralytics)")
    
    # Get class names
    classes = model.names
    print(f"✓ Available classes: {len(classes)}")
    
except Exception as e:
    print(f"X YOLOv8 loading failed: {e}")
    print("\nPlease install ultralytics:")
    print("pip install ultralytics")
    exit(1)

pygame.mixer.init()

def play_audio_simple(file_path):
    """Play audio file and delete it after playback"""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        
        time.sleep(0.2)
        
        if file_path.startswith('voice') and file_path.endswith('.mp3'):
            try:
                os.remove(file_path)
                print(f"Deleted audio file: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")
                time.sleep(1)
                try:
                    os.remove(file_path)
                    print(f"Deleted audio file (retry): {file_path}")
                except:
                    print(f"Failed to delete {file_path} - will be cleaned up later")
                
    except Exception as e:
        print(f"Audio playback error: {e}")
        if file_path.startswith('voice') and file_path.endswith('.mp3'):
            try:
                os.remove(file_path)
            except:
                pass

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_no = 0
inc = 0
last_audio_time = 0
last_detected_objects = set()

print("🚀 Starting Real-Time Object Detection with YOLOv8")
print("📹 Camera initialized")
print("🎤 Voice announcements enabled")
print("Press 'q' to quit")
print("-" * 50)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to read from camera")
        break
        
    frame_no += 1
    
    # Run YOLOv8 inference
    results = model.predict(
        source=frame,
        stream=False,
        conf=0.5,  # Confidence threshold
        imgsz=640,
        verbose=False,
        save=False
    )
    
    # Process detections
    current_objects = set()
    boxes = []
    confidences = []
    class_ids = []
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if confidence > 0.5:  # Filter by confidence
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class name
                    label = classes[cls]
                    current_objects.add(label)
                    
                    # Store for drawing
                    boxes.append([x1, y1, x2-x1, y2-y1])
                    confidences.append(confidence)
                    class_ids.append(cls)
                    
                    print(f"{label}: {confidence*100:.1f}%")
    
    # Check if objects changed
    objects_changed = current_objects != last_detected_objects
    
    if objects_changed:
        voice = "no object detected"
        if current_objects:
            # Get the most confident detection
            first_object = list(current_objects)[0]
            voice = str(first_object) + " in front of you"
        
        file_path = f'voice{inc}.mp3'
        inc += 1
        
        try:
            sound = gTTS(text=voice, lang='en')
            sound.save(file_path)
            
            if current_objects:
                play_audio_simple(file_path)
            else:
                # Use existing no_obj.mp3 if available
                if os.path.exists('no_obj.mp3'):
                    play_audio_simple('no_obj.mp3')
                else:
                    play_audio_simple(file_path)
            
            last_detected_objects = current_objects.copy()
            print(f"🔊 Objects changed! Announcing: {voice}")
            
        except Exception as e:
            print(f"❌ Audio generation error: {e}")
    else:
        if current_objects:
            print(f"Same objects detected: {', '.join(current_objects)}")
        else:
            print("No objects detected")
    
    # Draw detections
    if len(boxes) > 0:
        # Apply NMS
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw performance info
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_no}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Objects: {len(current_objects)}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display frame
    cv2.imshow("Real-Time Object Detection (YOLOv8)", frame)
    
    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Object detection stopped")