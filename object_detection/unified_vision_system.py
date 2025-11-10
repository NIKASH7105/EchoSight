#!/usr/bin/env python3
"""
Unified Real-Time Vision System
===============================
Combines Object Detection, OCR, and Traffic Light Detection with LLM Processing
"""

import cv2
import numpy as np
import time
import os
import threading
import queue
from datetime import datetime
from collections import defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Import Detection Systems
# ============================================================================

# YOLOv8 for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✓ YOLOv8 imported successfully")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"✗ YOLOv8 import failed: {e}")

# OCR System
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("✓ PaddleOCR imported successfully")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    print(f"✗ PaddleOCR import failed: {e}")

# LLM Processing
try:
    from together import Together
    TOGETHER_AVAILABLE = True
    print("✓ Together AI imported successfully")
except ImportError as e:
    TOGETHER_AVAILABLE = False
    print(f"✗ Together AI import failed: {e}")

# Text-to-Speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("✓ pyttsx3 imported successfully")
except ImportError as e:
    TTS_AVAILABLE = False
    print(f"✗ pyttsx3 import failed: {e}")

# Audio playback
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
    print("✓ pygame imported successfully")
except ImportError as e:
    PYGAME_AVAILABLE = False
    print(f"✗ pygame import failed: {e}")

# ============================================================================
# Configuration
# ============================================================================

# LLM Configuration
TOGETHER_API_KEY = ""
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# TTS Configuration
TTS_RATE = 150
TTS_VOLUME = 0.9

# Detection Configuration
OBJECT_CONFIDENCE = 0.5
TRAFFIC_LIGHT_CONFIDENCE = 0.3
OCR_CONFIDENCE = 0.3

# Performance Configuration
FRAME_SKIP = 2  # Process every 2nd frame for better performance
STABILITY_THRESHOLD = 3  # Frames before announcing changes

# ============================================================================
# Unified LLM Processor
# ============================================================================

class UnifiedLLMProcessor:
    """Process all detection outputs using LLM for unified scene description"""
    
    def __init__(self, api_key):
        if not TOGETHER_AVAILABLE:
            print("⚠ Together AI not available - LLM processing disabled")
            self.client = None
            return
            
        self.client = Together(api_key=api_key)
        self.last_processed_scene = ""
        self.lock = threading.Lock()
        
        print("✓ Unified LLM Processor initialized")
    
    def process_unified_scene(self, object_detections, ocr_texts, traffic_light_status):
        """
        Process all detection outputs into unified scene description
        
        Args:
            object_detections: List of object detection results
            ocr_texts: List of OCR text results
            traffic_light_status: Traffic light detection result
        
        Returns:
            Unified scene description string
        """
        if not self.client:
            return self._fallback_combination(object_detections, ocr_texts, traffic_light_status)
        
        # Prepare input for LLM
        scene_data = {
            'objects': object_detections,
            'texts': ocr_texts,
            'traffic_light': traffic_light_status
        }
        
        # Check for duplicate scenes
        scene_key = str(scene_data)
        with self.lock:
            if scene_key == self.last_processed_scene:
                return None
            self.last_processed_scene = scene_key
        
        # Create comprehensive prompt
        prompt = f"""You are an assistive AI helping visually impaired users understand their environment in real-time.

You will receive detection results from multiple computer vision systems and must create a clear, natural spoken description.

DETECTION RESULTS:
- Objects detected: {object_detections}
- Text found: {ocr_texts}
- Traffic light status: {traffic_light_status}

INSTRUCTIONS:
1. Create a single, clear spoken description of the current scene
2. Prioritize safety information (traffic lights, obstacles)
3. Include important text content (signs, labels)
4. Describe objects and their relationships naturally
5. Keep it concise but informative
6. Sound like a helpful human assistant

OUTPUT:
A single clear spoken description for a visually impaired user."""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.1,
                stop=["</s>", "\n\n"]
            )
            
            combined_description = response.choices[0].message.content.strip()
            return self._clean_output(combined_description)
            
        except Exception as e:
            print(f"✗ LLM processing error: {e}")
            return self._fallback_combination(object_detections, ocr_texts, traffic_light_status)
    
    def _fallback_combination(self, objects, texts, traffic_light):
        """Fallback combination without LLM"""
        descriptions = []
        
        # Traffic light priority
        if traffic_light and traffic_light != 'unknown':
            descriptions.append(f"Traffic light is {traffic_light}")
        
        # Objects
        if objects:
            if len(objects) == 1:
                descriptions.append(f"{objects[0]} in front of you")
            else:
                obj_str = ", ".join(objects[:-1]) + f" and {objects[-1]}"
                descriptions.append(f"{obj_str} in front of you")
        
        # Texts
        if texts:
            if len(texts) == 1:
                descriptions.append(f"Text says: {texts[0]}")
            else:
                text_str = ", ".join(texts)
                descriptions.append(f"Text content: {text_str}")
        
        if not descriptions:
            return "Scene ahead appears clear"
        
        return ". ".join(descriptions) + "."
    
    def _clean_output(self, text):
        """Clean LLM output"""
        # Remove common prefixes
        prefixes = [
            "Scene description:",
            "Description:",
            "Output:",
            "Result:",
            "Message:"
        ]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Remove quotes
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        return text

# ============================================================================
# Unified TTS Handler
# ============================================================================

class UnifiedTTSHandler:
    """Unified text-to-speech handler with interrupt capability"""
    
    def __init__(self, rate=150, volume=0.9):
        if not TTS_AVAILABLE:
            print("⚠ pyttsx3 not available - TTS disabled")
            self.engine = None
            return
            
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.is_speaking = False
        self.lock = threading.Lock()
        
        print("✓ Unified TTS Handler initialized")
    
    def speak(self, text):
        """Speak text with interrupt capability"""
        if not text or not text.strip() or not self.engine:
            return
        
        def _speak():
            with self.lock:
                self.is_speaking = True
            try:
                print(f"\n🔊 SPEAKING: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"✗ TTS error: {e}")
            finally:
                with self.lock:
                    self.is_speaking = False
        
        # Stop current speech and start new one
        if self.is_speaking:
            self.engine.stop()
            time.sleep(0.1)
        
        # Run in separate thread
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def is_currently_speaking(self):
        """Check if currently speaking"""
        if not self.engine:
            return False
        with self.lock:
            return self.is_speaking

# ============================================================================
# Detection Systems
# ============================================================================

class ObjectDetector:
    """Object detection with contextual relationships"""
    
    def __init__(self):
        if not YOLO_AVAILABLE:
            self.model = None
            return
            
        self.model = YOLO("yolov8n.pt")
        self.classes = self.model.names
        
        # Relationship rules
        self.HELD_ITEMS = ['cell phone', 'phone', 'cup', 'bottle', 'book', 'remote', 'knife', 'spoon', 'fork']
        self.SITTING_OBJECTS = ['chair', 'couch', 'bench', 'bed']
        self.VEHICLES = ['car', 'bus', 'truck', 'bicycle', 'motorcycle']
        self.ANIMALS = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow']
        
        print("✓ Object Detector initialized")
    
    def detect_objects(self, frame):
        """Detect objects with contextual relationships"""
        if not self.model:
            return []
        
        results = self.model.predict(
            source=frame,
            stream=False,
            conf=OBJECT_CONFIDENCE,
            imgsz=640,
            verbose=False,
            save=False
        )
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence > OBJECT_CONFIDENCE:
                        x1, y1, x2, y2 = box.xyxy[0]
                        label = self.classes[cls]
                        
                        detections.append({
                            'label': label,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        return self._generate_contextual_description(detections)
    
    def _generate_contextual_description(self, detections):
        """Generate contextual description of detected objects"""
        if not detections:
            return []
        
        # Group by class
        objects_dict = defaultdict(list)
        for det in detections:
            objects_dict[det['label']].append(det)
        
        descriptions = []
        people = objects_dict.get('person', [])
        processed_objects = set()
        
        # Analyze each person and their relationships
        for person in people:
            person_desc_parts = []
            person_items = []
            
            # Check for objects held by or near the person
            for label, items in objects_dict.items():
                if label == 'person':
                    continue
                
                for item in items:
                    item_id = f"{label}_{item['bbox']}"
                    if item_id in processed_objects:
                        continue
                    
                    # Check if item is near person
                    if self._is_near(person['bbox'], item['bbox']):
                        if label in self.HELD_ITEMS:
                            person_items.append(label)
                        processed_objects.add(item_id)
            
            # Build person description
            if person_items:
                if len(person_items) == 1:
                    person_desc_parts.append(f"a person holding a {person_items[0]}")
                else:
                    items_str = ", ".join(person_items[:-1]) + f" and {person_items[-1]}"
                    person_desc_parts.append(f"a person with {items_str}")
            else:
                person_desc_parts.append("a person")
            
            if person_desc_parts:
                descriptions.append(" ".join(person_desc_parts))
        
        # Add remaining objects
        for label, items in objects_dict.items():
            if label == 'person':
                continue
            
            for item in items:
                item_id = f"{label}_{item['bbox']}"
                if item_id not in processed_objects:
                    descriptions.append(f"a {label}")
        
        return descriptions
    
    def _is_near(self, box1, box2, threshold=0.3):
        """Check if two boxes are near each other"""
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        
        distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
        box1_size = max(box1[2] - box1[0], box1[3] - box1[1])
        
        return distance < box1_size * threshold

class OCRDetector:
    """OCR text detection and recognition"""
    
    def __init__(self):
        if not PADDLEOCR_AVAILABLE:
            self.ocr = None
            return
            
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            rec_batch_num=1
        )
        print("✓ OCR Detector initialized")
    
    def detect_text(self, frame):
        """Detect and recognize text in frame"""
        if not self.ocr:
            return []
        
        try:
            results = self.ocr.ocr(frame, cls=True)
            texts = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) > 1:
                        if isinstance(line[1], tuple):
                            text = line[1][0]
                            confidence = line[1][1]
                        else:
                            text = line[1]
                            confidence = 0.0
                        
                        if confidence > OCR_CONFIDENCE and text.strip():
                            texts.append(text.strip())
            
            return texts
            
        except Exception as e:
            print(f"✗ OCR error: {e}")
            return []

class TrafficLightDetector:
    """Traffic light detection and color recognition"""
    
    def __init__(self):
        if not YOLO_AVAILABLE:
            self.model = None
            return
            
        self.model = YOLO("yolov8n.pt")
        self.traffic_light_class_id = self._find_traffic_light_class_id()
        print("✓ Traffic Light Detector initialized")
    
    def _find_traffic_light_class_id(self):
        """Find traffic light class ID"""
        class_names = self.model.names
        
        traffic_light_names = ['traffic light', 'traffic_light', 'traffic-signal']
        
        for class_id, class_name in class_names.items():
            if any(name in class_name.lower() for name in traffic_light_names):
                return class_id
        
        # Fallback
        return 9
    
    def detect_traffic_light(self, frame):
        """Detect traffic light and determine color"""
        if not self.model:
            return 'unknown'
        
        results = self.model.predict(
            source=frame,
            stream=False,
            conf=TRAFFIC_LIGHT_CONFIDENCE,
            imgsz=640,
            verbose=False,
            save=False
        )
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if cls == self.traffic_light_class_id and confidence > TRAFFIC_LIGHT_CONFIDENCE:
                        x1, y1, x2, y2 = box.xyxy[0]
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        
                        color = self._detect_traffic_light_color(frame, bbox)
                        return color
        
        return 'unknown'
    
    def _detect_traffic_light_color(self, img, bbox):
        """Detect traffic light color within bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Extract ROI
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 'unknown'
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Color ranges
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
        
        yellow_pixels = cv2.countNonZero(cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 255, 255])))
        green_pixels = cv2.countNonZero(cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255])))
        
        # Determine dominant color
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        
        if max_pixels < 10:
            return 'unknown'
        
        if max_pixels == red_pixels and red_pixels > yellow_pixels * 1.5 and red_pixels > green_pixels * 1.5:
            return 'red'
        elif max_pixels == yellow_pixels and yellow_pixels > red_pixels * 1.5 and yellow_pixels > green_pixels * 1.5:
            return 'yellow'
        elif max_pixels == green_pixels and green_pixels > red_pixels * 1.5 and green_pixels > yellow_pixels * 1.5:
            return 'green'
        else:
            return 'unknown'

# ============================================================================
# Unified Vision System
# ============================================================================

class UnifiedVisionSystem:
    """Main unified vision system combining all detection capabilities"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        print("\n" + "="*60)
        print("🚀 Initializing Unified Vision System")
        print("="*60)
        
        # Initialize detection systems
        self.object_detector = ObjectDetector()
        self.ocr_detector = OCRDetector()
        self.traffic_light_detector = TrafficLightDetector()
        
        # Initialize processing systems
        self.llm_processor = UnifiedLLMProcessor(TOGETHER_API_KEY)
        self.tts_handler = UnifiedTTSHandler(rate=TTS_RATE, volume=TTS_VOLUME)
        
        # State tracking
        self.last_scene_description = ""
        self.scene_stable_count = 0
        
        print("✓ All systems initialized")
        print("="*60)
    
    def process_frame(self, frame):
        """Process single frame with all detection systems"""
        # Run all detections
        object_descriptions = self.object_detector.detect_objects(frame)
        ocr_texts = self.ocr_detector.detect_text(frame)
        traffic_light_status = self.traffic_light_detector.detect_traffic_light(frame)
        
        # Process with LLM
        unified_description = self.llm_processor.process_unified_scene(
            object_descriptions, ocr_texts, traffic_light_status
        )
        
        return {
            'objects': object_descriptions,
            'texts': ocr_texts,
            'traffic_light': traffic_light_status,
            'unified_description': unified_description
        }
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"✗ Cannot open camera {self.camera_id}")
            return
        
        print(f"\n🚀 Starting Unified Vision System")
        print(f"📹 Camera {self.camera_id} initialized")
        print(f"🎤 Voice announcements enabled")
        print(f"🔍 Multi-modal detection active")
        print("Press 'q' to quit")
        print("-" * 60)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Process every nth frame for performance
                if frame_count % FRAME_SKIP == 0:
                    result = self.process_frame(frame)
                    
                    # Check for scene changes
                    current_description = result['unified_description']
                    if current_description and current_description != self.last_scene_description:
                        self.scene_stable_count = 0
                    else:
                        self.scene_stable_count += 1
                    
                    # Announce if scene is stable and different
                    if (self.scene_stable_count == STABILITY_THRESHOLD and 
                        current_description and 
                        current_description != self.last_scene_description):
                        
                        print(f"\n{'='*60}")
                        print(f"🎯 SCENE CHANGE DETECTED")
                        print(f"{'='*60}")
                        print(f"Objects: {result['objects']}")
                        print(f"Texts: {result['texts']}")
                        print(f"Traffic Light: {result['traffic_light']}")
                        print(f"Unified: {current_description}")
                        print(f"{'='*60}")
                        
                        # Speak the unified description
                        self.tts_handler.speak(current_description)
                        self.last_scene_description = current_description
                    
                    # Draw information on frame
                    self._draw_info(frame, result, frame_count)
                
                # Display frame
                cv2.imshow("Unified Vision System", frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping system...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Unified Vision System stopped")
    
    def _draw_info(self, frame, result, frame_count):
        """Draw detection information on frame"""
        y_offset = 30
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Objects
        if result['objects']:
            obj_text = f"Objects: {', '.join(result['objects'][:2])}"
            cv2.putText(frame, obj_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        # Texts
        if result['texts']:
            text_preview = f"Text: {result['texts'][0][:30]}..."
            cv2.putText(frame, text_preview, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            y_offset += 20
        
        # Traffic light
        if result['traffic_light'] != 'unknown':
            tl_text = f"Traffic Light: {result['traffic_light'].upper()}"
            color = (0, 0, 255) if result['traffic_light'] == 'red' else \
                   (0, 255, 255) if result['traffic_light'] == 'yellow' else (0, 255, 0)
            cv2.putText(frame, tl_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        # Stability counter
        cv2.putText(frame, f"Stability: {self.scene_stable_count}/{STABILITY_THRESHOLD}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("🌟 Unified Real-Time Vision System")
    print("=" * 60)
    print("Combines:")
    print("  ✓ Object Detection (YOLOv8)")
    print("  ✓ OCR Text Recognition (PaddleOCR)")
    print("  ✓ Traffic Light Detection")
    print("  ✓ LLM Scene Understanding")
    print("  ✓ Text-to-Speech Output")
    print("=" * 60)
    
    # Check system availability
    if not YOLO_AVAILABLE:
        print("❌ YOLOv8 not available - object detection disabled")
    if not PADDLEOCR_AVAILABLE:
        print("❌ PaddleOCR not available - OCR disabled")
    if not TOGETHER_AVAILABLE:
        print("❌ Together AI not available - LLM processing disabled")
    if not TTS_AVAILABLE:
        print("❌ pyttsx3 not available - TTS disabled")
    
    if not any([YOLO_AVAILABLE, PADDLEOCR_AVAILABLE]):
        print("\n❌ No detection systems available. Please install required packages.")
        return
    
    # Initialize and run system
    try:
        system = UnifiedVisionSystem(camera_id=0)
        system.run()
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
