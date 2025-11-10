#!/usr/bin/env python3
"""
Unified Detection System - Complete Integration
==============================================
Uses the ENTIRE detect.py system as base and adds OCR + Traffic Light detection
All results combined into ONE speech output using detect.py's proven voice system
"""

import cv2
import numpy as np
import time
import os
import sys
from gtts import gTTS
import pygame
import threading
from collections import defaultdict
from ultralytics import YOLO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU Detection and Setup
# ============================================================================

def check_gpu_availability():
    """Check if GPU (CUDA) is available for PyTorch"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU detected: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("ℹ GPU not available, using CPU")
            return False
    except ImportError:
        print("ℹ PyTorch not available, GPU check skipped")
        return False
    except Exception as e:
        print(f"ℹ GPU check failed: {e}, using CPU")
        return False

# Detect GPU availability
USE_GPU = check_gpu_availability()

# Add project paths for imports
current_dir = os.path.dirname(os.path.abspath("E:\\traffic_new"))
sys.path.append(os.path.join(current_dir, 'OCR'))
sys.path.append(os.path.join(current_dir, 'object_detection'))
sys.path.append(os.path.join(current_dir, 'trafficli'))
# Only set TOGETHER_API_KEY if not already set in environment
if 'TOGETHER_API_KEY' not in os.environ or not os.environ.get('TOGETHER_API_KEY'):
    os.environ['TOGETHER_API_KEY'] = 'a60c2c24e4f37100bf8dea9930a9a8a0d354b122c597847eca8dad4ee1551efd'  # Will be handled gracefully if empty
# ============================================================================
# Import Additional Systems
# ============================================================================

# Import OCR System (using ocr_order.py and ocr1_llm.py)
try:
    from OCR.ocr_order import RealTimeOCRPipeline
    from OCR.ocr1_llm import TextProcessor, TTSHandler
    OCR_AVAILABLE = True
    print("✓ OCR System (ocr_order.py + ocr1_llm.py) connected")
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"✗ OCR System not available: {e}")

# Import Traffic Light System
try:
    from trafficli.realtime_traffic_light_system import RealTimeTrafficLightDetector
    TRAFFIC_AVAILABLE = True
    print("✓ Traffic Light System connected")
except ImportError as e:
    TRAFFIC_AVAILABLE = False
    print(f"✗ Traffic Light System not available: {e}")

# ============================================================================
# Load YOLOv8 model (from detect.py)
# ============================================================================

try:
    # Load model with appropriate device
    device = 'cuda' if USE_GPU else 'cpu'
    model = YOLO("yolov8n.pt")
    
    # Move model to appropriate device
    if USE_GPU:
        model.to(device)
        print(f"✓ Loaded YOLOv8n model (Ultralytics) - Using GPU")
    else:
        print(f"✓ Loaded YOLOv8n model (Ultralytics) - Using CPU")
    
    classes = model.names
    print(f"✓ Available classes: {len(classes)}")
except Exception as e:
    print(f"X YOLOv8 loading failed: {e}")
    print("\nPlease install ultralytics:")
    print("pip install ultralytics")
    exit(1)

pygame.mixer.init()

# ============================================================================
# All functions from detect.py (unchanged)
# ============================================================================

# Relationship rules for contextual descriptions
HELD_ITEMS = ['cell phone', 'phone', 'cup', 'bottle', 'book', 'remote', 'knife', 'spoon', 'fork']
SITTING_OBJECTS = ['chair', 'couch', 'bench', 'bed']
VEHICLES = ['car', 'bus', 'truck', 'bicycle', 'motorcycle']
ANIMALS = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow']

def calculate_distance(box1, box2):
    """Calculate distance between centers of two boxes"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return np.sqrt((x1_center - x2_center)*2 + (y1_center - y2_center)*2)

def is_inside_or_near(small_box, large_box, threshold=0.3):
    """Check if small box is inside or very near large box"""
    # Check if center of small box is inside large box
    small_center_x = (small_box[0] + small_box[2]) / 2
    small_center_y = (small_box[1] + small_box[3]) / 2
    
    if (large_box[0] <= small_center_x <= large_box[2] and 
        large_box[1] <= small_center_y <= large_box[3]):
        return True
    
    # Check if very close
    distance = calculate_distance(small_box, large_box)
    large_box_size = max(large_box[2] - large_box[0], large_box[3] - large_box[1])
    return distance < large_box_size * threshold

def generate_scene_description(detections):
    """Generate a natural language description of the scene"""
    if not detections:
        return "no objects detected"
    
    # Group detections by class
    objects_dict = defaultdict(list)
    for det in detections:
        objects_dict[det['label']].append(det)
    
    # Count objects
    object_counts = {label: len(items) for label, items in objects_dict.items()}
    
    # Build contextual relationships
    people = objects_dict.get('person', [])
    descriptions = []
    processed_objects = set()
    
    # Analyze each person and their relationships
    for person in people:
        person_desc_parts = []
        person_items = []
        person_near = []
        
        # Check for objects held by or near the person
        for label, items in objects_dict.items():
            if label == 'person':
                continue
                
            for item in items:
                item_id = f"{label}_{item['box']}"
                if item_id in processed_objects:
                    continue
                
                # Check if item is held (inside or very close to person box)
                if is_inside_or_near(item['box'], person['box'], threshold=0.4):
                    if label in HELD_ITEMS:
                        person_items.append(label)
                    elif label in SITTING_OBJECTS:
                        person_near.append(f"sitting on a {label}")
                    else:
                        person_items.append(label)
                    processed_objects.add(item_id)
        
        # Build person description
        if person_items:
            if len(person_items) == 1:
                person_desc_parts.append(f"a person holding a {person_items[0]}")
            else:
                items_str = ", ".join(person_items[:-1]) + f" and {person_items[-1]}"
                person_desc_parts.append(f"a person with {items_str}")
        elif person_near:
            person_desc_parts.append(f"a person {person_near[0]}")
        else:
            person_desc_parts.append("a person")
        
        if person_desc_parts:
            descriptions.append(" ".join(person_desc_parts))
    
    # Add remaining objects not associated with people
    remaining_objects = []
    for label, items in objects_dict.items():
        if label == 'person':
            continue
        
        for item in items:
            item_id = f"{label}_{item['box']}"
            if item_id not in processed_objects:
                remaining_objects.append(label)
    
    # Count and describe remaining objects
    remaining_counts = defaultdict(int)
    for obj in remaining_objects:
        remaining_counts[obj] += 1
    
    # Add remaining objects to description
    for label, count in remaining_counts.items():
        if count == 1:
            if label in VEHICLES:
                descriptions.append(f"a {label} nearby")
            elif label in ANIMALS:
                descriptions.append(f"a {label}")
            else:
                descriptions.append(f"a {label}")
        else:
            descriptions.append(f"{count} {label}s")
    
    # Combine all descriptions
    if len(descriptions) == 0:
        return "scene ahead"
    elif len(descriptions) == 1:
        return descriptions[0] + " in front of you"
    elif len(descriptions) == 2:
        return f"{descriptions[0]} and {descriptions[1]} in front of you"
    else:
        main_desc = ", ".join(descriptions[:-1])
        return f"{main_desc}, and {descriptions[-1]} in front of you"

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
    except Exception as e:
        print(f"Audio playback error: {e}")

# ============================================================================
# Unified Detection System
# ============================================================================

class UnifiedDetectionSystem:
    """Complete system using detect.py base with OCR and Traffic Light integration"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        # Initialize additional systems
        self.ocr_system = None
        self.traffic_system = None
        
        if OCR_AVAILABLE:
            try:
                # Initialize OCR pipeline (ocr_order.py) with GPU if available
                self.ocr_pipeline = RealTimeOCRPipeline(
                    output_file='ocr_output.txt',
                    use_gpu=USE_GPU,  # Use GPU if available, else CPU
                    camera_id=self.camera_id,
                    enable_llm_tts=False  # We'll handle TTS ourselves
                )
                # Initialize LLM and TTS components (ocr1_llm.py) - only if API key is available
                api_key = os.environ.get('TOGETHER_API_KEY', '').strip()
                if api_key:
                    try:
                        self.ocr_text_processor = TextProcessor(api_key)
                        self.ocr_tts_handler = TTSHandler()
                        device_info = "GPU" if USE_GPU else "CPU"
                        print(f"✓ OCR System (ocr_order.py + ocr1_llm.py) initialized - Using {device_info}")
                    except Exception as llm_error:
                        print(f"⚠ LLM initialization failed (OCR detection will still work): {llm_error}")
                        self.ocr_text_processor = None
                        self.ocr_tts_handler = None
                else:
                    print("⚠ TOGETHER_API_KEY not set - OCR text detection will work, but LLM processing disabled")
                    print("   To enable LLM: Set TOGETHER_API_KEY environment variable or edit unified_detection_system.py")
                    self.ocr_text_processor = None
                    self.ocr_tts_handler = None
                    device_info = "GPU" if USE_GPU else "CPU"
                    print(f"✓ OCR System (detection only) initialized - Using {device_info}")
            except Exception as e:
                print(f"✗ OCR System initialization failed: {e}")
                # Continue without OCR
                self.ocr_pipeline = None
                self.ocr_text_processor = None
                self.ocr_tts_handler = None
        
        if TRAFFIC_AVAILABLE:
            try:
                self.traffic_system = RealTimeTrafficLightDetector()
                print("✓ Traffic Light System initialized")
            except Exception as e:
                print(f"✗ Traffic Light System initialization failed: {e}")
                # Continue without traffic light
                self.traffic_system = None
        
        # Speech tracking (from detect.py)
        self.last_scene_description = ""
        self.scene_stable_count = 0
        self.STABILITY_THRESHOLD = 2  # Frames before announcing change (reduced for faster speech)
        self.inc = 0
        
        print("\n" + "="*60)
        print("🚀 Unified Detection System Ready")
        print("="*60)
        print("Features:")
        print(f"  📱 Device: {'GPU' if USE_GPU else 'CPU'}")
        print("  ✓ Object Detection (detect.py)")
        print(f"  ✓ OCR Detection: {'✓' if self.ocr_pipeline else '✗'}")
        print(f"  ✓ Traffic Light Detection: {'✓' if self.traffic_system else '✗'}")
        print("  ✓ Combined Speech Output")
        print("="*60)
    
    def run(self):
        """Main execution loop - Complete detect.py integration"""
        
        # Initialize camera (from detect.py)
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"✗ Cannot open camera {self.camera_id}")
            return
        
        print(f"\n🚀 Starting Unified Detection System")
        print(f"📹 Camera {self.camera_id} active")
        print(f"🎤 Voice announcements enabled")
        print(f"🔍 Multi-modal detection active")
        print("Press 'q' to quit")
        print("-" * 60)
        
        frame_no = 0
        
        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                    
                frame_no += 1
                
                # Collect all detection results
                all_descriptions = []
                
                # 1. OBJECT DETECTION (from detect.py)
                object_description = self._detect_objects(frame)
                if object_description and object_description != "no objects detected":
                    all_descriptions.append(object_description)
                    print(f"🎯 Object detected: {object_description}")
                
                # 2. OCR DETECTION (if available)
                if self.ocr_pipeline:
                    ocr_description = self._detect_text(frame)
                    if ocr_description:
                        all_descriptions.append(ocr_description)
                        print(f"📝 Text detected: {ocr_description}")
                
                # 3. TRAFFIC LIGHT DETECTION (if available)
                if self.traffic_system:
                    traffic_description = self._detect_traffic_light(frame)
                    if traffic_description:
                        all_descriptions.append(traffic_description)
                        print(f"🚦 Traffic light detected: {traffic_description}")
                
                # Combine all descriptions
                if all_descriptions:
                    current_scene = ". ".join(all_descriptions)
                    print(f"🔊 COMBINED: {current_scene}")
                else:
                    current_scene = "scene ahead"
                
                # Speech logic (from detect.py)
                self._handle_speech(current_scene)
                
                # Draw detections and info (from detect.py)
                self._draw_frame_info(frame, frame_no, start_time, current_scene)
                
                # Display frame
                cv2.imshow("Unified Detection System", frame)
                
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
            print("✅ Unified Detection System stopped")
    
    def _detect_objects(self, frame):
        """Object detection using detect.py logic"""
        try:
            # Run YOLOv8 inference with appropriate device
            device = 'cuda' if USE_GPU else 'cpu'
            results = model.predict(
                source=frame,
                stream=False,
                conf=0.5,
                imgsz=640,
                verbose=False,
                save=False,
                device=device
            )
            
            # Process detections
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            label = classes[cls]
                            
                            detections.append({
                                'label': label,
                                'confidence': confidence,
                                'box': [x1, y1, x2, y2]
                            })
            
            # Generate scene description
            return generate_scene_description(detections)
            
        except Exception as e:
            print(f"✗ Object detection error: {e}")
            return None
    
    def _detect_text(self, frame):
        """OCR text detection using ocr_order.py"""
        try:
            if not self.ocr_pipeline:
                return None
                
            # Use the OCR pipeline to process frame
            annotated_frame, detected_texts = self.ocr_pipeline.process_frame(frame, 1)
            
            if detected_texts:
                # Extract text fragments in reading order
                text_fragments = []
                texts_sorted = sorted(detected_texts, key=lambda x: x.get('reading_order', 0))
                for text_info in texts_sorted:
                    text_fragments.append(text_info['text'])
                
                if text_fragments:
                    # Process with LLM (ocr1_llm.py) if available
                    if self.ocr_text_processor:
                        try:
                            combined_text = self.ocr_text_processor.combine_texts(text_fragments)
                            if combined_text:
                                return f"Text says: {combined_text}"
                        except Exception as e:
                            print(f"⚠ LLM text processing error: {e}")
                            # Fall through to simple concatenation
                    
                    # Fallback to simple concatenation
                    return f"Text says: {', '.join(text_fragments)}"
            
            return None
            
        except Exception as e:
            print(f"✗ OCR detection error: {e}")
            return None
    
    def _detect_traffic_light(self, frame):
        """Traffic light detection with color information"""
        try:
            if not self.traffic_system:
                return None
                
            traffic_result = self.traffic_system.process_frame(frame)
            if traffic_result:
                # Check for current color (from state)
                current_color = traffic_result.get('current_color')
                if current_color and current_color != 'unknown':
                    # Format: "Traffic light is red" or similar
                    return f"Traffic light is {current_color}"
                
                # Fallback: Check detections directly
                if 'detections' in traffic_result:
                    detections = traffic_result['detections']
                    if detections:
                        # Get the most confident detection
                        best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                        color = best_detection.get('color', 'unknown')
                        if color and color != 'unknown':
                            return f"Traffic light is {color}"
                        else:
                            return "Traffic light detected"
            return None
            
        except Exception as e:
            print(f"✗ Traffic light detection error: {e}")
            return None
    
    def _handle_speech(self, current_scene):
        """Handle speech output (from detect.py)"""
        # SIMPLIFIED: Just speak if scene is different and not empty
        if (current_scene != self.last_scene_description and 
            current_scene != "scene ahead" and
            current_scene.strip() != ""):
            
            file_path = f'voice{self.inc}.mp3'
            self.inc += 1
            
            try:
                sound = gTTS(text=current_scene, lang='en', slow=False)
                sound.save(file_path)
                
                print(f"\n🔊 ANNOUNCING: {current_scene}")
                print("-" * 50)
                
                # Play in a separate thread to avoid blocking
                threading.Thread(target=play_audio_simple, args=(file_path,), daemon=True).start()
                
                self.last_scene_description = current_scene
                
            except Exception as e:
                print(f"❌ Audio generation error: {e}")
    
    def _draw_frame_info(self, frame, frame_no, start_time, current_scene):
        """Draw frame information (from detect.py)"""
        # Performance info
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_no}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # System status
        y_offset = 90
        systems = [
            ("Object Detection", True),
            ("OCR", self.ocr_pipeline is not None),
            ("Traffic Light", self.traffic_system is not None)
        ]
        
        for system_name, is_active in systems:
            color = (0, 255, 0) if is_active else (0, 0, 255)
            status = "✓" if is_active else "✗"
            cv2.putText(frame, f"{status} {system_name}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        # Current scene description
        y_offset = frame.shape[0] - 30
        cv2.putText(frame, current_scene[:80], (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("🌟 Unified Detection System")
    print("=" * 60)
    print("Complete integration using detect.py as base")
    print(f"  📱 Device: {'GPU' if USE_GPU else 'CPU'}")
    print("  ✓ Object Detection (detect.py)")
    print("  ✓ OCR Detection (unified_vision_system)")
    print("  ✓ Traffic Light Detection (realtime_traffic_light_system)")
    print("  ✓ Combined Speech Output")
    print("=" * 60)
    
    # Initialize and run system
    try:
        system = UnifiedDetectionSystem(camera_id=0)
        system.run()
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()