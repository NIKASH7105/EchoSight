#!/usr/bin/env python3
"""
Single Camera Live System - Real-Time Multi-Modal Detection with Immediate Speech
================================================================================
Uses ONE camera for all detection systems with immediate LLM processing and speech feedback
"""

import cv2
import numpy as np
import time
import threading
import queue
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Import Required Libraries
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

# Windows SAPI fallback
try:
    import subprocess
    WINDOWS_SAPI_AVAILABLE = True
    print("✓ Windows SAPI available")
except ImportError as e:
    WINDOWS_SAPI_AVAILABLE = False
    print(f"✗ Windows SAPI not available: {e}")

# ============================================================================
# Configuration
# ============================================================================

# LLM Configuration
TOGETHER_API_KEY = "a60c2c24e4f37100bf8dea9930a9a8a0d354b122c597847eca8dad4ee1551efd"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# TTS Configuration
TTS_RATE = 150
TTS_VOLUME = 0.9

# Detection Configuration
OBJECT_CONFIDENCE = 0.5
OCR_CONFIDENCE = 0.3
TRAFFIC_LIGHT_CONFIDENCE = 0.3

# ============================================================================
# Live LLM Processor
# ============================================================================

class LiveLLMProcessor:
    """Process detection results with LLM for immediate speech feedback"""
    
    def __init__(self, api_key):
        if not TOGETHER_AVAILABLE:
            print("⚠ Together AI not available - LLM processing disabled")
            self.client = None
            return
            
        self.client = Together(api_key=api_key)
        self.last_processed = ""
        self.lock = threading.Lock()
        
        print("✓ Live LLM Processor initialized")
    
    def process_detection(self, detection_type, detection_data):
        """
        Process detection result with LLM for immediate speech
        
        Args:
            detection_type: 'object', 'ocr', 'traffic'
            detection_data: The detection result data
        
        Returns:
            Processed sentence for speech
        """
        if not self.client or not detection_data:
            return None
        
        # Create SHORT, FAST prompts for real-time commentary
        if detection_type == 'object':
            prompt = f"""Object: {detection_data}
            Create a very short sentence. Example: "Person with phone ahead"
            Response:"""
            
        elif detection_type == 'ocr':
            prompt = f"""Text: {detection_data}
            Create a very short sentence. Example: "Text says Library"
            Response:"""
            
        elif detection_type == 'traffic':
            prompt = f"""Traffic light: {detection_data}
            Create a very short instruction. Example: "Red light, stop"
            Response:"""
        
        else:
            return None
        
        # Check for duplicate processing
        detection_key = f"{detection_type}:{detection_data}"
        with self.lock:
            if detection_key == self.last_processed:
                return None
            self.last_processed = detection_key
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,  # Very short responses for speed
                temperature=0.1,  # Lower temperature for faster, more predictable responses
                top_p=0.5,
                top_k=20,
                repetition_penalty=1.0,
                stop=["</s>", "\n\n", "."]
            )
            
            processed_text = response.choices[0].message.content.strip()
            return self._clean_output(processed_text)
            
        except Exception as e:
            print(f"✗ LLM processing error: {e}")
            return None
    
    def _clean_output(self, text):
        """Clean LLM output"""
        # Remove common prefixes
        prefixes = [
            "Response:",
            "Output:",
            "Result:",
            "Message:",
            "Description:"
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
# Live TTS Handler
# ============================================================================

class LiveTTSHandler:
    """Handle immediate text-to-speech with interrupt capability - Based on ocr1_llm.py"""
    
    def __init__(self, rate=150, volume=0.9):
        if not TTS_AVAILABLE:
            print("⚠ pyttsx3 not available - TTS disabled")
            self.engine = None
            return
            
        try:
            # Initialize exactly like ocr1_llm.py
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            self.speaking_queue = []
            self.is_speaking = False
            self.lock = threading.Lock()
            
            # Get available voices (optional: select different voice)
            voices = self.engine.getProperty('voices')
            if voices:
                # You can change voice here if desired
                # self.engine.setProperty('voice', voices[1].id)  # Female voice
                pass
            
            print("✓ Text-to-Speech engine initialized")
            
        except Exception as e:
            print(f"✗ TTS initialization failed: {e}")
            self.engine = None
    
    def speak_immediately(self, text):
        """Speak text immediately with blocking to ensure it's heard"""
        if not text or not text.strip() or not self.engine:
            return
        
        # Stop any current speech
        if self.is_speaking:
            try:
                self.engine.stop()
                time.sleep(0.1)
            except:
                pass
        
        try:
            print(f"\n🔊 SPEAKING NOW: {text}")
            with self.lock:
                self.is_speaking = True
            
            # Speak directly without threading to ensure it's heard
            self.engine.say(text)
            self.engine.runAndWait()
            
            print(f"✅ FINISHED SPEAKING: {text}")
            
        except Exception as e:
            print(f"✗ TTS error: {e}")
        finally:
            with self.lock:
                self.is_speaking = False
    
    def is_currently_speaking(self):
        """Check if TTS is currently speaking - Based on ocr1_llm.py"""
        if not self.engine:
            return False
        with self.lock:
            return self.is_speaking

# ============================================================================
# Detection Systems (No Camera Access)
# ============================================================================

class ObjectDetector:
    """Object detection without camera access"""
    
    def __init__(self):
        if not YOLO_AVAILABLE:
            self.model = None
            return
            
        self.model = YOLO("yolov8n.pt")
        self.classes = self.model.names
        self.last_detection = ""
        
        print("✓ Object Detector initialized")
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if not self.model:
            return None
        
        try:
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
                            label = self.classes[cls]
                            detections.append(label)
            
            if detections:
                # Create simple description
                if len(detections) == 1:
                    detection_text = f"a {detections[0]}"
                else:
                    detection_text = f"{', '.join(detections[:3])}"
                
                # Check if detection changed
                if detection_text != self.last_detection:
                    self.last_detection = detection_text
                    return detection_text
            
            return None
            
        except Exception as e:
            print(f"✗ Object detection error: {e}")
            return None

class OCRDetector:
    """OCR text detection without camera access"""
    
    def __init__(self):
        if not PADDLEOCR_AVAILABLE:
            self.ocr = None
            return
            
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            rec_batch_num=1
        )
        self.last_texts = []
        
        print("✓ OCR Detector initialized")
    
    def detect_text(self, frame):
        """Detect text in frame"""
        if not self.ocr:
            return None
        
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
            
            if texts and texts != self.last_texts:
                self.last_texts = texts.copy()
                return ', '.join(texts)
            
            return None
            
        except Exception as e:
            print(f"✗ OCR detection error: {e}")
            return None

class TrafficLightDetector:
    """Traffic light detection without camera access"""
    
    def __init__(self):
        if not YOLO_AVAILABLE:
            self.model = None
            return
            
        self.model = YOLO("yolov8n.pt")
        self.traffic_light_class_id = self._find_traffic_light_class_id()
        self.last_status = ""
        
        print("✓ Traffic Light Detector initialized")
    
    def _find_traffic_light_class_id(self):
        """Find traffic light class ID"""
        class_names = self.model.names
        
        traffic_light_names = ['traffic light', 'traffic_light', 'traffic-signal']
        
        for class_id, class_name in class_names.items():
            if any(name in class_name.lower() for name in traffic_light_names):
                return class_id
        
        return 9
    
    def detect_traffic_light(self, frame):
        """Detect traffic light and determine color"""
        if not self.model:
            return None
        
        try:
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
                            
                            # Check if status changed
                            if color != self.last_status and color != 'unknown':
                                self.last_status = color
                                return color
            
            return None
            
        except Exception as e:
            print(f"✗ Traffic light detection error: {e}")
            return None
    
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
# Single Camera Live System
# ============================================================================

class SingleCameraLiveSystem:
    """Main system with single camera access and immediate speech feedback"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        print("\n" + "="*60)
        print("🚀 Initializing Single Camera Live System")
        print("="*60)
        
        # Initialize detection systems (NO CAMERA ACCESS)
        self.object_detector = ObjectDetector()
        self.ocr_detector = OCRDetector()
        self.traffic_detector = TrafficLightDetector()
        
        # Initialize processing systems
        self.llm_processor = LiveLLMProcessor(TOGETHER_API_KEY)
        self.tts_handler = LiveTTSHandler(rate=TTS_RATE, volume=TTS_VOLUME)
        
        # Test TTS system
        print("\n🧪 Testing TTS system...")
        self.tts_handler.speak_immediately("TTS system test. Speech is working correctly.")
        time.sleep(3)  # Wait for speech to complete
        
        print("✓ All systems initialized")
        print("="*60)
    
    def run(self):
        """Main execution loop with single camera and immediate speech"""
        # SINGLE CAMERA ACCESS
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"✗ Cannot open camera {self.camera_id}")
            return
        
        print(f"\n🚀 Starting Single Camera Live System")
        print(f"📹 Camera {self.camera_id} initialized (SINGLE ACCESS)")
        print(f"🎤 Immediate speech feedback enabled")
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
                
                # Process all systems with the SAME frame - FASTER PROCESSING
                if frame_count % 2 == 0:  # Process every 2nd frame for faster response
                    
                    # Object detection - FAST COMMENTARY
                    obj_result = self.object_detector.detect_objects(frame)
                    if obj_result:
                        # Try LLM first, fallback to direct speech
                        llm_result = self.llm_processor.process_detection('object', obj_result)
                        if llm_result:
                            self.tts_handler.speak_immediately(llm_result)
                        else:
                            # Direct speech fallback for speed
                            self.tts_handler.speak_immediately(f"{obj_result} ahead")
                    
                    # OCR detection - FAST COMMENTARY
                    ocr_result = self.ocr_detector.detect_text(frame)
                    if ocr_result:
                        # Try LLM first, fallback to direct speech
                        llm_result = self.llm_processor.process_detection('ocr', ocr_result)
                        if llm_result:
                            self.tts_handler.speak_immediately(llm_result)
                        else:
                            # Direct speech fallback for speed
                            self.tts_handler.speak_immediately(f"Text: {ocr_result}")
                    
                    # Traffic light detection - FAST COMMENTARY
                    traffic_result = self.traffic_detector.detect_traffic_light(frame)
                    if traffic_result:
                        # Try LLM first, fallback to direct speech
                        llm_result = self.llm_processor.process_detection('traffic', traffic_result)
                        if llm_result:
                            self.tts_handler.speak_immediately(llm_result)
                        else:
                            # Direct speech fallback for speed
                            self.tts_handler.speak_immediately(f"Traffic light: {traffic_result}")
                
                # Draw frame info
                self._draw_frame_info(frame, frame_count)
                
                # Display frame
                cv2.imshow("Single Camera Live System", frame)
                
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
            print("✅ Single Camera Live System stopped")
    
    def _draw_frame_info(self, frame, frame_count):
        """Draw frame information"""
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "SINGLE CAMERA - LIVE SPEECH", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("🌟 Single Camera Live System")
    print("=" * 60)
    print("Features:")
    print("  ✓ Single camera access for all systems")
    print("  ✓ Object detection with LLM processing")
    print("  ✓ OCR text detection with LLM processing")
    print("  ✓ Traffic light detection with LLM processing")
    print("  ✓ Immediate speech feedback")
    print("  ✓ Real-time LLM sentence formation")
    print("=" * 60)
    
    # Check system availability
    if not any([YOLO_AVAILABLE, PADDLEOCR_AVAILABLE]):
        print("\n❌ No detection systems available. Please install required packages.")
        return
    
    # Initialize and run system
    try:
        system = SingleCameraLiveSystem(camera_id=0)
        system.run()
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
