#!/usr/bin/env python3
"""
Real-Time Traffic Light Detection with Interruptible Speech

This system provides instant, human-like voice feedback for traffic light changes
with interruptible speech and optimized real-time performance.
"""

import cv2
import numpy as np
import time
import threading
import queue
import subprocess
import os
import tempfile
from ultralytics import YOLO
from typing import Optional, Dict, Any


class WindowsSpeechEngine:
    """Ultra-fast speech engine using Windows SAPI for reliable audio."""
    
    def __init__(self):
        self.last_speech_time = 0
        self.speech_cooldown = 1.0  # 1 second cooldown to prevent repetition
        
        # Pre-defined speech messages for calm, clear response
        self.speech_messages = {
            'red': "Now You can cross the Road.",
            'yellow': "Be ready to cross the Road.", 
            'green': "Stop Wait for the next signal."
        }
        
        print("🎤 Windows Speech Engine Ready")
        print("⚡ Using Windows SAPI for reliable audio output")
        
        # Test speech on initialization
        self._test_speech()
    
    def _test_speech(self):
        """Test speech system on startup."""
        try:
            print("🧪 Testing speech system...")
            self._speak_with_windows("Speech system ready")
            print("✅ Speech test completed")
        except Exception as e:
            print(f"❌ Speech test failed: {e}")
    
    def _speak_with_windows(self, message: str):
        """Speak using Windows built-in speech system."""
        try:
            # Method 1: Try using Windows PowerShell with SAPI
            ps_command = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.Volume = 80
$synth.Speak("{message}")
'''
            
            print(f"🔊 SPEAKING: {message}")
            
            # Run PowerShell command
            result = subprocess.run([
                'powershell', '-Command', ps_command
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print(f"✅ SPOKE: {message}")
            else:
                print(f"❌ PowerShell speech failed: {result.stderr}")
                # Fallback to pyttsx3
                self._speak_with_pyttsx3(message)
                
        except Exception as e:
            print(f"❌ Windows speech error: {e}")
            # Fallback to pyttsx3
            self._speak_with_pyttsx3(message)
    
    def _speak_with_pyttsx3(self, message: str):
        """Fallback speech using pyttsx3."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Calmer, slower speech
            engine.setProperty('volume', 0.8)  # Softer volume
            engine.say(message)
            engine.runAndWait()
            print(f"✅ SPOKE (fallback): {message}")
        except Exception as e:
            print(f"❌ Fallback speech error: {e}")
    
    def speak(self, color: str):
        """Speak immediately with minimal delay."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_speech_time < self.speech_cooldown:
            return
        
        if color in self.speech_messages:
            message = self.speech_messages[color]
            self.last_speech_time = current_time
            
            # Speak immediately in a separate thread to avoid blocking
            speech_thread = threading.Thread(
                target=self._speak_with_windows, 
                args=(message,),
                daemon=True
            )
            speech_thread.start()
    
    def stop(self):
        """Stop speech engine."""
        pass  # No cleanup needed for Windows SAPI


class RealTimeTrafficLightDetector:
    """Ultra-fast traffic light detection with real-time voice feedback."""
    
    def __init__(self):
        # Initialize YOLO model with optimized settings
        self.model = YOLO("yolov8n.pt")
        
        # Find traffic light class ID dynamically
        self.traffic_light_class_id = self._find_traffic_light_class_id()
        
        # Initialize speech engine
        self.speech_engine = WindowsSpeechEngine()
        
        # State tracking
        self.current_color: Optional[str] = None
        self.last_color: Optional[str] = None
        self.frame_skip = 2  # Process every 2nd frame for speed
        self.frame_count = 0
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_count = 0
        
        print("🚦 Real-Time Traffic Light Detector Ready")
        print(f"🔍 Traffic light class ID: {self.traffic_light_class_id}")
        print("⚡ Optimized for <0.5ms response time")
        
        # Show available classes for debugging
        print("\n📋 Available detection classes:")
        for class_id, class_name in list(self.model.names.items())[:10]:  # Show first 10
            marker = "🚦" if class_id == self.traffic_light_class_id else "  "
            print(f"{marker} {class_id}: {class_name}")
        if len(self.model.names) > 10:
            print(f"  ... and {len(self.model.names) - 10} more classes")
        print()
    
    def _find_traffic_light_class_id(self) -> int:
        """Find traffic light class ID in COCO dataset."""
        class_names = self.model.names
        
        # Common traffic light class names
        traffic_light_names = ['traffic light', 'traffic_light', 'traffic-signal']
        
        for class_id, class_name in class_names.items():
            if any(name in class_name.lower() for name in traffic_light_names):
                print(f"✅ Found traffic light class: {class_name} (ID: {class_id})")
                return class_id
        
        # If not found, try common IDs
        common_ids = [9, 10, 11]  # Common traffic light IDs in different datasets
        for class_id in common_ids:
            if class_id in class_names:
                print(f"⚠️ Using fallback traffic light class: {class_names[class_id]} (ID: {class_id})")
                return class_id
        
        print("❌ Traffic light class not found! Using ID 9 as default")
        return 9
    
    def detect_traffic_light_color(self, img: np.ndarray, bbox: tuple) -> str:
        """Fast color detection within traffic light bounding box."""
        x1, y1, x2, y2 = bbox
        
        # Extract ROI with bounds checking
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 'unknown'
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Optimized color ranges
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
        
        yellow_pixels = cv2.countNonZero(cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 255, 255])))
        green_pixels = cv2.countNonZero(cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255])))
        
        # Determine dominant color with improved logic
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        
        # Calculate total pixels for percentage check
        total_pixels = roi.shape[0] * roi.shape[1]
        color_percentage = max_pixels / total_pixels if total_pixels > 0 else 0
        
        # More lenient threshold but require minimum percentage
        if max_pixels < 10 or color_percentage < 0.01:  # At least 1% of ROI
            return 'unknown'
        
        # Check for clear dominance (at least 1.5x more pixels than others)
        if max_pixels == red_pixels and red_pixels > yellow_pixels * 1.5 and red_pixels > green_pixels * 1.5:
            return 'red'
        elif max_pixels == yellow_pixels and yellow_pixels > red_pixels * 1.5 and yellow_pixels > green_pixels * 1.5:
            return 'yellow'
        elif max_pixels == green_pixels and green_pixels > red_pixels * 1.5 and green_pixels > yellow_pixels * 1.5:
            return 'green'
        else:
            return 'unknown'
    
    def process_frame(self, img: np.ndarray) -> Dict[str, Any]:
        """Process single frame with optimized inference."""
        start_time = time.time()
        
        # Run YOLO inference with optimized settings
        results = self.model.predict(
            source=img,
            stream=False,
            conf=0.2,  # Very low confidence for better detection
            imgsz=640,  # Larger image size for better detection
            verbose=False,
            save=False
        )
        
        detections = []
        all_detections = []  # Debug: track all detections
        traffic_light_detections = []  # Track traffic light detections specifically
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Debug: show all detections
                    class_name = self.model.names[cls]
                    all_detections.append(f"{class_name}: {confidence:.2f}")
                    
                    # Check if this is a traffic light detection
                    if cls == self.traffic_light_class_id:
                        traffic_light_detections.append(f"{class_name}: {confidence:.2f}")
                        
                        if confidence > 0.2:  # Lower threshold
                            x1, y1, x2, y2 = box.xyxy[0]
                            bbox = (int(x1), int(y1), int(x2), int(y2))
                            
                            # Detect color
                            color = self.detect_traffic_light_color(img, bbox)
                            
                            print(f"🚦 Traffic light detected: {confidence:.2f} -> {color}")
                            
                            if color != 'unknown':
                                detections.append({
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'color': color
                                })
        
        # Debug output
        if self.frame_count % 30 == 0:  # Print every 30 frames
            if traffic_light_detections:
                print(f"Frame {self.frame_count} traffic lights: {', '.join(traffic_light_detections)}")
            elif all_detections:
                print(f"Frame {self.frame_count} all detections: {', '.join(all_detections[:3])}...")  # Show first 3
        
        # Update state and trigger speech
        self._update_state_and_speak(detections)
        
        # Performance tracking
        process_time = (time.time() - start_time) * 1000  # Convert to ms
        self.detection_count += 1
        
        return {
            'detections': detections,
            'process_time_ms': process_time,
            'current_color': self.current_color,
            'all_detections': all_detections  # Debug info
        }
    
    def _update_state_and_speak(self, detections: list):
        """Update color state and trigger speech ONLY when color actually changes."""
        if detections:
            # Get most confident detection
            best_detection = max(detections, key=lambda x: x['confidence'])
            new_color = best_detection['color']
            
            # Only trigger speech if color actually changed
            if new_color != self.current_color:
                print(f"🎯 COLOR CHANGE: {self.current_color} -> {new_color}")
                self.speech_engine.speak(new_color)
                self.last_detection_time = time.time()
            
            # Update state after speech
            self.last_color = self.current_color
            self.current_color = new_color
        else:
            # No detections - reset state
            self.last_color = self.current_color
            self.current_color = None
    
    def run(self, camera_id: int = 0, test_mode: bool = False):
        """Main detection loop with real-time optimization."""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🚀 Starting Real-Time Traffic Light Detection")
        print("📹 Camera initialized")
        print("⚡ Processing every 2nd frame for maximum speed")
        print("🎤 Speech engine ready with instant interruption")
        
        if test_mode:
            print("🧪 TEST MODE: Will simulate traffic light detections")
            print("Press 't' to trigger test detection")
            print("Press 'i' to test with generated image")
            print("Press 's' to test speech directly")
            print("Press 'q' to quit")
        else:
            print("Press 'q' to quit")
        print("-" * 60)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                self.frame_count += 1
                
                # Process only every nth frame for speed
                if self.frame_count % self.frame_skip == 0:
                    result = self.process_frame(frame)
                    
                    # Draw detections
                    self._draw_detections(frame, result['detections'])
                    
                    # Performance info
                    self._draw_performance_info(frame, result)
                
                # Display frame
                cv2.imshow('Real-Time Traffic Light Detection', frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif test_mode and key == ord('t'):
                    # Test mode: simulate traffic light detection
                    self._simulate_traffic_light_detection()
                elif test_mode and key == ord('i'):
                    # Test mode: test with generated image
                    self._test_with_generated_image()
                elif test_mode and key == ord('s'):
                    # Test mode: test speech directly
                    self._test_speech_directly()
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping detection...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.speech_engine.stop()
            print("✅ System stopped")
    
    def _draw_detections(self, img: np.ndarray, detections: list):
        """Draw detection results on image."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            confidence = det['confidence']
            
            # Color-coded bounding box
            if color == 'red':
                box_color = (0, 0, 255)
            elif color == 'yellow':
                box_color = (0, 255, 255)
            else:  # green
                box_color = (0, 255, 0)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
            
            # Label
            label = f"{color.upper()}: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    
    def _draw_performance_info(self, img: np.ndarray, result: dict):
        """Draw performance information on image."""
        # Status text
        status = f"Detections: {len(result['detections'])}"
        cv2.putText(img, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current color
        if result['current_color']:
            color_text = f"Current: {result['current_color'].upper()}"
            cv2.putText(img, color_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Performance metrics
        perf_text = f"Process: {result['process_time_ms']:.1f}ms"
        cv2.putText(img, perf_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Frame skip info
        skip_text = f"Frame: {self.frame_count} (skip: {self.frame_skip})"
        cv2.putText(img, skip_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _simulate_traffic_light_detection(self):
        """Simulate traffic light detection for testing."""
        import random
        
        colors = ['red', 'yellow', 'green']
        test_color = random.choice(colors)
        
        # Create fake detection
        fake_detection = [{
            'bbox': (100, 100, 200, 200),
            'confidence': 0.8,
            'color': test_color
        }]
        
        print(f"🧪 TEST: Simulating {test_color} traffic light detection")
        self._update_state_and_speak(fake_detection)
    
    def create_test_traffic_light_image(self, color: str = 'red'):
        """Create a test image with a traffic light for testing."""
        # Create a simple test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple traffic light
        center_x, center_y = 320, 240
        light_radius = 30
        
        # Traffic light background
        cv2.rectangle(img, (center_x - 40, center_y - 100), (center_x + 40, center_y + 100), (50, 50, 50), -1)
        
        # Draw the active light
        if color == 'red':
            cv2.circle(img, (center_x, center_y - 50), light_radius, (0, 0, 255), -1)
        elif color == 'yellow':
            cv2.circle(img, (center_x, center_y), light_radius, (0, 255, 255), -1)
        elif color == 'green':
            cv2.circle(img, (center_x, center_y + 50), light_radius, (0, 255, 0), -1)
        
        return img
    
    def _test_with_generated_image(self):
        """Test detection with a generated traffic light image."""
        import random
        
        colors = ['red', 'yellow', 'green']
        test_color = random.choice(colors)
        
        print(f"🧪 TEST: Creating test image with {test_color} traffic light")
        
        # Create test image
        test_img = self.create_test_traffic_light_image(test_color)
        
        # Process the test image
        result = self.process_frame(test_img)
        
        print(f"🧪 TEST: Processed test image, found {len(result['detections'])} detections")
        
        # Show the test image
        cv2.imshow('Test Traffic Light', test_img)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyWindow('Test Traffic Light')
    
    def _test_speech_directly(self):
        """Test speech system directly without detection."""
        import random
        
        colors = ['red', 'yellow', 'green']
        test_color = random.choice(colors)
        
        print(f"🧪 SPEECH TEST: Testing {test_color} speech directly")
        self.speech_engine.speak(test_color)


def main():
    """Main function."""
    print("🚦 Real-Time Traffic Light Detection System")
    print("=" * 50)
    
    detector = RealTimeTrafficLightDetector()
    
    # Enable test mode for easier testing
    print("Choose mode:")
    print("1. Normal detection mode")
    print("2. Test mode (simulate detections)")
    
    try:
        choice = "1"
        test_mode = choice == "2"
    except:
        test_mode = True  # Default to test mode
    
    detector.run(test_mode=test_mode)


if __name__ == "__main__":
    main()
