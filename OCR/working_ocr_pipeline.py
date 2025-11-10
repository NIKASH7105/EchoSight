"""
Working Real-time OCR Pipeline
=============================
A simplified but working OCR pipeline that detects and recognizes text in real-time.

Features:
- Real-time camera feed capture
- Enhanced OpenCV text detection
- PaddleOCR text recognition (when available)
- Console output and file storage
- Visual feedback with bounding boxes

Requirements:
    pip install opencv-python paddleocr numpy
"""

import cv2
import numpy as np
import os
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("✓ PaddleOCR imported successfully")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    print(f"✗ PaddleOCR not available: {e}")
    print("Will use text detection only")


class TextDetector:
    """Enhanced text detection using OpenCV"""
    
    def __init__(self):
        print("Initializing text detector...")
    
    def detect_text_regions(self, image):
        """Detect text regions in image using multiple OpenCV methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = []
        
        # Method 1: Adaptive thresholding
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        # Method 2: Otsu thresholding
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 3: Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine methods
        combined = cv2.bitwise_or(thresh1, cv2.bitwise_or(thresh2, edges))
        
        # Morphological operations to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio for text-like regions
            if w > 20 and h > 15 and w < image.shape[1] * 0.8 and h < image.shape[0] * 0.8:
                aspect_ratio = w / float(h)
                area = w * h
                
                # Accept reasonable aspect ratios for text
                if 0.3 < aspect_ratio < 8 and area > 200:
                    # Convert to 4-point format
                    box = [x, y, x+w, y, x+w, y+h, x, y+h]
                    boxes.append(box)
        
        return boxes


class OCRPipeline:
    """Real-time OCR pipeline"""
    
    def __init__(self, output_file='output.txt', camera_id=0):
        self.output_file = output_file
        self.camera_id = camera_id
        
        # Initialize text detector
        print("1. Loading text detector...")
        self.detector = TextDetector()
        
        # Initialize OCR recognizer
        print("2. Loading text recognizer...")
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
                print("✓ PaddleOCR initialized successfully")
            except Exception as e:
                print(f"✗ PaddleOCR initialization failed: {e}")
                self.ocr = None
        else:
            self.ocr = None
        
        # Initialize output file
        self._initialize_output_file()
        print("✓ Pipeline initialization complete!")
    
    def _initialize_output_file(self):
        """Initialize output file with header"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("Real-time OCR Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def process_frame(self, frame, frame_count):
        """Process a single frame for text detection and recognition"""
        detected_texts = []
        annotated_frame = frame.copy()
        
        # Detect text regions
        boxes = self.detector.detect_text_regions(frame)
        
        # Process each detected text region
        for idx, box in enumerate(boxes):
            try:
                # Get bounding rectangle
                x_min = max(0, int(min(box[::2])))
                y_min = max(0, int(min(box[1::2])))
                x_max = min(frame.shape[1], int(max(box[::2])))
                y_max = min(frame.shape[0], int(max(box[1::2])))
                
                # Crop text region
                if x_max > x_min and y_max > y_min:
                    text_region = frame[y_min:y_max, x_min:x_max]
                    
                    # Skip very small regions
                    if text_region.shape[0] < 15 or text_region.shape[1] < 15:
                        continue
                    
                    # Recognize text using PaddleOCR
                    if self.ocr is not None:
                        try:
                            result = self.ocr.ocr(text_region, cls=True)
                            
                            if result and result[0]:
                                for line in result[0]:
                                    if line and len(line) > 1:
                                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                                        confidence = line[1][1] if isinstance(line[1], tuple) else 0.0
                                        
                                        # Only keep high confidence results
                                        if confidence > 0.5 and text.strip():
                                            detected_texts.append(text.strip())
                                            
                                            # Draw bounding box
                                            box_points = np.array(box).reshape(4, 2).astype(np.int32)
                                            cv2.polylines(annotated_frame, [box_points], True, (0, 255, 0), 2)
                                            
                                            # Draw recognized text
                                            cv2.putText(annotated_frame, f"{text[:20]}...", 
                                                      (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                      0.5, (0, 255, 0), 1)
                                            
                                            # Draw confidence score
                                            cv2.putText(annotated_frame, f"({confidence:.2f})", 
                                                      (x_min, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                                      0.3, (255, 255, 0), 1)
                        except Exception as e:
                            print(f"OCR error: {e}")
                            # Fallback: just draw bounding box
                            box_points = np.array(box).reshape(4, 2).astype(np.int32)
                            cv2.polylines(annotated_frame, [box_points], True, (0, 255, 0), 2)
                            cv2.putText(annotated_frame, "Text Region", (x_min, y_min - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        # No OCR available, just draw bounding box
                        box_points = np.array(box).reshape(4, 2).astype(np.int32)
                        cv2.polylines(annotated_frame, [box_points], True, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, "Text Region", (x_min, y_min - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            except Exception as e:
                print(f"Error processing box {idx}: {e}")
                continue
        
        return annotated_frame, detected_texts
    
    def save_results(self, frame_count, texts, timestamp):
        """Save recognized text to file"""
        if texts:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Frame {frame_count} at {timestamp} ---\n")
                for i, text in enumerate(texts, 1):
                    f.write(f"{i}. {text}\n")
                f.write("\n")
    
    def run(self, display=True, process_every_n_frames=3):
        """Run the real-time OCR pipeline"""
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera {self.camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"\n✓ Camera {self.camera_id} opened successfully")
        print("Starting real-time OCR pipeline...")
        print("Press 'q' to quit, 's' to save current frame\n")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                
                if not ret:
                    print("✗ Error: Could not read frame")
                    break
                
                frame_count += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                # Process every nth frame
                if frame_count % process_every_n_frames == 0:
                    # Process frame
                    annotated_frame, texts = self.process_frame(frame, frame_count)
                    
                    # Print results to console
                    if texts:
                        print(f"\n[Frame {frame_count} at {current_time}] Detected text:")
                        for i, text in enumerate(texts, 1):
                            print(f"  {i}. {text}")
                        
                        # Save to file
                        self.save_results(frame_count, texts, current_time)
                    else:
                        print(f"[Frame {frame_count}] No text detected")
                    
                    # Display processed frame
                    if display:
                        cv2.imshow('Real-time OCR Pipeline - Press Q to quit', annotated_frame)
                else:
                    # Display raw frame for non-processed frames
                    if display:
                        cv2.imshow('Real-time OCR Pipeline - Press Q to quit', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✓ Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"frame_{frame_count}_{current_time.replace(':', '-')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Frame saved as {filename}")
        
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Calculate and display statistics
            end_time = time.time()
            total_time = end_time - start_time
            fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"\n" + "="*60)
            print("OCR Pipeline Statistics")
            print("="*60)
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {fps:.2f}")
            print(f"Results saved to: {self.output_file}")
            print("="*60)


if __name__ == "__main__":
    # Configuration
    OUTPUT_FILE = 'output.txt'
    CAMERA_ID = 0
    DISPLAY_VIDEO = True
    PROCESS_EVERY_N_FRAMES = 3
    
    print("=" * 60)
    print("Working Real-time OCR Pipeline")
    print("=" * 60)
    print("Text Detection: Enhanced OpenCV methods")
    print("Text Recognition: PaddleOCR (when available)")
    print("=" * 60)
    print(f"Output File: {OUTPUT_FILE}")
    print(f"Camera ID: {CAMERA_ID}")
    print(f"Process every {PROCESS_EVERY_N_FRAMES} frames")
    print("=" * 60)
    
    # Initialize and run pipeline
    try:
        pipeline = OCRPipeline(output_file=OUTPUT_FILE, camera_id=CAMERA_ID)
        pipeline.run(display=DISPLAY_VIDEO, process_every_n_frames=PROCESS_EVERY_N_FRAMES)
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        print("Please check your camera connection and dependencies")
