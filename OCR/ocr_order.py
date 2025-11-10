"""
Complete OCR Pipeline with CRAFT + PaddleOCR - MAXIMUM TEXT COVERAGE
=====================================================================
Optimized for capturing ALL text in real-time scenarios with READING ORDER
"""

import cv2
import torch
import numpy as np
import os
from datetime import datetime
import warnings
import time
from collections import OrderedDict
from threading import Thread, Lock

warnings.filterwarnings('ignore')

# Import LLM and TTS dependencies
try:
    from together import Together
    TOGETHER_AVAILABLE = True
    print("✓ Together AI imported successfully")
except ImportError as e:
    TOGETHER_AVAILABLE = False
    print(f"✗ Together AI import failed: {e}")

try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("✓ pyttsx3 imported successfully")
except ImportError as e:
    TTS_AVAILABLE = False
    print(f"✗ pyttsx3 import failed: {e}")

# Import CRAFT model definition
try:
    import sys
    import os
    craft_path = os.path.join(os.path.dirname(__file__), 'CRAFT_pytorch')
    if craft_path not in sys.path:
        sys.path.insert(0, craft_path)
    
    from craft import CRAFT
    CRAFT_MODULE_AVAILABLE = True
    print("✓ CRAFT module imported successfully")
except ImportError as e:
    CRAFT_MODULE_AVAILABLE = False
    print(f"✗ CRAFT module import failed: {e}")

# Import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("✓ PaddleOCR imported successfully")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    print(f"✗ PaddleOCR import failed: {e}")
    exit(1)

# ============================================================================
# LLM and TTS Configuration
# ============================================================================

# LLM Configuration
TOGETHER_API_KEY = "a60c2c24e4f37100bf8dea9930a9a8a0d354b122c597847eca8dad4ee1551efd"  # Replace with your API key
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Text-to-Speech Configuration
TTS_RATE = 150  # Speaking rate (words per minute)
TTS_VOLUME = 0.9  # Volume (0.0 to 1.0)

# ============================================================================
# Text Processing with LLM
# ============================================================================

class TextProcessor:
    """Process fragmented OCR text using Together AI LLM"""
    
    def __init__(self, api_key):
        if not TOGETHER_AVAILABLE:
            print("⚠ Together AI not available - LLM processing disabled")
            self.client = None
            return
            
        self.client = Together(api_key=api_key)
        self.last_processed_texts = []
        self.lock = Lock()
        
        print("✓ Together AI client initialized")
    
    def combine_texts(self, text_fragments):
        """
        Combine fragmented OCR texts into coherent message using LLM
        
        Args:
            text_fragments: List of text strings from OCR
        
        Returns:
            Combined and refined message string
        """
        if not text_fragments or not self.client:
            return None
        
        # Join fragments preserving reading order
        raw_text = " ".join(text_fragments)
        
        # Check if we've already processed similar text recently
        with self.lock:
            if raw_text in self.last_processed_texts:
                return None
            self.last_processed_texts.append(raw_text)
            # Keep only last 10 processed texts to avoid memory buildup
            if len(self.last_processed_texts) > 10:
                self.last_processed_texts.pop(0)
        
        # Create prompt for LLM
        prompt = f"""You are an assistive AI helping visually impaired users understand scanned documents.
You will receive unordered OCR text fragments and must reconstruct them into a clear, natural language description that sounds like a human is reading the document aloud.

STRICT RULES

Use only the provided fragments — do not add or assume new information.

Reconstruct the content into a logical, meaningful sentence or structured spoken description.

Make it sound like a real voice assistant reading an ID card or document.

Preserve accuracy — never change names, numbers, institution, or meaning.

No analysis, no extra commentary, no instructions — only read out the content clearly.


OUTPUT

A single clear spoken-style description as if read aloud for a blind user.

Fragmented text: {raw_text}

Reconstructed message:"""
        
        try:
            # Call Together AI LLM
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,  # Low temperature for consistency
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.1,
                stop=["</s>", "\n\n"]
            )
            
            # Extract the message
            combined_text = response.choices[0].message.content.strip()
            
            # Clean up any residual formatting
            combined_text = self._clean_output(combined_text)
            
            return combined_text
            
        except Exception as e:
            print(f"✗ LLM processing error: {e}")
            return None
    
    def _clean_output(self, text):
        """Clean LLM output to ensure only the message is returned"""
        # Remove common prefixes
        prefixes = [
            "Reconstructed message:",
            "Combined text:",
            "Output:",
            "Result:",
            "Message:"
        ]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Remove quotes if the entire text is quoted
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        return text

# ============================================================================
# Text-to-Speech Handler
# ============================================================================

class TTSHandler:
    """Handle text-to-speech conversion with threading for real-time effect"""
    
    def __init__(self, rate=150, volume=0.9):
        if not TTS_AVAILABLE:
            print("⚠ pyttsx3 not available - TTS disabled")
            self.engine = None
            return
            
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.speaking_queue = []
        self.is_speaking = False
        self.lock = Lock()
        
        # Get available voices (optional: select different voice)
        voices = self.engine.getProperty('voices')
        if voices:
            # You can change voice here if desired
            # self.engine.setProperty('voice', voices[1].id)  # Female voice
            pass
        
        print("✓ Text-to-Speech engine initialized")
    
    def speak(self, text):
        """Speak text in a separate thread for non-blocking operation"""
        if not text or not text.strip() or not self.engine:
            return
        
        def _speak():
            with self.lock:
                self.is_speaking = True
            try:
                print(f"\n🔊 Speaking: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"✗ TTS error: {e}")
            finally:
                with self.lock:
                    self.is_speaking = False
        
        # Run TTS in separate thread
        thread = Thread(target=_speak, daemon=True)
        thread.start()
    
    def is_currently_speaking(self):
        """Check if TTS is currently speaking"""
        if not self.engine:
            return False
        with self.lock:
            return self.is_speaking


class CRAFTDetector:
    """CRAFT text detection with maximum coverage settings"""
    
    def __init__(self, use_cuda=True):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() and use_cuda:
            print("Running CRAFT on: GPU")
        else:
            print("Running CRAFT on: CPU")
        
        self.net = self._load_craft_model()
        if self.net is not None:
            self.net.to(self.device)
            self.net.eval()
            print("✓ Loaded local CRAFT model")
        else:
            print("⚠ CRAFT model not available, using fallback detection")
    
    def _load_craft_model(self):
        """Load pre-trained CRAFT model"""
        if not CRAFT_MODULE_AVAILABLE:
            return None
        
        try:
            model = CRAFT()
            possible_paths = [
                'craft_mlt_25k.pth',
                os.path.join(os.path.dirname(__file__), 'craft_mlt_25k.pth'),
                r'D:\trafficdl\OCR\craft_mlt_25k.pth'
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print("Model file not found, using fallback detection")
                return None
            
            state_dict = torch.load(model_path, map_location='cpu')
            state_dict = self._copy_state_dict(state_dict)
            model.load_state_dict(state_dict)
            
            print(f"✓ CRAFT model loaded from {model_path}")
            return model
            
        except Exception as e:
            print(f"Failed to load CRAFT model: {e}")
            return None
    
    def _copy_state_dict(self, state_dict):
        """Handle 'module.' prefix in state dict"""
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    def detect_text_regions(self, image, text_threshold=0.3, link_threshold=0.2, low_text=0.2):
        """
        Detect text regions with LOWER thresholds for maximum coverage
        
        Args:
            text_threshold: Lowered from 0.7 to 0.3 for more detections
            link_threshold: Lowered from 0.4 to 0.2
            low_text: Lowered from 0.4 to 0.2
        """
        if self.net is None:
            return self._fallback_text_detection(image)
        
        try:
            img_resized, target_ratio, size_heatmap = self._resize_aspect_ratio(
                image, square_size=1280, interpolation=cv2.INTER_LINEAR
            )
            
            x = self._normalize_mean_variance(img_resized)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                y, feature = self.net(x)
            
            score_text = y[0, :, :, 0].cpu().numpy()
            score_link = y[0, :, :, 1].cpu().numpy()
            
            boxes = self._get_boxes(score_text, score_link, text_threshold, link_threshold, low_text)
            boxes = self._adjust_result_coordinates(boxes, target_ratio, size_heatmap)
            
            return boxes
            
        except Exception as e:
            print(f"Error in CRAFT detection: {e}")
            return self._fallback_text_detection(image)
    
    def _resize_aspect_ratio(self, img, square_size=1280, interpolation=cv2.INTER_LINEAR):
        """Resize image maintaining aspect ratio"""
        height, width, channel = img.shape
        target_size = square_size
        ratio = min(target_size / height, target_size / width)
        target_h, target_w = int(height * ratio), int(width * ratio)
        
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
        
        target_h32 = target_h + (32 - target_h % 32) if target_h % 32 != 0 else target_h
        target_w32 = target_w + (32 - target_w % 32) if target_w % 32 != 0 else target_w
        
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
        resized[0:target_h, 0:target_w, :] = proc
        
        size_heatmap = (int(target_w32 / 2), int(target_h32 / 2))
        
        return resized, ratio, size_heatmap
    
    def _normalize_mean_variance(self, img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        """Normalize image for CRAFT"""
        img = img.astype(np.float32) / 255.0
        img -= np.array(mean)
        img /= np.array(variance)
        return img
    
    def _get_boxes(self, score_text, score_link, text_threshold, link_threshold, low_text):
        """Extract bounding boxes from score maps"""
        boxes = []
        
        text_score = score_text > text_threshold
        link_score = score_link > link_threshold
        text_score_comb = np.clip(text_score + link_score, 0, 1)
        text_score_comb = (text_score_comb * 255).astype(np.uint8)
        contours, _ = cv2.findContours(text_score_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Lower minimum area threshold
            if cv2.contourArea(contour) > 5:  # Changed from 10 to 5
                boxes.append(box.reshape(-1))
        
        return boxes
    
    def _adjust_result_coordinates(self, boxes, ratio, size_heatmap):
        """Adjust box coordinates to original size"""
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        boxes = boxes * 2
        boxes = boxes / ratio
        
        return boxes.astype(np.int32).tolist()
    
    def _fallback_text_detection(self, image):
        """Enhanced fallback detection with lower thresholds"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = []
            
            # Multiple thresholding methods
            thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            edges = cv2.Canny(gray, 50, 150)
            
            combined = cv2.bitwise_or(thresh1, cv2.bitwise_or(thresh2, edges))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # More permissive filtering
                if w > 10 and h > 8 and w < image.shape[1] * 0.95 and h < image.shape[0] * 0.95:
                    aspect_ratio = w / float(h)
                    area = w * h
                    
                    # Lower minimum area
                    if 0.1 < aspect_ratio < 15 and area > 50:  # Changed from 100 to 50
                        box = [x, y, x+w, y, x+w, y+h, x, y+h]
                        boxes.append(box)
            
            return boxes
        except Exception as e:
            print(f"Error in fallback detection: {e}")
            return []
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        try:
            def box_to_rect(box):
                box = np.array(box).reshape(4, 2)
                x_min, y_min = box.min(axis=0)
                x_max, y_max = box.max(axis=0)
                return x_min, y_min, x_max, y_max
            
            x1_min, y1_min, x1_max, y1_max = box_to_rect(box1)
            x2_min, y2_min, x2_max, y2_max = box_to_rect(box2)
            
            x_left = max(x1_min, x2_min)
            y_top = max(y1_min, y2_min)
            x_right = min(x1_max, x2_max)
            y_bottom = min(y1_max, y2_max)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            return 0.0
    
    def _calculate_spatial_relationship(self, box1, box2):
        """Determine spatial relationship between boxes"""
        try:
            def box_to_rect(box):
                box = np.array(box).reshape(4, 2)
                x_min, y_min = box.min(axis=0)
                x_max, y_max = box.max(axis=0)
                return x_min, y_min, x_max, y_max
            
            x1_min, y1_min, x1_max, y1_max = box_to_rect(box1)
            x2_min, y2_min, x2_max, y2_max = box_to_rect(box2)
            
            # Calculate centers
            c1_x, c1_y = (x1_min + x1_max) / 2, (y1_min + y1_max) / 2
            c2_x, c2_y = (x2_min + x2_max) / 2, (y2_min + y2_max) / 2
            
            # Calculate dimensions
            w1, h1 = x1_max - x1_min, y1_max - y1_min
            w2, h2 = x2_max - x2_min, y2_max - y2_min
            
            # Check if one box is nested in another
            if (x1_min <= x2_min and x1_max >= x2_max and y1_min <= y2_min and y1_max >= y2_max):
                return 'nested'
            if (x2_min <= x1_min and x2_max >= x1_max and y2_min <= y1_min and y2_max >= y1_max):
                return 'nested'
            
            # Calculate distances
            dx = abs(c1_x - c2_x)
            dy = abs(c1_y - c2_y)
            
            # Horizontal alignment (same line) - more permissive
            if dy < min(h1, h2) * 0.7 and dx < max(w1, w2) * 3:
                return 'horizontal'
            
            # Vertical alignment (same column)
            if dx < min(w1, w2) * 0.7 and dy < max(h1, h2) * 3:
                return 'vertical'
            
            return 'separate'
            
        except Exception as e:
            return 'separate'
    
    def smart_box_handler(self, boxes, iou_threshold=0.1):
        """
        INTELLIGENT BOX HANDLER with LOWER threshold for better coverage
        - Keeps nested boxes for multi-line text
        - Merges horizontal overlaps (split words)
        - Keeps vertical stacks separate
        - Removes only true duplicates
        """
        if len(boxes) <= 1:
            return boxes
        
        boxes_array = [np.array(box).reshape(4, 2) for box in boxes]
        kept_boxes = []
        processed = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if processed[i]:
                continue
            
            current_box = boxes[i]
            current_array = boxes_array[i]
            merged = False
            
            for j in range(i + 1, len(boxes)):
                if processed[j]:
                    continue
                
                iou = self._calculate_iou(current_box, boxes[j])
                relationship = self._calculate_spatial_relationship(current_box, boxes[j])
                
                # CASE 1: Very high IoU - true duplicates only
                if iou > 0.9:
                    area_i = cv2.contourArea(current_array)
                    area_j = cv2.contourArea(boxes_array[j])
                    
                    if area_j > area_i:
                        current_box = boxes[j]
                        current_array = boxes_array[j]
                    processed[j] = True
                    merged = True
                
                # CASE 2: Nested boxes - keep both
                elif relationship == 'nested':
                    continue
                
                # CASE 3: Horizontal overlap - merge more aggressively
                elif relationship == 'horizontal' and iou > 0.05:  # Lowered from 0.1
                    all_points = np.vstack([current_array, boxes_array[j]])
                    x_min, y_min = all_points.min(axis=0)
                    x_max, y_max = all_points.max(axis=0)
                    
                    current_box = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                    current_array = np.array(current_box).reshape(4, 2)
                    processed[j] = True
                    merged = True
                
                # CASE 4: Vertical relationship - keep separate
                elif relationship == 'vertical':
                    continue
                
                # CASE 5: Moderate overlap - only merge if very close
                elif 0.3 < iou < 0.9:
                    all_points = np.vstack([current_array, boxes_array[j]])
                    x_min, y_min = all_points.min(axis=0)
                    x_max, y_max = all_points.max(axis=0)
                    
                    current_box = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                    current_array = np.array(current_box).reshape(4, 2)
                    processed[j] = True
                    merged = True
            
            kept_boxes.append(current_box)
            processed[i] = True
        
        return kept_boxes
    
    def _expand_box(self, box, image_shape, width_margin=0.15, height_margin=0.25):
        """Expand box with LARGER margins for better text capture"""
        try:
            box = np.array(box).reshape(4, 2)
            x_min, y_min = box.min(axis=0)
            x_max, y_max = box.max(axis=0)
            
            width = x_max - x_min
            height = y_max - y_min
            
            width_expand = int(width * width_margin)
            height_expand = int(height * height_margin)
            
            x_min = max(0, x_min - width_expand)
            y_min = max(0, y_min - height_expand)
            x_max = min(image_shape[1], x_max + width_expand)
            y_max = min(image_shape[0], y_max + height_expand)
            
            return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            
        except Exception as e:
            return box
    
    def _preprocess_for_ocr(self, image):
        """Enhanced preprocessing for better OCR"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Slight sharpening
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Convert back to BGR
            enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            return image


def sort_boxes_reading_order(boxes, line_height_threshold_ratio=0.5):
    """
    Sort boxes in reading order: top-to-bottom, left-to-right
    Enhanced to better group boxes on the same line
    
    Args:
        boxes: List of boxes (each box has format [x1,y1,x2,y2,x3,y3,x4,y4])
        line_height_threshold_ratio: Ratio of box height to determine same line (0.5 = 50% overlap)
    
    Returns:
        Sorted list of (index, box) tuples
    """
    if not boxes:
        return []
    
    # Convert boxes to (index, center_y, center_x, y_min, y_max, x_min, height, box) format
    indexed_boxes = []
    for idx, box in enumerate(boxes):
        box_array = np.array(box).reshape(4, 2)
        y_min = int(box_array[:, 1].min())
        y_max = int(box_array[:, 1].max())
        x_min = int(box_array[:, 0].min())
        x_max = int(box_array[:, 0].max())
        
        center_y = (y_min + y_max) / 2
        center_x = (x_min + x_max) / 2
        height = y_max - y_min
        
        indexed_boxes.append((idx, center_y, center_x, y_min, y_max, x_min, height, box))
    
    # Sort by center_y first
    indexed_boxes.sort(key=lambda x: x[1])
    
    # Group boxes into lines using adaptive threshold
    lines = []
    current_line = [indexed_boxes[0]]
    
    for i in range(1, len(indexed_boxes)):
        curr_idx, curr_cy, curr_cx, curr_y_min, curr_y_max, curr_x_min, curr_height, curr_box = indexed_boxes[i]
        
        # Get the average height and y-position of current line
        line_heights = [item[6] for item in current_line]
        line_y_centers = [item[1] for item in current_line]
        avg_line_height = sum(line_heights) / len(line_heights)
        avg_line_y = sum(line_y_centers) / len(line_y_centers)
        
        # Calculate dynamic threshold based on average height
        dynamic_threshold = avg_line_height * line_height_threshold_ratio
        
        # Check if current box overlaps with the line's Y-range
        line_y_min = min(item[3] for item in current_line)
        line_y_max = max(item[4] for item in current_line)
        
        # Check for vertical overlap OR close proximity
        vertical_overlap = not (curr_y_max < line_y_min or curr_y_min > line_y_max)
        y_distance = abs(curr_cy - avg_line_y)
        
        if vertical_overlap or y_distance <= dynamic_threshold:
            # Same line
            current_line.append(indexed_boxes[i])
        else:
            # New line - save current line and start new one
            current_line.sort(key=lambda x: x[2])  # Sort by center_x (left to right)
            lines.append(current_line)
            current_line = [indexed_boxes[i]]
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda x: x[2])  # Sort by center_x
        lines.append(current_line)
    
    # Flatten lines back into single list
    sorted_boxes = []
    for line in lines:
        sorted_boxes.extend(line)
    
    return sorted_boxes


class RealTimeOCRPipeline:
    """Optimized OCR pipeline with maximum text coverage, reading order, LLM processing, and TTS"""
    
    def __init__(self, output_file='output.txt', use_gpu=True, camera_id=0, enable_llm_tts=True):
        self.output_file = output_file
        self.use_gpu = use_gpu
        self.camera_id = camera_id
        self.enable_llm_tts = enable_llm_tts
        
        print("\n" + "="*60)
        print("Initializing OCR Pipeline - MAXIMUM COVERAGE MODE")
        print("="*60)
        
        print("1. Loading CRAFT text detector...")
        self.detector = CRAFTDetector(use_cuda=use_gpu)
        
        print("2. Loading PaddleOCR...")
        try:
            # Enable angle classification for better accuracy
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Enabled for accuracy
                lang='en',
                rec_batch_num=1
            )
            print("✓ PaddleOCR initialized with angle classification")
        except Exception as e:
            print(f"✗ PaddleOCR initialization failed: {e}")
            exit(1)
        
        # Initialize LLM and TTS if enabled
        if self.enable_llm_tts:
            print("3. Initializing LLM and TTS...")
            self.text_processor = TextProcessor(TOGETHER_API_KEY)
            self.tts_handler = TTSHandler(rate=TTS_RATE, volume=TTS_VOLUME)
            print("✓ LLM and TTS initialized")
        else:
            self.text_processor = None
            self.tts_handler = None
            print("3. LLM and TTS disabled")
        
        self._initialize_output_file()
        print("✓ Pipeline ready - Maximum text coverage enabled!")
        print("✓ Reading order: Top-to-Bottom, Left-to-Right")
        if self.enable_llm_tts:
            print("✓ LLM processing: Enabled")
            print("✓ Text-to-Speech: Enabled")
        print("="*60)
    
    def _initialize_output_file(self):
        """Initialize output file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("Real-time OCR Results - Maximum Coverage Mode with Reading Order\n")
            f.write("=" * 50 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def process_frame(self, frame, frame_count):
        """
        ENHANCED FRAME PROCESSING with multiple preprocessing attempts and READING ORDER
        """
        detected_texts = []
        annotated_frame = frame.copy()
        
        # Step 1: Detect text regions
        boxes = self.detector.detect_text_regions(frame)
        
        if not boxes:
            return annotated_frame, detected_texts
        
        print(f"\n[Frame {frame_count}] Detected {len(boxes)} initial boxes")
        
        # Step 2: Smart box handling
        processed_boxes = self.detector.smart_box_handler(boxes, iou_threshold=0.1)
        print(f"After smart handling: {len(processed_boxes)} boxes")
        
        # Step 3: Sort boxes in reading order (top-to-bottom, left-to-right)
        sorted_boxes = sort_boxes_reading_order(processed_boxes)
        print(f"Sorted in reading order: {len(sorted_boxes)} boxes")
        
        # Step 4: Process each box in sorted order
        for reading_order, (original_idx, center_y, center_x, y_min, y_max, x_min, height, box) in enumerate(sorted_boxes, start=1):
            try:
                # Expand box generously
                expanded_box = self.detector._expand_box(box, frame.shape, 
                                                        width_margin=0.05, height_margin=0.10)
                
                box_array = np.array(expanded_box).reshape(4, 2).astype(np.int32)
                x_min = max(0, int(box_array[:, 0].min()))
                y_min = max(0, int(box_array[:, 1].min()))
                x_max = min(frame.shape[1], int(box_array[:, 0].max()))
                y_max = min(frame.shape[0], int(box_array[:, 1].max()))
                
                # Check for very small regions and expand more
                region_width = x_max - x_min
                region_height = y_max - y_min

                if region_width < 20 or region_height < 15:
                    pad = 5  # Reduced from 15 to 5
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(frame.shape[1], x_max + pad)
                    y_max = min(frame.shape[0], y_max + pad)
                
                if x_max > x_min and y_max > y_min:
                    text_region = frame[y_min:y_max, x_min:x_max]
                    
                    # Try multiple preprocessing methods
                    best_result = None
                    best_confidence = 0.0
                    
                    preprocessing_variants = [
                        ("enhanced", self.detector._preprocess_for_ocr(text_region)),
                        ("original", text_region),
                        ("brightened", cv2.convertScaleAbs(text_region, alpha=1.3, beta=20)),
                        ("contrasted", cv2.convertScaleAbs(text_region, alpha=1.5, beta=0))
                    ]
                    
                    for method_name, region_variant in preprocessing_variants:
                        try:
                            # Try with cls first
                            result = self.ocr.ocr(region_variant, cls=True)
                            
                            if result and result[0]:
                                for line in result[0]:
                                    if line and len(line) > 1:
                                        if isinstance(line[1], tuple):
                                            text = line[1][0]
                                            confidence = line[1][1]
                                        else:
                                            text = line[1]
                                            confidence = 0.0
                                        
                                        if confidence > best_confidence and text.strip():
                                            best_result = {
                                                'text': text.strip(),
                                                'confidence': confidence,
                                                'method': method_name
                                            }
                                            best_confidence = confidence
                            
                            # If low confidence, try without cls
                            if best_confidence < 0.5:
                                result_no_cls = self.ocr.ocr(region_variant, cls=False)
                                if result_no_cls and result_no_cls[0]:
                                    for line in result_no_cls[0]:
                                        if line and len(line) > 1:
                                            if isinstance(line[1], tuple):
                                                text = line[1][0]
                                                confidence = line[1][1]
                                            else:
                                                text = line[1]
                                                confidence = 0.0
                                            
                                            if confidence > best_confidence and text.strip():
                                                best_result = {
                                                    'text': text.strip(),
                                                    'confidence': confidence,
                                                    'method': f"{method_name}_no_cls"
                                                }
                                                best_confidence = confidence
                        
                        except Exception as e:
                            continue
                    
                    # Use result with LOWER confidence threshold
                    if best_result and best_confidence > 0.1:  # Lowered from 0.3 to 0.1
                        detected_texts.append({
                            'text': best_result['text'],
                            'confidence': best_result['confidence'],
                            'bbox': [x_min, y_min, x_max, y_max],
                            'method': best_result['method'],
                            'reading_order': reading_order  # Add reading order
                        })
                        
                        # Draw results with reading order number
                        cv2.polylines(annotated_frame, [box_array], True, (0, 255, 0), 2)
                        
                        # Draw reading order number in a circle
                        center_x = (x_min + x_max) // 2
                        center_y = y_min - 25
                        cv2.circle(annotated_frame, (center_x, center_y), 15, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, str(reading_order), 
                                  (center_x - 8, center_y + 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (0, 0, 0), 2)
                        
                        # Draw text
                        cv2.putText(annotated_frame, best_result['text'][:25], 
                                  (x_min, y_min - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 1)
                        cv2.putText(annotated_frame, f"{best_result['confidence']:.2f}", 
                                  (x_min, y_min + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.4, (255, 255, 0), 1)
                        
                        print(f"  [{reading_order}] '{best_result['text']}' (conf: {best_result['confidence']:.2f}, method: {best_result['method']})")
                    else:
                        # Draw box even if no text recognized
                        cv2.polylines(annotated_frame, [box_array], True, (0, 0, 255), 1)
                        if best_result:
                            print(f"  [{reading_order}] Low confidence: '{best_result['text']}' ({best_confidence:.2f})")
                        else:
                            print(f"  [{reading_order}] No text recognized")
            
            except Exception as e:
                print(f"Error processing box {reading_order}: {e}")
                continue
        
        # Process with LLM and TTS if enabled and text was detected
        if self.enable_llm_tts and detected_texts and self.text_processor and self.tts_handler:
            self._process_text_with_llm_tts(detected_texts, frame_count)
        
        return annotated_frame, detected_texts
    
    def _process_text_with_llm_tts(self, detected_texts, frame_count):
        """Process detected text with LLM and TTS"""
        try:
            # Extract text fragments in reading order
            text_fragments = []
            texts_sorted = sorted(detected_texts, key=lambda x: x.get('reading_order', 0))
            for text_info in texts_sorted:
                text_fragments.append(text_info['text'])
            
            if not text_fragments:
                return
            
            print(f"\n{'='*60}")
            print(f"🤖 Processing {len(text_fragments)} text fragment(s) with LLM...")
            print(f"{'='*60}")
            print("Text fragments:")
            for i, frag in enumerate(text_fragments, 1):
                print(f"  [{i}] {frag}")
            
            # Process with LLM
            combined_message = self.text_processor.combine_texts(text_fragments)
            
            if combined_message:
                print(f"\n✓ LLM Combined: {combined_message}")
                
                # Log the result
                self._log_llm_result(text_fragments, combined_message, frame_count)
                
                # Speak the message
                self.tts_handler.speak(combined_message)
                
                print(f"{'='*60}\n")
            else:
                print("⚠ No new message to process (duplicate or empty)")
                print(f"{'='*60}\n")
                
        except Exception as e:
            print(f"✗ LLM/TTS processing error: {e}")
    
    def _log_llm_result(self, fragments, combined, frame_count):
        """Log the LLM processed message"""
        try:
            log_file = "llm_processed_messages.txt"
            with open(log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n--- Frame {frame_count} at {timestamp} ---\n")
                f.write(f"Fragments: {fragments}\n")
                f.write(f"Combined: {combined}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"✗ Error logging LLM result: {e}")
    
    def save_results(self, frame_count, texts, timestamp):
        """Save results to file in reading order"""
        if texts:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Frame {frame_count} at {timestamp} ---\n")
                # Sort by reading order before saving
                texts_sorted = sorted(texts, key=lambda x: x.get('reading_order', 0))
                for text_info in texts_sorted:
                    order = text_info.get('reading_order', 0)
                    text = text_info['text']
                    confidence = text_info['confidence']
                    method = text_info.get('method', 'unknown')
                    f.write(f"[{order}] {text} (conf: {confidence:.2f}, method: {method})\n")
                f.write("\n")
    
    def process_image(self, image_path, display=True):
        """Process single image"""
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return []
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Could not load image: {image_path}")
            return []
        
        print(f"\nProcessing: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        annotated_image, texts = self.process_frame(image, 1)
        
        if texts:
            print(f"\n✓ Detected {len(texts)} text region(s) in reading order:")
            # Sort by reading order
            texts_sorted = sorted(texts, key=lambda x: x.get('reading_order', 0))
            for t in texts_sorted:
                order = t.get('reading_order', 0)
                print(f"  [{order}] {t['text']} (conf: {t['confidence']:.2f})")
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.save_results(1, texts, timestamp)
        else:
            print("✗ No text detected")
        
        if display:
            cv2.imshow(f'OCR Results - {os.path.basename(image_path)}', annotated_image)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return texts
    
    def process_images_batch(self, image_folder, display=True):
        """Batch process images"""
        if not os.path.exists(image_folder):
            print(f"✗ Folder not found: {image_folder}")
            return
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                       if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"✗ No images found in: {image_folder}")
            return
        
        print(f"Found {len(image_files)} image(s) to process\n")
        
        total_texts = []
        for i, img_path in enumerate(image_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing image {i}/{len(image_files)}")
            print(f"{'='*60}")
            texts = self.process_image(img_path, display)
            total_texts.extend(texts)
            
            if display and i < len(image_files):
                print("\nPress any key to continue to next image...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✓ Batch processing complete!")
        print(f"Total texts detected: {len(total_texts)}")
        print(f"{'='*60}")
        return total_texts
    
    def run(self, display=True, process_every_n_frames=1):
        """Run real-time OCR from camera"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"✗ Cannot open camera {self.camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"\n{'='*60}")
        print(f"✓ Camera {self.camera_id} opened successfully")
        print(f"Real-time OCR started - Maximum coverage mode with reading order")
        print(f"Processing every {process_every_n_frames} frame(s)")
        print(f"{'='*60}")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print(f"{'='*60}\n")
        
        frame_count = 0
        start_time = time.time()
        total_detections = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("✗ Cannot read frame")
                    break
                
                frame_count += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                if frame_count % process_every_n_frames == 0:
                    annotated_frame, texts = self.process_frame(frame, frame_count)
                    
                    if texts:
                        total_detections += len(texts)
                        print(f"\n[Frame {frame_count} @ {current_time}] ✓ Detected {len(texts)} text(s)")
                        self.save_results(frame_count, texts, current_time)
                    
                    if display:
                        # Add frame info overlay
                        info_text = f"Frame: {frame_count} | Detections: {len(texts)} | Total: {total_detections}"
                        cv2.putText(annotated_frame, info_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, info_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        cv2.imshow('Real-time OCR - Press Q to quit', annotated_frame)
                else:
                    if display:
                        cv2.imshow('Real-time OCR - Press Q to quit', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✓ Quitting...")
                    break
                elif key == ord('s'):
                    filename = f"frame_{frame_count}_{current_time.replace(':', '-')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Frame saved as {filename}")
        
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            end_time = time.time()
            total_time = end_time - start_time
            fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"\n{'='*60}")
            print("OCR Pipeline Statistics")
            print(f"{'='*60}")
            print(f"Total frames captured: {frame_count}")
            print(f"Total frames processed: {frame_count // process_every_n_frames}")
            print(f"Total text detections: {total_detections}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {fps:.2f}")
            print(f"Results saved to: {self.output_file}")
            print(f"{'='*60}")


# ============================================================================
# Main Execution
# ============================================================================

def get_user_choice():
    """Get processing mode"""
    print("\n" + "=" * 60)
    print("OCR Pipeline - Choose Processing Mode")
    print("=" * 60)
    print("1. Camera Mode - Real-time text detection with LLM + TTS")
    print("2. Single Image Mode - Process one image with LLM + TTS")
    print("3. Batch Image Mode - Process folder of images with LLM + TTS")
    print("4. Camera Mode - Real-time text detection (OCR only)")
    print("5. Single Image Mode - Process one image (OCR only)")
    print("6. Batch Image Mode - Process folder of images (OCR only)")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Enter choice (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return int(choice)
            else:
                print("Please enter 1, 2, 3, 4, 5, or 6")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)

def get_image_path():
    """Get image path from user"""
    while True:
        try:
            path = input("Enter image path: ").strip().strip('"\'')
            if os.path.exists(path):
                return path
            else:
                print(f"File not found: {path}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)

def get_folder_path():
    """Get folder path from user"""
    while True:
        try:
            path = input("Enter folder path: ").strip().strip('"\'')
            if os.path.exists(path) and os.path.isdir(path):
                return path
            else:
                print(f"Folder not found: {path}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)

if __name__ == "__main__":
    # Configuration - MAXIMUM COVERAGE MODE WITH READING ORDER
    OUTPUT_FILE = 'output.txt'
    CAMERA_ID = 0
    USE_GPU = torch.cuda.is_available()
    DISPLAY_VIDEO = True
    PROCESS_EVERY_N_FRAMES = 1  # Process every frame for maximum coverage
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 60)
    print("Real-Time OCR Pipeline - MAXIMUM TEXT COVERAGE")
    print("=" * 60)
    print("CRAFT: Text Detection (Low thresholds for max coverage)")
    print("PaddleOCR: Text Recognition (Multi-preprocessing)")
    print("Reading Order: Top-to-Bottom, Left-to-Right")
    print("=" * 60)
    print(f"GPU Available: {USE_GPU}")
    print(f"Output File: {OUTPUT_FILE}")
    print("=" * 60)
    print("\nOptimizations enabled:")
    print("  ✓ Lower CRAFT thresholds (0.3/0.2/0.2)")
    print("  ✓ Lower confidence threshold (0.1)")
    print("  ✓ Larger box expansion (15%/25%)")
    print("  ✓ Multiple preprocessing methods")
    print("  ✓ Intelligent overlap handling")
    print("  ✓ Angle classification enabled")
    print("  ✓ Reading order sorting (Top→Bottom, Left→Right)")
    print("=" * 60)
    
    # Get user's processing mode choice
    mode = get_user_choice()
    
    # Determine if LLM/TTS should be enabled
    enable_llm_tts = mode in [1, 2, 3]  # Modes 1-3 have LLM + TTS
    
    # Initialize pipeline
    try:
        pipeline = RealTimeOCRPipeline(
            output_file=OUTPUT_FILE, 
            use_gpu=USE_GPU, 
            camera_id=CAMERA_ID,
            enable_llm_tts=enable_llm_tts
        )
        
        if mode in [1, 4]:  # Camera modes
            mode_name = "Camera Mode with LLM + TTS" if mode == 1 else "Camera Mode (OCR only)"
            print(f"\nStarting {mode_name}...")
            pipeline.run(display=DISPLAY_VIDEO, process_every_n_frames=PROCESS_EVERY_N_FRAMES)
            
        elif mode in [2, 5]:  # Single image modes
            mode_name = "Single Image Mode with LLM + TTS" if mode == 2 else "Single Image Mode (OCR only)"
            print(f"\nStarting {mode_name}...")
            image_path = get_image_path()
            if image_path:
                pipeline.process_image(image_path, display=DISPLAY_VIDEO)
            else:
                print("No image selected. Exiting...")
                
        elif mode in [3, 6]:  # Batch image modes
            mode_name = "Batch Image Mode with LLM + TTS" if mode == 3 else "Batch Image Mode (OCR only)"
            print(f"\nStarting {mode_name}...")
            folder_path = get_folder_path()
            if folder_path:
                pipeline.process_images_batch(folder_path, display=DISPLAY_VIDEO)
            else:
                print("No folder selected. Exiting...")
                
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("  1. Camera connection (for camera mode)")
        print("  2. CRAFT model file (craft_mlt_25k.pth)")
        print("  3. PaddleOCR installation")
        print("  4. Image file paths and formats")