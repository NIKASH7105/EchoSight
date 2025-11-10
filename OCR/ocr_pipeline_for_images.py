"""
Complete OCR Pipeline with CRAFT + PaddleOCR
===========================================
This script provides multiple modes for text detection and recognition:
1. Real-time camera processing
2. Single image processing
3. Batch image processing

Features:
- Real-time camera feed capture
- Single image processing
- Batch processing of multiple images
- CRAFT text detection algorithm (local pretrained model)
- PaddleOCR text recognition (English language)
- Console output and file storage
- GPU acceleration support
- Multiple detections per frame/image
- Visual feedback with bounding boxes
- Support for common image formats (JPG, PNG, BMP, TIFF)

Requirements:
    pip install opencv-python torch torchvision paddlepaddle paddleocr numpy scipy scikit-image
"""

import cv2
import torch
import numpy as np
import os
from datetime import datetime
import warnings
import time
from collections import OrderedDict

warnings.filterwarnings('ignore')

# Import CRAFT model definition
try:
    import sys
    import os
    # Add CRAFT_pytorch to Python path
    craft_path = os.path.join(os.path.dirname(__file__), 'CRAFT_pytorch')
    if craft_path not in sys.path:
        sys.path.insert(0, craft_path)
    
    from craft import CRAFT  # type: ignore
    CRAFT_MODULE_AVAILABLE = True
    print("✓ CRAFT module imported successfully")
except ImportError as e:
    CRAFT_MODULE_AVAILABLE = False
    print(f"✗ CRAFT module import failed: {e}")
    print("Please ensure CRAFT_pytorch directory is in the same directory")

# Import PaddleOCR - required dependency
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("✓ PaddleOCR imported successfully")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    print(f"✗ PaddleOCR import failed: {e}")
    print("Please install PaddleOCR: pip install paddlepaddle paddleocr")
    exit(1)
except Exception as e:
    PADDLEOCR_AVAILABLE = False
    print(f"✗ PaddleOCR initialization error: {e}")
    exit(1)


# ============================================================================
# CRAFT Text Detection Model
# ============================================================================

class CRAFTDetector:
    """CRAFT text detection model implementation with proper CRAFT inference"""
    
    def __init__(self, use_cuda=True):
        """
        Initialize CRAFT detector with local pretrained model
        
        Args:
            use_cuda: Whether to use GPU acceleration
        """
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() and use_cuda:
            print("Running CRAFT on: GPU")
        else:
            print("Running CRAFT on: CPU")
        
        # Load CRAFT model locally
        self.net = self._load_craft_model()
        if self.net is not None:
            self.net.to(self.device)
            self.net.eval()
            print("✓ Loaded local CRAFT model from craft_mlt_25k.pth")
        else:
            print("⚠ CRAFT model not available, using fallback detection")
    
    def _load_craft_model(self):
        """Load pre-trained CRAFT model from local files"""
        if not CRAFT_MODULE_AVAILABLE:
            print("CRAFT module not available, using fallback detection")
            return None
        
        try:
            # Initialize CRAFT model
            model = CRAFT()
            
            # Load pretrained weights from local file
            # Try multiple possible paths for the model file
            possible_paths = [
                'craft_mlt_25k.pth',  # Relative to current directory
                os.path.join(os.path.dirname(__file__), 'craft_mlt_25k.pth'),  # Relative to script location
                r'D:\trafficdl\OCR\craft_mlt_25k.pth'  # Absolute path (user's current setup)
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print(f"Model file not found in any of these locations:")
                for path in possible_paths:
                    print(f"  - {path}")
                print("Using fallback detection")
                return None
            
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Handle keys that start with "module."
            state_dict = self._copy_state_dict(state_dict)
            
            # Load the state dict into the model
            model.load_state_dict(state_dict)
            
            print(f"✓ CRAFT model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            print(f"Failed to load local CRAFT model: {e}")
            print("Using fallback text detection method")
            return None
    
    def _copy_state_dict(self, state_dict):
        """Handle keys that start with 'module.' in state dict"""
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # Remove 'module.' prefix
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    def detect_text_regions(self, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
        """
        Detect text regions in image using CRAFT model
        
        Args:
            image: Input image (BGR format from OpenCV)
            text_threshold: Threshold for text confidence
            link_threshold: Threshold for link confidence
            low_text: Lower bound for text confidence
            
        Returns:
            boxes: List of detected text bounding boxes
        """
        if self.net is None:
            # Use fallback detection if CRAFT model unavailable
            return self._fallback_text_detection(image)
        
        try:
            # Prepare image for CRAFT inference
            img_resized, target_ratio, size_heatmap = self._resize_aspect_ratio(
                image, square_size=1280, interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize image for CRAFT
            x = self._normalize_mean_variance(img_resized)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            # CRAFT forward pass
            with torch.no_grad():
                y, feature = self.net(x)
            
            # Extract score maps from CRAFT output
            score_text = y[0, :, :, 0].cpu().numpy()
            score_link = y[0, :, :, 1].cpu().numpy()
            
            # Post-process to get bounding boxes
            boxes = self._get_boxes(score_text, score_link, text_threshold, link_threshold, low_text)
            
            # Adjust coordinates to original image size
            boxes = self._adjust_result_coordinates(boxes, target_ratio, size_heatmap)
            
            return boxes
            
        except Exception as e:
            print(f"Error in CRAFT detection: {e}")
            return self._fallback_text_detection(image)
    
    def _resize_aspect_ratio(self, img, square_size=1280, interpolation=cv2.INTER_LINEAR):
        """Resize image while maintaining aspect ratio for CRAFT input"""
        height, width, channel = img.shape
        target_size = square_size
        ratio = min(target_size / height, target_size / width)
        target_h, target_w = int(height * ratio), int(width * ratio)
        
        # Resize image
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
        
        # Padding to make it square (CRAFT requires square input divisible by 32)
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
        resized[0:target_h, 0:target_w, :] = proc
        
        size_heatmap = (int(target_w32 / 2), int(target_h32 / 2))
        
        return resized, ratio, size_heatmap
    
    def _normalize_mean_variance(self, img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        """Normalize image using ImageNet statistics for CRAFT"""
        img = img.astype(np.float32) / 255.0
        img -= np.array(mean)
        img /= np.array(variance)
        return img
    
    def _get_boxes(self, score_text, score_link, text_threshold, link_threshold, low_text):
        """Extract bounding boxes from CRAFT score maps"""
        boxes = []
        
        # Apply thresholds to score maps
        text_score = score_text > text_threshold
        link_score = score_link > link_threshold
        
        # Combine text and link scores
        text_score_comb = np.clip(text_score + link_score, 0, 1)
        
        # Find connected components in the combined score map
        text_score_comb = (text_score_comb * 255).astype(np.uint8)
        contours, _ = cv2.findContours(text_score_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Filter small boxes
            if cv2.contourArea(contour) > 10:
                boxes.append(box.reshape(-1))
        
        return boxes
    
    def _adjust_result_coordinates(self, boxes, ratio, size_heatmap):
        """Adjust box coordinates to original image size"""
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        boxes = boxes * 2  # Score map is half size
        boxes = boxes / ratio  # Adjust to original ratio
        
        return boxes.astype(np.int32).tolist()
    
    def _fallback_text_detection(self, image):
        """Enhanced fallback text detection using multiple OpenCV methods"""
        try:
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
                
                # More lenient filtering to catch more text
                if w > 15 and h > 10 and w < image.shape[1] * 0.9 and h < image.shape[0] * 0.9:
                    aspect_ratio = w / float(h)
                    area = w * h
                    
                    # Accept wider range of aspect ratios for text
                    if 0.2 < aspect_ratio < 10 and area > 100:
                        # Convert to 4-point format
                        box = [x, y, x+w, y, x+w, y+h, x, y+h]
                        boxes.append(box)
            
            return boxes
        except Exception as e:
            print(f"Error in fallback text detection: {e}")
            return []
    
    def _expand_box(self, box, image_shape, width_margin=0.1, height_margin=0.2):
        """
        Expand bounding box by specified margins to avoid tight cropping
        
        Args:
            box: 4-point bounding box [x1, y1, x2, y2, x3, y3, x4, y4]
            image_shape: (height, width, channels) of the image
            width_margin: Fraction to expand width (default 10%)
            height_margin: Fraction to expand height (default 20%)
            
        Returns:
            expanded_box: Expanded bounding box
        """
        try:
            # Convert to numpy array and reshape
            box = np.array(box).reshape(4, 2)
            
            # Get bounding rectangle
            x_min, y_min = box.min(axis=0)
            x_max, y_max = box.max(axis=0)
            
            # Calculate current dimensions
            width = x_max - x_min
            height = y_max - y_min
            
            # Calculate expansion amounts
            width_expand = int(width * width_margin)
            height_expand = int(height * height_margin)
            
            # Expand coordinates
            x_min = max(0, x_min - width_expand)
            y_min = max(0, y_min - height_expand)
            x_max = min(image_shape[1], x_max + width_expand)
            y_max = min(image_shape[0], y_max + height_expand)
            
            # Return as 4-point box
            expanded_box = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            return expanded_box
            
        except Exception as e:
            print(f"Error expanding box: {e}")
            return box
    
    def _upsample_small_region(self, image, min_size=32):
        """
        Upsample small text regions using cubic interpolation
        
        Args:
            image: Input image region
            min_size: Minimum size for width and height
            
        Returns:
            upsampled_image: Upsampled image if needed
        """
        try:
            h, w = image.shape[:2]
            
            # Check if upsampling is needed
            if h < min_size or w < min_size:
                # Calculate scale factor to make the smaller dimension at least min_size
                scale_h = min_size / h if h < min_size else 1.0
                scale_w = min_size / w if w < min_size else 1.0
                scale = max(scale_h, scale_w)
                
                # Calculate new dimensions
                new_h = int(h * scale)
                new_w = int(w * scale)
                
                # Upsample using cubic interpolation
                upsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                return upsampled
            
            return image
            
        except Exception as e:
            print(f"Error upsampling region: {e}")
            return image
    
    def _enhance_image_quality(self, image):
        """
        Enhance image quality with multiple approaches for better OCR
        
        Args:
            image: Input image region
            
        Returns:
            enhanced_image: Enhanced image
        """
        try:
            # Method 1: Gentle enhancement (preferred for most text)
            enhanced_gentle = self._gentle_enhancement(image)
            
            # Method 2: Strong enhancement (for very poor quality images)
            enhanced_strong = self._strong_enhancement(image)
            
            # Return the gentle enhancement as default
            return enhanced_gentle
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image
    
    def _gentle_enhancement(self, image):
        """Gentle enhancement suitable for most text"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply gentle CLAHE with lower clip limit
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply gentle sharpening
            kernel = np.array([[0, -0.5, 0],
                              [-0.5, 3, -0.5],
                              [0, -0.5, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Ensure values are in valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Convert back to BGR for PaddleOCR
            enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            print(f"Error in gentle enhancement: {e}")
            return image
    
    def _strong_enhancement(self, image):
        """Strong enhancement for very poor quality images"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply stronger CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply stronger sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Ensure values are in valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Convert back to BGR for PaddleOCR
            enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            print(f"Error in strong enhancement: {e}")
            return image
    
    def _binary_enhancement(self, image):
        """Binary enhancement for clear text with high contrast"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive thresholding for binary image
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Convert back to BGR
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            return binary_bgr
            
        except Exception as e:
            print(f"Error in binary enhancement: {e}")
            return image
    
    def _denoise_enhancement(self, image):
        """Denoising enhancement for noisy images"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply gentle CLAHE
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Convert back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            print(f"Error in denoise enhancement: {e}")
            return image
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes
        
        Args:
            box1, box2: 4-point bounding boxes [x1, y1, x2, y2, x3, y3, x4, y4]
            
        Returns:
            iou: IoU value between 0 and 1
        """
        try:
            # Convert to rectangles
            def box_to_rect(box):
                box = np.array(box).reshape(4, 2)
                x_min, y_min = box.min(axis=0)
                x_max, y_max = box.max(axis=0)
                return x_min, y_min, x_max, y_max
            
            x1_min, y1_min, x1_max, y1_max = box_to_rect(box1)
            x2_min, y2_min, x2_max, y2_max = box_to_rect(box2)
            
            # Calculate intersection
            x_left = max(x1_min, x2_min)
            y_top = max(y1_min, y2_min)
            x_right = min(x1_max, x2_max)
            y_bottom = min(y1_max, y2_max)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating IoU: {e}")
            return 0.0
    
    def _merge_overlapping_boxes(self, boxes, iou_threshold=0.3):
        """
        Merge overlapping bounding boxes using IoU-based logic
        
        Args:
            boxes: List of 4-point bounding boxes
            iou_threshold: IoU threshold for merging (default 0.3)
            
        Returns:
            merged_boxes: List of merged bounding boxes
        """
        try:
            if len(boxes) <= 1:
                return boxes
            
            merged_boxes = []
            used = [False] * len(boxes)
            
            for i in range(len(boxes)):
                if used[i]:
                    continue
                
                current_box = boxes[i]
                merged_box = current_box.copy()
                used[i] = True
                
                # Find overlapping boxes to merge
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    
                    iou = self._calculate_iou(current_box, boxes[j])
                    if iou > iou_threshold:
                        # Merge boxes by taking the union
                        box1 = np.array(current_box).reshape(4, 2)
                        box2 = np.array(boxes[j]).reshape(4, 2)
                        
                        # Get bounding rectangle of both boxes
                        all_points = np.vstack([box1, box2])
                        x_min, y_min = all_points.min(axis=0)
                        x_max, y_max = all_points.max(axis=0)
                        
                        # Create merged box
                        merged_box = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                        used[j] = True
                
                merged_boxes.append(merged_box)
            
            return merged_boxes
            
        except Exception as e:
            print(f"Error merging boxes: {e}")
            return boxes


# ============================================================================
# OCR Pipeline
# ============================================================================

class RealTimeOCRPipeline:
    """Complete real-time OCR pipeline with CRAFT detection and PaddleOCR recognition"""
    
    def __init__(self, output_file='output.txt', use_gpu=True, camera_id=0):
        """
        Initialize OCR pipeline
        
        Args:
            output_file: Path to output text file
            use_gpu: Whether to use GPU acceleration
            camera_id: Camera device ID
        """
        self.output_file = output_file
        self.use_gpu = use_gpu
        self.camera_id = camera_id
        
        # Initialize text detector
        print("\n" + "="*60)
        print("Initializing OCR Pipeline Components")
        print("="*60)
        
        print("1. Loading CRAFT text detector...")
        self.detector = CRAFTDetector(use_cuda=use_gpu)
        
        # Initialize OCR recognizer
        print("2. Loading PaddleOCR text recognizer...")
        try:
            # Initialize PaddleOCR for text recognition only (no detection)
            # Use CPU mode to avoid device-related issues
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en' # Force CPU mode to avoid device issues
            )
            self.ocr_type = 'paddleocr'
            print("✓ PaddleOCR initialized successfully for text recognition")
        except Exception as e:
            print(f"✗ PaddleOCR initialization failed: {e}")
            print("Trying alternative PaddleOCR configuration...")
            try:
                # Try with minimal configuration
                self.ocr = PaddleOCR(lang='en')
                self.ocr_type = 'paddleocr'
                print("✓ PaddleOCR initialized with minimal configuration")
            except Exception as e2:
                print(f"✗ PaddleOCR initialization failed again: {e2}")
                print("Please ensure PaddleOCR is properly installed:")
                print("pip install paddlepaddle paddleocr")
                exit(1)
        
        # Initialize output file
        self._initialize_output_file()
        
        print("✓ Pipeline initialization complete!")
        print("="*60)
    
    def _initialize_output_file(self):
        """Initialize output file with header"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("Real-time OCR Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def process_frame(self, frame, frame_count):
        """
        Enhanced process_frame function for perfect text recognition in challenging conditions
        
        Args:
            frame: Input frame from camera
            frame_count: Frame number for logging
            
        Returns:
            annotated_frame: Frame with bounding boxes and text labels
            detected_texts: List of recognized text strings with confidence scores
        """
        detected_texts = []
        annotated_frame = frame.copy()
        
        # Detect text regions using CRAFT model
        boxes = self.detector.detect_text_regions(frame)
        
        if not boxes:
            return annotated_frame, detected_texts
        
        # Step 1: Merge overlapping boxes to avoid duplicate detections and handle split words
        merged_boxes = self.detector._merge_overlapping_boxes(boxes, iou_threshold=0.2)  # Lower threshold for better merging
        
        # Step 2: Process each detected text region with enhanced pipeline
        for idx, box in enumerate(merged_boxes):
            try:
                # Step 3: Expand bounding box to avoid tight cropping
                expanded_box = self.detector._expand_box(box, frame.shape, width_margin=0.1, height_margin=0.2)
                
                # Convert expanded box to numpy array and get bounding rectangle
                box_array = np.array(expanded_box).reshape(4, 2).astype(np.int32)
                x_min = max(0, int(box_array[:, 0].min()))
                y_min = max(0, int(box_array[:, 1].min()))
                x_max = min(frame.shape[1], int(box_array[:, 0].max()))
                y_max = min(frame.shape[0], int(box_array[:, 1].max()))
                
                # Step 4: Crop text region (no size filtering - accept all boxes)
                if x_max > x_min and y_max > y_min:
                    text_region = frame[y_min:y_max, x_min:x_max]
                    
                    # Step 5: Upsample small regions using cubic interpolation
                    upsampled_region = self.detector._upsample_small_region(text_region, min_size=32)
                    
                    # Step 6: Enhance image quality (CLAHE, sharpening, grayscale)
                    enhanced_region = self.detector._enhance_image_quality(upsampled_region)
                    
                    # Step 7: Recognize text using improved PaddleOCR approach
                    text_recognized = False
                    best_result = None
                    best_confidence = 0.0
                    
                    # Create additional enhanced versions
                    gentle_enhanced = self.detector._gentle_enhancement(upsampled_region)
                    strong_enhanced = self.detector._strong_enhancement(upsampled_region)
                    
                    # Create specialized versions for different text types
                    binary_enhanced = self.detector._binary_enhancement(upsampled_region)
                    denoised_enhanced = self.detector._denoise_enhancement(upsampled_region)
                    
                    # Try multiple preprocessing approaches and pick the best result
                    preprocessing_methods = [
                        ("gentle", gentle_enhanced),
                        ("upsampled", upsampled_region),
                        ("original", text_region),
                        ("binary", binary_enhanced),
                        ("denoised", denoised_enhanced),
                        ("strong", strong_enhanced)
                    ]
                    
                    for method_name, region in preprocessing_methods:
                        try:
                            # Try with cls=True first
                            result = self.ocr.ocr(region, cls=True)
                            
                            if result and result[0]:
                                for line in result[0]:
                                    if line and len(line) > 1:
                                        if isinstance(line[1], tuple):
                                            text = line[1][0]
                                            confidence = line[1][1]
                                        else:
                                            text = line[1]
                                            confidence = 0.0
                                        
                                        # Keep the result with highest confidence
                                        if confidence > best_confidence and text.strip():
                                            best_result = {
                                                'text': text.strip(),
                                                'confidence': confidence,
                                                'method': method_name
                                            }
                                            best_confidence = confidence
                            
                            # If cls=True didn't work well, try cls=False
                            if best_confidence < 0.5:
                                result_no_cls = self.ocr.ocr(region, cls=False)
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
                            print(f"OCR failed for {method_name}: {e}")
                            continue
                    
                    # Use the best result found with adaptive threshold
                    # Lower threshold for better coverage, but still reasonable
                    confidence_threshold = 0.2 if best_confidence > 0.1 else 0.1
                    
                    if best_result and best_confidence > confidence_threshold:
                        detected_texts.append({
                            'text': best_result['text'],
                            'confidence': best_result['confidence'],
                            'bbox': [x_min, y_min, x_max, y_max]
                        })
                        
                        # Draw bounding box around detected text
                        cv2.polylines(annotated_frame, [box_array], True, (0, 255, 0), 2)
                        
                        # Draw recognized text label
                        cv2.putText(annotated_frame, f"{best_result['text'][:20]}", 
                                  (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 1)
                        
                        # Draw confidence score
                        cv2.putText(annotated_frame, f"({best_result['confidence']:.2f})", 
                                  (x_min, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.3, (255, 255, 0), 1)
                        
                        # Draw processing method
                        cv2.putText(annotated_frame, f"{best_result['method']}", 
                                  (x_min, y_min + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.3, (0, 255, 255), 1)
                        
                        text_recognized = True
                        print(f"Region {idx}: '{best_result['text']}' (conf: {best_result['confidence']:.3f}, method: {best_result['method']})")
                    else:
                        # Last resort: try with very low confidence threshold
                        if best_result and best_confidence > 0.05:
                            print(f"Region {idx}: Using low confidence result '{best_result['text']}' (conf: {best_confidence:.3f})")
                            detected_texts.append({
                                'text': best_result['text'],
                                'confidence': best_result['confidence'],
                                'bbox': [x_min, y_min, x_max, y_max]
                            })
                            
                            # Draw bounding box with low confidence indicator
                            cv2.polylines(annotated_frame, [box_array], True, (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"{best_result['text'][:20]}", 
                                      (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0), 1)
                            cv2.putText(annotated_frame, f"({best_result['confidence']:.2f})", 
                                      (x_min, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.3, (255, 255, 0), 1)
                            cv2.putText(annotated_frame, f"{best_result['method']}", 
                                      (x_min, y_min + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.3, (0, 255, 255), 1)
                        else:
                            # Draw bounding box even if no text recognized
                            cv2.polylines(annotated_frame, [box_array], True, (0, 255, 0), 2)
                            cv2.putText(annotated_frame, "No Text", (x_min, y_min - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                            print(f"Region {idx}: No text recognized (best confidence: {best_confidence:.3f})")
            
            except Exception as e:
                print(f"Error processing box {idx}: {e}")
                continue
        
        return annotated_frame, detected_texts
    
    def _process_ocr_result(self, result, detected_texts, box_array, x_min, y_min, x_max, y_max, annotated_frame, idx):
        """
        Process PaddleOCR result and update detected_texts and annotated_frame
        
        Returns:
            bool: True if text was recognized, False otherwise
        """
        try:
            text_recognized = False
            
            for line in result[0]:
                if line and len(line) > 1:
                    # Handle different PaddleOCR result formats
                    if isinstance(line[1], tuple):
                        text = line[1][0]
                        confidence = line[1][1]
                    else:
                        text = line[1]
                        confidence = 0.0
                    
                    print(f"Recognized text: '{text}' with confidence: {confidence}")
                    
                    # Accept all recognized text (no confidence filtering for maximum coverage)
                    if text.strip():
                        detected_texts.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': [x_min, y_min, x_max, y_max]
                        })
                        
                        # Draw bounding box around detected text
                        cv2.polylines(annotated_frame, [box_array], True, (0, 255, 0), 2)
                        
                        # Draw recognized text label
                        cv2.putText(annotated_frame, f"{text[:20]}", 
                                  (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 1)
                        
                        # Draw confidence score
                        cv2.putText(annotated_frame, f"({confidence:.2f})", 
                                  (x_min, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.3, (255, 255, 0), 1)
                        
                        # Draw processing info
                        cv2.putText(annotated_frame, f"Enhanced", 
                                  (x_min, y_min + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.3, (0, 255, 255), 1)
                        
                        text_recognized = True
            
            return text_recognized
            
        except Exception as e:
            print(f"Error processing OCR result: {e}")
            return False
    
    def save_results(self, frame_count, texts, timestamp):
        """Save recognized text to file with enhanced information"""
        if texts:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Frame {frame_count} at {timestamp} ---\n")
                for i, text_info in enumerate(texts, 1):
                    if isinstance(text_info, dict):
                        # New format with confidence and bbox
                        text = text_info['text']
                        confidence = text_info['confidence']
                        bbox = text_info['bbox']
                        f.write(f"{i}. Text: {text}\n")
                        f.write(f"   Confidence: {confidence:.3f}\n")
                        f.write(f"   BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n")
                    else:
                        # Legacy format (string)
                        f.write(f"{i}. {text_info}\n")
                f.write("\n")
    
    def process_image(self, image_path, display=True):
        """
        Process a single image for text detection and recognition
        
        Args:
            image_path: Path to the input image
            display: Whether to display the processed image
            
        Returns:
            detected_texts: List of recognized text strings
        """
        if not os.path.exists(image_path):
            print(f"✗ Error: Image file not found: {image_path}")
            return []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Error: Could not load image: {image_path}")
            print("Supported formats: JPG, JPEG, PNG, BMP, TIFF, TIF")
            return []
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Process the image
        annotated_image, texts = self.process_frame(image, 1)
        
        # Print results to console
        if texts:
            print(f"\n[Image: {os.path.basename(image_path)}] Detected text:")
            for i, text_info in enumerate(texts, 1):
                if isinstance(text_info, dict):
                    text = text_info['text']
                    confidence = text_info['confidence']
                    print(f"  {i}. {text} (conf: {confidence:.3f})")
                else:
                    print(f"  {i}. {text_info}")
            
            # Save to file
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.save_results(1, texts, timestamp)
        else:
            print(f"[Image: {os.path.basename(image_path)}] No text detected")
        
        # Display processed image
        if display:
            cv2.imshow(f'OCR Results - {os.path.basename(image_path)}', annotated_image)
            print("\nPress any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return texts
    
    def process_images_batch(self, image_folder, display=True):
        """
        Process multiple images from a folder
        
        Args:
            image_folder: Path to folder containing images
            display: Whether to display each processed image
        """
        if not os.path.exists(image_folder):
            print(f"✗ Error: Folder not found: {image_folder}")
            return
        
        # Supported image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        # Get all image files
        image_files = []
        for file in os.listdir(image_folder):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(image_folder, file))
        
        if not image_files:
            print(f"✗ No image files found in: {image_folder}")
            return
        
        print(f"Found {len(image_files)} image(s) to process")
        
        total_texts = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n--- Processing image {i}/{len(image_files)} ---")
            texts = self.process_image(image_path, display)
            total_texts.extend(texts)
            
            if display and i < len(image_files):
                print("Press any key to continue to next image...")
                cv2.waitKey(0)
        
        print(f"\n✓ Batch processing complete! Total texts detected: {len(total_texts)}")
        return total_texts

    def run(self, display=True, process_every_n_frames=5):
        """
        Run the real-time OCR pipeline
        
        Args:
            display: Whether to display the video feed
            process_every_n_frames: Process every nth frame to reduce load
        """
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera {self.camera_id}")
            return
        
        # Set camera properties for better performance
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
                
                # Process every nth frame to reduce computational load
                if frame_count % process_every_n_frames == 0:
                    # Process frame for text detection and recognition
                    annotated_frame, texts = self.process_frame(frame, frame_count)
                    
                    # Print results to console
                    if texts:
                        print(f"\n[Frame {frame_count} at {current_time}] Detected text:")
                        for i, text_info in enumerate(texts, 1):
                            if isinstance(text_info, dict):
                                text = text_info['text']
                                confidence = text_info['confidence']
                                print(f"  {i}. {text} (conf: {confidence:.3f})")
                            else:
                                print(f"  {i}. {text_info}")
                        
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


# ============================================================================
# Main Execution
# ============================================================================

def get_user_choice():
    """Get user's choice for processing mode"""
    print("\n" + "=" * 60)
    print("OCR Pipeline - Choose Processing Mode")
    print("=" * 60)
    print("1. Camera Mode - Real-time text detection from webcam")
    print("2. Single Image Mode - Process one image file")
    print("3. Batch Image Mode - Process all images in a folder")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            else:
                print("Please enter 1, 2, or 3")
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
    # Configuration
    OUTPUT_FILE = 'output.txt'
    CAMERA_ID = 0  # Use 0 for default webcam
    USE_GPU = torch.cuda.is_available()  # Auto-detect GPU
    DISPLAY_VIDEO = True  # Set to False to run headless
    PROCESS_EVERY_N_FRAMES = 1  # Process every frame for maximum text coverage
    
    # Change to script directory to ensure relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 60)
    print("Real-time OCR Pipeline with CRAFT + PaddleOCR")
    print("=" * 60)
    print("CRAFT: Text Detection (local pretrained model)")
    print("PaddleOCR: Text Recognition (English)")
    print("=" * 60)
    print(f"GPU Available: {USE_GPU}")
    print(f"Output File: {OUTPUT_FILE}")
    print("=" * 60)
    
    # Get user's processing mode choice
    mode = get_user_choice()
    
    # Initialize pipeline
    try:
        pipeline = RealTimeOCRPipeline(
            output_file=OUTPUT_FILE, 
            use_gpu=USE_GPU, 
            camera_id=CAMERA_ID
        )
        
        if mode == 1:
            # Camera mode
            print(f"\nStarting Camera Mode...")
            print(f"Camera ID: {CAMERA_ID}")
            print(f"Process every {PROCESS_EVERY_N_FRAMES} frame(s) - Enhanced mode for maximum coverage")
            print("Press 'q' to quit, 's' to save current frame")
            pipeline.run(display=DISPLAY_VIDEO, process_every_n_frames=PROCESS_EVERY_N_FRAMES)
            
        elif mode == 2:
            # Single image mode
            print(f"\nStarting Single Image Mode...")
            image_path = get_image_path()
            if image_path:
                pipeline.process_image(image_path, display=DISPLAY_VIDEO)
            else:
                print("No image selected. Exiting...")
                
        elif mode == 3:
            # Batch image mode
            print(f"\nStarting Batch Image Mode...")
            folder_path = get_folder_path()
            if folder_path:
                pipeline.process_images_batch(folder_path, display=DISPLAY_VIDEO)
            else:
                print("No folder selected. Exiting...")
                
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        print("Please check your camera connection and dependencies")
