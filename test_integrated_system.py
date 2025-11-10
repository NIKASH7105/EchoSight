#!/usr/bin/env python3
"""
Test Integrated System - Quick test to verify connections
========================================================
"""

import sys
import os

# Add project paths to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'OCR'))
sys.path.append(os.path.join(current_dir, 'object_detection'))
sys.path.append(os.path.join(current_dir, 'traffic_light'))

print("🧪 Testing Integrated Vision System Connections...")
print("=" * 60)

# Test detect.py import
try:
    from detect import generate_scene_description, model, classes
    print("✓ detect.py imported successfully")
    print(f"  - Model: {type(model)}")
    print(f"  - Classes: {len(classes)} available")
except ImportError as e:
    print(f"✗ detect.py import failed: {e}")

# Test OCR system import
try:
    from ocr1_llm import TextProcessor, TTSHandler
    print("✓ OCR system imported successfully")
except ImportError as e:
    print(f"✗ OCR system import failed: {e}")

# Test unified vision system import
try:
    from unified_vision_system import UnifiedVisionSystem
    print("✓ Unified vision system imported successfully")
except ImportError as e:
    print(f"✗ Unified vision system import failed: {e}")

# Test traffic light system import
try:
    from realtime_traffic_light_system import RealTimeTrafficLightDetector
    print("✓ Traffic light system imported successfully")
except ImportError as e:
    print(f"✗ Traffic light system import failed: {e}")

print("\n" + "=" * 60)
print("✅ Connection test completed")
print("=" * 60)
print("\nTo run the integrated system:")
print("python integrated_vision_system.py")
