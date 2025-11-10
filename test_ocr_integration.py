#!/usr/bin/env python3
"""
Test OCR Integration - Quick test to verify OCR systems work
===========================================================
"""

import sys
import os

# Add project paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'OCR'))

# Set API key
os.environ['TOGETHER_API_KEY'] = 'a60c2c24e4f37100bf8dea9930a9a8a0d354b122c597847eca8dad4ee1551efd'

print("🧪 Testing OCR Integration...")
print("=" * 60)

# Test ocr_order.py import
try:
    from ocr_order import RealTimeOCRPipeline
    print("✓ ocr_order.py imported successfully")
except ImportError as e:
    print(f"✗ ocr_order.py import failed: {e}")

# Test ocr1_llm.py import
try:
    from ocr1_llm import TextProcessor, TTSHandler
    print("✓ ocr1_llm.py imported successfully")
except ImportError as e:
    print(f"✗ ocr1_llm.py import failed: {e}")

# Test TextProcessor initialization
try:
    text_processor = TextProcessor(os.environ['TOGETHER_API_KEY'])
    print("✓ TextProcessor initialized successfully")
except Exception as e:
    print(f"✗ TextProcessor initialization failed: {e}")

# Test TTSHandler initialization
try:
    tts_handler = TTSHandler()
    print("✓ TTSHandler initialized successfully")
except Exception as e:
    print(f"✗ TTSHandler initialization failed: {e}")

print("\n" + "=" * 60)
print("✅ OCR Integration test completed")
print("=" * 60)
print("\nTo run the unified system:")
print("python unified_detection_system.py")
