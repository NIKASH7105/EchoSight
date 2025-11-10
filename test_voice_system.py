#!/usr/bin/env python3
"""
Test Voice System - Quick test to verify speech works
====================================================
"""

import cv2
import numpy as np
import time
import os
from gtts import gTTS
import pygame
import threading
from ultralytics import YOLO

# Initialize pygame
pygame.mixer.init()

# Load YOLOv8 model
try:
    model = YOLO("yolov8n.pt")
    print("✓ YOLOv8 model loaded")
except Exception as e:
    print(f"✗ YOLOv8 loading failed: {e}")
    exit(1)

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

def test_voice():
    """Test voice system"""
    print("🧪 Testing voice system...")
    
    file_path = 'voice_test.mp3'
    
    try:
        sound = gTTS(text="Hello, this is a test of the voice system", lang='en', slow=False)
        sound.save(file_path)
        
        print("🔊 ANNOUNCING: Hello, this is a test of the voice system")
        
        # Play in a separate thread
        threading.Thread(target=play_audio_simple, args=(file_path,), daemon=True).start()
        
        print("✅ Voice test completed")
        
    except Exception as e:
        print(f"❌ Voice test failed: {e}")

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Voice System Test")
    print("=" * 60)
    
    # Test voice system
    test_voice()
    
    # Wait for voice to complete
    time.sleep(3)
    
    print("\n✅ Test completed")

if __name__ == "__main__":
    main()
