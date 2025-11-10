#!/usr/bin/env python3
"""
Working Voice System for Traffic Light Detection

This version uses direct speech calls to ensure the correct instructions are spoken.
"""

import re
import time
import threading
import pyttsx3
from datetime import datetime


class WorkingVoiceSystem:
    """
    A working voice system that speaks directly without complex queuing.
    """
    
    def __init__(self, confidence_threshold: float = 0.4, cooldown_seconds: float = 2.0):
        """
        Initialize the working voice system.
        """
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # State tracking
        self._lock = threading.Lock()
        self._last_color = None
        self._last_speech_time = 0.0
        self._first_detection = True
        
        # Initialize speech engine
        self._speech_engine = None
        self._initialize_speech_engine()
        
        # Instructions
        self._instructions = {
            'red': "Stop! Red light detected.",
            'yellow': "Caution! Yellow light detected. Prepare to stop.",
            'green': "Go! Green light detected. Safe to proceed."
        }
        
        # Change notifications
        self._change_instructions = {
            'red': "Traffic light changed to red. Stop immediately!",
            'yellow': "Traffic light changed to yellow. Prepare to stop!",
            'green': "Traffic light changed to green. You may proceed!"
        }
        
        print("Working Voice System Ready")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"Cooldown: {cooldown_seconds} seconds")
        print("-" * 50)
    
    def _initialize_speech_engine(self):
        """Initialize the speech engine."""
        try:
            self._speech_engine = pyttsx3.init()
            self._speech_engine.setProperty('rate', 150)
            self._speech_engine.setProperty('volume', 1.0)
            
            # Test speech
            print("Testing speech engine...")
            self._speech_engine.say("Voice system ready")
            self._speech_engine.runAndWait()
            print("Speech engine test completed")
            
        except Exception as e:
            print(f"Error initializing speech engine: {e}")
            self._speech_engine = None
    
    def _speak_direct(self, text: str):
        """Speak text directly and immediately."""
        try:
            print(f"SPEAKING: {text}")
            
            # Create a new speech engine for each message to avoid corruption
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Select a voice
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            
            engine.say(text)
            engine.runAndWait()
            
            # Clean up the engine
            del engine
            
            print(f"SPOKE: {text}")
        except Exception as e:
            print(f"SPEECH ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def _is_color_change(self, new_color: str) -> bool:
        """Check if this is a color change."""
        with self._lock:
            if self._last_color is None:
                return False
            return self._last_color != new_color
    
    def _should_speak(self, is_change: bool, is_first_detection: bool = False) -> bool:
        """Determine if we should speak."""
        current_time = time.time()
        
        with self._lock:
            # Always speak on first valid detection
            if is_first_detection:
                return True
            
            time_since_last = current_time - self._last_speech_time
            
            # Always speak on color changes (shorter cooldown)
            if is_change:
                return time_since_last > 0.5
            
            # For same color, use normal cooldown
            return time_since_last > self.cooldown_seconds
    
    def _update_state(self, color: str, confidence: float):
        """Update internal state."""
        with self._lock:
            self._last_color = color
            self._last_speech_time = time.time()
            self._first_detection = False
    
    def process_detection(self, color: str, confidence: float):
        """
        Process a detection and speak if appropriate.
        
        Args:
            color: Detected color (red, yellow, green)
            confidence: Detection confidence (0.0 to 1.0)
        """
        if confidence < self.confidence_threshold:
            print(f"Confidence too low: {color} (conf: {confidence:.2f} < {self.confidence_threshold}) - NOT SPOKEN")
            return
        
        # Check if this is a color change
        is_change = self._is_color_change(color)
        is_first_detection = self._first_detection
        
        # Check if we should speak
        should_speak = self._should_speak(is_change, is_first_detection)
        
        if not should_speak:
            print(f"[{color}] [conf: {confidence:.2f}] [NOT SPOKEN - cooldown/repetition]")
            return
        
        # Get appropriate instruction
        if is_first_detection:
            instruction = self._instructions.get(color, "")
            print(f"[{color}] [conf: {confidence:.2f}] [FIRST DETECTION] -> {instruction}")
        elif is_change:
            instruction = self._change_instructions.get(color, "")
            print(f"[{color}] [conf: {confidence:.2f}] [COLOR CHANGE: {self._last_color} -> {color}] -> {instruction}")
        else:
            instruction = self._instructions.get(color, "")
            print(f"[{color}] [conf: {confidence:.2f}] [SAME COLOR REPEAT] -> {instruction}")
        
        if not instruction:
            print(f"No instruction found for color: {color}")
            return
        
        # Speak directly
        self._speak_direct(instruction)
        
        # Update state
        self._update_state(color, confidence)
        
        print(f"[{color}] [conf: {confidence:.2f}] [SPOKEN] - {instruction}")
    
    def process_line(self, line: str):
        """Process a line of model output."""
        line = line.strip()
        if not line:
            return
        
        # Extract color
        colors = ['red', 'yellow', 'green']
        detected_color = None
        for color in colors:
            if color in line.lower():
                detected_color = color
                break
        
        if detected_color:
            print(f"Detected color: {detected_color} from line: {line}")
            # For simplicity, assume confidence is high enough
            self.process_detection(detected_color, 0.8)
        else:
            print("No color detected in line")
    
    def cleanup(self):
        """Clean up resources."""
        print("Voice system stopped")


def test_working_voice_system():
    """Test the working voice system."""
    print("Testing Working Voice System")
    print("=" * 50)
    
    system = WorkingVoiceSystem(confidence_threshold=0.4, cooldown_seconds=1.0)
    
    # Test data
    test_detections = [
        ("red", 0.8),
        ("green", 0.7),
        ("yellow", 0.6),
        ("red", 0.9),
    ]
    
    print("Testing direct detection processing...")
    for color, confidence in test_detections:
        print(f"\n{'='*50}")
        print(f"Testing: {color} (confidence: {confidence})")
        system.process_detection(color, confidence)
        time.sleep(3)  # Wait to hear speech
    
    print("\nTest completed!")
    system.cleanup()


if __name__ == "__main__":
    test_working_voice_system()
