"""
Real-time OCR Text Processing with LLM and Text-to-Speech
==========================================================
Combines fragmented OCR text using Together AI LLM and speaks it aloud
"""

import os
import time
import re
from datetime import datetime
from together import Together
import pyttsx3
from threading import Thread, Lock
import warnings
import subprocess
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

TOGETHER_API_KEY = ""  # Replace with your API key
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
OCR_OUTPUT_FILE = "output.txt"  # Output file from the OCR script
PROCESSED_LOG_FILE = "processed_messages.txt"  # Log of processed messages

# Text-to-Speech Configuration
TTS_RATE = 150  # Speaking rate (words per minute)
TTS_VOLUME = 0.9  # Volume (0.0 to 1.0)


# ============================================================================
# Text Processing with LLM
# ============================================================================

class TextProcessor:
    """Process fragmented OCR text using Together AI LLM"""
    
    def __init__(self, api_key):
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
        if not text_fragments:
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
        if not text or not text.strip():
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
        with self.lock:
            return self.is_speaking


# ============================================================================
# OCR File Monitor
# ============================================================================

class OCRFileMonitor:
    """Monitor OCR output file for new text entries"""
    
    def __init__(self, file_path, start_from_beginning=False):
        self.file_path = file_path
        self.last_position = 0
        self.last_frame_number = 0
        
        # Create file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("")
            print(f"✓ Created OCR output file: {file_path}")
        else:
            if start_from_beginning:
                # Start monitoring from beginning of file
                self.last_position = 0
                print(f"✓ Monitoring existing file from BEGINNING: {file_path}")
                print(f"   File size: {os.path.getsize(file_path)} bytes")
            else:
                # Get initial file size (skip existing content)
                self.last_position = os.path.getsize(file_path)
                print(f"✓ Monitoring existing file from END: {file_path}")
                print(f"   Starting at position: {self.last_position} bytes")
    
    def check_for_new_entries(self):
        """
        Check for new OCR entries in the file
        
        Returns:
            List of text fragments from new frame, or None if no new entries
        """
        try:
            current_size = os.path.getsize(self.file_path)
            
            if current_size <= self.last_position:
                return None
            
            # Read new content
            with open(self.file_path, 'r', encoding='utf-8') as f:
                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = current_size
            
            if not new_content.strip():
                return None
            
            # Debug: Show what was read
            print(f"\n[DEBUG] Read {len(new_content)} characters from file")
            print(f"[DEBUG] Content preview: {new_content[:200]}...")
            
            # Parse the new content for frame entries
            texts = self._parse_frame_entry(new_content)
            
            if texts:
                print(f"[DEBUG] Parsed {len(texts)} text fragments")
            else:
                print(f"[DEBUG] No valid text fragments found in content")
            
            return texts if texts else None
            
        except Exception as e:
            print(f"✗ Error reading OCR file: {e}")
            return None
    
    def _parse_frame_entry(self, content):
        """
        Parse frame entry and extract text fragments in order
        
        Format:
        - Frame X at HH:MM:SS --- OR --- Frame X at HH:MM:SS ---
        [1] text1 (conf: 0.XX, method: xxx)
        [2] text2 (conf: 0.XX, method: xxx)
        ...
        """
        lines = content.strip().split('\n')
        texts = []
        current_frame = None
        
        for line in lines:
            line = line.strip()
            
            # Check for frame marker (flexible - matches both "- Frame" and "--- Frame")
            if 'Frame' in line and 'at' in line and ('---' in line or line.startswith('-')):
                # If we have texts from previous frame, return them
                if texts and current_frame:
                    print(f"[DEBUG] Found new frame marker, returning {len(texts)} texts from frame {current_frame}")
                    return texts
                
                # Extract frame number
                match = re.search(r'Frame (\d+)', line)
                if match:
                    frame_num = int(match.group(1))
                    if frame_num > self.last_frame_number:
                        current_frame = frame_num
                        self.last_frame_number = frame_num
                        texts = []
                        print(f"[DEBUG] Starting to parse frame {frame_num}")
            
            # Extract text from reading order lines
            elif line.startswith('['):
                # Format: [order] text (conf: X.XX, method: xxx)
                match = re.match(r'\[(\d+)\]\s+(.+?)\s+\(conf:', line)
                if match:
                    order = int(match.group(1))
                    text = match.group(2).strip()
                    if text and current_frame:
                        texts.append(text)
                        print(f"[DEBUG] Extracted text from line [{order}]: '{text}'")
                    elif not current_frame:
                        print(f"[DEBUG] Skipping text (no current frame): '{line}'")
        
        # Return texts from the last parsed frame
        if texts and current_frame:
            print(f"[DEBUG] Returning {len(texts)} texts from final frame {current_frame}")
            return texts
        else:
            if not current_frame:
                print(f"[DEBUG] No valid frame marker found")
            elif not texts:
                print(f"[DEBUG] Frame {current_frame} found but no texts extracted")
            return None


# ============================================================================
# Main Pipeline
# ============================================================================

class RealTimeTextProcessor:
    """Main pipeline: Monitor OCR → Process with LLM → Speak"""
    
    def __init__(self, api_key, ocr_file, log_file, process_existing=False):
        print("\n" + "=" * 60)
        print("Real-Time OCR Text Processor with LLM and TTS")
        print("=" * 60)
        
        self.text_processor = TextProcessor(api_key)
        self.tts_handler = TTSHandler(rate=TTS_RATE, volume=TTS_VOLUME)
        self.ocr_monitor = OCRFileMonitor(ocr_file, start_from_beginning=process_existing)
        self.log_file = log_file
        
        # Initialize log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Processed Messages Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
        
        print(f"✓ Log file: {log_file}")
        print("=" * 60)
        
        if process_existing:
            print("\n⚠ Processing EXISTING content in OCR file first...")
            self._process_existing_content()
        
        print("\nPipeline ready!")
        print("Waiting for NEW OCR input...")
        print("(Press Ctrl+C to stop)\n")
    
    def _process_existing_content(self):
        """Process any existing content in the OCR file"""
        text_fragments = self.ocr_monitor.check_for_new_entries()
        
        if text_fragments:
            print(f"\n{'='*60}")
            print(f"Processing Existing Content - {len(text_fragments)} fragment(s)")
            print(f"{'='*60}")
            print("Fragments:")
            for i, frag in enumerate(text_fragments, 1):
                print(f"  [{i}] {frag}")
            
            # Process with LLM
            print("\n⚙️  Processing with LLM...")
            combined_message = self.text_processor.combine_texts(text_fragments)
            
            if combined_message:
                print(f"✓ Combined: {combined_message}")
                self._log_message(text_fragments, combined_message)
                self.tts_handler.speak(combined_message)
                
                # Wait for speech to complete
                time.sleep(2)
                while self.tts_handler.is_currently_speaking():
                    time.sleep(0.5)
            else:
                print("⚠ No message to process")
            
            print(f"{'='*60}\n")
        else:
            print("⚠ No existing content found in OCR file\n")
    
    def run(self, check_interval=0.2):
        """
        Run the monitoring loop
        
        Args:
            check_interval: Time in seconds between file checks
        """
        try:
            while True:
                # Check for new OCR entries
                text_fragments = self.ocr_monitor.check_for_new_entries()
                print(".", end="", flush=True) 
                if text_fragments:
                    print(f"\n{'='*60}")
                    print(f"New OCR Detection - {len(text_fragments)} fragment(s)")
                    print(f"{'='*60}")
                    print("Fragments:")
                    for i, frag in enumerate(text_fragments, 1):
                        print(f"  [{i}] {frag}")
                    
                    # Process with LLM
                    print("\n⚙️  Processing with LLM...")
                    combined_message = self.text_processor.combine_texts(text_fragments)
                    
                    if combined_message:
                        print(f"✓ Combined: {combined_message}")
                        
                        # Log the result
                        self._log_message(text_fragments, combined_message)
                        
                        # Speak the message
                        self.tts_handler.speak(combined_message)
                    else:
                        print("⚠ No new message to process (duplicate or empty)")
                    
                    print(f"{'='*60}\n")
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n✓ Stopping processor...")
            print("Goodbye!")
    
    def _log_message(self, fragments, combined):
        """Log the processed message"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n--- {timestamp} ---\n")
                f.write(f"Fragments: {fragments}\n")
                f.write(f"Combined: {combined}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"✗ Error logging message: {e}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point"""
    
    # Check for API key
    api_key = os.getenv('TOGETHER_API_KEY', TOGETHER_API_KEY)
    
    if not api_key or api_key == "your_together_api_key_here":
        print("\n" + "=" * 60)
        print("ERROR: Together AI API Key Required")
        print("=" * 60)
        print("\nPlease set your API key:")
        print("1. Edit this script and replace TOGETHER_API_KEY value")
        print("2. OR set environment variable: TOGETHER_API_KEY")
        print("\nGet your API key from: https://api.together.xyz/")
        print("=" * 60 + "\n")
        return
    
    # Check if OCR output file exists
    if not os.path.exists(OCR_OUTPUT_FILE):
        print(f"\n⚠ Warning: OCR output file not found: {OCR_OUTPUT_FILE}")
        print("Make sure the OCR script is running and generating output.")
        print("Creating empty file for monitoring...\n")
        process_existing = False
    else:
        # Check file size
        file_size = os.path.getsize(OCR_OUTPUT_FILE)
        print(f"\n✓ OCR output file found: {OCR_OUTPUT_FILE}")
        print(f"  File size: {file_size} bytes")
        
        if file_size > 0:
            print("\nDo you want to process existing content in the file?")
            print("1. Yes - Process existing content first, then monitor for new")
            print("2. No - Only monitor for NEW content (skip existing)")
            choice = input("Enter choice (1 or 2): ").strip()
            process_existing = (choice == "1")
        else:
            process_existing = False
    
    # Initialize and run pipeline
    try:
        processor = RealTimeTextProcessor(
            api_key=api_key,
            ocr_file=OCR_OUTPUT_FILE,
            log_file=PROCESSED_LOG_FILE,
            process_existing=process_existing
        )
        processor.run(check_interval=0.5)
        
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()