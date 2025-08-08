import whisper
import numpy as np
import asyncio
import threading
import queue
import pyaudio
import wave
import io
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uvicorn
import torch
from collections import deque
import time
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingWhisperTranscriber:
    def __init__(self, model_name="base", input_source="mic", websocket_host="localhost", websocket_port=8000):
        """
        Initialize the streaming Whisper transcriber.
        
        Args:
            model_name (str): Whisper model name (tiny, base, small, medium, large)
            input_source (str): "mic" for microphone or "websocket" for WebSocket
            websocket_host (str): Host for WebSocket server
            websocket_port (int): Port for WebSocket server
        """
        self.model_name = model_name
        self.input_source = input_source
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = 5  # 5 second chunks for better accuracy
        self.overlap_duration = 1  # 1 second overlap
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap_size = int(self.sample_rate * self.overlap_duration)
        
        # Audio buffer with overlap handling
        self.audio_buffer = deque(maxlen=self.chunk_size * 3)  # Larger buffer
        self.processing_queue = queue.Queue()
        
        # Streaming transcription state
        self.full_transcription = ""
        self.last_chunk_text = ""
        self.sentence_buffer = []
        self.silence_counter = 0
        self.max_silence_chunks = 2  # Reset after 3 silent chunks
        
        # Control flags
        self.is_running = False
        self.is_listening = False
        
        # Load Whisper model on GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info("Whisper model loaded successfully")
        
        # Initialize audio components based on input source
        if self.input_source == "mic":
            self._init_microphone()
        elif self.input_source == "websocket":
            self._init_websocket()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        
    def _init_microphone(self):
        """Initialize PyAudio for microphone input"""
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def _init_websocket(self):
        """Initialize FastAPI WebSocket server"""
        self.app = FastAPI()
        self.websocket_connection = None
        
        @self.app.websocket("/transcribe")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connection = websocket
            logger.info("WebSocket client connected")
            
            try:
                while True:
                    # Receive audio data from WebSocket
                    data = await websocket.receive_bytes()
                    audio_np = np.frombuffer(data, dtype=np.float32)
                    
                    # Add to buffer
                    self.audio_buffer.extend(audio_np)
                    
                    # Process if buffer has enough data
                    if len(self.audio_buffer) >= self.chunk_size:
                        self._process_audio_chunk()
                        
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                self.websocket_connection = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert audio data to numpy array
        audio_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer.extend(audio_np)
        
        # Process if buffer has enough data
        if len(self.audio_buffer) >= self.chunk_size:
            self._process_audio_chunk()
            
        return (None, pyaudio.paContinue)
    
    def _process_audio_chunk(self):
        """Process audio chunk with overlap"""
        if len(self.audio_buffer) < self.chunk_size:
            return
        
        # Get chunk with overlap
        audio_chunk = np.array(list(self.audio_buffer)[-self.chunk_size:])
        
        # Remove processed part (keeping overlap)
        overlap_data = list(self.audio_buffer)[-self.overlap_size:]
        self.audio_buffer.clear()
        self.audio_buffer.extend(overlap_data)
        
        # Add to processing queue
        self.processing_queue.put(audio_chunk)
    
    def _is_silence(self, audio_chunk, threshold=0.01):
        """Check if audio chunk contains mostly silence"""
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms < threshold
    
    def _clean_text(self, text):
        """Clean and normalize transcribed text"""
        text = text.strip()
        # Remove common artifacts
        artifacts = ["", " ", ".", "...", "Thank you.", "Thanks for watching.", "you", "You"]
        if text in artifacts:
            return ""
        return text
    
    def _update_streaming_display(self, new_text):
        """Update the streaming display with new text"""
        if not new_text:
            return
        
        # Clear the current line and move cursor to beginning
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        
        # Add new text to full transcription
        if self.full_transcription and not self.full_transcription.endswith(' '):
            self.full_transcription += ' '
        self.full_transcription += new_text
        
        # Display the updated transcription
        display_text = self.full_transcription
        if len(display_text) > 100:  # Limit display length
            display_text = "..." + display_text[-97:]
        
        sys.stdout.write(f"🎙️  {display_text}")
        sys.stdout.flush()
    
    def _processing_loop(self):
        """Main processing loop for transcription"""
        while self.is_running:
            try:
                # Get audio chunk from queue (with timeout)
                audio_chunk = self.processing_queue.get(timeout=1.0)
                try:
                    # Check for silence
                    if self._is_silence(audio_chunk):
                        self.silence_counter += 1
                        if self.silence_counter >= self.max_silence_chunks:
                            if hasattr(self, 'orchestrator'):
                                self.orchestrator.notify_silence()
                            if self.full_transcription and not self.full_transcription.endswith(('.', '!', '?', '\n')):
                                self.full_transcription += '.'
                                sys.stdout.write('.\n')
                                sys.stdout.flush()
                        continue  # still task_done in finally
                    else:
                        self.silence_counter = 0

                    # Transcribe audio chunk (ensure float32 dtype)
                    audio_chunk = audio_chunk.astype(np.float32)
                    result = self.model.transcribe(
                        audio_chunk,
                        task="transcribe",
                        no_speech_threshold=0.3,
                        logprob_threshold=-1.0,
                        compression_ratio_threshold=2.4,
                        language="en",  # Force English transcription only
                        fp16=torch.cuda.is_available()
                    )

                    # Clean and process transcription
                    new_text = self._clean_text(result["text"])

                    if new_text and new_text != self.last_chunk_text:
                        # Avoid repetition of the same chunk
                        if not (self.full_transcription.endswith(new_text) or
                                new_text in self.full_transcription.split()[-5:]):  # Check last 5 words
                            self._update_streaming_display(new_text)
                            self.last_chunk_text = new_text
                finally:
                    # Always mark the queue item as done
                    self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def start_transcription(self):
        """Start the streaming transcription"""
        if self.is_running:
            logger.warning("Transcription already running")
            return
        
        self.is_running = True
        self.processing_thread.start()
        
        if self.input_source == "mic":
            self._start_microphone_transcription()
        elif self.input_source == "websocket":
            self._start_websocket_transcription()
    
    def _start_microphone_transcription(self):
        """Start microphone-based transcription"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=512,  # Smaller buffer for lower latency
                stream_callback=self._audio_callback
            )
            
            print("🎤 Streaming transcription started... (Press Ctrl+C to stop)")
            print("💬 Your speech will appear below in real-time:")
            print("-" * 80)
            self.stream.start_stream()
            
            # Keep the main thread alive
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n" + "=" * 80)
            print("📝 Final Transcription:")
            print(self.full_transcription)
            print("=" * 80)
            print("⏹️  Stopping transcription...")
            self.stop_transcription()
        except Exception as e:
            logger.error(f"Error in microphone transcription: {e}")
            self.stop_transcription()
    
    def _start_websocket_transcription(self):
        """Start WebSocket-based transcription"""
        print(f"🌐 Starting WebSocket server on {self.websocket_host}:{self.websocket_port}")
        print("📡 Waiting for WebSocket connections...")
        
        try:
            uvicorn.run(
                self.app,
                host=self.websocket_host,
                port=self.websocket_port,
                log_level="info"
            )
        except KeyboardInterrupt:
            print("\n⏹️  Stopping WebSocket server...")
            self.stop_transcription()
    
    def stop_transcription(self):
        """Stop the transcription"""
        self.is_running = False
        
        if self.input_source == "mic" and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        
        # Wait for processing thread to finish
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        print("\n✅ Transcription stopped")
    
    def get_full_transcription(self):
        """Get the complete transcription text"""
        return self.full_transcription
    
    def clear_transcription(self):
        """Clear the current transcription"""
        self.full_transcription = ""
        self.last_chunk_text = ""
        self.sentence_buffer = []
        print("\n🧹 Transcription cleared")
    
    def test_whisper(self, test_audio_path=None):
        """
        Test Whisper model functionality
        
        Args:
            test_audio_path (str): Path to test audio file (optional)
        """
        print("🧪 Testing Whisper model...")
        
        if test_audio_path:
            # Test with provided audio file
            result = self.model.transcribe(test_audio_path)
            print(f"Test transcription: {result['text']}")
        else:
            # Test with synthetic audio (sine wave) - ensure float32 dtype
            print("Generating test audio (sine wave)...")
            duration = 3.0
            frequency = 440.0  # A4 note
            t = np.linspace(0, duration, int(self.sample_rate * duration), dtype=np.float32)
            test_audio = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
            
            result = self.model.transcribe(test_audio)
            print(f"Test result: {result['text']}")
            print("Note: Sine wave may not produce meaningful text")
        
        print("✅ Whisper model test completed")
    def _reset_audio_buffer(self):
        """Reset audio buffer after processing"""
        self.audio_buffer.clear()
        self.processing_queue = queue.Queue()  # Fresh queue
    
    def reset_for_new_conversation(self):
        logger.info("🔄 Resetting transcriber state...")

        self.audio_buffer.clear()
        self.processing_queue = queue.Queue()

        self.full_transcription = ""
        self.last_chunk_text = ""   # ✅ Clear last chunk
        self.sentence_buffer = []
        self.silence_counter = 0

        # ✅ Ensure we don't compare to stale orchestrator text
        if hasattr(self, 'orchestrator'):
            self.orchestrator._last_processed_text = ""

        logger.info("✅ Transcriber reset complete")






# Example usage and testing
if __name__ == "__main__":
    # Example 1: Microphone input with streaming
    print("=== Streaming Real-time Whisper Transcription ===\n")
    
    # Initialize transcriber with microphone input
    transcriber = StreamingWhisperTranscriber(
        model_name="base",  # or "tiny", "small", "medium", "large"
        input_source="mic"
    )
    
    # Test Whisper functionality
    transcriber.test_whisper()
    print()
    
    # Start streaming transcription
    transcriber.start_transcription()
    
    # Example 2: WebSocket input (uncomment to use)
    # transcriber_ws = StreamingWhisperTranscriber(
    #     model_name="base",
    #     input_source="websocket",
    #     websocket_host="0.0.0.0",
    #     websocket_port=8000
    # )
    # transcriber_ws.start_transcription()