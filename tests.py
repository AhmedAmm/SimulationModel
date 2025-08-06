import asyncio
import time
import numpy as np
import pyaudio
import whisper
import threading
from typing import Optional, Callable, List
from dataclasses import dataclass
import logging
import json
from datetime import datetime
from enum import Enum
from orchestrator_test import RealTimePresentationOrchestrator, WhisperChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WhisperChunk:
    """Represents a 5-second audio chunk from Whisper"""
    timestamp: float
    text: str
    confidence: float = 0.0
    speaker_id: Optional[str] = None
    chunk_id: str = ""
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"chunk_{int(self.timestamp)}_{hash(self.text) % 10000}"

class RealTimeWhisperProcessor:
    """
    Real-time Whisper speech-to-text processor that captures audio in 5-second chunks
    and processes them using OpenAI's Whisper model
    """
    
    def __init__(self, 
                 model_size: str = "small",
                 chunk_duration: float = 5.0,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None):
        
        # Audio configuration
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.bytes_per_sample = 2  # 16-bit audio
        
        # Whisper model
        self.model_size = model_size
        self.whisper_model = None
        
        # Audio processing
        self.audio_interface = None
        self.audio_stream = None
        self.is_recording = False
        self.is_processing = False
        
        # Buffers
        self.audio_buffer = []
        self.chunk_counter = 0
        
        # Callbacks
        self.on_chunk_processed: Optional[Callable[[WhisperChunk], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Event loop for async callbacks
        self.event_loop = event_loop
        
        # Performance tracking
        self.processing_times = []
        self.chunk_stats = []
    
    def initialize(self) -> bool:
        """Initialize Whisper model and audio interface"""
        
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
            
            # Initialize PyAudio
            self.audio_interface = pyaudio.PyAudio()
            logger.info("Audio interface initialized")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize Whisper processor: {str(e)}"
            logger.error(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return False
    
    def start_recording(self) -> bool:
        """Start real-time audio recording and processing"""
        
        if self.is_recording:
            logger.warning("Already recording")
            return False
        
        if not self.whisper_model or not self.audio_interface:
            logger.error("Whisper processor not initialized")
            return False
        
        try:
            # Open audio stream
            self.audio_stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start processing thread
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Start audio stream
            self.audio_stream.start_stream()
            self.is_recording = True
            
            logger.info("Started real-time audio recording and processing")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start recording: {str(e)}"
            logger.error(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return False
    
    def stop_recording(self):
        """Stop audio recording and processing"""
        
        if not self.is_recording:
            return
        
        try:
            self.is_recording = False
            
            # Stop audio stream
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            # Stop processing thread
            self.stop_event.set()
            if self.processing_thread:
                self.processing_thread.join(timeout=5.0)
                self.processing_thread = None
            
            # Process any remaining audio in buffer
            if len(self.audio_buffer) > 0:
                self._process_audio_chunk(self.audio_buffer.copy())
                self.audio_buffer.clear()
            
            logger.info("Stopped audio recording and processing")
            
        except Exception as e:
            error_msg = f"Error stopping recording: {str(e)}"
            logger.error(error_msg)
            if self.on_error:
                self.on_error(error_msg)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for receiving audio data"""
        
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_buffer.extend(audio_data)
        
        # Check if we have enough data for a 5-second chunk
        samples_needed = int(self.sample_rate * self.chunk_duration)
        
        if len(self.audio_buffer) >= samples_needed:
            # Extract chunk and keep remainder
            chunk_data = self.audio_buffer[:samples_needed]
            self.audio_buffer = self.audio_buffer[samples_needed:]
            
            # Queue for processing
            self._queue_audio_chunk(chunk_data)
        
        return (None, pyaudio.paContinue)
    
    def _queue_audio_chunk(self, audio_data: List[int]):
        """Queue audio chunk for processing"""
        
        self.chunk_counter += 1
        chunk_info = {
            'data': audio_data,
            'timestamp': time.time(),
            'chunk_id': self.chunk_counter
        }
        
        # In a more robust implementation, you'd use a proper queue
        if hasattr(self, '_pending_chunks'):
            self._pending_chunks.append(chunk_info)
        else:
            self._pending_chunks = [chunk_info]
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        
        self._pending_chunks = []
        
        while not self.stop_event.is_set():
            try:
                # Process any pending chunks
                if self._pending_chunks:
                    chunk_info = self._pending_chunks.pop(0)
                    self._process_audio_chunk(chunk_info['data'], 
                                            chunk_info['timestamp'], 
                                            chunk_info['chunk_id'])
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                if self.on_error:
                    self.on_error(str(e))
                time.sleep(1.0)
    
    def _process_audio_chunk(self, audio_data: List[int], 
                           timestamp: float = None, 
                           chunk_id: int = None):
        """Process a single audio chunk with Whisper"""
        
        if self.is_processing:
            logger.warning("Skipping chunk - already processing")
            return
        
        self.is_processing = True
        start_time = time.time()
        
        try:
            if timestamp is None:
                timestamp = time.time()
            if chunk_id is None:
                chunk_id = self.chunk_counter
            
            # Convert to numpy array and normalize
            audio_array = np.array(audio_data, dtype=np.float32) / 32768.0
            
            # Ensure minimum length (Whisper needs some audio)
            if len(audio_array) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                logger.debug("Audio chunk too short, skipping")
                return
            
            # Process with Whisper
            logger.debug(f"Processing audio chunk {chunk_id} ({len(audio_array)} samples)")
            
            result = self.whisper_model.transcribe(
                audio_array,
                language='en',  # Set to None for auto-detection
                fp16=True,     # Set to True if you have GPU with FP16 support
                verbose=False
            )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Extract transcription
            text = result.get('text', '').strip()
            
            # Calculate confidence
            confidence = self._estimate_confidence(result, processing_time)
            
            # Create WhisperChunk
            whisper_chunk = WhisperChunk(
                timestamp=timestamp,
                text=text,
                confidence=confidence,
                speaker_id="salesperson",
                chunk_id=f"chunk_{chunk_id}_{int(timestamp)}"
            )
            
            # Log and track
            self.chunk_stats.append({
                'chunk_id': chunk_id,
                'timestamp': timestamp,
                'text_length': len(text),
                'processing_time': processing_time,
                'confidence': confidence
            })
            
            if text:
                logger.info(f"Transcribed chunk {chunk_id}: '{text[:50]}...' (confidence: {confidence:.2f})")
                
                # Send to callback (orchestrator)
                if self.on_chunk_processed:
                    self.on_chunk_processed(whisper_chunk)
            else:
                logger.debug(f"No speech detected in chunk {chunk_id}")
            
        except Exception as e:
            error_msg = f"Error processing audio chunk: {str(e)}"
            logger.error(error_msg)
            if self.on_error:
                self.on_error(error_msg)
        
        finally:
            self.is_processing = False
    
    def _estimate_confidence(self, whisper_result: dict, processing_time: float) -> float:
        """Estimate confidence score based on Whisper result"""
        
        text = whisper_result.get('text', '')
        
        # Simple heuristic
        base_confidence = 0.7
        if len(text.strip()) == 0:
            return 0.0
        
        length_bonus = min(0.2, len(text) / 100)
        time_bonus = max(0, (5.0 - processing_time) / 20)
        artifacts = ['[', ']', '(', ')', 'inaudible', 'unclear']
        artifact_penalty = sum(0.1 for artifact in artifacts if artifact in text.lower())
        
        confidence = base_confidence + length_bonus + time_bonus - artifact_penalty
        return max(0.0, min(1.0, confidence))
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        
        if not self.processing_times:
            return {"status": "No chunks processed yet"}
        
        return {
            "chunks_processed": len(self.processing_times),
            "average_processing_time": np.mean(self.processing_times),
            "min_processing_time": np.min(self.processing_times),
            "max_processing_time": np.max(self.processing_times),
            "total_chunks": self.chunk_counter,
            "is_recording": self.is_recording,
            "model_size": self.model_size
        }
    
    def set_callbacks(self, 
                     on_chunk_processed: Callable[[WhisperChunk], None] = None,
                     on_error: Callable[[str], None] = None):
        """Set callback functions"""
        
        self.on_chunk_processed = on_chunk_processed
        self.on_error = on_error
    
    def cleanup(self):
        """Cleanup resources"""
        
        self.stop_recording()
        
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None
        
        self.whisper_model = None
        logger.info("Whisper processor cleaned up")

async def real_time_interaction():
    """Real-time speech interaction with Whisper and Orchestrator"""
    
    # Get the current event loop
    loop = asyncio.get_running_loop()
    
    # Initialize components
    whisper_processor = RealTimeWhisperProcessor(model_size="base", event_loop=loop)
    orchestrator = RealTimePresentationOrchestrator()
    
    # Set up orchestrator callbacks
    def on_client_response(response: str):
        print(f"🎯 Client Response: {response}")
    
    def on_session_update(status: dict):
        print(f"📊 Session: {status['state']} - Exchanges: {status.get('exchanges_count', 0)}")
    
    def on_orchestrator_error(error: str):
        print(f"❌ Orchestrator Error: {error}")
    
    orchestrator.set_callbacks(
        on_client_response=on_client_response,
        on_session_update=on_session_update,
        on_error=on_orchestrator_error
    )
    
    # Set up Whisper callbacks to feed orchestrator
    def on_whisper_chunk(chunk: WhisperChunk):
        print(f"🎙️ Transcribed: '{chunk.text[:50]}...'")
        if whisper_processor.event_loop:
            asyncio.run_coroutine_threadsafe(orchestrator.process_whisper_chunk(chunk), whisper_processor.event_loop)
        else:
            logger.error("No event loop available for processing whisper chunk")
    
    def on_whisper_error(error: str):
        print(f"❌ Whisper Error: {error}")
    
    whisper_processor.set_callbacks(
        on_chunk_processed=on_whisper_chunk,
        on_error=on_whisper_error
    )
    
    # Initialize components
    if not whisper_processor.initialize():
        print("❌ Failed to initialize Whisper processor")
        return
    
    success = await orchestrator.initialize_session(
        vendor_name="TechCorp",
        product_name="CloudPlatform Pro",
        client_company="Manufacturing Inc",
        client_role="CTO",
        client_industry="Manufacturing",
        company_size="Enterprise"
    )
    
    if not success:
        print("❌ Failed to initialize orchestrator session")
        return
    
    print("🚀 System ready! Start speaking...")
    print("📢 Your speech will be transcribed and the client will respond")
    print("🛑 Press Ctrl+C to stop")
    
    # Start recording
    if whisper_processor.start_recording():
        try:
            # Keep running until interrupted
            last_stat_time = 0
            while True:
                await asyncio.sleep(1)
                
                # Print stats every 30 seconds, handle empty stats
                current_time = int(time.time())
                if current_time % 30 == 0 and current_time != last_stat_time:
                    stats = whisper_processor.get_stats()
                    if 'chunks_processed' in stats:
                        print(f"📈 Stats: {stats['chunks_processed']} chunks, "
                              f"avg {stats.get('average_processing_time', 0):.2f}s processing time")
                    else:
                        print(f"📈 Stats: {stats.get('status', 'No stats available')}")
                    last_stat_time = current_time
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        
        finally:
            # Cleanup
            whisper_processor.stop_recording()
            # Add delay to ensure pending chunks are processed
            await asyncio.sleep(2)  # Wait for orchestrator to process any pending chunks
            final_report = await orchestrator.end_session()
            whisper_processor.cleanup()
            
            if final_report and 'error' not in final_report:
                report_filename = f"presentation_report_{final_report['session_summary']['session_id']}.json"
                with open(report_filename, 'w') as f:
                    json.dump(final_report, f, indent=2)
                
                print(f"✅ Session completed! Report saved to: {report_filename}")
                print(f"📈 Summary:")
                print(f"   • Duration: {final_report['session_summary']['duration_minutes']:.1f} minutes")
                print(f"   • Total Exchanges: {final_report['session_summary']['total_exchanges']}")
                # Safely access engagement_level
                engagement_level = (final_report.get('session_quality_assessment', {})
                                   .get('client_engagement', {})
                                   .get('engagement_level', 'N/A'))
                print(f"   • Client Engagement: {engagement_level}")
                print(f"   • Overall Quality: {final_report['session_quality_assessment']['overall_rating']}")
            else:
                print("❌ Session ended with errors: ", final_report.get('error', 'Unknown error'))
if __name__ == "__main__":
    asyncio.run(real_time_interaction())