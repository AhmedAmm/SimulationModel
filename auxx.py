import json
import whisper
import numpy as np
import asyncio
import threading
import queue
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uvicorn
import torch
from collections import deque
import time
import logging
import sys
from TTS import generate_speak_and_sync_payload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingWhisperTranscriber:
    def __init__(self, model_name="base", input_source="websocket", websocket_host="localhost", websocket_port=8000):
        """
        Initialize the streaming Whisper transcriber.
        
        Args:
            model_name (str): Whisper model name (tiny, base, small, medium, large)
            input_source (str): Must be "websocket" for WebSocket input
            websocket_host (str): Host for WebSocket server
            websocket_port (int): Port for WebSocket server
        """
        if input_source != "websocket":
            raise ValueError("Only 'websocket' input source is supported")
        
        self.model_name = model_name
        self.input_source = input_source
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = 5
        self.overlap_duration = 1
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap_size = int(self.sample_rate * self.overlap_duration)
        
        # Audio buffer with overlap handling
        self.audio_buffer = deque(maxlen=self.chunk_size * 3)
        self.processing_queue = queue.Queue()
        
        # Streaming transcription state
        self.full_transcription = ""
        self.last_chunk_text = ""
        self.sentence_buffer = []
        self.silence_counter = 0
        self.max_silence_chunks = 2
        
        # Control flags
        self.is_running = False
        self.is_listening = False
        self.server_should_stop = False
        
        # Orchestrator reference
        self.orchestrator = None
        
        # Load Whisper model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info("Whisper model loaded successfully")
        
        # Initialize WebSocket
        self._init_websocket()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.loop = asyncio.new_event_loop()

    def set_orchestrator(self, orchestrator):
        """Set reference to ConversationalOrchestrator"""
        self.orchestrator = orchestrator
        logger.info("Orchestrator set for transcriber")

    def _init_websocket(self):
        """Initialize FastAPI WebSocket server"""
        self.app = FastAPI()
        self.websocket_connection = None
        self.server = None
        self.loop = asyncio.new_event_loop()

        @self.app.websocket("/transcribe")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connection = websocket
            logger.info("WebSocket client connected")
            try:
                while True:
                    data = await websocket.receive_bytes()
                    audio_np = np.frombuffer(data, dtype=np.float32)
                    logger.debug(f"Received audio chunk: {len(audio_np)} samples")
                    if len(audio_np) == 0:
                        logger.warning("Empty audio chunk received")
                        continue
                    self.audio_buffer.extend(audio_np)
                    if len(self.audio_buffer) >= self.chunk_size:
                        self._process_audio_chunk()
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                self.websocket_connection = None
            except Exception as e:
                logger.error(f"WebSocket error: {type(e).__name__}: {str(e)}")
                self.websocket_connection = None

        def run_server():
            config = uvicorn.Config(
                self.app,
                host=self.websocket_host,
                port=self.websocket_port,
                log_level="info",
                loop=self.loop
            )
            self.server = uvicorn.Server(config)
            asyncio.set_event_loop(self.loop)
            logger.debug(f"Starting Uvicorn with loop: {self.loop}")
            try:
                while not self.server_should_stop:
                    self.loop.run_until_complete(self.server.serve())
                    if self.server_should_stop:
                        break
                    time.sleep(0.1)  # Prevent tight loop
            except Exception as e:
                logger.error(f"Uvicorn server error: {type(e).__name__}: {str(e)}")
            finally:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                if not self.loop.is_closed():
                    self.loop.close()
                    logger.info("Event loop closed in server thread")

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def send_transcription_to_ws(self, data):
        """Send both transcription and LLM response to WebSocket"""
        if not self.websocket_connection:
            logger.warning("No active websocket connection. Cannot send data.")
            return
        transcription = data.get("transcription", "")
        response = generate_speak_and_sync_payload(data.get("response", ""))
        logger.info(f"Preparing to send data: transcription='{transcription}', response='{response}'")
        async def send_text():
            try:
                payload = {"transcription": transcription, "response": response}
                await self.websocket_connection.send_text(json.dumps(payload))
                logger.info(f"Sent data to WebSocket: {payload}")
            except Exception as e:
                logger.error(f"Failed to send data via WebSocket: {type(e).__name__}: {str(e)}")
                self.websocket_connection = None

        print("Starting new thread to send data")
        def run_send_text():
            asyncio.run(send_text())

        send_thread = threading.Thread(target=run_send_text, daemon=True)
        send_thread.start()
        send_thread.join(timeout=2.0)
        print("Finished sending data attempt")

    def _process_audio_chunk(self):
        """Process audio chunk with overlap"""
        if len(self.audio_buffer) < self.chunk_size:
            return
        
        audio_chunk = np.array(list(self.audio_buffer)[-self.chunk_size:])
        overlap_data = list(self.audio_buffer)[-self.overlap_size:]
        self.audio_buffer.clear()
        self.audio_buffer.extend(overlap_data)
        self.processing_queue.put(audio_chunk)
    
    def _is_silence(self, audio_chunk, threshold=0.01):
        """Check if audio chunk contains mostly silence"""
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms < threshold
    
    def _clean_text(self, text):
        """Clean and normalize transcribed text"""
        text = text.strip()
        artifacts = ["", " ", ".", "...", "Thank you.", "Thanks for watching.", "you", "You"]
        if text in artifacts:
            return ""
        return text
    
    def _update_streaming_display(self, new_text):
        """Update the streaming display with new text"""
        if not new_text:
            return
        
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        if self.full_transcription and not self.full_transcription.endswith(' '):
            self.full_transcription += ' '
        self.full_transcription += new_text
        
        display_text = self.full_transcription
        if len(display_text) > 100:
            display_text = "..." + display_text[-97:]
        
        sys.stdout.write(f"🎙️  {display_text}")
        sys.stdout.flush()
    
    def _processing_loop(self):
        """Main processing loop for transcription"""
        while self.is_running:
            try:
                audio_chunk = self.processing_queue.get(timeout=1.0)
                
                if self._is_silence(audio_chunk):
                    self.silence_counter += 1
                    if self.silence_counter >= self.max_silence_chunks:
                        if self.full_transcription and not self.full_transcription.endswith(('.', '!', '?', '\n')):
                            self.full_transcription += '.'
                            sys.stdout.write('.\n')
                            sys.stdout.flush()
                    continue
                else:
                    self.silence_counter = 0
                
                audio_chunk = audio_chunk.astype(np.float32)
                result = self.model.transcribe(
                    audio_chunk,
                    task="transcribe",
                    no_speech_threshold=0.3,
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                    language="en",
                    fp16=torch.cuda.is_available()
                )
                
                new_text = self._clean_text(result["text"])
                
                if new_text and new_text != self.last_chunk_text:
                    if not (self.full_transcription.endswith(new_text) or 
                           new_text in self.full_transcription.split()[-5:]):
                        print(" \n --------------------------------------------------------------------------------------Processing Transcription----------------------")
                        logger.info(f"Transcribed text: {new_text}")
                        if self.orchestrator:
                            self.orchestrator.response_callback(new_text)
                        else:
                            logger.warning("No orchestrator set, cannot process transcription")
                        print("\n --------------------------------------------------------------------------------------DONE-----------------------------------")
                        self._update_streaming_display(new_text)
                        self.last_chunk_text = new_text
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {type(e).__name__}: {str(e)}")
    
    def start_transcription(self):
        """Start the streaming transcription"""
        if self.is_running:
            logger.warning("Transcription already running")
            return
        
        self.is_running = True
        self.processing_thread.start()
        self._start_websocket_transcription()
    
    def _start_websocket_transcription(self):
        """Start WebSocket-based transcription"""
        print(f"🌐 Starting WebSocket server on {self.websocket_host}:{self.websocket_port}")
        print("📡 Waiting for WebSocket connections...")
        logger.info(f"WebSocket server started on {self.websocket_host}:{self.websocket_port}")
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping WebSocket server...")
            self.stop_transcription()
    
    def stop_transcription(self):
        """Stop the transcription"""
        self.is_running = False
        self.server_should_stop = True
        
        if self.processing_thread.is_alive() and threading.current_thread() != self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.websocket_connection:
            try:
                # Check if WebSocket is still connected
                if not self.websocket_connection.client_state == WebSocketDisconnect:
                    # Run WebSocket close in the correct event loop
                    future = asyncio.run_coroutine_threadsafe(self.websocket_connection.close(), self.loop)
                    future.result(timeout=5.0)
                    logger.info("WebSocket connection closed successfully")
                else:
                    logger.info("WebSocket already disconnected, no need to close")
            except TimeoutError:
                logger.warning("Timeout while closing WebSocket connection, proceeding with cleanup")
            except Exception as e:
                logger.error(f"Failed to close WebSocket connection: {type(e).__name__}: {str(e)}")
            finally:
                self.websocket_connection = None
        
        # Shut down Uvicorn server
        if self.server:
            try:
                async def shutdown_server():
                    # Log pending tasks for debugging
                    tasks = [task for task in asyncio.all_tasks(self.loop) if task is not asyncio.current_task(self.loop)]
                    logger.debug(f"Pending tasks in loop: {len(tasks)}")
                    for task in tasks:
                        logger.debug(f"Task: {task}")
                    await self.server.shutdown()
                    await self.loop.shutdown_asyncgens()
                    if not self.loop.is_closed():
                        self.loop.close()
                
                future = asyncio.run_coroutine_threadsafe(shutdown_server(), self.loop)
                future.result(timeout=10.0)
                logger.info("Uvicorn server shut down successfully")
            except TimeoutError:
                logger.warning("Timeout while shutting down Uvicorn server, forcing cleanup")
                if not self.loop.is_closed():
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                    self.loop.close()
                    logger.info("Event loop forcibly closed")
            except Exception as e:
                logger.error(f"Failed to shut down Uvicorn server: {type(e).__name__}: {str(e)}")
            finally:
                self.server = None
        
        print("\n✅ Transcription stopped")
        logger.info("Transcription stopped")
    
    def get_full_transcription(self):
        """Get the complete transcription text"""
        return self.full_transcription
    
    def clear_transcription(self):
        """Clear the current transcription"""
        self.full_transcription = ""
        self.last_chunk_text = ""
        self.sentence_buffer = []
        print("\n🧹 Transcription cleared")
        logger.info("Transcription cleared")
    
    def test_whisper(self, test_audio_path=None):
        """
        Test Whisper model functionality
        """
        print("🧪 Testing Whisper model...")
        logger.info("Testing Whisper model")
        
        if test_audio_path:
            result = self.model.transcribe(test_audio_path)
            print(f"Test transcription: {result['text']}")
            logger.info(f"Test transcription: {result['text']}")
        else:
            print("Generating test audio (sine wave)...")
            logger.info("Generating test audio (sine wave)")
            duration = 3.0
            frequency = 440.0
            t = np.linspace(0, duration, int(self.sample_rate * duration), dtype=np.float32)
            test_audio = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
            
            result = self.model.transcribe(test_audio)
            print(f"Test result: {result['text']}")
            print("Note: Sine wave may not produce meaningful text")
            logger.info(f"Test result: {result['text']}")
        
        print("✅ Whisper model test completed")
        logger.info("Whisper model test completed")
    
    def _reset_audio_buffer(self):
        """Reset audio buffer after processing"""
        self.audio_buffer.clear()
        self.processing_queue = queue.Queue()
        logger.info("Audio buffer reset")
    
    def reset_for_new_conversation(self):
        """Reset all transcription state for fresh conversation"""
        logger.info("🔄 Resetting transcriber state...")
        
        self.audio_buffer.clear()
        self.full_transcription = ""
        self.last_chunk_text = ""
        self.sentence_buffer = []
        self.silence_counter = 0
        
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("✅ Transcriber reset complete")

if __name__ == "__main__":
    print("=== Streaming Real-time Whisper Transcription ===\n")
    
    transcriber = StreamingWhisperTranscriber(
        model_name="base",
        input_source="websocket"
    )
    
    transcriber.test_whisper()
    print()
    
    transcriber.start_transcription()
