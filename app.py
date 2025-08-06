import asyncio
import json
import time
import numpy as np
import pyaudio
import whisper
import threading
import base64
import io
import wave
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import tempfile
import os
from Llm import llm as llm1 # Assuming you have a separate LLM module
import io
import soundfile as sf  # pip install soundfile
import numpy as np
import io
import numpy as np
import ffmpeg
import numpy as np
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Simple LLM class for demonstration (replace with your actual LLM implementation)
class SimpleLLM:
    def __init__(self):
        self.conversation_history = []
    
    async def ask_gemini(self, prompt):
        return llm1.ask_gemini(prompt)  # Replace with actual LLM call

# Global LLM instance
llm = SimpleLLM()

# Data classes
@dataclass
class WhisperChunk:
    timestamp: float
    text: str
    confidence: float = 0.0
    speaker_id: Optional[str] = None
    chunk_id: str = ""
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"chunk_{int(self.timestamp)}_{hash(self.text) % 10000}"

class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"

# Conversation Manager
class ConversationManager:
    def __init__(self):
        self.state = ConversationState.IDLE
        self.whisper_model = None
        self.conversation_history = []
        self.session_start_time = None
        self.current_audio_buffer = []
        self.silence_threshold = 3.0  # seconds
        self.last_audio_time = 0
        self.processing_lock = threading.Lock()
        
    def initialize_whisper(self):
        """Initialize Whisper model"""
        try:
            logger.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return False
    
    def start_conversation(self):
        """Start a new conversation"""
        if self.state != ConversationState.IDLE:
            return False
        
        if not self.whisper_model:
            if not self.initialize_whisper():
                return False
        
        self.state = ConversationState.LISTENING
        self.session_start_time = time.time()
        self.conversation_history = []
        self.current_audio_buffer = []
        
        logger.info("Conversation started")
        return True
    
    def pause_conversation(self):
        """Pause the conversation"""
        if self.state == ConversationState.LISTENING:
            self.state = ConversationState.PAUSED
            return True
        return False
    
    def resume_conversation(self):
        """Resume the conversation"""
        if self.state == ConversationState.PAUSED:
            self.state = ConversationState.LISTENING
            return True
        return False
    
    def end_conversation(self):
        """End the conversation"""
        self.state = ConversationState.ENDED
        logger.info("Conversation ended")
        return self.get_conversation_summary()
    
    def process_audio_chunk(self, audio_data: bytes) -> Optional[WhisperChunk]:
        if self.state not in [ConversationState.LISTENING, ConversationState.PAUSED]:
            return None

        try:
            # 1. WebM/Opus → 16 kHz mono 16-bit PCM via FFmpeg
            out, _ = (
                ffmpeg
                .input('pipe:', format='webm')
                .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=16000)
                .run(input=audio_data, capture_stdout=True, quiet=True)
            )
            pcm = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

            # 2. Transcribe
            result = self.whisper_model.transcribe(pcm, language='en', fp16=False)
            text = result.get('text', '').strip()
            if text:
                socketio.emit('transcription', {'text': text})
                return WhisperChunk(timestamp=time.time(), text=text, confidence=0.9)

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg decode error: {e.stderr.decode()}")
            socketio.emit('error', {'message': 'Audio decode failed'})
        except Exception as e:
            logger.error(f"General error: {e}")
            socketio.emit('error', {'message': str(e)})

        return None

    def _is_complete_speech(self, text):
        """Determine if the speech segment is complete"""
        # Simple heuristics for speech completion
        indicators = [
            text.endswith('.'),
            text.endswith('?'),
            text.endswith('!'),
            len(text.split()) > 5,  # More than 5 words
            any(phrase in text.lower() for phrase in [
                'what do you think', 'how about', 'can you', 'what is', 'tell me'
            ])
        ]
        return any(indicators)
    
    def _process_complete_speech(self, text):
        """Process complete speech and generate response"""
        if self.state != ConversationState.LISTENING:
            return
        
        with self.processing_lock:
            self.state = ConversationState.PROCESSING
            socketio.emit('state_change', {'state': 'processing'})
            
            # Add user message to history
            user_message = {
                'role': 'user',
                'content': text,
                'timestamp': time.time()
            }
            self.conversation_history.append(user_message)
            
            # Emit user message to frontend
            socketio.emit('user_message', {
                'text': text,
                'timestamp': time.time()
            })
            
            # Generate response in a separate thread to avoid blocking
            threading.Thread(target=self._generate_response, args=(text,)).start()
    
    def _generate_response(self, user_text):
        """Generate LLM response"""
        try:
            self.state = ConversationState.RESPONDING
            socketio.emit('state_change', {'state': 'responding'})
            
            # Create context-aware prompt
            context = self._build_context()
            prompt = f"{context}\n\nUser: {user_text}\n\nAs a helpful AI assistant, please provide a thoughtful and engaging response:"
            
            # Generate response using LLM (this is async but we're in a thread)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(llm.ask_gemini(prompt))
            loop.close()
            
            # Add assistant message to history
            assistant_message = {
                'role': 'assistant',
                'content': response,
                'timestamp': time.time()
            }
            self.conversation_history.append(assistant_message)
            
            # Emit response to frontend
            socketio.emit('assistant_response', {
                'text': response,
                'timestamp': time.time()
            })
            
            # Return to listening state
            if self.state == ConversationState.RESPONDING:
                self.state = ConversationState.LISTENING
                socketio.emit('state_change', {'state': 'listening'})
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            socketio.emit('error', {'message': f'Response generation error: {str(e)}'})
            self.state = ConversationState.ERROR
            socketio.emit('state_change', {'state': 'error'})
    
    def _build_context(self):
        """Build conversation context for the LLM"""
        if not self.conversation_history:
            return "This is the start of a conversation."
        
        # Get last few exchanges for context
        recent_history = self.conversation_history[-4:]  # Last 4 messages
        context_parts = []
        
        for msg in recent_history:
            role = "Human" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "Recent conversation context:\n" + "\n".join(context_parts)
    
    def get_conversation_summary(self):
        """Get conversation summary"""
        if not self.session_start_time:
            return {"error": "No active session"}
        
        duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        return {
            "session_duration": duration,
            "total_exchanges": len([msg for msg in self.conversation_history if msg['role'] == 'user']),
            "conversation_history": self.conversation_history,
            "state": self.state.value
        }
    
    def get_status(self):
        """Get current conversation status"""
        return {
            "state": self.state.value,
            "session_active": self.session_start_time is not None,
            "total_messages": len(self.conversation_history),
            "duration": time.time() - self.session_start_time if self.session_start_time else 0
        }

# Global conversation manager
conversation_manager = ConversationManager()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

# SocketIO events
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'status': 'Connected to server'})
    emit('state_change', {'state': conversation_manager.state.value})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('start_conversation')
def handle_start_conversation():
    try:
        success = conversation_manager.start_conversation()
        if success:
            emit('conversation_started', {'status': 'success'})
            emit('state_change', {'state': 'listening'})
        else:
            emit('error', {'message': 'Failed to start conversation'})
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        emit('error', {'message': str(e)})

@socketio.on('pause_conversation')
def handle_pause_conversation():
    try:
        success = conversation_manager.pause_conversation()
        if success:
            emit('conversation_paused', {'status': 'success'})
            emit('state_change', {'state': 'paused'})
        else:
            emit('error', {'message': 'Cannot pause conversation'})
    except Exception as e:
        logger.error(f"Error pausing conversation: {e}")
        emit('error', {'message': str(e)})

@socketio.on('resume_conversation')
def handle_resume_conversation():
    try:
        success = conversation_manager.resume_conversation()
        if success:
            emit('conversation_resumed', {'status': 'success'})
            emit('state_change', {'state': 'listening'})
        else:
            emit('error', {'message': 'Cannot resume conversation'})
    except Exception as e:
        logger.error(f"Error resuming conversation: {e}")
        emit('error', {'message': str(e)})

@socketio.on('end_conversation')
def handle_end_conversation():
    try:
        summary = conversation_manager.end_conversation()
        emit('conversation_ended', summary)
        emit('state_change', {'state': 'ended'})
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        emit('error', {'message': str(e)})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    try:
        if 'audio' not in data:
            emit('error', {'message': 'No audio data received'})
            return
        
        # Decode base64 audio data
        audio_data = base64.b64decode(data['audio'])
        
        # Process the audio chunk
        result = conversation_manager.process_audio_chunk(audio_data)
        
        if result is None:
            # Audio processing failed or not ready
            pass
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        emit('error', {'message': f'Audio processing error: {str(e)}'})

@socketio.on('get_status')
def handle_get_status():
    try:
        status = conversation_manager.get_status()
        emit('status_update', status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        emit('error', {'message': str(e)})

@socketio.on('clear_conversation')
def handle_clear_conversation():
    try:
        conversation_manager.conversation_history = []
        emit('conversation_cleared', {'status': 'success'})
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    # Initialize Whisper model on startup
    logger.info("Initializing conversation system...")
    
    try:
        # Pre-load Whisper model
        conversation_manager.initialize_whisper()
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"System initialization error: {e}")
    
    # Run the Flask-SocketIO app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)