import time
import re
import threading
import queue
from typing import Optional, Callable
from collections import deque
import pyttsx3
import logging
from dataclasses import dataclass
from enum import Enum

# Import your existing modules
from auxx import StreamingWhisperTranscriber
from Llm import RotatingGeminiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    LISTENING = "listening"
    PROCESSING = "processing" 
    SPEAKING = "speaking"
    IDLE = "idle"


@dataclass
class SentenceFragment:
    text: str
    timestamp: float
    confidence: float = 1.0


class ConversationalOrchestrator:
    def __init__(self, 
                 llm_client: RotatingGeminiClient,
                 use_tts: bool = True,
                 sentence_timeout: float = 6.0,
                 min_sentence_length: int = 5,
                 conversation_context_limit: int = 20,
                 extended_timeout: float = 12.0):
        """
        Initialize the conversational orchestrator.
        
        Args:
            llm_client: RotatingGeminiClient instance
            use_tts: Whether to use text-to-speech
            sentence_timeout: Time to wait before considering sentence complete (seconds)
            min_sentence_length: Minimum length to consider as a complete sentence
            conversation_context_limit: Number of recent exchanges to keep in context
            extended_timeout: Extended timeout for complex queries before forcing completion
        """
        self.llm_client = llm_client
        self.use_tts = use_tts
        self.sentence_timeout = sentence_timeout
        self.extended_timeout = extended_timeout
        self.min_sentence_length = min_sentence_length
        self.conversation_context_limit = conversation_context_limit
        
        # Conversation state
        self.state = ConversationState.IDLE
        self.current_sentence = ""
        self.sentence_buffer = deque(maxlen=100)  # Larger buffer for complex queries
        self.last_fragment_time = 0.0
        self.sentence_start_time = 0.0
        
        # Enhanced sentence tracking
        self.accumulated_sentence = ""  # Keeps building even after timeouts
        self.last_sent_sentence = ""    # Track what was last sent to avoid duplicates
        self.last_user_sentence = ""
        
        # Conversation history for context
        self.conversation_history = deque(maxlen=conversation_context_limit * 2)
        
        # Threading and processing
        self.processing_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None
        self.tts_thread = None
        
        # Add reference to transcriber for resetting
        self.transcriber = None
        
        # Initialize TTS
        if self.use_tts:
            try:
                self.tts_engine = pyttsx3.init()
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 0.8)
                logger.info("TTS engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                self.use_tts = False
        
        # Enhanced patterns
        self.sentence_endings = re.compile(r'[.!?]+\s*$')
        self.question_patterns = re.compile(r'\b(what|how|when|where|why|who|is|are|can|could|would|will|do|does|solve|find|calculate|explain|tell|show)\b', re.IGNORECASE)
        self.trigger_phrases = re.compile(r'\b(hey|hello|assistant|excuse me|listen|please|help)\b', re.IGNORECASE)
        self.conversation_enders = re.compile(r'\b(goodbye|bye|see you|talk later|end|stop|quit|exit)\b', re.IGNORECASE)
        
        # Math and code patterns - these need special handling
        self.math_patterns = re.compile(r'\b(equation|equal|equals|plus|minus|multiply|divide|solve|find|x|y|z)\b', re.IGNORECASE)
        self.code_patterns = re.compile(r'\b(code|function|class|variable|python|javascript|html|css)\b', re.IGNORECASE)
        self.complex_query_patterns = re.compile(r'\b(explain|describe|analyze|compare|create|build|write|implement)\b', re.IGNORECASE)
    
    def set_transcriber(self, transcriber: StreamingWhisperTranscriber):
        """Set reference to transcriber for resetting"""
        self.transcriber = transcriber
        
    def start(self):
        """Start the orchestrator"""
        if self.is_running:
            logger.warning("Orchestrator already running")
            return
        
        self.is_running = True
        self.state = ConversationState.LISTENING
        
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        if self.use_tts:
            self.tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
            self.tts_thread.start()
        
        logger.info("🤖 Conversational Orchestrator started")
        print("💬 Ready for conversation! Say something...")
    
    def stop(self):
        """Stop the orchestrator"""
        self.is_running = False
        self.state = ConversationState.IDLE
        
        self.processing_queue.put(None)
        if self.use_tts:
            self.response_queue.put(None)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)
        
        logger.info("🛑 Conversational Orchestrator stopped")
    
    def _is_duplicate(self, sentence: str) -> bool:
        """
        Determine if a sentence is a near-duplicate of the last one.
        Allows small differences (e.g., punctuation, minor rewording).
        """
        if not self.last_sent_sentence:
            return False

        # Exact match
        if sentence.strip() == self.last_sent_sentence.strip():
            return True

        # Ignore case and punctuation for comparison
        simplified_new = re.sub(r'[^\w\s]', '', sentence.lower())
        simplified_last = re.sub(r'[^\w\s]', '', self.last_sent_sentence.lower())

        # If similarity is high, treat as duplicate
        return simplified_new.startswith(simplified_last) or simplified_last.startswith(simplified_new)

    def response_callback(self, new_text: str):
        if not self.is_running or self.state != ConversationState.LISTENING:
            return

        current_time = time.time()

        if hasattr(self, '_last_processed_text') and new_text.strip() == self._last_processed_text.strip():
            return

        cleaned_text = self._clean_text(new_text)
        if not cleaned_text:
            return
        if cleaned_text.strip().lower() == self.last_user_sentence.strip().lower():
            logger.info("🛑 Ignored repeated user message.")
            return
        self._last_processed_text = new_text

        # 🚫 REMOVE: self._reset_current_sentence(reset_accumulated=True)

        fragment = SentenceFragment(text=cleaned_text, timestamp=current_time)
        self.sentence_buffer.append(fragment)

        self.current_sentence = self._rebuild_current_sentence()
        self._update_accumulated_sentence()
        self.last_fragment_time = current_time

        print(f"\r🎙️  {self.accumulated_sentence}", end="", flush=True)

        completion_result = self._check_completion_conditions()
        if completion_result['should_complete']:
            sentence_to_send = self.accumulated_sentence
            if sentence_to_send and not self._is_duplicate(sentence_to_send):
                self.processing_queue.put(sentence_to_send)
                self.last_sent_sentence = sentence_to_send
                self.last_user_sentence = sentence_to_send
            self._reset_current_sentence(reset_accumulated=True)  # reset only after sending

    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        text = text.strip()
        
        # More comprehensive artifact removal
        artifacts = ["", " ", ".", "...", "thank you", "thanks for watching", "you", "um", "uh", "ah", "hmm"]
        if text.lower() in artifacts:
            return ""
        
        # Remove excessive whitespace but preserve single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Don't remove numbers or mathematical symbols
        return text
    
    def _rebuild_current_sentence(self) -> str:
        """Enhanced sentence rebuilding with better context preservation"""
        if not self.sentence_buffer:
            return ""
        
        current_time = time.time()
        # Use longer window for complex queries (15 seconds instead of 10)
        window_time = 15.0 if self._is_complex_query(self.accumulated_sentence) else 10.0
        
        recent_fragments = [
            f for f in self.sentence_buffer 
            if current_time - f.timestamp < window_time
        ]
        
        if not recent_fragments:
            return ""
        
        sentence = " ".join(f.text for f in recent_fragments)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        return sentence
    
    def _update_accumulated_sentence(self):
        """Update the accumulated sentence with new content"""
        if not self.current_sentence:
            return
        
        # If current sentence extends beyond what we had accumulated, update it
        current_clean = self.current_sentence.strip()
        accumulated_clean = self.accumulated_sentence.strip()
        
        if len(current_clean) > len(accumulated_clean):
            self.accumulated_sentence = current_clean
        elif current_clean and not accumulated_clean.endswith(current_clean):
            # Handle cases where speech recognition corrects itself
            if len(current_clean) > 3:  # Only update with substantial content
                self.accumulated_sentence = current_clean
    
    def _check_completion_conditions(self) -> dict:
        """
        Only complete when silence is detected for sentence_timeout seconds.
        """
        if not self.current_sentence:
            return {'should_complete': False, 'sentence': '', 'reset_accumulated': False}

        sentence = self.current_sentence.strip()
        accumulated = self.accumulated_sentence.strip()
        current_time = time.time()

        silence_duration = current_time - self.last_fragment_time
        if silence_duration > self.sentence_timeout:
            sentence_to_send = accumulated if len(accumulated) > len(sentence) else sentence
            if len(sentence_to_send) >= self.min_sentence_length:
                return {
                    'should_complete': True,
                    'sentence': sentence_to_send,
                    'reset_accumulated': True
                }
        return {'should_complete': False, 'sentence': '', 'reset_accumulated': False}
    
    def _is_question(self, sentence: str) -> bool:
        """Enhanced question detection"""
        sentence_lower = sentence.lower().strip()
        
        # Direct question words at start
        question_starters = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 
                           'is', 'are', 'can', 'could', 'would', 'will', 'do', 'does', 'did']
        
        first_word = sentence_lower.split()[0] if sentence_lower.split() else ""
        if first_word in question_starters:
            return True
        
        # Question patterns anywhere in sentence
        if self.question_patterns.search(sentence):
            return True
        
        return False
    
    def _is_complex_query(self, sentence: str) -> bool:
        """Detect complex queries that might need longer processing time"""
        if not sentence:
            return False
        
        # Math problems
        if self.math_patterns.search(sentence):
            return True
        
        # Code-related queries
        if self.code_patterns.search(sentence):
            return True
        
        # Complex instruction patterns
        if self.complex_query_patterns.search(sentence):
            return True
        
        # Long sentences are often complex
        if len(sentence.split()) > 10:
            return True
        
        return False
    
    def _reset_current_sentence(self, reset_accumulated: bool = True):
        """Reset current sentence state"""
        self.current_sentence = ""
        self.sentence_buffer.clear()
        if reset_accumulated:
            self.accumulated_sentence = ""
        self.last_fragment_time = 0.0
        self.sentence_start_time = 0.0
    
    def _processing_loop(self):
        """Enhanced processing loop"""
        while self.is_running:
            try:
                sentence = self.processing_queue.get(timeout=1.0)
                if sentence is None:
                    break
                
                self.state = ConversationState.PROCESSING
                print(f"\n👤 User: {sentence}")
                
                # Check for conversation enders
                if self.conversation_enders.search(sentence):
                    response = "Goodbye! It was nice talking with you."
                    self._handle_response(response, sentence)
                    self.stop()
                    break
                
                # Generate response
                try:
                    response = self._generate_response(sentence)
                    if response:
                        self._handle_response(response, sentence)
                    else:
                        logger.warning("Received empty response from LLM")
                        
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    fallback_response = "I'm sorry, I'm having trouble processing that right now. Could you rephrase your question?"
                    self._handle_response(fallback_response, sentence)
                
                finally:
                    # CRITICAL FIX: Ensure we return to listening state
                    self.state = ConversationState.LISTENING
                    self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def _generate_response(self, user_input: str) -> str:
        """Enhanced response generation with better context"""
        context = self._build_conversation_context()
        
        # Determine response style based on query type
        if self._is_complex_query(user_input):
            instruction = "The user has asked a detailed question that may involve math, code, or complex concepts. Please provide a thorough, step-by-step response."
        elif self._is_question(user_input):
            instruction = "The user has asked a question. Please provide a clear, direct answer."
        else:
            instruction = "Please respond naturally and helpfully to the user's message."
        
        prompt = f"""You are a helpful, conversational AI assistant. {instruction}

Conversation history:
{context}

User: {user_input}

Response guidelines:
- Be conversational and natural
- For math problems, show your work step by step
- For code questions, provide clear explanations
- Keep responses concise unless detail is specifically needed
- If the input seems incomplete or unclear, ask for clarification"""
        
        response = self.llm_client.ask_gemini(prompt)
        return response.strip()
    
    def _build_conversation_context(self) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return "No previous conversation."
        
        context_lines = []
        for entry in list(self.conversation_history)[-8:]:  # More context for complex queries
            context_lines.append(entry)
        
        return "\n".join(context_lines) if context_lines else "No previous conversation."
    
    def _handle_response(self, response: str, user_input: str):
        """Enhanced response handling with proper state reset"""
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")
        
        print(f"🤖 Assistant: {response}")
        
        if self.use_tts:
            self.response_queue.put(response)
        
        # CRITICAL FIX: Complete reset after response AND reset transcriber
        self._complete_reset_after_response()
        
    def _complete_reset_after_response(self):
        """Complete state reset after successful response"""
        # Reset orchestrator state
        self.current_sentence = ""
        self.sentence_buffer.clear()
        self.accumulated_sentence = ""
        self.last_sent_sentence = ""
        self.last_fragment_time = 0.0
        self.sentence_start_time = 0.0
        
        # CRITICAL FIX: Check if transcriber exists and has reset method
        if hasattr(self, 'transcriber') and self.transcriber:
            try:
                self.transcriber.reset_for_new_conversation()
                logger.info("🔄 Transcriber state reset")
            except Exception as e:
                logger.warning(f"Could not reset transcriber: {e}")
        
        # Ensure we're back to listening state
        self.state = ConversationState.LISTENING
        print("\n💬 Ready for next question...")
    
    def _tts_loop(self):
        """Text-to-speech loop"""
        while self.is_running:
            try:
                response = self.response_queue.get(timeout=1.0)
                if response is None:
                    break

                self.state = ConversationState.SPEAKING

                try:
                    logger.info("🔊 Speaking...")
                    self.tts_engine.say(response)
                    self.tts_engine.runAndWait()
                    logger.info("✅ Finished speaking")
                except Exception as e:
                    logger.error(f"TTS error: {e}")

                finally:
                    self.response_queue.task_done()
                    self.resume_listening()  # <-- Add this line

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS loop: {e}")
    
    def get_conversation_history(self) -> list:
        """Get the current conversation history"""
        return list(self.conversation_history)
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.accumulated_sentence = ""
        self.last_sent_sentence = ""
        print("🧹 Conversation history cleared")
    
    def get_status(self) -> dict:
        """Get current orchestrator status"""
        return {
            'state': self.state.value,
            'is_running': self.is_running,
            'current_sentence': self.current_sentence,
            'accumulated_sentence': self.accumulated_sentence,
            'conversation_length': len(self.conversation_history),
            'use_tts': self.use_tts,
            'processing_queue_size': self.processing_queue.qsize(),
            'response_queue_size': self.response_queue.qsize() if self.use_tts else 0
        }
    def pause_listening(self):
        """Pause listening (ignore incoming speech fragments) but keep context."""
        self.state = ConversationState.IDLE
        logger.info("Listening paused. Context preserved.")

    def resume_listening(self):
        """Resume listening and continue with preserved context."""
        self.state = ConversationState.LISTENING
        logger.info("Listening resumed.")
    
    def notify_silence(self):
        if time.time() - self.last_fragment_time > self.sentence_timeout:
            if self.current_sentence and len(self.current_sentence) >= self.min_sentence_length:
                # ✅ Prevent resending same sentence
                if not self._is_duplicate(self.current_sentence):
                    self.processing_queue.put(self.current_sentence)
                    self.last_sent_sentence = self.current_sentence



# FIXED: Enhanced integration function
def setup_integrated_conversation_system(
    whisper_model_name: str = "base",
    use_tts: bool = True,
    sentence_timeout: float = 6.0,
    extended_timeout: float = 12.0
) -> tuple[StreamingWhisperTranscriber, ConversationalOrchestrator]:
    """
    Setup enhanced integrated conversation system.
    
    Args:
        whisper_model_name: Whisper model to use
        use_tts: Whether to use text-to-speech
        sentence_timeout: Standard timeout for sentence completion
        extended_timeout: Extended timeout for complex queries
    
    Returns:
        tuple: (transcriber, orchestrator)
    """
    llm_client = RotatingGeminiClient(
        base_cooldown=60,
        exponential_backoff=True
    )
    
    orchestrator = ConversationalOrchestrator(
        llm_client=llm_client,
        use_tts=use_tts,
        sentence_timeout=sentence_timeout,
        extended_timeout=extended_timeout
    )
    
    transcriber = StreamingWhisperTranscriber(
        model_name=whisper_model_name,
        input_source="mic"
    )
    
    # CRITICAL FIX: Set transcriber reference in orchestrator
    orchestrator.set_transcriber(transcriber)
    
    # Enhanced monkey patch with better integration
    original_update_display = transcriber._update_streaming_display
    
    def enhanced_update_display(new_text):
        # Only update display if we're listening
        if orchestrator.state == ConversationState.LISTENING:
            original_update_display(new_text)
            orchestrator.response_callback(new_text)
    
    transcriber._update_streaming_display = enhanced_update_display
    
    return transcriber, orchestrator

if __name__ == "__main__":
    try:
        transcriber, orchestrator = setup_integrated_conversation_system(
            whisper_model_name="base",
            use_tts=True,
            sentence_timeout=4.0,
            extended_timeout=12.0
        )
        transcriber.orchestrator = orchestrator  
        orchestrator.start()
        transcriber.start_transcription()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down conversation system...")
        orchestrator.stop()
    except Exception as e:
        print(f"❌ Error: {e}")