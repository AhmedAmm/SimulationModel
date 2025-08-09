import time
import re
import threading
import queue
from typing import Optional, Callable
from collections import deque
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
    IDLE = "idle"

@dataclass
class SentenceFragment:
    text: str
    timestamp: float
    confidence: float = 1.0

class ConversationalOrchestrator:
    def __init__(self, 
                 llm_client: RotatingGeminiClient,
                 sentence_timeout: float = 6.0,
                 min_sentence_length: int = 5,
                 conversation_context_limit: int = 20,
                 extended_timeout: float = 12.0):
        """
        Initialize the conversational orchestrator.
        
        Args:
            llm_client: RotatingGeminiClient instance
            sentence_timeout: Time to wait before considering sentence complete (seconds)
            min_sentence_length: Minimum length to consider as a complete sentence
            conversation_context_limit: Number of recent exchanges to keep in context
            extended_timeout: Extended timeout for complex queries before forcing completion
        """
        self.llm_client = llm_client
        self.sentence_timeout = sentence_timeout
        self.extended_timeout = extended_timeout
        self.min_sentence_length = min_sentence_length
        self.conversation_context_limit = conversation_context_limit
        
        # Conversation state
        self.state = ConversationState.IDLE
        self.current_sentence = ""
        self.sentence_buffer = deque(maxlen=100)
        self.last_fragment_time = 0.0
        self.sentence_start_time = 0.0
        
        # Enhanced sentence tracking
        self.accumulated_sentence = ""
        self.last_sent_sentence = ""
        
        # Conversation history for context
        self.conversation_history = deque(maxlen=conversation_context_limit * 2)
        
        # Threading and processing
        self.processing_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None
        
        # Add reference to transcriber for resetting
        self.transcriber = None
        
        # Enhanced patterns
        self.sentence_endings = re.compile(r'[.!?]+\s*$')
        self.question_patterns = re.compile(r'\b(what|how|when|where|why|who|is|are|can|could|would|will|do|does|solve|find|calculate|explain|tell|show)\b', re.IGNORECASE)
        self.trigger_phrases = re.compile(r'\b(hey|hello|assistant|excuse me|listen|please|help)\b', re.IGNORECASE)
        self.conversation_enders = re.compile(r'\b(goodbye|bye|see you|talk later|end|stop|quit|exit)\b', re.IGNORECASE)
        
        # Math and code patterns
        self.math_patterns = re.compile(r'\b(equation|equal|equals|plus|minus|multiply|divide|solve|find|x|y|z)\b', re.IGNORECASE)
        self.code_patterns = re.compile(r'\b(code|function|class|variable|python|javascript|html|css)\b', re.IGNORECASE)
        self.complex_query_patterns = re.compile(r'\b(explain|describe|analyze|compare|create|build|write|implement)\b', re.IGNORECASE)
    
    def set_transcriber(self, transcriber: StreamingWhisperTranscriber):
        """Set reference to transcriber for resetting"""
        self.transcriber = transcriber
        logger.info("Transcriber set for orchestrator")
        
    def start(self):
        """Start the orchestrator"""
        if self.is_running:
            logger.warning("Orchestrator already running")
            return
        
        self.is_running = True
        self.state = ConversationState.LISTENING
        
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("🤖 Conversational Orchestrator started")
        print("💬 Ready for conversation! Say something...")
    
    def stop(self):
        """Stop the orchestrator"""
        self.is_running = False
        self.state = ConversationState.IDLE
        
        self.processing_queue.put(None)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("🛑 Conversational Orchestrator stopped")
    
    def _is_duplicate(self, sentence: str) -> bool:
        """
        Determine if a sentence is a near-duplicate of the last one.
        Allows small differences (e.g., punctuation, minor rewording).
        """
        if not self.last_sent_sentence:
            return False

        if sentence.strip() == self.last_sent_sentence.strip():
            return True

        simplified_new = re.sub(r'[^\w\s]', '', sentence.lower())
        simplified_last = re.sub(r'[^\w\s]', '', self.last_sent_sentence.lower())
        return simplified_new.startswith(simplified_last) or simplified_last.startswith(simplified_new)

    def response_callback(self, new_text: str):
        """
        Enhanced callback with duplicate prevention and WebSocket integration
        """
        if not self.is_running or self.state != ConversationState.LISTENING:
            logger.debug("Skipping callback: orchestrator not running or not listening")
            return

        current_time = time.time()
        
        if hasattr(self, '_last_processed_text') and new_text.strip() == self._last_processed_text.strip():
            logger.debug("🔄 Skipping duplicate text")
            return
        
        self._last_processed_text = new_text
        
        cleaned_text = self._clean_text(new_text)
        if not cleaned_text:
            logger.debug("Empty or invalid text after cleaning, skipping")
            return
        
        fragment = SentenceFragment(text=cleaned_text, timestamp=current_time)
        self.sentence_buffer.append(fragment)
        
        self.current_sentence = self._rebuild_current_sentence()
        self.last_fragment_time = current_time
        
        print(f"\r🎙️  {self.current_sentence}", end="", flush=True)
        
        completion_result = self._check_completion_conditions()
        if completion_result['should_complete']:
            sentence_to_send = completion_result['sentence']
            if sentence_to_send and not self._is_duplicate(sentence_to_send):
                self.processing_queue.put(sentence_to_send)
                self.last_sent_sentence = sentence_to_send
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        text = text.strip()
        artifacts = ["", " ", ".", "...", "thank you", "thanks for watching", "you", "um", "uh", "ah", "hmm"]
        if text.lower() in artifacts:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _rebuild_current_sentence(self) -> str:
        """Enhanced sentence rebuilding with better context preservation"""
        if not self.sentence_buffer:
            return ""
        
        current_time = time.time()
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
        
        current_clean = self.current_sentence.strip()
        accumulated_clean = self.accumulated_sentence.strip()
        
        if len(current_clean) > len(accumulated_clean):
            self.accumulated_sentence = current_clean
        elif current_clean and not accumulated_clean.endswith(current_clean):
            if len(current_clean) > 3:
                self.accumulated_sentence = current_clean
    
    def _check_completion_conditions(self) -> dict:
        """
        Enhanced completion checking with multiple strategies.
        Returns dict with completion decision and sentence to send.
        """
        if not self.current_sentence:
            return {'should_complete': False, 'sentence': '', 'reset_accumulated': False}
        
        sentence = self.current_sentence.strip()
        accumulated = self.accumulated_sentence.strip()
        current_time = time.time()
        
        sentence_to_check = accumulated if len(accumulated) > len(sentence) else sentence
        
        if len(sentence_to_check) < self.min_sentence_length:
            return {'should_complete': False, 'sentence': '', 'reset_accumulated': False}
        
        if self.sentence_endings.search(sentence_to_check):
            return {
                'should_complete': True, 
                'sentence': sentence_to_check,
                'reset_accumulated': True
            }
        
        if self.conversation_enders.search(sentence_to_check):
            return {
                'should_complete': True, 
                'sentence': sentence_to_check,
                'reset_accumulated': True
            }
        
        if self._is_question(sentence_to_check):
            timeout = self.sentence_timeout * 0.8
            if current_time - self.last_fragment_time > timeout:
                return {
                    'should_complete': True, 
                    'sentence': sentence_to_check,
                    'reset_accumulated': True
                }
        
        if self._is_complex_query(sentence_to_check):
            time_since_start = current_time - self.sentence_start_time
            silence_time = current_time - self.last_fragment_time
            if time_since_start > self.extended_timeout or silence_time > self.sentence_timeout * 1.5:
                return {
                    'should_complete': True, 
                    'sentence': sentence_to_check,
                    'reset_accumulated': True
                }
        
        if self.trigger_phrases.search(sentence_to_check):
            if current_time - self.last_fragment_time > self.sentence_timeout * 0.7:
                return {
                    'should_complete': True, 
                    'sentence': sentence_to_check,
                    'reset_accumulated': True
                }
        
        silence_duration = current_time - self.last_fragment_time
        if silence_duration > self.sentence_timeout:
            if len(accumulated) > len(sentence) + 5:
                return {
                    'should_complete': True, 
                    'sentence': accumulated,
                    'reset_accumulated': True
                }
            elif len(sentence_to_check) >= self.min_sentence_length:
                return {
                    'should_complete': True, 
                    'sentence': sentence_to_check,
                    'reset_accumulated': False
                }
        
        return {'should_complete': False, 'sentence': '', 'reset_accumulated': False}
    
    def _is_question(self, sentence: str) -> bool:
        """Enhanced question detection"""
        sentence_lower = sentence.lower().strip()
        question_starters = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 
                           'is', 'are', 'can', 'could', 'would', 'will', 'do', 'does', 'did']
        first_word = sentence_lower.split()[0] if sentence_lower.split() else ""
        if first_word in question_starters:
            return True
        if self.question_patterns.search(sentence):
            return True
        return False
    
    def _is_complex_query(self, sentence: str) -> bool:
        """Detect complex queries that might need longer processing time"""
        if not sentence:
            return False
        if self.math_patterns.search(sentence):
            return True
        if self.code_patterns.search(sentence):
            return True
        if self.complex_query_patterns.search(sentence):
            return True
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
                
                if self.conversation_enders.search(sentence):
                    response = "Goodbye! It was nice talking with you."
                    self._handle_response(response, sentence)
                    self.stop()
                    break
                
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
                    self.state = ConversationState.LISTENING
                    self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def _generate_response(self, user_input: str) -> str:
        """Enhanced response generation with better context"""
        context = self._build_conversation_context()
        if self._is_complex_query(user_input):
            instruction = "The user has asked a detailed question that may involve math, code, or complex concepts. Please provide a thorough, step-by-step response."
        elif self._is_question(user_input):
            instruction = "The user has asked a question. Please provide a clear, direct answer in two lines."
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
        for entry in list(self.conversation_history)[-8:]:
            context_lines.append(entry)
        
        return "\n".join(context_lines) if context_lines else "No previous conversation."
    
    def _handle_response(self, response: str, user_input: str):
        """Enhanced response handling with WebSocket integration"""
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")
        
        print(f"🤖 Assistant: {response}")
        
        # Send both transcription and response to transcriber
        if self.transcriber:
            try:
                self.transcriber.send_transcription_to_ws({
                    "transcription": user_input,
                    "response": response
                })
                logger.info(f"Sent transcription and response to transcriber: {user_input} -> {response}")
            except Exception as e:
                logger.error(f"Failed to send to transcriber: {type(e).__name__}: {str(e)}")
        
        self._complete_reset_after_response()
        
    def _complete_reset_after_response(self):
        """Complete state reset after successful response"""
        self.current_sentence = ""
        self.sentence_buffer.clear()
        self.accumulated_sentence = ""
        self.last_sent_sentence = ""
        self.last_fragment_time = 0.0
        self.sentence_start_time = 0.0
        
        if hasattr(self, 'transcriber') and self.transcriber:
            try:
                self.transcriber.reset_for_new_conversation()
                logger.info("🔄 Transcriber state reset")
            except Exception as e:
                logger.warning(f"Could not reset transcriber: {e}")
        
        self.state = ConversationState.LISTENING
        print("\n💬 Ready for next question...")
    
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
            'processing_queue_size': self.processing_queue.qsize()
        }

def setup_integrated_conversation_system(
    whisper_model_name: str = "base",
    sentence_timeout: float = 6.0,
    extended_timeout: float = 12.0
) -> tuple[StreamingWhisperTranscriber, ConversationalOrchestrator]:
    """
    Setup integrated conversation system.
    
    Args:
        whisper_model_name: Whisper model to use
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
        sentence_timeout=sentence_timeout,
        extended_timeout=extended_timeout
    )
    
    transcriber = StreamingWhisperTranscriber(
        model_name=whisper_model_name,
        input_source="websocket"
    )
    
    orchestrator.set_transcriber(transcriber)
    transcriber.set_orchestrator(orchestrator)
    
    return transcriber, orchestrator

def test_enhanced_orchestrator():
    """Test function for the orchestrator"""
    print("🧪 Testing Conversational Orchestrator...")
    print("=" * 60)
    
    try:
        llm_client = RotatingGeminiClient(base_cooldown=60)
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {e}")
        return
    
    orchestrator = ConversationalOrchestrator(
        llm_client=llm_client,
        sentence_timeout=6.0,
        extended_timeout=12.0,
        min_sentence_length=3
    )
    
    orchestrator.start()
    
    test_scenarios = [
        {
            "name": "Math Problem - Fragmented",
            "fragments": ["I have", " x plus", " 2 equal", " to 5", ", find x"],
            "expected": "Math equation solving"
        },
        {
            "name": "Complex Question",
            "fragments": ["Can you", " explain how", " to solve", " quadratic", " equations?"],
            "expected": "Educational explanation"
        },
        {
            "name": "Code Request",
            "fragments": ["Write a", " python function", " that calculates", " factorial"],
            "expected": "Code generation"
        },
        {
            "name": "Interrupted Speech",
            "fragments": ["Hello can you", " answer me about", " the question", "?", " Question I have"],
            "expected": "Proper handling of interruptions"
        }
    ]
    
    def simulate_speech_fragments(fragments, delay=0.4):
        for fragment in fragments:
            time.sleep(delay)
            orchestrator.response_callback(fragment)
        time.sleep(orchestrator.sentence_timeout + 1.0)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔍 Test {i}: {scenario['name']}")
        print("-" * 40)
        
        simulate_speech_fragments(scenario['fragments'])
        time.sleep(3.0)
        
        status = orchestrator.get_status()
        print(f"✅ Test completed. Accumulated: '{status['accumulated_sentence']}'")
    
    print(f"\n📊 Final Status: {orchestrator.get_status()}")
    print(f"📜 Conversation History:")
    for entry in orchestrator.get_conversation_history():
        print(f"  {entry}")
    
    orchestrator.stop()
    print("\n✅ Test completed!")

if __name__ == "__main__":
    try:
        transcriber, orchestrator = setup_integrated_conversation_system(
            whisper_model_name="base",
            sentence_timeout=6.0,
            extended_timeout=12.0
        )
        orchestrator.start()
        transcriber.start_transcription()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down conversation system...")
        orchestrator.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
