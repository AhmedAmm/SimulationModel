import time
import re
import threading
import queue
from typing import Optional, Callable, Tuple
from collections import deque
import pyttsx3
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

# Import existing modules
from auxx import StreamingWhisperTranscriber
from agent import ClientSimulationAgent  # ✅ Using our updated agent

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
    def __init__(
        self,
        agent: ClientSimulationAgent,
        use_tts: bool = True,
        sentence_timeout: float = 4.0,
        min_sentence_length: int = 5,
        conversation_context_limit: int = 20,
        extended_timeout: float = 12.0,
    ):
        """
        Initialize the conversational orchestrator.

        Args:
            agent: ClientSimulationAgent instance
            use_tts: Whether to use text-to-speech
            sentence_timeout: Time to wait before considering sentence complete (seconds)
            min_sentence_length: Minimum length to consider as a complete sentence
            conversation_context_limit: Number of recent exchanges to keep in context
            extended_timeout: Extended timeout for complex queries before forcing completion
        """
        self.agent = agent
        self.use_tts = use_tts
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

        # Sentence tracking
        self.accumulated_sentence = ""
        self.last_sent_sentence = ""
        self.last_user_sentence = ""

        # Conversation history (for orchestrator only)
        self.conversation_history = deque(maxlen=conversation_context_limit * 2)

        # Threading primitives
        self.processing_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.tts_thread: Optional[threading.Thread] = None

        # Transcriber ref
        self.transcriber: Optional[StreamingWhisperTranscriber] = None

        # Init TTS
        if self.use_tts:
            try:
                self.tts_engine = pyttsx3.init()
                voices = self.tts_engine.getProperty("voices")
                if voices:
                    self.tts_engine.setProperty("voice", voices[0].id)
                self.tts_engine.setProperty("rate", 180)
                self.tts_engine.setProperty("volume", 0.8)
                logger.info("TTS engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                self.use_tts = False

        # Patterns
        self.sentence_endings = re.compile(r"[.!?]+\s*$")
        self.question_patterns = re.compile(
            r"\b(what|how|when|where|why|who|is|are|can|could|would|will|do|does|solve|find|calculate|explain|tell|show)\b",
            re.IGNORECASE,
        )
        self.trigger_phrases = re.compile(
            r"\b(hey|hello|assistant|excuse me|listen|please|help)\b", re.IGNORECASE
        )
        self.conversation_enders = re.compile(
            r"\b(goodbye|bye|see you|talk later|end|stop|quit|exit)\b", re.IGNORECASE
        )

        self.math_patterns = re.compile(
            r"\b(equation|equal|equals|plus|minus|multiply|divide|solve|find|x|y|z)\b",
            re.IGNORECASE,
        )
        self.code_patterns = re.compile(
            r"\b(code|function|class|variable|python|javascript|html|css)\b", re.IGNORECASE
        )
        self.complex_query_patterns = re.compile(
            r"\b(explain|describe|analyze|compare|create|build|write|implement)\b",
            re.IGNORECASE,
        )

    def set_transcriber(self, transcriber: StreamingWhisperTranscriber):
        self.transcriber = transcriber

    def start(self):
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
        self.is_running = False
        self.state = ConversationState.IDLE

        # Signal threads to stop
        self.processing_queue.put(None)
        if self.use_tts:
            self.response_queue.put(None)

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)

        logger.info("🛑 Conversational Orchestrator stopped")

    def _is_duplicate(self, sentence: str) -> bool:
        if not self.last_sent_sentence:
            return False
        if sentence.strip() == self.last_sent_sentence.strip():
            return True
        simplified_new = re.sub(r"[^\w\s]", "", sentence.lower())
        simplified_last = re.sub(r"[^\w\s]", "", self.last_sent_sentence.lower())
        return simplified_new.startswith(simplified_last) or simplified_last.startswith(simplified_new)

    def response_callback(self, new_text: str):
        if not self.is_running or self.state != ConversationState.LISTENING:
            return
        current_time = time.time()
        if hasattr(self, "_last_processed_text") and new_text.strip() == self._last_processed_text.strip():
            return
        cleaned_text = self._clean_text(new_text)
        if not cleaned_text:
            return
        if cleaned_text.strip().lower() == self.last_user_sentence.strip().lower():
            logger.info("🛑 Ignored repeated user message.")
            return
        self._last_processed_text = new_text
        fragment = SentenceFragment(text=cleaned_text, timestamp=current_time)
        self.sentence_buffer.append(fragment)
        self.current_sentence = self._rebuild_current_sentence()
        self._update_accumulated_sentence()
        self.last_fragment_time = current_time
        print(f"\r🎙️  {self.accumulated_sentence}", end="", flush=True)
        completion_result = self._check_completion_conditions()
        if completion_result["should_complete"]:
            sentence_to_send = self.accumulated_sentence
            if sentence_to_send and not self._is_duplicate(sentence_to_send):
                self.processing_queue.put(sentence_to_send)
                self.last_sent_sentence = sentence_to_send
                self.last_user_sentence = sentence_to_send
            self._reset_current_sentence(reset_accumulated=True)

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        artifacts = ["", " ", ".", "...", "thank you", "thanks for watching", "you", "um", "uh", "ah", "hmm"]
        if text.lower() in artifacts:
            return ""
        return re.sub(r"\s+", " ", text)

    def _rebuild_current_sentence(self) -> str:
        if not self.sentence_buffer:
            return ""
        current_time = time.time()
        window_time = 15.0 if self._is_complex_query(self.accumulated_sentence) else 10.0
        recent_fragments = [f for f in self.sentence_buffer if current_time - f.timestamp < window_time]
        if not recent_fragments:
            return ""
        return re.sub(r"\s+", " ", " ".join(f.text for f in recent_fragments)).strip()

    def _update_accumulated_sentence(self):
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
        if not self.current_sentence:
            return {"should_complete": False, "sentence": "", "reset_accumulated": False}
        sentence = self.current_sentence.strip()
        accumulated = self.accumulated_sentence.strip()
        current_time = time.time()
        silence_duration = current_time - self.last_fragment_time
        if silence_duration > self.sentence_timeout:
            sentence_to_send = accumulated if len(accumulated) > len(sentence) else sentence
            if len(sentence_to_send) >= self.min_sentence_length:
                return {"should_complete": True, "sentence": sentence_to_send, "reset_accumulated": True}
        return {"should_complete": False, "sentence": "", "reset_accumulated": False}

    def _is_complex_query(self, sentence: str) -> bool:
        if not sentence:
            return False
        if self.math_patterns.search(sentence) or self.code_patterns.search(sentence) or self.complex_query_patterns.search(sentence):
            return True
        if len(sentence.split()) > 10:
            return True
        return False

    def _reset_current_sentence(self, reset_accumulated: bool = True):
        self.current_sentence = ""
        self.sentence_buffer.clear()
        if reset_accumulated:
            self.accumulated_sentence = ""
        self.last_fragment_time = 0.0
        self.sentence_start_time = 0.0

    def _processing_loop(self):
        while self.is_running:
            try:
                sentence = self.processing_queue.get(timeout=1.0)
                if sentence is None:
                    break
                self.state = ConversationState.PROCESSING
                print(f"\n👤 User: {sentence}")
                # If user ends conversation
                if self.conversation_enders.search(sentence):
                    response = "Goodbye! It was nice talking with you."
                    self._handle_response(response, sentence)
                    print("\n📜 Full conversation log (JSON):")
                    print(self.get_full_discussion_json())
                    self.stop()
                    break
                try:
                    response = self._generate_response(sentence)
                    if response:
                        self._handle_response(response, sentence)
                    else:
                        logger.warning("Received empty response from agent")
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    self._handle_response("I'm sorry, there was a problem generating a response.", sentence)
                finally:
                    self.state = ConversationState.LISTENING
                    self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")

    def _generate_response(self, user_input: str) -> str:
        """Use ClientSimulationAgent's async simulate_client_response"""
        # simulate_client_response is async and returns a string
        try:
            return asyncio.run(self.agent.simulate_client_response(user_input))
        except RuntimeError:
            # If an event loop is already running (e.g. in certain environments),
            # fall back to creating a new loop.
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.agent.simulate_client_response(user_input))
            finally:
                loop.close()

    def _handle_response(self, response: str, user_input: str):
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")
        print(f"🤖 Assistant: {response}")
        if self.use_tts:
            self.response_queue.put(response)
        self._complete_reset_after_response()

    def _complete_reset_after_response(self):
        self.current_sentence = ""
        self.sentence_buffer.clear()
        self.accumulated_sentence = ""
        self.last_sent_sentence = ""
        self.last_fragment_time = 0.0
        self.sentence_start_time = 0.0
        if hasattr(self, "transcriber") and self.transcriber:
            try:
                self.transcriber.reset_for_new_conversation()
                logger.info("🔄 Transcriber state reset")
            except Exception as e:
                logger.warning(f"Could not reset transcriber: {e}")
        self.state = ConversationState.LISTENING
        print("\n💬 Ready for next question...")

    def _tts_loop(self):
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
                    self.resume_listening()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS loop: {e}")

    def resume_listening(self):
        self.state = ConversationState.LISTENING
        logger.info("Listening resumed.")

    def notify_silence(self):
        if time.time() - self.last_fragment_time > self.sentence_timeout:
            if self.current_sentence and len(self.current_sentence) >= self.min_sentence_length:
                if not self._is_duplicate(self.current_sentence):
                    self.processing_queue.put(self.current_sentence)
                    self.last_sent_sentence = self.current_sentence

    # Public helper methods to access agent state:
    def get_agent_conversation_log(self):
        """Return the agent's raw conversation context (list of dicts)."""
        return self.agent.conversation_context

    def get_full_discussion_json(self) -> str:
        """
        Return the full discussion history in JSON format.
        Includes both user and client turns from the agent's conversation_context.
        """
        try:
            history = self.agent.conversation_context
            return json.dumps(history, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error converting discussion to JSON: {e}")
            return "[]"


def setup_integrated_conversation_system(
    agent: ClientSimulationAgent,
    whisper_model_name: str = "base",
    use_tts: bool = True,
    sentence_timeout: float = 4.0,
    extended_timeout: float = 12.0,
) -> Tuple[StreamingWhisperTranscriber, ConversationalOrchestrator]:
    orchestrator = ConversationalOrchestrator(
        agent=agent, use_tts=use_tts, sentence_timeout=sentence_timeout, extended_timeout=extended_timeout
    )
    transcriber = StreamingWhisperTranscriber(model_name=whisper_model_name, input_source="mic")
    orchestrator.set_transcriber(transcriber)

    # Hook the transcriber's streaming display update so orchestrator gets text fragments
    original_update_display = transcriber._update_streaming_display

    def enhanced_update_display(new_text: str):
        if orchestrator.state == ConversationState.LISTENING:
            original_update_display(new_text)
            orchestrator.response_callback(new_text)

    transcriber._update_streaming_display = enhanced_update_display
    return transcriber, orchestrator


if __name__ == "__main__":
    try:
        # Instantiate the agent (no persona args required for compatibility)
        agent = ClientSimulationAgent(fastapi_base_url="http://localhost:8000")

        # Optionally: initialize agent session before starting conversation.
        # Uncomment and modify if you want the agent to run initialize_session automatically:
        asyncio.run(agent.initialize_session(
            user_company="Orange",
            user_product="internet services",
            client_company="talan",
            client_role="CTO",
            client_industry="provide Services",
            company_size="Enterprise",
        ))

        transcriber, orchestrator = setup_integrated_conversation_system(
            agent=agent, whisper_model_name="base", use_tts=True, sentence_timeout=4.0, extended_timeout=12.0
        )
        transcriber.orchestrator = orchestrator
        orchestrator.start()
        transcriber.start_transcription()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down conversation system...")
        try:
            orchestrator.stop()
        except Exception:
            pass
    except Exception as e:
        print(f"❌ Error: {e}")
