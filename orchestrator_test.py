# orchestrator_test.py
import time
import re
import threading
import queue
import logging
import asyncio
from typing import Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import the transcriber type for hints (not circular at runtime if wired properly)
from auxx import StreamingWhisperTranscriber
from agent import ClientSimulationAgent  # ensure this module exists in your project


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
        sentence_timeout: float = 5.0,
        min_sentence_length: int = 5,
        conversation_context_limit: int = 20,
        extended_timeout: float = 5.0,
    ):
        self.agent = agent
        self.use_tts = use_tts
        self.sentence_timeout = sentence_timeout
        self.min_sentence_length = min_sentence_length
        self.extended_timeout = extended_timeout
        self.conversation_context_limit = conversation_context_limit

        # state
        self.state = ConversationState.IDLE
        self.current_sentence = ""
        self.sentence_buffer = deque(maxlen=200)
        self.last_fragment_time = time.time()
        self.sentence_start_time = time.time()

        self.accumulated_sentence = ""
        self.last_sent_sentence = ""
        self._last_processed_text = ""

        # conversation history
        self.conversation_history = deque(maxlen=conversation_context_limit * 2)

        # threading & processing
        self.processing_queue = queue.Queue()
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None

        # transcriber ref
        self.transcriber: Optional[StreamingWhisperTranscriber] = None

        # reentrancy guard for notify_silence
        self._is_handling_silence = False
        self._last_silence_notified_at = 0.0

        # patterns
        self.sentence_endings = re.compile(r"[.!?]+\s*$")
        self.question_patterns = re.compile(r"\b(what|how|when|where|why|who|is|are|can|could|would|will|do|does)\b", re.IGNORECASE)
        self.conversation_enders = re.compile(r"\b(goodbye|bye|see you|end|stop|quit|exit)\b", re.IGNORECASE)

    def set_transcriber(self, transcriber: StreamingWhisperTranscriber):
        self.transcriber = transcriber
        transcriber.orchestrator = self
        # sync thresholds if desired
        # transcriber.silence_seconds_threshold = self.sentence_timeout

    def start(self):
        if self.is_running:
            logger.warning("Orchestrator already running")
            return
        self.is_running = True
        self.state = ConversationState.LISTENING
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Conversational Orchestrator started. Ready for conversation!")

    def stop(self):
        """
        Stop the orchestrator. If called from the processing thread itself,
        skip joining the thread to avoid `RuntimeError: cannot join current thread`.
        """
        self.is_running = False
        try:
            # attempt to wake the processing loop if it's waiting
            self.processing_queue.put(None)
        except Exception:
            pass

        # Only join if we're not the processing thread
        try:
            if (
                self.processing_thread
                and self.processing_thread.is_alive()
                and threading.current_thread() is not self.processing_thread
            ):
                self.processing_thread.join(timeout=2.0)
        except RuntimeError:
            # defensive fallback: if join still fails, just log and move on
            logger.exception("Failed to join processing thread (current thread may be the same)")

        logger.info("Conversational Orchestrator stopped")


    def _processing_loop(self):
        logger.info("Processing loop started")
        while self.is_running:
            try:
                sentence = self.processing_queue.get(timeout=1.0)
                if sentence is None:
                    break
                self.state = ConversationState.PROCESSING
                logger.info(f"RECEIVED FOR PROCESSING: '{sentence}'")
                # handle conversation enders
                if self.conversation_enders.search(sentence):
                    response = "Goodbye! It was nice talking with you."
                    self._handle_response(response, sentence)
                    self.stop()
                    break
                # generate response via agent (agent.simulate_client_response is async)
                try:
                    response = self._generate_response(sentence)
                    if response:
                        self._handle_response(response, sentence)
                    else:
                        self._handle_response("I'm sorry, I didn't understand that. Could you repeat?", sentence)
                except Exception:
                    logger.exception("Error generating response")
                    self._handle_response("There was an error generating a response.", sentence)
                finally:
                    self.state = ConversationState.LISTENING
                    self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Error in orchestrator processing loop")
        logger.info("Processing loop ended")

    def _generate_response(self, user_input: str) -> str:
        try:
            return asyncio.run(self.agent.simulate_client_response(user_input))
        except RuntimeError:
            # fallback if event loop already running
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.agent.simulate_client_response(user_input))
            finally:
                loop.close()

    def _handle_response(self, response: str, user_input: str):
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")
        print(f"\n🤖 Assistant: {response}")
        # after responding, reset waiting timers so we don't self-trigger on silence
        self._complete_reset_after_response()

    def _complete_reset_after_response(self):
        self.current_sentence = ""
        self.sentence_buffer.clear()
        self.accumulated_sentence = ""
        self.last_sent_sentence = ""
        # mark last fragment time to now (prevent immediate false silence)
        self.last_fragment_time = time.time()
        self.sentence_start_time = time.time()
        # reset transcriber state if attached
        if self.transcriber:
            try:
                self.transcriber.reset_for_new_conversation()
            except Exception:
                logger.exception("Could not reset transcriber")
        self.state = ConversationState.LISTENING
        logger.info("State reset to LISTENING")

    def notify_silence(self):
        # idempotent guard
        if self._is_handling_silence:
            logger.debug("notify_silence ignored: already handling silence")
            return
        # small cooldown
        if time.time() - self._last_silence_notified_at < 0.5:
            logger.debug("notify_silence ignored: within cooldown")
            return

        self._is_handling_silence = True
        self._last_silence_notified_at = time.time()
        try:
            logger.info("Orchestrator received silence notification")
            current_time = time.time()
            silence_duration = current_time - self.last_fragment_time
            timeout = self.extended_timeout if self._is_complex_query(self.accumulated_sentence) else self.sentence_timeout
            logger.info(f"Silence duration: {silence_duration:.2f}s, required timeout: {timeout:.2f}s")
            if silence_duration >= timeout:
                sentence_to_process = self.accumulated_sentence if self.accumulated_sentence else self.current_sentence
                if sentence_to_process and len(sentence_to_process.strip()) >= self.min_sentence_length:
                    if not self._is_duplicate(sentence_to_process):
                        self.processing_queue.put(sentence_to_process)
                        self.last_sent_sentence = sentence_to_process
                        # reset current sentence (accumulated cleared)
                        self._reset_current_sentence(reset_accumulated=True)
                    else:
                        logger.info("Duplicate sentence ignored")
                else:
                    logger.info("Not enough content to process on silence")
            else:
                logger.info("Silence duration insufficient to trigger processing")
        finally:
            self._is_handling_silence = False

    # ---------- callbacks from transcriber ----------
    def response_callback(self, new_text: str):
        """Called by transcriber for partial fragments."""
        logger.info(f"Response callback received: '{new_text}'")
        if not self.is_running or self.state != ConversationState.LISTENING:
            logger.debug("Not in listening state; ignoring callback")
            return

        if new_text.strip() == self._last_processed_text.strip():
            logger.debug("Duplicate fragment; ignoring")
            return

        cleaned = self._clean_text(new_text)
        if not cleaned:
            logger.debug("Cleaned text empty; ignoring")
            return

        self._last_processed_text = new_text
        fragment = SentenceFragment(text=cleaned, timestamp=time.time())
        self.sentence_buffer.append(fragment)
        self.current_sentence = self._rebuild_current_sentence()
        self._update_accumulated_sentence()
        # update last fragment time (we got speech)
        self.last_fragment_time = time.time()
        # check structural completion (punctuation / question)
        res = self._check_completion_conditions()
        logger.info(f"Completion check: {res}")
        if res["should_complete"]:
            s = res["sentence"]
            if s and not self._is_duplicate(s):
                self.processing_queue.put(s)
                self.last_sent_sentence = s
                self._reset_current_sentence(reset_accumulated=res["reset_accumulated"])

    # ---------- helpers for sentence assembly ----------
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        artifacts = ["", " ", ".", ".", "thank you", "thanks for watching", "you", "um", "uh", "ah", "hmm"]
        if t.lower() in artifacts:
            return ""
        return re.sub(r"\s+", " ", t)

    def _rebuild_current_sentence(self) -> str:
        if not self.sentence_buffer:
            return ""
        current_time = time.time()
        window_time = 15.0 if self._is_complex_query(self.accumulated_sentence) else 10.0
        recent = [f for f in self.sentence_buffer if current_time - f.timestamp < window_time]
        if not recent:
            return ""
        return " ".join(f.text for f in recent).strip()

    def _update_accumulated_sentence(self):
        if not self.current_sentence:
            return
        cur = self.current_sentence.strip()
        acc = self.accumulated_sentence.strip()
        if len(cur) > len(acc):
            self.accumulated_sentence = cur
        elif cur and not acc.endswith(cur):
            if len(cur) > 3:
                self.accumulated_sentence = cur

    def _check_completion_conditions(self) -> dict:
        if not self.current_sentence:
            return {"should_complete": False, "sentence": "", "reset_accumulated": False}
        sentence = self.current_sentence.strip()
        accumulated = self.accumulated_sentence.strip()
        # fallback silence-based
        now = time.time()
        silence_duration = now - self.last_fragment_time
        timeout = self.extended_timeout if self._is_complex_query(accumulated) else self.sentence_timeout
        if silence_duration >= timeout:
            candidate = accumulated if len(accumulated) > len(sentence) else sentence
            if len(candidate) >= self.min_sentence_length:
                return {"should_complete": True, "sentence": candidate, "reset_accumulated": True}
        return {"should_complete": False, "sentence": "", "reset_accumulated": False}

    def _is_complex_query(self, sentence: str) -> bool:
        if not sentence:
            return False
        if len(sentence.split()) > 10:
            return True
        return False

    def _is_duplicate(self, sentence: str) -> bool:
        if not self.last_sent_sentence:
            return False
        a = re.sub(r"[^\w\s]", "", sentence.lower()).strip()
        b = re.sub(r"[^\w\s]", "", self.last_sent_sentence.lower()).strip()
        return a.startswith(b) or b.startswith(a)

    def _reset_current_sentence(self, reset_accumulated: bool = True):
        self.current_sentence = ""
        self.sentence_buffer.clear()
        if reset_accumulated:
            self.accumulated_sentence = ""
        # mark times so subsequent silence is measured from now
        self.last_fragment_time = time.time()
        self.sentence_start_time = time.time()
