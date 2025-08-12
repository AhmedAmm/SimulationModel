# auxx.py
import threading
import queue
import time
import sys
import logging
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import torch
import whisper
import pyaudio
import uvicorn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StreamingWhisperTranscriber:
    """
    Streaming transcriber that:
      - keeps 5s chunks for Whisper,
      - scans each 5s chunk at small frame resolution (250ms) to detect speech energy,
      - notifies an attached orchestrator once when continuous silence crosses
        a configurable threshold (default 5.0s).
    """

    def __init__(
        self,
        model_name: str = "base",
        input_source: str = "mic",
        websocket_host: str = "0.0.0.0",
        websocket_port: int = 9000,
        callback: Optional[callable] = None,
    ):
        # Audio / chunk params
        self.sample_rate = 16000
        self.chunk_duration = 5.0
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        # Frame-level scan inside chunk (for intra-chunk speech detection)
        self.frame_duration = 0.25  # 250 ms
        self.frame_size = int(self.sample_rate * self.frame_duration)
        self.frame_hop = self.frame_size  # non-overlapping (tune if desired)

        # RMS energy threshold for a frame => speech (tune for your mic)
        self.frame_energy_threshold = 0.01

        # Silence config (default 5s). Set to None to disable automatic silence notifying.
        self.silence_seconds_threshold = 5.0

        # State for silence tracking
        self.last_speech_time = time.time()
        self.silence_notified = False

        # Buffers & threads
        self.audio_buffer = deque = list  # placeholder, we replace with deque below
        from collections import deque as _dq
        self.audio_buffer = _dq()
        self.processing_queue = queue.Queue()
        self.full_transcription = ""
        self.last_chunk_text = ""
        self.sentence_buffer = []

        self.input_source = input_source
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.callback = callback  # optional external callback (orchestrator.response_callback)

        # load Whisper model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info("Whisper model loaded successfully")

        # audio init
        self.audio = None
        self.stream = None

        # websocket app for direct websocket mode
        self.app = None
        self.websocket_connection: Optional[WebSocket] = None

        # control
        self.is_running = False
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)

        # orchestrator (set externally)
        self.orchestrator = None

    # ---------- helpers ----------
    def _frame_has_speech(self, frame: np.ndarray) -> bool:
        """Return True if frame RMS energy exceeds threshold."""
        if frame is None or frame.size == 0:
            return False
        frame = frame.astype(np.float32)
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        return rms >= self.frame_energy_threshold

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        artifacts = ["", " ", ".", "...", "thank you", "thanks for watching", "you", "um", "uh", "ah", "hmm"]
        if text.lower() in artifacts:
            return ""
        return " ".join(text.split())

    def _update_streaming_display(self, new_text: str):
        """Update terminal display and call external callback if provided."""
        if not new_text:
            return

        if self.full_transcription and not self.full_transcription.endswith(" "):
            self.full_transcription += " "
        self.full_transcription += new_text

        # Trim long display
        display_text = self.full_transcription
        if len(display_text) > 120:
            display_text = "..." + display_text[-117:]

        # Print inline
        sys.stdout.write("\r" + " " * 140 + "\r")
        sys.stdout.write(f"🎙️  {display_text}")
        sys.stdout.flush()

        # call callback (orchestrator.response_callback) if present
        try:
            if self.callback:
                self.callback(new_text)
        except Exception:
            logger.exception("Error calling transcriber callback")

    # ---------- audio ingest helpers ----------
    def _init_microphone(self):
        self.audio = pyaudio.PyAudio()

    def _init_websocket(self):
        self.app = FastAPI()

        @self.app.websocket("/transcribe")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connection = websocket
            logger.info("✅ WebSocket client connected")
            try:
                while True:
                    data = await websocket.receive_bytes()
                    audio_np = np.frombuffer(data, dtype=np.float32)
                    # append into buffer
                    self.audio_buffer.extend(audio_np.tolist())
                    # if enough data, process chunk (keeps overlap logic inside)
                    if len(self.audio_buffer) >= self.chunk_size:
                        self._process_audio_chunk()
            except WebSocketDisconnect:
                logger.info("❌ WebSocket client disconnected")
                self.websocket_connection = None
            except Exception:
                logger.exception("Websocket receive loop error")
                self.websocket_connection = None

    def _process_audio_chunk(self):
        """Grabs the most recent chunk_size samples (keeping overlap) and enqueue for processing."""
        if len(self.audio_buffer) < self.chunk_size:
            return
        # take last chunk_size samples
        # convert to numpy array float32
        arr = np.array(list(self.audio_buffer)[-self.chunk_size:], dtype=np.float32)
        # preserve an overlap of 0.5s (optional). To keep it simple, keep 0.5s overlap:
        overlap_seconds = 0.5
        overlap_size = int(self.sample_rate * overlap_seconds)
        # drop everything except last overlap_size to keep sliding window
        keep = list(self.audio_buffer)[-overlap_size:] if overlap_size > 0 else []
        self.audio_buffer.clear()
        self.audio_buffer.extend(keep)
        # enqueue chunk
        self.processing_queue.put(arr)

    # ---------- main processing loop ----------
    def _processing_loop(self):
        logger.info("Transcriber processing loop started")
        while self.is_running:
            try:
                audio_chunk = self.processing_queue.get(timeout=1.0)
                try:
                    if audio_chunk is None:
                        continue

                    # ensure float32
                    audio_chunk = audio_chunk.astype(np.float32)

                    # Frame-level scan inside this 5s chunk
                    saw_speech_in_chunk = False
                    n = len(audio_chunk)
                    pos = 0
                    while pos < n:
                        frame = audio_chunk[pos: pos + self.frame_size]
                        if self._frame_has_speech(frame):
                            saw_speech_in_chunk = True
                            # update last_speech_time to now
                            self.last_speech_time = time.time()
                            break
                        pos += self.frame_hop

                    if not saw_speech_in_chunk:
                        now = time.time()
                        silence_duration = now - self.last_speech_time
                        logger.debug(f"Chunk scanned: no speech; silence_duration={silence_duration:.2f}s")

                        if (self.silence_seconds_threshold is not None
                                and silence_duration >= self.silence_seconds_threshold
                                and not self.silence_notified):
                            logger.info(f"Silence threshold reached (>= {self.silence_seconds_threshold}s). Notifying orchestrator.")
                            if hasattr(self, "orchestrator") and self.orchestrator:
                                try:
                                    self.orchestrator.notify_silence()
                                except Exception:
                                    logger.exception("Error notifying orchestrator")
                            self.silence_notified = True

                        # don't transcribe fully silent chunk
                        continue
                    else:
                        # speech seen -> reset notification flag
                        self.silence_notified = False

                    # Transcribe the chunk with Whisper
                    try:
                        result = self.model.transcribe(
                            audio_chunk,
                            task="transcribe",
                            language="en",
                            fp16=torch.cuda.is_available(),
                            # keep thresholds conservative — we already gate by frame energy
                        )
                    except Exception:
                        logger.exception("Whisper transcription failed")
                        continue

                    new_text = self._clean_text(result.get("text", ""))
                    if new_text and new_text != self.last_chunk_text:
                        # update display and possibly call orchestrator via callback
                        self._update_streaming_display(new_text)
                        self.last_chunk_text = new_text
                finally:
                    try:
                        self.processing_queue.task_done()
                    except Exception:
                        pass
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Error in transcriber processing loop")

        logger.info("Transcriber processing loop stopped")

    # ---------- control ----------
    def start_transcription(self):
        if self.is_running:
            logger.warning("Transcription already running")
            return
        # init input
        if self.input_source == "mic":
            self._init_microphone()
        elif self.input_source == "websocket":
            self._init_websocket()
        # start threads
        self.is_running = True
        if not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
        # if websocket mode, run uvicorn in main thread if desired -- usually the outer server handles this
        logger.info("Transcription started")

    def stop_transcription(self):
        self.is_running = False
        # drain queue
        try:
            self.processing_queue.put(None)
        except Exception:
            pass
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        if self.input_source == "mic" and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()
            except Exception:
                pass
        logger.info("Transcription stopped")

    def reset_for_new_conversation(self):
        logger.info("Resetting transcriber state for a new conversation")
        self.audio_buffer.clear()
        # re-create processing queue to drop stale tasks
        self.processing_queue = queue.Queue()
        self.full_transcription = ""
        self.last_chunk_text = ""
        self.silence_notified = False
        self.last_speech_time = time.time()
        # clear orchestrator last processed text if present (defensive)
        if hasattr(self, "orchestrator") and self.orchestrator:
            try:
                self.orchestrator._last_processed_text = ""
            except Exception:
                pass

    # small helper for quick testing
    def test_whisper(self, test_audio_path=None):
        print("Testing Whisper model")
        if test_audio_path:
            res = self.model.transcribe(test_audio_path)
            print("Result:", res.get("text", ""))
        else:
            duration = 2.0
            t = np.linspace(0, duration, int(self.sample_rate * duration), dtype=np.float32)
            test_audio = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            res = self.model.transcribe(test_audio)
            print("Result:", res.get("text", ""))


# if run standalone for quick tests
if __name__ == "__main__":
    tr = StreamingWhisperTranscriber(model_name="base", input_source="mic")
    tr.start_transcription()
