# websocket_server.py
import asyncio
import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from auxx import StreamingWhisperTranscriber
from orchestrator_test import ConversationalOrchestrator, ConversationState
from agent import ClientSimulationAgent  # ensure this module exists
from TTS import generate_speak_and_sync_payload  # ensure this module exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
agent: ClientSimulationAgent = None
transcriber: StreamingWhisperTranscriber = None
orchestrator: ConversationalOrchestrator = None
send_queue: asyncio.Queue = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, transcriber, orchestrator, send_queue
    logger.info("🚀 Starting conversation system (lifespan)...")

    send_queue = asyncio.Queue()

    # 1) Initialize agent
    agent = ClientSimulationAgent(fastapi_base_url="http://localhost:8000")
    await agent.initialize_session(
        user_company="Orange",
        user_product="internet services",
        client_company="talan",
        client_role="CTO",
        client_industry="provide Services",
        company_size="Enterprise",
    )
    logger.info("✅ Agent session initialized")

    # 2) Orchestrator
    orchestrator = ConversationalOrchestrator(agent=agent, use_tts=False, sentence_timeout=5.0)
    # 3) Transcriber (websocket mode). Keep 5s chunks.
    transcriber = StreamingWhisperTranscriber(
        model_name="base",
        input_source="websocket",
        websocket_host="0.0.0.0",
        websocket_port=9000,
        callback=orchestrator.response_callback,
    )
    # wire them
    orchestrator.set_transcriber(transcriber)
    # tune defaults (if desired)
    transcriber.silence_seconds_threshold = 3.5
    transcriber.frame_energy_threshold = 0.01

    # Give transcriber a hook to post partials into the main event loop
    main_loop = asyncio.get_running_loop()
    original_update = transcriber._update_streaming_display

    def enhanced_update(new_text: str):
        try:
            original_update(new_text)
        except Exception:
            logger.exception("Error in original_update")
        # schedule partial transcription to websocket clients
        """ try:
            main_loop.call_soon_threadsafe(send_queue.put_nowait, {"type": "partial_transcription", "text": transcriber.full_transcription})
        except Exception:
            logger.exception("Failed to enqueue partial_transcription") """

    transcriber._update_streaming_display = enhanced_update

    # When orchestrator emits completed responses, forward them to clients via send_queue
    original_handle = orchestrator._handle_response if hasattr(orchestrator, "_handle_response") else None

    def enhanced_handle_response(response: str, user_input: str):
        if original_handle:
            try:
                original_handle(response, user_input)
            except Exception:
                logger.exception("Error in original handle response")
        try:
            #main_loop.call_soon_threadsafe(send_queue.put_nowait, {"type": "user_sentence", "text": user_input})
            data_back = generate_speak_and_sync_payload(response, message_id="from_orchestrator")
            main_loop.call_soon_threadsafe(send_queue.put_nowait, {"response" : data_back} )
        except Exception:
            logger.exception("Failed to schedule messages")

    if original_handle:
        orchestrator._handle_response = enhanced_handle_response

    # Start orchestrator and transcriber processing thread (we do not run transcriber's uvicorn here)
    orchestrator.start()
    transcriber.is_running = True
    if not transcriber.processing_thread.is_alive():
        transcriber.processing_thread.start()
    logger.info("✅ Orchestrator started and transcriber processing thread running")

    yield

    # shutdown
    logger.info("🛑 Shutting down conversation system (lifespan).")
    try:
        if orchestrator:
            orchestrator.stop()
    except Exception:
        logger.exception("Error stopping orchestrator")
    try:
        if transcriber:
            transcriber.stop_transcription()
    except Exception:
        logger.exception("Error stopping transcriber")
    try:
        if agent and agent.rag_initialized:
            await agent.cleanup_rag()
    except Exception:
        logger.exception("Error cleaning up agent RAG")
    logger.info("✅ Shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.websocket("/transcribe")
async def transcribe_endpoint(ws: WebSocket):
    """
    Accepts float32 PCM audio bytes from the frontend and forwards:
      - partial transcriptions and assistant responses back to the client via send_queue
    """
    global transcriber, send_queue
    await ws.accept()
    logger.info("✅ WebSocket client connected")
    if transcriber:
        transcriber.websocket_connection = ws

    async def ws_sender():
        try:
            while True:
                msg = await send_queue.get()
                if msg is None:
                    break
                try:
                    await ws.send_json(msg)
                except Exception:
                    logger.exception("Failed to send JSON over websocket; breaking sender")
                    break
        finally:
            logger.info("ws_sender stopped")

    sender_task = asyncio.create_task(ws_sender())

    try:
        while True:
            data = await ws.receive_bytes()
            audio_np = np.frombuffer(data, dtype=np.float32)
            # append to transcriber's buffer (deque-like)
            if hasattr(transcriber, "audio_buffer"):
                transcriber.audio_buffer.extend(audio_np.tolist())
                if len(transcriber.audio_buffer) >= transcriber.chunk_size:
                    transcriber._process_audio_chunk()
    except WebSocketDisconnect:
        logger.info("❌ WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket receive loop error")
    finally:
        try:
            await send_queue.put(None)
        except Exception:
            pass
        sender_task.cancel()
        if transcriber:
            transcriber.websocket_connection = None


if __name__ == "__main__":
    uvicorn.run("websocket_server:app", host="0.0.0.0", port=9000, log_level="info")
