import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from Llm import llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    PAUSED = "paused"
    ERROR = "error"

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

@dataclass
class ConversationExchange:
    """Represents a direct user-LLM exchange"""
    id: str
    timestamp: float
    user_speech: str           # What the user said (from Whisper)
    llm_response: str          # Direct LLM response
    whisper_chunks: List[WhisperChunk] = field(default_factory=list)
    response_time: float = 0.0
    confidence_score: float = 0.0
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationSession:
    """Complete conversation session data"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    exchanges: List[ConversationExchange] = field(default_factory=list)
    session_metrics: Dict[str, Any] = field(default_factory=dict)
    conversation_context: List[Dict[str, str]] = field(default_factory=list)

class DirectLLMOrchestrator:
    """
    Direct orchestrator for user speech -> LLM response
    No background simulation, just direct interaction
    """
    
    def __init__(self):
        # Session management
        self.current_session: Optional[ConversationSession] = None
        self.state = ConversationState.IDLE
        
        # Real-time processing
        self.whisper_buffer: List[WhisperChunk] = []
        self.current_speech_buffer: List[WhisperChunk] = []
        self.silence_threshold = 3.0  # seconds of silence before processing
        self.last_activity_time = 0.0
        
        # Processing control
        self.processing_lock = asyncio.Lock()
        self.auto_process_task: Optional[asyncio.Task] = None
        
        # Conversation context for LLM
        self.conversation_history: List[Dict[str, str]] = []
        self.max_context_length = 10  # Keep last 10 exchanges
        
        # Callbacks for UI updates
        self.on_state_change: Optional[Callable[[ConversationState], None]] = None
        self.on_user_speech: Optional[Callable[[str], None]] = None
        self.on_llm_response: Optional[Callable[[str], None]] = None
        self.on_processing_update: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # Performance tracking
        self.response_times: List[float] = []
        self.processing_errors: List[str] = []

    async def start_conversation(self) -> bool:
        """Start a new direct conversation session"""
        
        if self.state != ConversationState.IDLE:
            logger.error(f"Cannot start conversation in state: {self.state}")
            return False
        
        try:
            self._change_state(ConversationState.LISTENING)
            logger.info("Starting direct LLM conversation...")
            
            # Create new session
            session_id = f"direct_session_{int(time.time())}"
            
            self.current_session = ConversationSession(
                session_id=session_id,
                start_time=time.time()
            )
            
            # Reset state
            self.whisper_buffer = []
            self.current_speech_buffer = []
            self.conversation_history = []
            self.response_times = []
            self.processing_errors = []
            
            # Start auto-processing task
            self.auto_process_task = asyncio.create_task(self._auto_process_loop())
            
            logger.info(f"Conversation session started: {session_id}")
            
            # Initial system message
            initial_context = {
                "role": "system",
                "content": "You are Claude, an AI assistant. Have a natural, helpful conversation with the user. Keep responses conversational and engaging."
            }
            self.conversation_history.append(initial_context)
            
            return True
            
        except Exception as e:
            self._change_state(ConversationState.ERROR)
            error_msg = f"Error starting conversation: {str(e)}"
            logger.error(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return False
    
    async def process_whisper_chunk(self, whisper_chunk: WhisperChunk) -> bool:
        """Process incoming speech chunk from user"""
        
        if self.state not in [ConversationState.LISTENING, ConversationState.PAUSED]:
            logger.warning(f"Ignoring whisper chunk in state: {self.state}")
            return False
        
        try:
            # Add to buffers
            self.whisper_buffer.append(whisper_chunk)
            self.current_speech_buffer.append(whisper_chunk)
            self.last_activity_time = time.time()
            
            logger.debug(f"User speech chunk: '{whisper_chunk.text[:30]}...'")
            
            # Update UI with partial speech
            partial_speech = self._combine_chunks(self.current_speech_buffer)
            if self.on_processing_update:
                self.on_processing_update({
                    "partial_speech": partial_speech,
                    "chunk_count": len(self.current_speech_buffer),
                    "is_speaking": True
                })
            
            # Check for immediate processing triggers
            if self._should_process_immediately(whisper_chunk):
                await self._process_user_speech()
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing speech chunk: {str(e)}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return False
    
    def _should_process_immediately(self, chunk: WhisperChunk) -> bool:
        """Determine if we should process speech immediately"""
        
        text_lower = chunk.text.lower().strip()
        
        # Immediate triggers for natural conversation breaks
        immediate_triggers = [
            text_lower.endswith('?'),  # Questions
            text_lower.endswith('.'),  # Statements
            text_lower.endswith('!'),  # Exclamations
            any(phrase in text_lower for phrase in [
                'okay', 'alright', 'so', 'well', 'anyway', 'actually',
                'by the way', 'speaking of', 'that reminds me'
            ]),
            len(self.current_speech_buffer) >= 8,  # 40 seconds of speech
        ]
        
        return any(immediate_triggers)
    
    async def _auto_process_loop(self):
        """Automatically process speech after silence periods"""
        
        while self.state in [ConversationState.LISTENING, ConversationState.PAUSED, ConversationState.PROCESSING]:
            try:
                await asyncio.sleep(0.5)  # Check twice per second
                
                if self.state == ConversationState.PAUSED:
                    continue
                
                # Process if silence detected
                if (self.current_speech_buffer and 
                    time.time() - self.last_activity_time >= self.silence_threshold):
                    
                    await self._process_user_speech()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-process loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_user_speech(self):
        """Process accumulated user speech and generate LLM response"""
        
        async with self.processing_lock:
            if not self.current_speech_buffer:
                return
            
            self._change_state(ConversationState.PROCESSING)
            
            try:
                start_time = time.time()
                
                # Combine speech chunks
                user_speech = self._combine_chunks(self.current_speech_buffer)
                
                if not user_speech.strip():
                    logger.info("No meaningful speech to process")
                    self.current_speech_buffer = []
                    self._change_state(ConversationState.LISTENING)
                    return
                
                logger.info(f"Processing user speech: '{user_speech[:60]}...'")
                
                # Notify UI of user speech
                if self.on_user_speech:
                    self.on_user_speech(user_speech)
                
                # Add user message to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_speech
                })
                
                # Keep context manageable
                if len(self.conversation_history) > self.max_context_length * 2:
                    # Keep system message and recent exchanges
                    system_msg = self.conversation_history[0]
                    recent_msgs = self.conversation_history[-(self.max_context_length * 2 - 1):]
                    self.conversation_history = [system_msg] + recent_msgs
                
                # Generate LLM response
                self._change_state(ConversationState.RESPONDING)
                
                # Create conversation context for LLM
                context_for_llm = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in self.conversation_history[-6:]  # Last 6 messages for context
                ])
                
                # Generate response using Gemini
                llm_response = await llm.ask_gemini(user_speech, context=context_for_llm)
                
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                # Add LLM response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": llm_response
                })
                
                # Calculate confidence based on response quality
                confidence = self._calculate_confidence(user_speech, llm_response, response_time)
                
                # Create exchange record
                exchange = ConversationExchange(
                    id=f"exchange_{len(self.current_session.exchanges) + 1}",
                    timestamp=time.time(),
                    user_speech=user_speech,
                    llm_response=llm_response,
                    whisper_chunks=self.current_speech_buffer.copy(),
                    response_time=response_time,
                    confidence_score=confidence,
                    processing_metadata={
                        "chunk_count": len(self.current_speech_buffer),
                        "speech_length": len(user_speech),
                        "response_length": len(llm_response),
                        "context_length": len(self.conversation_history)
                    }
                )
                
                self.current_session.exchanges.append(exchange)
                self.current_session.conversation_context = self.conversation_history.copy()
                
                # Clear speech buffer
                self.current_speech_buffer = []
                
                logger.info(f"Generated response ({response_time:.2f}s): '{llm_response[:60]}...'")
                
                # Notify UI of LLM response
                if self.on_llm_response:
                    self.on_llm_response(llm_response)
                
                # Return to listening state
                self._change_state(ConversationState.LISTENING)
                
            except Exception as e:
                error_msg = f"Error processing user speech: {str(e)}"
                logger.error(error_msg)
                self.processing_errors.append(error_msg)
                if self.on_error:
                    self.on_error(error_msg)
                self._change_state(ConversationState.ERROR)
    
    def _combine_chunks(self, chunks: List[WhisperChunk]) -> str:
        """Intelligently combine speech chunks"""
        
        if not chunks:
            return ""
        
        # Sort by timestamp
        sorted_chunks = sorted(chunks, key=lambda x: x.timestamp)
        
        # Combine with intelligent spacing
        combined_parts = []
        
        for i, chunk in enumerate(sorted_chunks):
            text = chunk.text.strip()
            if not text:
                continue
            
            if i == 0:
                combined_parts.append(text)
            else:
                # Add appropriate spacing
                prev_chunk = sorted_chunks[i-1]
                time_gap = chunk.timestamp - prev_chunk.timestamp
                
                if time_gap > 2.0:  # Significant pause
                    combined_parts.append('. ' + text.capitalize())
                elif combined_parts[-1].endswith(('.', '!', '?')):
                    combined_parts.append(' ' + text.capitalize())
                else:
                    combined_parts.append(' ' + text)
        
        return ''.join(combined_parts).strip()
    
    def _calculate_confidence(self, user_speech: str, llm_response: str, response_time: float) -> float:
        """Calculate confidence score for the exchange"""
        
        base_confidence = 0.8
        
        # Adjust based on speech clarity (length and coherence)
        speech_quality = min(0.1, len(user_speech) / 50)
        
        # Response quality (not too short, not too long)
        response_quality = 0.1 if 10 < len(llm_response) < 500 else 0.0
        
        # Response time (faster is generally better for conversation)
        time_quality = max(0, (10.0 - response_time) / 50)
        
        confidence = base_confidence + speech_quality + response_quality + time_quality
        return max(0.0, min(1.0, confidence))
    
    def _change_state(self, new_state: ConversationState):
        """Change conversation state and notify listeners"""
        
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.debug(f"State change: {old_state.value} -> {new_state.value}")
            
            if self.on_state_change:
                self.on_state_change(new_state)
    
    async def pause_conversation(self) -> bool:
        """Pause the conversation"""
        
        if self.state not in [ConversationState.LISTENING, ConversationState.PROCESSING]:
            return False
        
        self._change_state(ConversationState.PAUSED)
        return True
    
    async def resume_conversation(self) -> bool:
        """Resume the conversation"""
        
        if self.state != ConversationState.PAUSED:
            return False
        
        self._change_state(ConversationState.LISTENING)
        return True
    
    async def end_conversation(self) -> Dict[str, Any]:
        """End the conversation and generate report"""
        
        if not self.current_session:
            return {"error": "No active conversation"}
        
        try:
            # Process any remaining speech
            if self.current_speech_buffer:
                await self._process_user_speech()
            
            # Stop auto-processing
            if self.auto_process_task:
                self.auto_process_task.cancel()
                try:
                    await self.auto_process_task
                except asyncio.CancelledError:
                    pass
            
            # Finalize session
            self.current_session.end_time = time.time()
            
            # Calculate metrics
            duration = self.current_session.end_time - self.current_session.start_time
            
            self.current_session.session_metrics = {
                "duration_seconds": duration,
                "duration_minutes": duration / 60,
                "total_exchanges": len(self.current_session.exchanges),
                "total_speech_chunks": len(self.whisper_buffer),
                "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                "processing_errors": len(self.processing_errors),
                "average_confidence": sum(ex.confidence_score for ex in self.current_session.exchanges) / len(self.current_session.exchanges) if self.current_session.exchanges else 0
            }
            
            # Generate final report
            report = self._generate_conversation_report()
            
            self._change_state(ConversationState.IDLE)
            logger.info(f"Conversation ended: {self.current_session.session_id}")
            
            return report
            
        except Exception as e:
            error_msg = f"Error ending conversation: {str(e)}"
            logger.error(error_msg)
            self._change_state(ConversationState.ERROR)
            return {"error": error_msg}
    
    def _generate_conversation_report(self) -> Dict[str, Any]:
        """Generate comprehensive conversation report"""
        
        if not self.current_session:
            return {"error": "No session data"}
        
        # Convert exchanges to serializable format
        exchanges_data = []
        for exchange in self.current_session.exchanges:
            exchange_data = {
                "id": exchange.id,
                "timestamp": exchange.timestamp,
                "datetime": datetime.fromtimestamp(exchange.timestamp).isoformat(),
                "user_speech": exchange.user_speech,
                "llm_response": exchange.llm_response,
                "response_time_seconds": exchange.response_time,
                "confidence_score": exchange.confidence_score,
                "processing_metadata": exchange.processing_metadata,
                "whisper_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "timestamp": chunk.timestamp,
                        "text": chunk.text,
                        "confidence": chunk.confidence
                    }
                    for chunk in exchange.whisper_chunks
                ]
            }
            exchanges_data.append(exchange_data)
        
        # Build report
        report = {
            "session_summary": {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time,
                "end_time": self.current_session.end_time,
                "start_datetime": datetime.fromtimestamp(self.current_session.start_time).isoformat(),
                "end_datetime": datetime.fromtimestamp(self.current_session.end_time).isoformat() if self.current_session.end_time else None,
                "duration_minutes": self.current_session.session_metrics.get("duration_minutes", 0),
                "total_exchanges": len(self.current_session.exchanges),
                "conversation_type": "direct_llm_interaction"
            },
            
            "conversation_exchanges": exchanges_data,
            
            "technical_metrics": {
                "whisper_processing": {
                    "total_chunks_received": len(self.whisper_buffer),
                    "chunks_per_exchange": len(self.whisper_buffer) / len(self.current_session.exchanges) if self.current_session.exchanges else 0
                },
                "llm_performance": {
                    "average_response_time": self.current_session.session_metrics.get("average_response_time", 0),
                    "response_times": self.response_times,
                    "average_confidence": self.current_session.session_metrics.get("average_confidence", 0)
                },
                "processing_errors": self.processing_errors
            },
            
            "conversation_analysis": {
                "user_engagement": {
                    "average_speech_length": sum(len(ex.user_speech) for ex in self.current_session.exchanges) / len(self.current_session.exchanges) if self.current_session.exchanges else 0,
                    "total_user_words": sum(len(ex.user_speech.split()) for ex in self.current_session.exchanges),
                    "questions_asked": sum(1 for ex in self.current_session.exchanges if '?' in ex.user_speech)
                },
                "llm_response_quality": {
                    "average_response_length": sum(len(ex.llm_response) for ex in self.current_session.exchanges) / len(self.current_session.exchanges) if self.current_session.exchanges else 0,
                    "total_response_words": sum(len(ex.llm_response.split()) for ex in self.current_session.exchanges),
                    "conversation_flow": "natural" if len(self.current_session.exchanges) > 3 else "brief"
                }
            }
        }
        
        return report
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current conversation status"""
        
        return {
            "state": self.state.value,
            "session_id": self.current_session.session_id if self.current_session else None,
            "exchanges_count": len(self.current_session.exchanges) if self.current_session else 0,
            "buffer_chunks": len(self.current_speech_buffer),
            "last_activity": self.last_activity_time,
            "conversation_active": self.current_session is not None
        }
    
    def set_callbacks(self, 
                     on_state_change: Callable[[ConversationState], None] = None,
                     on_user_speech: Callable[[str], None] = None,
                     on_llm_response: Callable[[str], None] = None,
                     on_processing_update: Callable[[Dict], None] = None,
                     on_error: Callable[[str], None] = None):
        """Set callback functions for UI integration"""
        
        self.on_state_change = on_state_change
        self.on_user_speech = on_user_speech
        self.on_llm_response = on_llm_response
        self.on_processing_update = on_processing_update
        self.on_error = on_error


# Example usage
async def example_direct_conversation():
    """Example of direct user-LLM conversation"""
    
    orchestrator = DirectLLMOrchestrator()
    
    # Set up callbacks
    def on_state_change(state: ConversationState):
        print(f"🔄 State: {state.value}")
    
    def on_user_speech(speech: str):
        print(f"👤 You: {speech}")
    
    def on_llm_response(response: str):
        print(f"🤖 Claude: {response}")
    
    def on_processing_update(update: Dict):
        if update.get("is_speaking"):
            print(f"🎙️ Speaking: {update['partial_speech'][:30]}...")
    
    def on_error(error: str):
        print(f"❌ Error: {error}")
    
    orchestrator.set_callbacks(
        on_state_change=on_state_change,
        on_user_speech=on_user_speech,
        on_llm_response=on_llm_response,
        on_processing_update=on_processing_update,
        on_error=on_error
    )
    
    # Start conversation
    success = await orchestrator.start_conversation()
    
    if success:
        print("🚀 Direct conversation started!")
        print("🎙️ Start speaking - I'll respond directly...")
        
        # Simulate some user speech chunks
        user_chunks = [
            WhisperChunk(time.time(), "Hello Claude, how are you today?", 0.95),
            WhisperChunk(time.time() + 5, "I wanted to ask you about artificial intelligence", 0.90),
            WhisperChunk(time.time() + 10, "and how it's changing the world.", 0.88),
        ]
        
        for chunk in user_chunks:
            await orchestrator.process_whisper_chunk(chunk)
            await asyncio.sleep(1)  # Simulate real-time speech
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Add another exchange
        more_chunks = [
            WhisperChunk(time.time() + 30, "That's really interesting.", 0.92),
            WhisperChunk(time.time() + 35, "What do you think about the future of AI?", 0.94),
        ]
        
        for chunk in more_chunks:
            await orchestrator.process_whisper_chunk(chunk)
            await asyncio.sleep(1)
        
        # Wait and end conversation
        await asyncio.sleep(5)
        final_report = await orchestrator.end_conversation()
        
        print(f"\n✅ Conversation completed!")
        print(f"📊 {final_report['session_summary']['total_exchanges']} exchanges")
        print(f"⏱️ {final_report['session_summary']['duration_minutes']:.1f} minutes")
        
        return final_report
    
    else:
        print("❌ Failed to start conversation")
        return None


if __name__ == "__main__":
    asyncio.run(example_direct_conversation())