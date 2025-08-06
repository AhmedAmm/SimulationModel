import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from agent import ClientSimulationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PresentationState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
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
class PresentationExchange:
    """Represents a complete exchange in the presentation"""
    id: str
    timestamp: float
    salesperson_speech: str    # What the salesperson said (from Whisper)
    client_response: str       # How the simulated client responded
    whisper_chunks: List[WhisperChunk] = field(default_factory=list)
    response_time: float = 0.0
    meeting_stage: str = ""
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PresentationSession:
    """Complete presentation session data"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    client_profile: Dict[str, Any] = field(default_factory=dict)
    product_context: Dict[str, Any] = field(default_factory=dict)
    exchanges: List[PresentationExchange] = field(default_factory=list)
    session_metrics: Dict[str, Any] = field(default_factory=dict)
    final_analysis: Dict[str, Any] = field(default_factory=dict)

class RealTimePresentationOrchestrator:
    """
    Orchestrates real-time presentation simulation by managing Whisper input
    and ClientSimulationAgent responses
    """
    
    def __init__(self, fastapi_base_url: str = "http://localhost:8000"):
        self.fastapi_base_url = fastapi_base_url
        self.client_agent = ClientSimulationAgent(fastapi_base_url)
        
        # Session management
        self.current_session: Optional[PresentationSession] = None
        self.state = PresentationState.IDLE
        
        # Real-time processing
        self.whisper_buffer: List[WhisperChunk] = []
        self.current_exchange_buffer: List[WhisperChunk] = []
        self.silence_threshold = 5.0  # seconds of silence before processing
        self.last_activity_time = 0.0
        
        # Processing control
        self.processing_lock = asyncio.Lock()
        self.auto_process_task: Optional[asyncio.Task] = None
        
        # Callbacks for external integration
        self.on_client_response: Optional[Callable[[str], None]] = None
        self.on_session_update: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # Metrics tracking
        self.response_times: List[float] = []
        self.processing_errors: List[str] = []
    
    async def initialize_session(self, 
                                user_company: str,
                                user_product: str,
                                client_company: str = None,
                                client_role: str = "Decision Maker",
                                client_industry: str = "Technology",
                                company_size: str = "Enterprise",
                                additional_context: Dict[str, Any] = None) -> bool:
        """Initialize a new presentation session"""
        
        if self.state != PresentationState.IDLE:
            logger.error(f"Cannot initialize session in state: {self.state}")
            return False
        
        try:
            self.state = PresentationState.INITIALIZING
            logger.info("Initializing presentation session...")
            
            # Initialize the client simulation agent
            success = await self.client_agent.initialize_session(
                user_company=user_company,
                user_product=user_product,
                client_company=client_company,
                client_role=client_role,
                client_industry=client_industry,
                company_size=company_size,
                additional_context=additional_context or {}
            )
            
            if not success:
                self.state = PresentationState.ERROR
                logger.error("Failed to initialize client agent")
                return False
            
            # Create session
            session_id = f"session_{int(time.time())}_{hash(user_company + user_product) % 10000}"
            
            # Get client profile and product context from agent
            agent_status = self.client_agent.get_enhanced_session_status()
            
            self.current_session = PresentationSession(
                session_id=session_id,
                start_time=time.time(),
                client_profile=agent_status.get('client_profile', {}),
                product_context=agent_status.get('product_context', {})
            )
            
            # Reset processing state
            self.whisper_buffer = []
            self.current_exchange_buffer = []
            self.response_times = []
            self.processing_errors = []
            
            # Start auto-processing task
            self.auto_process_task = asyncio.create_task(self._auto_process_loop())
            
            self.state = PresentationState.ACTIVE
            logger.info(f"Session initialized: {session_id}")
            
            # Notify external systems
            if self.on_session_update:
                self.on_session_update(self._get_session_status())
            
            return True
            
        except Exception as e:
            self.state = PresentationState.ERROR
            error_msg = f"Error initializing session: {str(e)}"
            logger.error(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return False
    
    async def process_whisper_chunk(self, whisper_chunk: WhisperChunk) -> bool:
        """Process a new 5-second chunk from Whisper module (capturing salesperson's speech)"""
        
        if self.state not in [PresentationState.ACTIVE, PresentationState.PAUSED]:
            logger.warning(f"Ignoring whisper chunk in state: {self.state}")
            return False
        
        if self.state == PresentationState.PAUSED:
            logger.info("Session paused, buffering salesperson speech chunk")
        
        try:
            # Add to buffers
            self.whisper_buffer.append(whisper_chunk)
            self.current_exchange_buffer.append(whisper_chunk)
            self.last_activity_time = time.time()
            
            logger.info(f"Captured salesperson speech: {whisper_chunk.chunk_id} - '{whisper_chunk.text[:50]}...'")
            
            # If there's enough content and we detect a natural pause, process immediately
            if self._should_process_immediately(whisper_chunk):
                await self._process_accumulated_input()
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing salesperson speech chunk: {str(e)}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return False
    
    def _should_process_immediately(self, chunk: WhisperChunk) -> bool:
        """Determine if we should process input immediately"""
        
        # Check for natural conversation breaks
        text_lower = chunk.text.lower().strip()
        
        # Immediate processing triggers
        immediate_triggers = [
            text_lower.endswith('?'),  # Questions
            any(phrase in text_lower for phrase in [
                'what do you think', 'how does that sound', 'any questions',
                'does that make sense', 'what are your thoughts'
            ]),
            len(self.current_exchange_buffer) >= 6,  # 30 seconds of content
        ]
        
        return any(immediate_triggers)
    
    async def _auto_process_loop(self):
        """Automatically process accumulated input after silence periods"""
        
        while self.state in [PresentationState.ACTIVE, PresentationState.PAUSED]:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                if self.state == PresentationState.PAUSED:
                    continue
                
                # Check if we should process due to silence
                if (self.current_exchange_buffer and 
                    time.time() - self.last_activity_time >= self.silence_threshold):
                    
                    await self._process_accumulated_input()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-process loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_accumulated_input(self):
        """Process accumulated salesperson speech chunks into a client response"""
        
        async with self.processing_lock:
            if not self.current_exchange_buffer:
                return
            
            try:
                start_time = time.time()
                
                # Combine chunks into coherent salesperson speech
                salesperson_speech = self._combine_chunks(self.current_exchange_buffer)
                
                if not salesperson_speech.strip():
                    logger.info("No meaningful salesperson speech to process")
                    self.current_exchange_buffer = []
                    return
                
                logger.info(f"Processing salesperson speech: '{salesperson_speech[:100]}...'")
                
                # Generate client response using the agent
                client_response = await self.client_agent.simulate_client_response(salesperson_speech)
                
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                # Create exchange record
                exchange = PresentationExchange(
                    id=f"exchange_{len(self.current_session.exchanges) + 1}",
                    timestamp=time.time(),
                    salesperson_speech=salesperson_speech,
                    client_response=client_response,
                    whisper_chunks=self.current_exchange_buffer.copy(),
                    response_time=response_time,
                    meeting_stage=self.client_agent.meeting_stage,
                    quality_metrics=self._calculate_exchange_metrics(salesperson_speech, client_response)
                )
                
                self.current_session.exchanges.append(exchange)
                
                # Clear buffer for next exchange
                self.current_exchange_buffer = []
                
                logger.info(f"Generated client response ({response_time:.2f}s): '{client_response[:100]}...'")
                
                # Notify external systems (this would trigger TTS to speak the client response)
                if self.on_client_response:
                    self.on_client_response(client_response)
                
                if self.on_session_update:
                    self.on_session_update(self._get_session_status())
                
                # Check if session should end
                if self.client_agent._should_end_meeting():
                    await self._finalize_session()
                
            except Exception as e:
                error_msg = f"Error processing salesperson speech: {str(e)}"
                logger.error(error_msg)
                self.processing_errors.append(error_msg)
                if self.on_error:
                    self.on_error(error_msg)
    
    def _combine_chunks(self, chunks: List[WhisperChunk]) -> str:
        """Intelligently combine whisper chunks into coherent text"""
        
        if not chunks:
            return ""
        
        # Sort by timestamp
        sorted_chunks = sorted(chunks, key=lambda x: x.timestamp)
        
        # Combine text with intelligent spacing
        combined_parts = []
        last_chunk = None
        
        for chunk in sorted_chunks:
            text = chunk.text.strip()
            if not text:
                continue
            
            # Handle sentence continuation vs new sentences
            if last_chunk and (chunk.timestamp - last_chunk.timestamp) < 1.0:
                # Likely continuation of same thought
                if not combined_parts[-1].endswith(('.', '!', '?')):
                    combined_parts.append(' ' + text)
                else:
                    combined_parts.append(' ' + text.capitalize())
            else:
                # New thought or significant pause
                combined_parts.append(' ' + text.capitalize() if combined_parts else text.capitalize())
            
            last_chunk = chunk
        
        return ''.join(combined_parts).strip()
    
    def _calculate_exchange_metrics(self, input_text: str, response: str) -> Dict[str, Any]:
        """Calculate quality metrics for an exchange"""
        
        return {
            "input_length": len(input_text),
            "response_length": len(response),
            "input_word_count": len(input_text.split()),
            "response_word_count": len(response.split()),
            "has_question": '?' in response,
            "response_complexity": len(response.split('.')) + len(response.split(','))
        }
    
    async def pause_session(self) -> bool:
        """Pause the current session"""
        
        if self.state != PresentationState.ACTIVE:
            return False
        
        self.state = PresentationState.PAUSED
        logger.info("Session paused")
        
        if self.on_session_update:
            self.on_session_update(self._get_session_status())
        
        return True
    
    async def resume_session(self) -> bool:
        """Resume a paused session"""
        
        if self.state != PresentationState.PAUSED:
            return False
        
        self.state = PresentationState.ACTIVE
        logger.info("Session resumed")
        
        if self.on_session_update:
            self.on_session_update(self._get_session_status())
        
        return True
    
    async def force_process(self) -> bool:
        """Force process current buffer (manual trigger)"""
        
        if self.state not in [PresentationState.ACTIVE, PresentationState.PAUSED]:
            return False
        
        await self._process_accumulated_input()
        return True
    
    async def end_session(self) -> Dict[str, Any]:
        """Manually end the current session"""
        
        if not self.current_session:
            return {"error": "No active session"}
        
        return await self._finalize_session()
    
    async def _finalize_session(self) -> Dict[str, Any]:
        """Finalize the session and generate comprehensive report"""
        
        if not self.current_session:
            return {"error": "No session to finalize"}
        
        try:
            # Process any remaining input
            if self.current_exchange_buffer:
                await self._process_accumulated_input()
            
            # Stop auto-processing
            if self.auto_process_task:
                self.auto_process_task.cancel()
                try:
                    await self.auto_process_task
                except asyncio.CancelledError:
                    pass
            
            # Finalize session data
            self.current_session.end_time = time.time()
            
            # Get comprehensive analysis from client agent
            agent_analysis = self.client_agent.export_comprehensive_summary()
            self.current_session.final_analysis = agent_analysis
            
            # Calculate session metrics
            session_duration = self.current_session.end_time - self.current_session.start_time
            
            self.current_session.session_metrics = {
                "duration_seconds": session_duration,
                "duration_minutes": session_duration / 60,
                "total_exchanges": len(self.current_session.exchanges),
                "total_whisper_chunks": len(self.whisper_buffer),
                "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                "processing_errors": self.processing_errors,
                "chunks_per_exchange": len(self.whisper_buffer) / len(self.current_session.exchanges) if self.current_session.exchanges else 0
            }
            
            # Generate final JSON report
            final_report = self._generate_final_report()
            
            # Cleanup
            await self.client_agent.cleanup_rag()
            self.state = PresentationState.COMPLETED
            
            logger.info(f"Session finalized: {self.current_session.session_id}")
            
            # Notify external systems
            if self.on_session_update:
                self.on_session_update(self._get_session_status())
            
            return final_report
            
        except Exception as e:
            error_msg = f"Error finalizing session: {str(e)}"
            logger.error(error_msg)
            self.state = PresentationState.ERROR
            if self.on_error:
                self.on_error(error_msg)
            return {"error": error_msg}
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final JSON report"""
        
        if not self.current_session:
            return {"error": "No session data"}
        
        # Convert exchanges to serializable format
        exchanges_data = []
        for exchange in self.current_session.exchanges:
            exchange_data = {
                "id": exchange.id,
                "timestamp": exchange.timestamp,
                "datetime": datetime.fromtimestamp(exchange.timestamp).isoformat(),
                "salesperson_speech": exchange.salesperson_speech,  # What the salesperson said
                "client_response": exchange.client_response,        # How the client responded
                "response_time_seconds": exchange.response_time,
                "meeting_stage": exchange.meeting_stage,
                "quality_metrics": exchange.quality_metrics,
                "whisper_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "timestamp": chunk.timestamp,
                        "text": chunk.text,  # Salesperson's speech captured by Whisper
                        "confidence": chunk.confidence,
                        "speaker_id": chunk.speaker_id or "salesperson"
                    }
                    for chunk in exchange.whisper_chunks
                ]
            }
            exchanges_data.append(exchange_data)
        
        # Build comprehensive report
        final_report = {
            "session_summary": {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time,
                "end_time": self.current_session.end_time,
                "start_datetime": datetime.fromtimestamp(self.current_session.start_time).isoformat(),
                "end_datetime": datetime.fromtimestamp(self.current_session.end_time).isoformat() if self.current_session.end_time else None,
                "duration_minutes": self.current_session.session_metrics.get("duration_minutes", 0),
                "total_exchanges": len(self.current_session.exchanges),
                "session_state": self.state.value
            },
            
            "participants": {
                "client_profile": self.current_session.client_profile,
                "product_context": self.current_session.product_context
            },
            
            "conversation_exchanges": exchanges_data,
            
            "technical_metrics": {
                "whisper_processing": {
                    "total_chunks_received": len(self.whisper_buffer),
                    "chunks_per_exchange": self.current_session.session_metrics.get("chunks_per_exchange", 0),
                    "average_chunk_length": sum(len(chunk.text) for chunk in self.whisper_buffer) / len(self.whisper_buffer) if self.whisper_buffer else 0
                },
                "response_performance": {
                    "average_response_time": self.current_session.session_metrics.get("average_response_time", 0),
                    "response_times": self.response_times,
                    "processing_errors": self.processing_errors
                },
                "orchestrator_config": {
                    "silence_threshold": self.silence_threshold,
                    "fastapi_base_url": self.fastapi_base_url
                }
            },
            
            "sales_analysis": self.current_session.final_analysis,
            
            "session_quality_assessment": {
                "overall_rating": self._calculate_overall_session_quality(),
                "conversation_flow": self._assess_conversation_flow(),
                "client_engagement": self._assess_client_engagement(),
                "technical_performance": self._assess_technical_performance()
            }
        }
        
        return final_report
    
    def _calculate_overall_session_quality(self) -> str:
        """Calculate overall session quality rating"""
        
        if not self.current_session.exchanges:
            return "Poor"
        
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        error_rate = len(self.processing_errors) / len(self.current_session.exchanges)
        
        # Simple quality scoring
        if avg_response_time < 2.0 and error_rate < 0.1 and len(self.current_session.exchanges) >= 5:
            return "Excellent"
        elif avg_response_time < 3.0 and error_rate < 0.2 and len(self.current_session.exchanges) >= 3:
            return "Good"
        elif avg_response_time < 5.0 and error_rate < 0.3:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_conversation_flow(self) -> Dict[str, Any]:
        """Assess the flow and natural progression of conversation"""
        
        if not self.current_session.exchanges:
            return {"rating": "Poor", "details": "No exchanges"}
        
        # Analyze exchange patterns
        exchange_gaps = []
        for i in range(1, len(self.current_session.exchanges)):
            gap = (self.current_session.exchanges[i].timestamp - 
                   self.current_session.exchanges[i-1].timestamp)
            exchange_gaps.append(gap)
        
        avg_gap = sum(exchange_gaps) / len(exchange_gaps) if exchange_gaps else 0
        
        return {
            "rating": "Good" if 10 <= avg_gap <= 60 else "Fair",
            "average_exchange_gap_seconds": avg_gap,
            "total_exchanges": len(self.current_session.exchanges),
            "conversation_rhythm": "Natural" if 15 <= avg_gap <= 45 else "Variable"
        }
    
    def _assess_client_engagement(self) -> Dict[str, Any]:
        """Assess client engagement based on response patterns"""
        
        if not self.current_session.exchanges:
            return {"rating": "Poor", "details": "No data"}
        
        # Analyze response characteristics
        avg_response_length = sum(len(ex.client_response) for ex in self.current_session.exchanges) / len(self.current_session.exchanges)
        questions_asked = sum(1 for ex in self.current_session.exchanges if '?' in ex.client_response)
        
        return {
            "rating": "High" if avg_response_length > 100 and questions_asked > 2 else "Medium",
            "average_response_length": avg_response_length,
            "questions_asked": questions_asked,
            "engagement_level": "Active" if questions_asked / len(self.current_session.exchanges) > 0.3 else "Passive"
        }
    
    def _assess_technical_performance(self) -> Dict[str, Any]:
        """Assess technical performance of the orchestrator"""
        
        return {
            "rating": "Good" if len(self.processing_errors) == 0 else "Fair",
            "error_count": len(self.processing_errors),
            "average_processing_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "whisper_integration": "Stable",
            "agent_integration": "Stable"
        }
    
    def _get_session_status(self) -> Dict[str, Any]:
        """Get current session status for real-time updates"""
        
        if not self.current_session:
            return {"state": self.state.value, "session": None}
        
        return {
            "state": self.state.value,
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time,
            "exchanges_count": len(self.current_session.exchanges),
            "buffer_chunks": len(self.current_exchange_buffer),
            "last_activity": self.last_activity_time,
            "current_meeting_stage": getattr(self.client_agent, 'meeting_stage', 'unknown')
        }
    
    def set_callbacks(self, 
                     on_client_response: Callable[[str], None] = None,
                     on_session_update: Callable[[Dict], None] = None,
                     on_error: Callable[[str], None] = None):
        """Set callback functions for external integration"""
        
        self.on_client_response = on_client_response
        self.on_session_update = on_session_update
        self.on_error = on_error
    
    def configure_processing(self, 
                           silence_threshold: float = None):
        """Configure processing parameters"""
        
        if silence_threshold is not None:
            self.silence_threshold = silence_threshold
            logger.info(f"Updated silence threshold to {silence_threshold}s")

# Example usage and integration patterns
async def example_usage():
    """Example of how to use the orchestrator"""
    
    # Initialize orchestrator
    orchestrator = RealTimePresentationOrchestrator()
    
    # Set up callbacks for real-time updates
    def on_response(response: str):
        print(f"🎯 Client Response: {response}")
        # In real implementation, this could trigger TTS to speak the client's response
    
    def on_update(status: Dict):
        print(f"📊 Session Status: {status['state']} - Exchanges: {status.get('exchanges_count', 0)}")
    
    def on_error(error: str):
        print(f"❌ Error: {error}")
    
    orchestrator.set_callbacks(
        on_client_response=on_response,
        on_session_update=on_update,
        on_error=on_error
    )
    
    # Initialize session - YOU are TechCorp selling CloudPlatform Pro to Manufacturing Inc
    success = await orchestrator.initialize_session(
        user_company="TechCorp",           # Your company (the seller)
        user_product="CloudPlatform Pro",  # Your product
        client_company="Manufacturing Inc", # The prospective client
        client_role="CTO",                 # Who you're presenting to
        client_industry="Manufacturing",
        company_size="Enterprise"
    )
    
    if success:
        print("🚀 Presentation session initialized successfully!")
        print("💬 Start speaking about your product - Whisper will capture your speech...")
        
        # Simulate what the SALESPERSON (you) would say - captured by Whisper
        salesperson_speech_chunks = [
            WhisperChunk(time.time(), "Good morning! Thank you for taking the time to meet with us today.", 0.95),
            WhisperChunk(time.time() + 5, "I'm excited to show you how CloudPlatform Pro can revolutionize", 0.90),
            WhisperChunk(time.time() + 10, "your manufacturing operations with our advanced cloud infrastructure.", 0.88),
            WhisperChunk(time.time() + 15, "We've specifically designed this platform for enterprises like yours", 0.92),
            WhisperChunk(time.time() + 20, "with real-time analytics and 99.99% uptime guarantee.", 0.87),
            WhisperChunk(time.time() + 25, "What are your biggest challenges with your current IT infrastructure?", 0.93),
        ]
        
        print("\n🎙️ Simulating salesperson speech (Whisper input):")
        for chunk in salesperson_speech_chunks:
            print(f"🗣️ Salesperson: {chunk.text}")
            await orchestrator.process_whisper_chunk(chunk)
            await asyncio.sleep(6)  # Simulate real-time speech pace
        
        # Wait a bit for final processing
        await asyncio.sleep(3)
        
        # End session and get final report
        final_report = await orchestrator.end_session()
        
        # Save report to file
        report_filename = f"presentation_report_{final_report['session_summary']['session_id']}.json"
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n✅ Presentation completed! Full conversation report saved to: {report_filename}")
        
        # Print summary
        print(f"\n📈 Session Summary:")
        print(f"   • Duration: {final_report['session_summary']['duration_minutes']:.1f} minutes")
        print(f"   • Total Exchanges: {final_report['session_summary']['total_exchanges']}")
        print(f"   • Client Engagement: {final_report['session_quality_assessment']['client_engagement']['engagement_level']}")
        print(f"   • Overall Quality: {final_report['session_quality_assessment']['overall_rating']}")
        
        return final_report
    
    else:
        print("❌ Failed to initialize presentation session")
        return None

if __name__ == "__main__":
    asyncio.run(example_usage())