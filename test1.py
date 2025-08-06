import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from types import ModuleType
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a mock LLM module for testing
llm_module = ModuleType('Llm')
sys.modules['Llm'] = llm_module

async def mock_ask_gemini(prompt):
    """Mock Gemini LLM response for testing"""
    await asyncio.sleep(0.5)  # Simulate processing delay
    
    # Simple rule-based responses for demonstration
    if "pricing" in prompt.lower():
        return "I'm concerned about the total cost of ownership. Can you provide a detailed breakdown of all costs involved?"
    elif "implementation" in prompt.lower():
        return "Implementation timeline is critical for us. What resources will we need to allocate, and how long will the transition take?"
    elif "competitors" in prompt.lower():
        return "How does your solution compare specifically to Competitor A and Competitor B in terms of features and reliability?"
    elif "roi" in prompt.lower():
        return "ROI is important, but we need concrete metrics. Can you share case studies from similar manufacturing companies?"
    else:
        return "That's interesting. Can you elaborate on how this would integrate with our existing systems?"

llm_module.ask_gemini = mock_ask_gemini

# Data classes from orchestrator_test.py
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

# SimplifiedClientSimulationAgent from orchestrator_test.py
class SimplifiedClientSimulationAgent:
    """
    Simplified client simulation agent using Gemini via llm.ask_gemini()
    """
    
    def __init__(self):
        self.conversation_context = []
        self.session_initialized = False
        self.meeting_stage = "opening"
        self.client_profile = None
        self.product_context = None
    
    async def initialize_simple_session(self, client_company: str = "TechCorp Industries", 
                                      client_role: str = "VP of Technology",
                                      client_industry: str = "Manufacturing",
                                      company_size: str = "Enterprise",
                                      product_name: str = "Enterprise Solution",
                                      vendor_name: str = "SolutionTech"):
        """Initialize a simple session with basic context (no RAG required)"""
        # Reset session state
        self.conversation_context = []
        self.session_initialized = True
        self.meeting_stage = "opening"
        
        # Create simple client profile
        self.client_profile = {
            "company_name": client_company,
            "industry": client_industry,
            "company_size": company_size,
            "role": client_role,
            "pain_points": ["operational efficiency", "cost reduction", "scalability"],
            "budget_range": "Enterprise budget",
            "timeline": "3-6 months",
            "decision_criteria": ["ROI", "implementation complexity", "vendor stability"]
        }
        
        # Create simple product context  
        self.product_context = {
            "name": product_name,
            "vendor": vendor_name,
            "category": "Enterprise Software",
            "main_competitors": ["Competitor A", "Competitor B"],
            "pricing_model": "Contact for pricing"
        }
        
        logger.info(f"Simple session initialized: {client_role} at {client_company} evaluating {product_name}")
        return True
    
    async def generate_llm_reply(self, user_input: str) -> str:
        """Generate client response using Gemini via llm.ask_gemini()."""
        try:
            # Use existing profile if available, otherwise create default
            if self.client_profile and self.product_context:
                client_name = self.client_profile.get("company_name", "TechCorp Industries")
                client_role = self.client_profile.get("role", "VP of Technology")
                client_industry = self.client_profile.get("industry", "Manufacturing")
                client_size = self.client_profile.get("company_size", "Enterprise")
                product_name = self.product_context.get("name", "your solution")
                vendor_name = self.product_context.get("vendor", "your company")
            else:
                # Default sophisticated client persona
                client_name = "TechCorp Industries"
                client_role = "VP of Technology"
                client_industry = "Manufacturing"
                client_size = "Enterprise"
                product_name = "your solution"
                vendor_name = "your company"
            
            # Get recent conversation context (last 2 exchanges)
            recent_context = ""
            if len(self.conversation_context) > 0:
                recent_exchanges = self.conversation_context[-2:]
                context_summary = []
                for exchange in recent_exchanges:
                    role = exchange.get('role', 'unknown')
                    content = exchange.get('content', '')[:150]
                    context_summary.append(f"{role}: {content}")
                recent_context = f"\n\nRecent conversation:\n" + "\n".join(context_summary)
            
            # Update meeting stage based on conversation progress
            self._update_meeting_stage()
            
            # Create sophisticated client prompt
            prompt = f"""You are a {client_role} at {client_name}, a {client_size} {client_industry} company. You're evaluating {product_name} from {vendor_name} in a professional sales meeting.

Your characteristics:
- Experienced business executive with deep industry knowledge
- Analytical and thorough in evaluation processes
- Focused on ROI, risk mitigation, and strategic fit
- Ask probing questions about implementation, costs, and competitive advantages
- Demonstrate sophisticated understanding of technology and business implications
- Professional but appropriately skeptical tone

Current meeting stage: {self.meeting_stage}

The salesperson just said: "{user_input}"
{recent_context}

Respond as this sophisticated client would:
1. Acknowledge their input professionally
2. Show your analytical mindset and business priorities  
3. Ask a relevant follow-up question that demonstrates your expertise
4. Keep response focused and business-appropriate (2-4 sentences)

Generate only your response as the {client_role}, nothing else."""

            # Generate response using Gemini via llm.ask_gemini()
            response = await llm_module.ask_gemini(prompt)
            
            # Add to conversation context
            self.conversation_context.append({
                "role": "salesperson", 
                "content": user_input,
                "timestamp": time.time()
            })
            
            self.conversation_context.append({
                "role": "client",
                "content": response,
                "timestamp": time.time(),
                "method": "gemini_llm"
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Get a fallback response when LLM is unavailable"""
        fallback_responses = [
            "That's interesting. Can you provide more specific details about the implementation timeline and resource requirements?",
            "I appreciate the information. How does this compare to your main competitors in terms of total cost of ownership?",
            "Good point. What kind of support and training do you provide during the transition period?",
            "I see. Can you share some specific ROI metrics from similar implementations in our industry?",
            "Understood. What are the potential risks and how do you typically mitigate them during deployment?"
        ]
        
        # Use conversation length to vary fallback responses
        fallback_index = len(self.conversation_context) % len(fallback_responses)
        return fallback_responses[fallback_index]
    
    def _update_meeting_stage(self):
        """Update meeting stage based on conversation progress"""
        exchange_count = len([ctx for ctx in self.conversation_context if ctx.get('role') == 'salesperson'])
        
        if exchange_count <= 2:
            self.meeting_stage = "opening"
        elif exchange_count <= 5:
            self.meeting_stage = "discovery"
        elif exchange_count <= 8:
            self.meeting_stage = "presentation"
        elif exchange_count <= 10:
            self.meeting_stage = "objection_handling"
        else:
            self.meeting_stage = "closing"
    
    def _should_end_meeting(self) -> bool:
        """Determine if meeting should naturally end"""
        exchange_count = len([ctx for ctx in self.conversation_context if ctx.get('role') == 'salesperson'])
        return exchange_count >= 12  # End after 12+ exchanges
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status"""
        return {
            "session_initialized": self.session_initialized,
            "meeting_stage": self.meeting_stage,
            "exchange_count": len([ctx for ctx in self.conversation_context if ctx.get('role') == 'salesperson']),
            "client_profile": self.client_profile,
            "product_context": self.product_context
        }
    
    def export_conversation_summary(self) -> Dict[str, Any]:
        """Export conversation summary and analysis"""
        return {
            "conversation_exchanges": self.conversation_context,
            "final_meeting_stage": self.meeting_stage,
            "total_exchanges": len([ctx for ctx in self.conversation_context if ctx.get('role') == 'salesperson']),
            "client_engagement_metrics": {
                "avg_response_length": sum(len(ctx.get('content', '')) for ctx in self.conversation_context if ctx.get('role') == 'client') / max(1, len([ctx for ctx in self.conversation_context if ctx.get('role') == 'client'])),
                "questions_asked": sum(1 for ctx in self.conversation_context if ctx.get('role') == 'client' and '?' in ctx.get('content', '')),
                "engagement_level": "High" if len(self.conversation_context) > 10 else "Medium"
            }
        }

# RealTimePresentationOrchestrator from orchestrator_test.py (simplified)
class RealTimePresentationOrchestrator:
    """
    Orchestrator using Gemini via llm.ask_gemini()
    """
    
    def __init__(self):
        self.client_agent = SimplifiedClientSimulationAgent()
        
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
        
        # Callbacks for external integration
        self.on_client_response: Optional[Callable[[str], None]] = None
        self.on_session_update: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # Metrics tracking
        self.response_times: List[float] = []
        self.processing_errors: List[str] = []

    
    async def initialize_session(self, 
                                client_company: str = "TechCorp Industries",
                                client_role: str = "VP of Technology", 
                                client_industry: str = "Manufacturing",
                                company_size: str = "Enterprise",
                                product_name: str = "Enterprise Solution",
                                vendor_name: str = "SolutionTech") -> bool:
        """Initialize a new presentation session with simplified setup"""
        
        if self.state != PresentationState.IDLE:
            logger.error(f"Cannot initialize session in state: {self.state}")
            return False
        
        try:
            self.state = PresentationState.INITIALIZING
            logger.info("Initializing simplified presentation session...")
            
            # Initialize the client simulation agent with simple setup
            success = await self.client_agent.initialize_simple_session(
                client_company=client_company,
                client_role=client_role,
                client_industry=client_industry,
                company_size=company_size,
                product_name=product_name,
                vendor_name=vendor_name
            )
            
            if not success:
                self.state = PresentationState.ERROR
                logger.error("Failed to initialize client agent")
                return False
            
            # Create session
            session_id = f"session_{int(time.time())}_{hash(client_company + product_name) % 10000}"
            
            # Get client profile and product context from agent
            agent_status = self.client_agent.get_session_status()
            
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
        
        try:
            # Add to buffers
            self.whisper_buffer.append(whisper_chunk)
            self.current_exchange_buffer.append(whisper_chunk)
            self.last_activity_time = time.time()
            
            logger.info(f"Captured salesperson speech: {whisper_chunk.chunk_id} - '{whisper_chunk.text[:50]}...'")
            
            # Process immediately for testing
            await self._process_accumulated_input()
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing salesperson speech chunk: {str(e)}"
            logger.error(error_msg)
            self.processing_errors.append(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return False
    
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
                
                # Generate client response using the simplified agent
                client_response = await self.client_agent.generate_llm_reply(salesperson_speech)
                
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
            
            # Finalize session data
            self.current_session.end_time = time.time()
            
            # Get comprehensive analysis from client agent
            agent_analysis = self.client_agent.export_conversation_summary()
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
                "salesperson_speech": exchange.salesperson_speech,
                "client_response": exchange.client_response,
                "response_time_seconds": exchange.response_time,
                "meeting_stage": exchange.meeting_stage,
                "quality_metrics": exchange.quality_metrics,
                "whisper_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "timestamp": chunk.timestamp,
                        "text": chunk.text,
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
                    "agent_type": "gemini_llm"
                }
            },
            
            "conversation_analysis": self.current_session.final_analysis,
            
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
            "agent_integration": "Gemini LLM"
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
            "current_meeting_stage": self.client_agent.meeting_stage
        }
    
    def set_callbacks(self, 
                     on_client_response: Callable[[str], None] = None,
                     on_session_update: Callable[[Dict], None] = None,
                     on_error: Callable[[str], None] = None):
        """Set callback functions for external integration"""
        
        self.on_client_response = on_client_response
        self.on_session_update = on_session_update
        self.on_error = on_error

# Test function
async def test_real_time_conversation():
    """Test real-time conversation between salesperson and AI client"""
    
    # Initialize orchestrator
    orchestrator = RealTimePresentationOrchestrator()
    
    # Set up callbacks for real-time updates
    def on_response(response: str):
        print(f"\n🤖 Client: {response}")
    
    def on_update(status: Dict):
        print(f"\n📊 Status: {status['state']} - Exchanges: {status.get('exchanges_count', 0)}")
    
    def on_error(error: str):
        print(f"\n❌ Error: {error}")
    
    orchestrator.set_callbacks(
        on_client_response=on_response,
        on_session_update=on_update,
        on_error=on_error
    )
    
    # Initialize session
    success = await orchestrator.initialize_session(
        client_company="Manufacturing Corp",
        client_role="VP of Operations",
        client_industry="Manufacturing",
        company_size="Enterprise",
        product_name="CloudPlatform Pro",
        vendor_name="TechCorp"
    )
    
    if not success:
        print("❌ Failed to initialize session")
        return
    
    print("\n🚀 Session initialized successfully!")
    print("💬 Starting simulated conversation...\n")
    
    # Simulate salesperson speech (as if captured by Whisper)
    salesperson_inputs = [
        "Good morning! Thank you for taking the time to meet with us today.",
        "I'm excited to show you how CloudPlatform Pro can revolutionize your manufacturing operations.",
        "Our platform provides real-time analytics and 99.99% uptime guarantee.",
        "What are your biggest challenges with your current IT infrastructure?",
        "That's a great question about integration. Our platform uses standard APIs and can connect to most ERP systems.",
        "How important is real-time data visibility for your decision making?",
        "Regarding pricing, we offer flexible subscription models based on your usage.",
        "What ROI metrics are most important for your team when evaluating new solutions?",
        "We provide dedicated implementation specialists to ensure a smooth transition.",
        "Do you have any other concerns about deployment or support?"
    ]
    
    # Process each salesperson input
    for i, text in enumerate(salesperson_inputs):
        print(f"\n🗣️ Salesperson: {text}")
        
        # Create WhisperChunk (simulating Whisper output)
        chunk = WhisperChunk(
            timestamp=time.time(),
            text=text,
            confidence=0.9,
            speaker_id="salesperson",
            chunk_id=f"chunk_{i+1}"
        )
        
        # Process the chunk
        await orchestrator.process_whisper_chunk(chunk)
        
        # Wait for processing
        await asyncio.sleep(2)
    
    # Wait for final processing
    await asyncio.sleep(3)
    
    # End session and get report
    print("\n🛑 Ending session...")
    final_report = await orchestrator.end_session()
    
    # Print summary
    print("\n📈 Session Summary:")
    print(f"   • Duration: {final_report['session_summary']['duration_minutes']:.1f} minutes")
    print(f"   • Total Exchanges: {final_report['session_summary']['total_exchanges']}")
    print(f"   • Client Engagement: {final_report['session_quality_assessment']['client_engagement']['engagement_level']}")
    print(f"   • Overall Quality: {final_report['session_quality_assessment']['overall_rating']}")
    
    # Save report to file
    with open("conversation_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n✅ Full conversation report saved to: conversation_report.json")
    
    return final_report

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_real_time_conversation())