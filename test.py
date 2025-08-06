import asyncio
import signal
import sys
import time
import json
import logging
from typing import Dict, Any

# Import existing classes
from whisper1 import RealTimeWhisperProcessor, WhisperChunk
from orchestrator import RealTimePresentationOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealTimePresentationSystem:
    """Main system that integrates Whisper and Orchestrator using OOP principles"""
    
    def __init__(self, 
                 whisper_model_size: str = "base",
                 fastapi_url: str = "http://localhost:8000"):
        
        # Initialize components using composition
        self.whisper_processor = RealTimeWhisperProcessor(model_size=whisper_model_size)
        self.orchestrator = RealTimePresentationOrchestrator(fastapi_url)
        
        # System state
        self.is_running = False
        self.session_active = False
        
        # Statistics
        self.start_time = None
        self.total_chunks_processed = 0
        self.total_responses_generated = 0
    
    async def initialize(self) -> bool:
        """Initialize both Whisper and Orchestrator components"""
        
        logger.info("🚀 Initializing Real-Time Presentation System...")
        
        # Initialize Whisper processor
        if not self.whisper_processor.initialize():
            logger.error("❌ Failed to initialize Whisper processor")
            return False
        
        logger.info("✅ Whisper processor initialized")
        
        # Set up callbacks to connect components
        self._setup_component_callbacks()
        
        logger.info("✅ Presentation system ready")
        return True
    
    def _setup_component_callbacks(self):
        """Set up callbacks to connect Whisper output to Orchestrator input"""
        
        # Whisper callbacks
        def on_whisper_chunk(chunk: WhisperChunk):
            if chunk.text.strip():  # Only process non-empty transcriptions
                logger.info(f"🎙️  Salesperson: '{chunk.text[:60]}...'")
                self.total_chunks_processed += 1
                
                # Send to orchestrator asynchronously
                if self.session_active:
                    asyncio.create_task(self.orchestrator.process_whisper_chunk(chunk))
        
        def on_whisper_error(error: str):
            logger.error(f"🎙️  Whisper Error: {error}")
        
        self.whisper_processor.set_callbacks(
            on_chunk_processed=on_whisper_chunk,
            on_error=on_whisper_error
        )
        
        # Orchestrator callbacks
        def on_client_response(response: str):
            logger.info(f"🎯 Client: '{response[:60]}...'")
            self.total_responses_generated += 1
            
            # Display client response
            print(f"\n🗣️  CLIENT RESPONSE:")
            print(f"   {response}")
            print(f"   (Continue speaking as the salesperson...)\n")
        
        def on_session_update(status: Dict):
            exchanges = status.get('exchanges_count', 0)
            state = status.get('state', 'unknown')
            
            if exchanges > 0:
                logger.info(f"📊 Session Status: {state} | Exchanges: {exchanges}")
        
        def on_orchestrator_error(error: str):
            logger.error(f"🎯 Orchestrator Error: {error}")
        
        self.orchestrator.set_callbacks(
            on_client_response=on_client_response,
            on_session_update=on_session_update,
            on_error=on_orchestrator_error
        )
    
    async def start_presentation(self, 
                               user_company: str,
                               user_product: str, 
                               client_company: str = None,
                               client_role: str = "Decision Maker",
                               client_industry: str = "Technology",
                               company_size: str = "Enterprise") -> bool:
        """Start a new presentation session using existing Orchestrator"""
        
        if self.session_active:
            logger.warning("Session already active")
            return False
        
        logger.info("🎬 Starting presentation session...")
        
        # Initialize orchestrator session using its existing method
        success = await self.orchestrator.initialize_session(
            user_company=user_company,
            user_product=user_product,
            client_company=client_company or f"{client_industry} Corp",
            client_role=client_role,
            client_industry=client_industry,
            company_size=company_size
        )
        
        if not success:
            logger.error("❌ Failed to initialize orchestrator session")
            return False
        
        # Start Whisper recording using its existing method
        if not self.whisper_processor.start_recording():
            logger.error("❌ Failed to start audio recording")
            await self.orchestrator.end_session()
            return False
        
        self.session_active = True
        self.start_time = time.time()
        self.is_running = True
        
        logger.info("✅ Presentation session started successfully!")
        self._print_instructions()
        
        return True
    
    def _print_instructions(self):
        """Print user instructions"""
        
        print("\n" + "="*60)
        print("🎤 REAL-TIME PRESENTATION SIMULATION ACTIVE")
        print("="*60)
        print("📢 Start speaking as the SALESPERSON")
        print("🎯 The AI CLIENT will respond to your presentation")
        print("⏱️  Speech is processed in 5-second chunks")
        print("🛑 Press Ctrl+C to end the session")
        print("="*60)
        print("\n💬 Begin your presentation now...\n")
    
    async def stop_presentation(self) -> Dict[str, Any]:
        """Stop the presentation using existing component methods"""
        
        if not self.session_active:
            logger.warning("No active session to stop")
            return {"error": "No active session"}
        
        logger.info("🛑 Stopping presentation session...")
        
        # Stop recording using Whisper's existing method
        self.whisper_processor.stop_recording()
        
        # End orchestrator session using its existing method
        final_report = await self.orchestrator.end_session()
        
        self.session_active = False
        self.is_running = False
        
        # Add system statistics to report
        if 'error' not in final_report:
            session_duration = time.time() - self.start_time if self.start_time else 0
            
            final_report['system_stats'] = {
                'whisper_chunks_processed': self.total_chunks_processed,
                'client_responses_generated': self.total_responses_generated,
                'total_session_duration': session_duration,
                'whisper_stats': self.whisper_processor.get_stats()
            }
        
        logger.info("✅ Presentation session ended")
        return final_report
    
    async def run_interactive_session(self):
        """Run an interactive presentation session"""
        
        print("\n🎯 SALES PRESENTATION SIMULATOR")
        print("=" * 50)
        
        # Get user input for session configuration
        print("\n📋 Session Configuration:")
        user_company = input("Your company name (default: TechCorp): ").strip() or "TechCorp"
        user_product = input("Your product name (default: CloudPlatform Pro): ").strip() or "CloudPlatform Pro"
        
        print("\n👥 Client Configuration:")
        client_company = input("Client company (default: Manufacturing Inc): ").strip() or "Manufacturing Inc"
        client_role = input("Client role (default: CTO): ").strip() or "CTO"
        client_industry = input("Client industry (default: Manufacturing): ").strip() or "Manufacturing"
        company_size = input("Company size (default: Enterprise): ").strip() or "Enterprise"
        
        # Start presentation
        success = await self.start_presentation(
            user_company=user_company,
            user_product=user_product,
            client_company=client_company,
            client_role=client_role,
            client_industry=client_industry,
            company_size=company_size
        )
        
        if not success:
            print("❌ Failed to start presentation")
            return
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("🛑 Interrupt received, stopping session...")
            asyncio.create_task(self._graceful_shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Keep running until stopped
            while self.is_running:
                await asyncio.sleep(1)
                
                # Print periodic stats
                if int(time.time()) % 30 == 0:
                    self._print_periodic_stats()
        
        except KeyboardInterrupt:
            logger.info("🛑 Manual stop requested")
        
        finally:
            await self._graceful_shutdown()
    
    def _print_periodic_stats(self):
        """Print periodic statistics"""
        
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"\n📊 Session Stats: {duration/60:.1f}min | "
                  f"Chunks: {self.total_chunks_processed} | "
                  f"Responses: {self.total_responses_generated}")
    
    async def _graceful_shutdown(self):
        """Gracefully shutdown the system"""
        
        if not self.session_active:
            return
        
        print("\n🔄 Ending session and generating report...")
        
        # Stop presentation and get final report
        final_report = await self.stop_presentation()
        
        if 'error' not in final_report:
            # Save report
            timestamp = int(time.time())
            filename = f"presentation_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            # Print summary
            self._print_session_summary(final_report, filename)
        else:
            print(f"❌ Session ended with error: {final_report['error']}")
        
        # Cleanup using Whisper's existing method
        self.whisper_processor.cleanup()
        self.is_running = False
    
    def _print_session_summary(self, report: Dict, filename: str):
        """Print final session summary"""
        
        print("\n" + "="*60)
        print("📊 PRESENTATION SESSION SUMMARY")
        print("="*60)
        
        summary = report.get('session_summary', {})
        quality = report.get('session_quality_assessment', {})
        
        print(f"⏱️  Duration: {summary.get('duration_minutes', 0):.1f} minutes")
        print(f"💬 Total Exchanges: {summary.get('total_exchanges', 0)}")
        print(f"🎯 Overall Quality: {quality.get('overall_rating', 'Unknown')}")
        print(f"👥 Client Engagement: {quality.get('client_engagement', {}).get('engagement_level', 'Unknown')}")
        
        system_stats = report.get('system_stats', {})
        print(f"🎙️  Audio Chunks Processed: {system_stats.get('whisper_chunks_processed', 0)}")
        print(f"🤖 Client Responses Generated: {system_stats.get('client_responses_generated', 0)}")
        
        print(f"\n📁 Full report saved to: {filename}")
        print("="*60)

# Main execution
async def main():
    """Main entry point"""
    
    # Check if FastAPI server is needed
    print("🔧 Make sure your FastAPI agent server is running on http://localhost:8000")
    print("   (This system uses the existing RAG service)")
    
    response = input("\n🚀 Press Enter to continue or 'q' to quit: ").strip().lower()
    if response == 'q':
        print("👋 Goodbye!")
        return
    
    # Initialize system using existing components
    system = RealTimePresentationSystem(
        whisper_model_size="base",  # Use "small", "medium", or "large" for better accuracy
        fastapi_url="http://localhost:8000"
    )
    
    if not await system.initialize():
        print("❌ Failed to initialize system")
        return
    
    # Run interactive session
    await system.run_interactive_session()
    
    print("\n👋 Session completed successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ System error: {e}")