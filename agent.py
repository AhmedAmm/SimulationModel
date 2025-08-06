import json
import hashlib
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import httpx


class QuestionPriority(Enum):
    CRITICAL = "critical"  # Deal-breaker questions
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

class QuestionCategory(Enum):
    PRICING = "pricing"
    COMPETITORS = "competitors"
    PRODUCT_FEATURES = "product_features"
    COMPANY_REPUTATION = "company_reputation"
    TECHNICAL_SPECS = "technical_specs"
    SUPPORT_SERVICE = "support_service"
    IMPLEMENTATION = "implementation"
    ROI_VALUE = "roi_value"
    SECURITY_COMPLIANCE = "security_compliance"
    INTEGRATION = "integration"
    CASE_STUDIES = "case_studies"
    SCALABILITY = "scalability"
    CUSTOMIZATION = "customization"
    VENDOR_STABILITY = "vendor_stability"
    OTHER = "other"


@dataclass
class ClientProfile:
    """Enhanced client profile with detailed context"""
    company_name: str
    industry: str = "Technology"
    company_size: str = "Enterprise"  # SMB, Mid-market, Enterprise
    role: str = "Decision Maker"
    pain_points: List[str] = field(default_factory=list)
    budget_range: str = "Not specified"
    timeline: str = "Not specified"
    technical_requirements: List[str] = field(default_factory=list)
    compliance_needs: List[str] = field(default_factory=list)
    current_solutions: List[str] = field(default_factory=list)
    decision_criteria: List[str] = field(default_factory=list)

@dataclass
class ProductContext:
    """Enhanced product context"""
    name: str
    vendor: str
    category: str = "Software Solution"
    key_features: List[str] = field(default_factory=list)
    target_market: List[str] = field(default_factory=list)
    pricing_model: str = "Not specified"
    main_competitors: List[str] = field(default_factory=list)
    unique_value_props: List[str] = field(default_factory=list)
    technical_specs: Dict[str, Any] = field(default_factory=dict)
    integration_capabilities: List[str] = field(default_factory=list)

@dataclass
class Question:
    id: str
    text: str
    category: QuestionCategory
    priority: QuestionPriority
    context_needed: List[str]
    follow_up_triggers: List[str]
    business_rationale: str = ""  # Why this question matters to the client
    expected_answer_type: str = ""  # What kind of answer they're looking for
    answered: bool = False
    answer_summary: str = ""
    answer_quality_score: float = 0.0  # How well the answer addressed the question
    timestamp_asked: Optional[float] = None
    timestamp_answered: Optional[float] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:8]

class ClientSimulationAgent:
    def __init__(self, fastapi_base_url: str = "http://localhost:8000"):
        """
        Initialize the client simulation agent to use FastAPI RAG service
        
        Args:
            fastapi_base_url: Base URL of the FastAPI RAG service
        """
        self.fastapi_base_url = fastapi_base_url.rstrip('/')
        self.question_stack: List[Question] = []
        self.answered_questions: Dict[str, Question] = {}
        self.all_generated_questions: Dict[str, Question] = {}
        self.conversation_context: List[Dict] = []
        self.session_initialized = False
        self.rag_initialized = False
        
        # Enhanced context tracking
        self.client_profile: Optional[ClientProfile] = None
        self.product_context: Optional[ProductContext] = None
        self.meeting_stage = "opening"  # opening, discovery, presentation, objection_handling, closing
        self.comprehensive_context: Dict[str, Any] = {}
        
        # Configure timeout settings
        self.timeout_config = httpx.Timeout(
            connect=30.0,    # Connection timeout
            read=3600.0,      # Read timeout (30 minutes for scraping operations)
            write=30.0,      # Write timeout
            pool=30.0        # Pool timeout
        )
        
    async def initialize_rag(self) -> bool:
        """Initialize the RAG system via FastAPI"""
        try:
            timeout = aiohttp.ClientTimeout(total=1800)  # 1 minute timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.fastapi_base_url}/initialize-rag") as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"RAG initialization: {result['message']}")
                        self.rag_initialized = True
                        return True
                    else:
                        error_text = await response.text()
                        print(f"Error initializing RAG system: {response.status} - {error_text}")
                        return False
        except asyncio.TimeoutError:
            print("Timeout while initializing RAG system")
            return False
        except Exception as e:
            print(f"Error connecting to RAG service: {e}")
            return False
    
    async def cleanup_rag(self):
        """Clean up the RAG system via FastAPI"""
        if not self.rag_initialized:
            return
            
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.fastapi_base_url}/cleanup-rag") as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"RAG cleanup: {result['message']}")
                        self.rag_initialized = False
                    else:
                        error_text = await response.text()
                        print(f"Error cleaning up RAG system: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            print("Timeout while cleaning up RAG system")
        except Exception as e:
            print(f"Error connecting to RAG service for cleanup: {e}")
    
    async def query_rag(self, question: str, output_format: str = None) -> str:
        """Query the RAG system via FastAPI with enhanced context"""
        if not self.rag_initialized:
            return "RAG system not initialized"
        
        try:
            # Enhance the query with context
            enhanced_query = self._enhance_query_with_context(question)
            
            payload = {"question": enhanced_query}
            if output_format:
                payload["output_format"] = output_format
            
            timeout = aiohttp.ClientTimeout(total=300)  # 2 minutes for RAG queries
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.fastapi_base_url}/query-rag",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["answer"]
                    else:
                        error_text = await response.text()
                        return f"Error querying RAG: {response.status} - {error_text}"
        except asyncio.TimeoutError:
            return "RAG query timed out"
        except Exception as e:
            return f"Error connecting to RAG service: {str(e)}"
    
    def _enhance_query_with_context(self, query: str) -> str:
        """Enhance RAG queries with comprehensive context"""
        context_parts = []
        
        if self.client_profile:
            context_parts.append(f"Client Context: {self.client_profile.company_name} ({self.client_profile.industry}, {self.client_profile.company_size})")
            if self.client_profile.pain_points:
                context_parts.append(f"Client Pain Points: {', '.join(self.client_profile.pain_points)}")
            if self.client_profile.current_solutions:
                context_parts.append(f"Current Solutions: {', '.join(self.client_profile.current_solutions)}")
        
        if self.product_context:
            context_parts.append(f"Product Context: {self.product_context.name} by {self.product_context.vendor}")
            if self.product_context.main_competitors:
                context_parts.append(f"Main Competitors: {', '.join(self.product_context.main_competitors)}")
        
        context_parts.append(f"Meeting Stage: {self.meeting_stage}")
        
        if context_parts:
            enhanced_query = f"{' | '.join(context_parts)}\n\nQuery: {query}"
        else:
            enhanced_query = query
            
        return enhanced_query
    
    async def initialize_session(self, user_company: str, user_product: str, 
                                client_company: str = None, client_role: str = "Decision Maker",
                                client_industry: str = "Technology", company_size: str = "Enterprise",
                                additional_context: Dict[str, Any] = None):
        """Initialize a comprehensive client simulation session"""
        if not self.rag_initialized:
            if not await self.initialize_rag():
                return False
        
        # Build comprehensive context first
        await self._build_comprehensive_context(user_company, user_product, client_company, 
                                               client_role, client_industry, company_size, 
                                               additional_context or {})
        
        # Reset session state
        self.conversation_context = []
        self.question_stack = []
        self.answered_questions = {}
        self.meeting_stage = "opening"
        
        # Generate initial questions with enhanced context
        await self._generate_enhanced_initial_questions()
        self.session_initialized = True
        
        print(f"Enhanced session initialized with {len(self.question_stack)} contextual questions")
        print(f"Client Profile: {self.client_profile.company_name} ({self.client_profile.industry})")
        print(f"Product: {self.product_context.name} by {self.product_context.vendor}")
        return True
    
    async def _build_comprehensive_context(self, user_company: str, user_product: str,
                                          client_company: str, client_role: str,
                                          client_industry: str, company_size: str,
                                          additional_context: Dict[str, Any]):
        """Build comprehensive context through detailed RAG queries"""
        
        # Detailed context queries
        context_queries = {
            "vendor_analysis": f"Provide comprehensive information about {user_company}: company history, market position, financial stability, recent news, customer base, reputation, strengths and weaknesses",
            
            "product_deep_dive": f"Detailed analysis of {user_product}: core features, technical specifications, pricing models, target market, implementation requirements, support offerings, recent updates",
            
            "competitive_landscape": f"Complete competitive analysis for {user_product} and {user_company}: main competitors, market share, pricing comparison, feature comparison, competitive advantages and disadvantages",
            
            "industry_context": f"Industry analysis for {client_industry} sector: current trends, common challenges, regulatory requirements, technology adoption patterns, typical procurement processes",
            
            "client_segment_analysis": f"Analysis of {company_size} companies in {client_industry}: typical pain points, budget ranges, decision-making processes, key evaluation criteria, common objections",
            
            "implementation_realities": f"Real-world implementation experience for {user_product}: typical timelines, common challenges, success factors, resource requirements, support needs",
            
            "roi_and_value": f"ROI and value proposition analysis for {user_product}: quantifiable benefits, cost justification methods, case studies, payback periods, value metrics",
            
            "integration_ecosystem": f"Integration capabilities and ecosystem for {user_product}: supported platforms, API capabilities, data migration, existing system compatibility"
        }
        
        # Execute all context queries with error handling
        print("Building comprehensive context...")
        context_data = {}
        for category, query in context_queries.items():
            try:
                print(f"  Analyzing: {category}")
                result = await self.query_rag(query)
                context_data[category] = result
                # Add small delay to avoid overwhelming the service
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"  Warning: Failed to analyze {category}: {str(e)}")
                context_data[category] = f"Context unavailable: {str(e)}"
        
        self.comprehensive_context = context_data
        
        # Extract structured information to build profiles
        await self._extract_structured_profiles(context_data, user_company, user_product, 
                                               client_company, client_role, client_industry, 
                                               company_size, additional_context)
    
    async def _extract_structured_profiles(self, context_data: Dict[str, str], 
                                          user_company: str, user_product: str,
                                          client_company: str, client_role: str,
                                          client_industry: str, company_size: str,
                                          additional_context: Dict[str, Any]):
        """Extract structured profiles from context data"""
        
        # Extract client profile information
        client_extraction_prompt = f"""
        Based on the industry and company analysis, extract typical characteristics for a {company_size} {client_industry} company client:
        
        Context: {context_data.get('industry_context', '')[:1000]}
        Client Segment: {context_data.get('client_segment_analysis', '')[:1000]}
        
        Extract and format as JSON:
        {{
            "typical_pain_points": ["pain1", "pain2", "pain3"],
            "budget_range": "typical budget range",
            "decision_criteria": ["criteria1", "criteria2", "criteria3"],
            "technical_requirements": ["req1", "req2", "req3"],
            "compliance_needs": ["compliance1", "compliance2"],
            "current_solutions": ["solution1", "solution2"],
            "timeline": "typical evaluation timeline"
        }}
        """
        
        try:
            client_data_str = await self.query_rag(client_extraction_prompt, "json")
            client_data = json.loads(client_data_str) if client_data_str.strip().startswith('{') else {}
        except:
            client_data = {}
        
        # Extract product profile information
        product_extraction_prompt = f"""
        Based on the product and competitive analysis, extract structured product information:
        
        Product Analysis: {context_data.get('product_deep_dive', '')[:1000]}
        Competitive Analysis: {context_data.get('competitive_landscape', '')[:1000]}
        
        Extract and format as JSON:
        {{
            "key_features": ["feature1", "feature2", "feature3"],
            "target_market": ["market1", "market2"],
            "pricing_model": "pricing approach",
            "main_competitors": ["competitor1", "competitor2", "competitor3"],
            "unique_value_props": ["prop1", "prop2", "prop3"],
            "integration_capabilities": ["integration1", "integration2"],
            "technical_specs": {{"spec1": "value1", "spec2": "value2"}}
        }}
        """
        
        try:
            product_data_str = await self.query_rag(product_extraction_prompt, "json")
            product_data = json.loads(product_data_str) if product_data_str.strip().startswith('{') else {}
        except:
            product_data = {}
        
        # Build client profile
        self.client_profile = ClientProfile(
            company_name=client_company or f"Prospective {client_industry} Company",
            industry=client_industry,
            company_size=company_size,
            role=client_role,
            pain_points=client_data.get('typical_pain_points', []),
            budget_range=client_data.get('budget_range', 'To be determined'),
            timeline=client_data.get('timeline', '3-6 months'),
            technical_requirements=client_data.get('technical_requirements', []),
            compliance_needs=client_data.get('compliance_needs', []),
            current_solutions=client_data.get('current_solutions', []),
            decision_criteria=client_data.get('decision_criteria', [])
        )
        
        # Build product context
        self.product_context = ProductContext(
            name=user_product,
            vendor=user_company,
            category=additional_context.get('product_category', 'Enterprise Software'),
            key_features=product_data.get('key_features', []),
            target_market=product_data.get('target_market', []),
            pricing_model=product_data.get('pricing_model', 'Contact for pricing'),
            main_competitors=product_data.get('main_competitors', []),
            unique_value_props=product_data.get('unique_value_props', []),
            technical_specs=product_data.get('technical_specs', {}),
            integration_capabilities=product_data.get('integration_capabilities', [])
        )
    
    async def _generate_enhanced_initial_questions(self):
        """Generate highly detailed, context-aware initial questions"""
        
        enhanced_prompt = f"""
        You are simulating a sophisticated {self.client_profile.role} from {self.client_profile.company_name}, 
        a {self.client_profile.company_size} {self.client_profile.industry} company, evaluating {self.product_context.name} 
        from {self.product_context.vendor}.
        
        CLIENT CONTEXT:
        - Company: {self.client_profile.company_name} ({self.client_profile.industry}, {self.client_profile.company_size})
        - Role: {self.client_profile.role}
        - Key Pain Points: {', '.join(self.client_profile.pain_points) if self.client_profile.pain_points else 'Standard business challenges'}
        - Budget Considerations: {self.client_profile.budget_range}
        - Timeline: {self.client_profile.timeline}
        - Current Solutions: {', '.join(self.client_profile.current_solutions) if self.client_profile.current_solutions else 'Legacy systems'}
        - Decision Criteria: {', '.join(self.client_profile.decision_criteria) if self.client_profile.decision_criteria else 'ROI, functionality, support'}
        - Technical Requirements: {', '.join(self.client_profile.technical_requirements) if self.client_profile.technical_requirements else 'Standard enterprise requirements'}
        - Compliance Needs: {', '.join(self.client_profile.compliance_needs) if self.client_profile.compliance_needs else 'Industry standard compliance'}
        
        PRODUCT/VENDOR CONTEXT:
        - Product: {self.product_context.name} by {self.product_context.vendor}
        - Key Features: {', '.join(self.product_context.key_features) if self.product_context.key_features else 'Core functionality'}
        - Main Competitors: {', '.join(self.product_context.main_competitors) if self.product_context.main_competitors else 'Market alternatives'}
        - Pricing Model: {self.product_context.pricing_model}
        - Target Market: {', '.join(self.product_context.target_market) if self.product_context.target_market else 'Enterprise clients'}
        
        COMPREHENSIVE MARKET INTELLIGENCE:
        Vendor Analysis: {self.comprehensive_context.get('vendor_analysis', '')[:500]}...
        Product Details: {self.comprehensive_context.get('product_deep_dive', '')[:500]}...
        Competitive Landscape: {self.comprehensive_context.get('competitive_landscape', '')[:500]}...
        Industry Context: {self.comprehensive_context.get('industry_context', '')[:500]}...
        
        Generate 12-15 highly specific, detailed questions that this sophisticated client would ask. Each question should:
        1. Be specific to the client's industry, company size, and role
        2. Reference specific business challenges or requirements
        3. Show deep understanding of the market and competitive landscape
        4. Include specific details that demonstrate research and preparation
        5. Address real business concerns and decision criteria
        6. Be sophisticated enough for a {self.client_profile.role} at a {self.client_profile.company_size} company
        
        Question categories to cover:
        - Competitive differentiation (specific competitor comparisons)
        - ROI and value quantification (specific metrics and timeframes)
        - Technical architecture and scalability
        - Implementation complexity and resource requirements
        - Integration with existing systems
        - Security and compliance (industry-specific)
        - Vendor stability and roadmap
        - Support and service levels
        - Pricing and contract terms
        - Case studies and references (industry-specific)
        - Risk mitigation and contingency planning
        - Customization and flexibility
        
        Format as JSON array:
        [
            {{
                "text": "Highly detailed, specific question with business context...",
                "category": "category_name",
                "priority": "critical|high|medium|low",
                "context_needed": ["specific context area 1", "specific context area 2"],
                "follow_up_triggers": ["keyword1", "keyword2"],
                "business_rationale": "Why this question is critical for the client's decision",
                "expected_answer_type": "What kind of answer the client is looking for"
            }}
        ]
        """
        
        try:
            response = await self.query_rag(enhanced_prompt)
            
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                questions_data = json.loads(json_str)
            else:
                questions_data = json.loads(response)
            
            # Convert to enhanced Question objects
            for q_data in questions_data:
                question = Question(
                    id="",
                    text=q_data["text"],
                    category=QuestionCategory(q_data["category"]),
                    priority=QuestionPriority(q_data["priority"]),
                    context_needed=q_data["context_needed"],
                    follow_up_triggers=q_data.get("follow_up_triggers", []),
                    business_rationale=q_data.get("business_rationale", ""),
                    expected_answer_type=q_data.get("expected_answer_type", ""),
                    timestamp_asked=time.time()
                )
                self.question_stack.append(question)
                self.all_generated_questions[question.id] = question
                
            # Sort by priority (critical first, then high, medium, low)
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            self.question_stack.sort(key=lambda x: priority_order[x.priority.value])
            
        except Exception as e:
            print(f"Error generating enhanced questions: {e}")
            await self._add_enhanced_fallback_questions()
    
    async def _add_enhanced_fallback_questions(self):
        """Add sophisticated fallback questions if generation fails"""
        fallback_questions = [
            Question("", f"How does {self.product_context.name} specifically differentiate from {', '.join(self.product_context.main_competitors[:2]) if self.product_context.main_competitors else 'leading competitors'} in terms of {', '.join(self.client_profile.decision_criteria[:2]) if self.client_profile.decision_criteria else 'functionality and ROI'}?", 
                    QuestionCategory.COMPETITORS, QuestionPriority.CRITICAL, 
                    ["competitive analysis", "differentiation"], ["competitors", "differentiation"],
                    f"Critical for {self.client_profile.role} to justify vendor selection", "Specific feature and value comparisons"),
            
            Question("", f"What specific ROI metrics and timeframes can you provide for {self.client_profile.company_size} {self.client_profile.industry} companies, and how do you measure success against our current {', '.join(self.client_profile.current_solutions[:1]) if self.client_profile.current_solutions else 'existing systems'}?", 
                    QuestionCategory.ROI_VALUE, QuestionPriority.CRITICAL, 
                    ["roi metrics", "case studies"], ["roi", "metrics", "timeframe"],
                    "Essential for business case development and stakeholder buy-in", "Quantifiable metrics with industry benchmarks"),
            
            Question("", f"Given our {self.client_profile.company_size} scale and {', '.join(self.client_profile.technical_requirements[:2]) if self.client_profile.technical_requirements else 'enterprise requirements'}, what are the specific implementation challenges and resource requirements we should anticipate?", 
                    QuestionCategory.IMPLEMENTATION, QuestionPriority.HIGH, 
                    ["implementation process", "resource planning"], ["implementation", "resources", "timeline"],
                    "Critical for project planning and resource allocation", "Detailed implementation roadmap with resource estimates"),
            
            Question("", f"How does {self.product_context.vendor} ensure compliance with {', '.join(self.client_profile.compliance_needs[:2]) if self.client_profile.compliance_needs else 'industry regulations'} and what security certifications do you maintain relevant to {self.client_profile.industry} sector?", 
                    QuestionCategory.SECURITY_COMPLIANCE, QuestionPriority.CRITICAL, 
                    ["security measures", "compliance"], ["security", "compliance", "certifications"],
                    "Mandatory for risk management and regulatory compliance", "Specific certifications and compliance documentation")
        ]
        
        self.question_stack.extend(fallback_questions)
        for question in fallback_questions:
            self.all_generated_questions[question.id] = question
    
    async def _generate_contextual_questions(self, user_input: str):
        """Generate sophisticated follow-up questions based on conversation context"""
        
        if len(self.question_stack) < 3:  # Maintain question pipeline
            contextual_prompt = f"""
            You are a {self.client_profile.role} from {self.client_profile.company_name} ({self.client_profile.industry}, {self.client_profile.company_size}).
            
            The salesperson just said: "{user_input}"
            
            Recent conversation context:
            {json.dumps(self.conversation_context[-3:], indent=2)}
            
            Client Profile Context:
            - Pain Points: {', '.join(self.client_profile.pain_points) if self.client_profile.pain_points else 'Standard challenges'}
            - Decision Criteria: {', '.join(self.client_profile.decision_criteria) if self.client_profile.decision_criteria else 'ROI and functionality'}
            - Technical Requirements: {', '.join(self.client_profile.technical_requirements) if self.client_profile.technical_requirements else 'Enterprise standards'}
            
            Meeting Stage: {self.meeting_stage}
            
            Generate 2-4 sophisticated follow-up questions that:
            1. Build naturally on what was just discussed
            2. Probe deeper into business implications
            3. Reference your specific industry context and company needs
            4. Show strategic thinking appropriate for your role
            5. Address potential concerns or risks
            6. Demonstrate thorough evaluation process
            
            Make questions specific, detailed, and business-focused. Include metrics, timelines, and specific business scenarios where relevant.
            
            Use the same JSON format as before with business_rationale and expected_answer_type.
            """
            
            try:
                response = await self.query_rag(contextual_prompt)
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    questions_data = json.loads(json_str)
                else:
                    questions_data = json.loads(response)
                
                for q_data in questions_data:
                    if not self._is_duplicate_question(q_data["text"]):
                        question = Question(
                            id="",
                            text=q_data["text"],
                            category=QuestionCategory(q_data["category"]),
                            priority=QuestionPriority(q_data["priority"]),
                            context_needed=q_data["context_needed"],
                            follow_up_triggers=q_data.get("follow_up_triggers", []),
                            business_rationale=q_data.get("business_rationale", ""),
                            expected_answer_type=q_data.get("expected_answer_type", ""),
                            timestamp_asked=time.time()
                        )
                        self.question_stack.append(question)
                        self.all_generated_questions[question.id] = question
                        
            except Exception as e:
                print(f"Error generating contextual questions: {e}")
    
    def _is_duplicate_question(self, new_question_text: str) -> bool:
        """Check if a question is duplicate or very similar to existing ones"""
        new_question_lower = new_question_text.lower()
        for existing_question in self.all_generated_questions.values():
            existing_lower = existing_question.text.lower()
            # Simple similarity check - could be enhanced with more sophisticated comparison
            if len(set(new_question_lower.split()) & set(existing_lower.split())) > len(new_question_lower.split()) * 0.6:
                return True
        return False
    
    def _update_answered_questions(self, user_input: str):
        """Enhanced question answering detection with quality scoring"""
        for question in self.question_stack[:]:
            if not question.answered:
                input_lower = user_input.lower()
                
                # Enhanced matching logic
                trigger_matches = sum(1 for keyword in question.follow_up_triggers 
                                    if keyword.lower() in input_lower)
                context_matches = sum(1 for context in question.context_needed 
                                    if any(word in input_lower for word in context.lower().split()))
                
                # Calculate match score
                total_triggers = len(question.follow_up_triggers) + len(question.context_needed)
                if total_triggers > 0:
                    match_score = (trigger_matches + context_matches) / total_triggers
                    
                    if match_score >= 0.4:  # 40% match threshold
                        question.answered = True
                        question.answer_summary = user_input[:300] + "..." if len(user_input) > 300 else user_input
                        question.answer_quality_score = min(match_score * 2, 1.0)  # Convert to quality score
                        question.timestamp_answered = time.time()
                        self.answered_questions[question.id] = question
    
    def _update_meeting_stage(self):
        """Update meeting stage based on conversation progress"""
        answered_count = len(self.answered_questions)
        conversation_length = len(self.conversation_context)
        
        if conversation_length <= 2:
            self.meeting_stage = "opening"
        elif answered_count <= 3:
            self.meeting_stage = "discovery"
        elif answered_count <= 8:
            self.meeting_stage = "presentation"
        elif answered_count <= 12:
            self.meeting_stage = "objection_handling"
        else:
            self.meeting_stage = "closing"
    
    async def simulate_client_response(self, user_input: str) -> str:
        """Enhanced client simulation with comprehensive context awareness"""
        if not self.session_initialized:
            return "Error: Session not initialized. Please call initialize_session() first."
        
        try:
            # Add user input to conversation context
            self.conversation_context.append({
                "role": "salesperson", 
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Update meeting stage
            self._update_meeting_stage()
            
            # Analyze answered questions
            self._update_answered_questions(user_input)
            
            # Generate contextual follow-ups
            await self._generate_contextual_questions(user_input)
            
            # Get next question
            next_question = self._get_next_question()
            
            # Check if meeting should end
            if not next_question and self._should_end_meeting():
                return await self._generate_enhanced_meeting_conclusion()
            
            # Generate sophisticated client response
            response_prompt = f"""
            You are {self.client_profile.role} at {self.client_profile.company_name}, a {self.client_profile.company_size} {self.client_profile.industry} company.
            
            CLIENT PROFILE:
            - Role: {self.client_profile.role}
            - Company: {self.client_profile.company_name} ({self.client_profile.industry}, {self.client_profile.company_size})
            - Key Pain Points: {', '.join(self.client_profile.pain_points) if self.client_profile.pain_points else 'Standard business challenges'}
            - Decision Criteria: {', '.join(self.client_profile.decision_criteria) if self.client_profile.decision_criteria else 'ROI and functionality'}
            - Budget Range: {self.client_profile.budget_range}
            - Timeline: {self.client_profile.timeline}
            - Current Solutions: {', '.join(self.client_profile.current_solutions) if self.client_profile.current_solutions else 'Legacy systems'}
            
            PRODUCT CONTEXT:
            - Evaluating: {self.product_context.name} by {self.product_context.vendor}
            - Key Competitors: {', '.join(self.product_context.main_competitors) if self.product_context.main_competitors else 'Market alternatives'}
            
            MEETING CONTEXT:
            - Current Stage: {self.meeting_stage}
            - Questions Answered: {len(self.answered_questions)}
            - Conversation Length: {len(self.conversation_context)} exchanges
            
            The salesperson just said: "{user_input}"
            
            Recent conversation: {json.dumps(self.conversation_context[-2:], indent=2)}
            
            RESPONSE GUIDELINES:
            1. Respond as a sophisticated {self.client_profile.role} would
            2. Show appropriate business acumen for your role and company size
            3. Reference your specific industry context and challenges
            4. Demonstrate that you're conducting a thorough evaluation
            5. Show appropriate skepticism and due diligence
            6. Ask the next question naturally as part of your response
            
            Next question to incorporate naturally: "{next_question.text if next_question else 'Continue the evaluation'}"
            
            Business rationale for next question: {next_question.business_rationale if next_question else 'Ongoing evaluation'}
            
            Generate a professional, contextually aware response (3-5 sentences) that:
            - Acknowledges the salesperson's input appropriately
            - Shows your evaluation mindset and business priorities
            - Naturally transitions to the next question
            - Reflects your role's perspective and concerns
            """
            
            client_response = await self.query_rag(response_prompt)
            
            # Add client response to context
            self.conversation_context.append({
                "role": "client",
                "content": client_response,
                "timestamp": time.time(),
                "meeting_stage": self.meeting_stage
            })
            
            return client_response
            
        except Exception as e:
            error_msg = f"Error generating client response: {str(e)}"
            print(error_msg)
            
            # Enhanced fallback response
            next_question = self._get_next_question()
            if not next_question and self._should_end_meeting():
                return await self._generate_enhanced_meeting_conclusion()
            
            fallback_responses = [
                f"That's interesting from a {self.client_profile.industry} perspective. {next_question.text if next_question else 'Can you elaborate further?'}",
                f"As a {self.client_profile.role}, I need to understand this better. {next_question.text if next_question else 'What else should we consider?'}",
                f"Given our {self.client_profile.company_size} scale, {next_question.text if next_question else 'how does this apply to our situation?'}"
            ]
            
            return fallback_responses[len(self.conversation_context) % len(fallback_responses)]
    
    def _get_next_question(self) -> Optional[Question]:
        """Get the next prioritized question"""
        # Remove answered questions from stack
        self.question_stack = [q for q in self.question_stack if not q.answered]
        
        if not self.question_stack:
            return None
        
        # Prioritize by meeting stage and question priority
        stage_priorities = {
            "opening": [QuestionCategory.COMPANY_REPUTATION, QuestionCategory.PRODUCT_FEATURES],
            "discovery": [QuestionCategory.COMPETITORS, QuestionCategory.ROI_VALUE, QuestionCategory.TECHNICAL_SPECS],
            "presentation": [QuestionCategory.IMPLEMENTATION, QuestionCategory.INTEGRATION, QuestionCategory.CASE_STUDIES],
            "objection_handling": [QuestionCategory.PRICING, QuestionCategory.SUPPORT_SERVICE, QuestionCategory.SECURITY_COMPLIANCE],
            "closing": [QuestionCategory.VENDOR_STABILITY, QuestionCategory.CUSTOMIZATION]
        }
        
        preferred_categories = stage_priorities.get(self.meeting_stage, [])
        
        # First try to get a question from preferred categories
        for category in preferred_categories:
            for question in self.question_stack:
                if question.category == category:
                    self.question_stack.remove(question)
                    return question
        
        # Fall back to highest priority question
        self.question_stack.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x.priority.value])
        return self.question_stack.pop(0)
    
    def _should_end_meeting(self) -> bool:
        """Enhanced meeting end detection"""
        # Check various end conditions
        no_questions = len(self.question_stack) == 0
        sufficient_conversation = len(self.conversation_context) >= 10
        
        # Check critical questions answered
        critical_questions = [q for q in self.all_generated_questions.values() if q.priority == QuestionPriority.CRITICAL]
        critical_answered = sum(1 for q in critical_questions if q.answered)
        critical_coverage = critical_answered / len(critical_questions) if critical_questions else 1.0
        
        # Check category coverage
        important_categories = [
            QuestionCategory.PRICING, QuestionCategory.COMPETITORS, 
            QuestionCategory.PRODUCT_FEATURES, QuestionCategory.ROI_VALUE,
            QuestionCategory.IMPLEMENTATION, QuestionCategory.SECURITY_COMPLIANCE
        ]
        
        covered_categories = set(
            q.category for q in self.all_generated_questions.values() 
            if q.answered and q.category in important_categories
        )
        
        category_coverage = len(covered_categories) / len(important_categories)
        
        # End meeting if comprehensive evaluation completed
        return (no_questions and sufficient_conversation) or \
               (critical_coverage >= 0.8 and category_coverage >= 0.6) or \
               (self.meeting_stage == "closing" and sufficient_conversation)
    
    async def _generate_enhanced_meeting_conclusion(self) -> str:
        """Generate sophisticated meeting conclusion"""
        
        # Calculate meeting metrics
        total_questions = len(self.all_generated_questions)
        answered_questions = len([q for q in self.all_generated_questions.values() if q.answered])
        avg_quality_score = sum(q.answer_quality_score for q in self.answered_questions.values()) / len(self.answered_questions) if self.answered_questions else 0
        
        covered_categories = list(set(q.category.value for q in self.answered_questions.values()))
        high_priority_answered = len([q for q in self.answered_questions.values() if q.priority in [QuestionPriority.CRITICAL, QuestionPriority.HIGH]])
        
        conclusion_prompt = f"""
        You are concluding a sales meeting as {self.client_profile.role} at {self.client_profile.company_name}.
        
        CLIENT CONTEXT:
        - Company: {self.client_profile.company_name} ({self.client_profile.industry}, {self.client_profile.company_size})
        - Role: {self.client_profile.role}
        - Timeline: {self.client_profile.timeline}
        - Decision Criteria: {', '.join(self.client_profile.decision_criteria) if self.client_profile.decision_criteria else 'Standard criteria'}
        
        MEETING ANALYSIS:
        - Questions Asked: {answered_questions}/{total_questions}
        - Categories Covered: {', '.join(covered_categories)}
        - High Priority Topics Addressed: {high_priority_answered}
        - Answer Quality Average: {avg_quality_score:.1f}/1.0
        - Meeting Duration: {len(self.conversation_context)} exchanges
        - Final Stage: {self.meeting_stage}
        
        CONVERSATION SUMMARY:
        {json.dumps(self.conversation_context[-3:], indent=2)}
        
        Generate a professional, role-appropriate conclusion that:
        1. Thanks the salesperson professionally
        2. Acknowledges the information quality and completeness
        3. Summarizes key evaluation points relevant to your decision criteria
        4. Outlines realistic next steps based on your company's process
        5. Indicates appropriate level of interest based on the meeting quality
        6. Mentions internal stakeholders or processes as appropriate for your role
        7. Provides clear timeline expectations
        
        Make it sound natural for a {self.client_profile.role} at a {self.client_profile.company_size} {self.client_profile.industry} company.
        Keep it professional and substantive (4-6 sentences).
        """
        
        try:
            conclusion = await self.query_rag(conclusion_prompt)
            
            # Add conclusion to conversation context
            self.conversation_context.append({
                "role": "client",
                "content": conclusion,
                "type": "meeting_conclusion",
                "timestamp": time.time(),
                "meeting_metrics": {
                    "questions_answered": answered_questions,
                    "categories_covered": len(covered_categories),
                    "avg_quality_score": avg_quality_score
                }
            })
            
            return conclusion
            
        except Exception as e:
            print(f"Error generating enhanced conclusion: {e}")
            # Sophisticated fallback based on meeting quality
            if avg_quality_score > 0.7 and answered_questions > 8:
                return f"Thank you for the comprehensive discussion today. You've addressed our key concerns around {', '.join(covered_categories[:3])} very thoroughly. I'll be presenting this to our {self.client_profile.industry} leadership team and our technical evaluation committee. Based on what we've discussed, I'm optimistic about moving forward. We should have a decision within our {self.client_profile.timeline} timeline. I'll follow up with any additional questions from our stakeholders."
            elif avg_quality_score > 0.5 and answered_questions > 5:
                return f"I appreciate the time you've taken to walk through {self.product_context.name} with me today. We've covered the main areas I needed to understand. I'll need to discuss this internally with our team and compare it against our other options. We'll get back to you within the next two weeks with our evaluation results and next steps."
            else:
                return f"Thank you for the introduction to {self.product_context.name}. I'll need to gather some additional information internally and discuss this with our evaluation committee. We'll be in touch if we decide to move forward with a more detailed evaluation. Have a good day."
    
    # Enhanced status and reporting methods
    def get_enhanced_session_status(self) -> Dict[str, Any]:
        """Get comprehensive session status with detailed metrics"""
        if not self.session_initialized:
            return {"error": "Session not initialized"}
        
        # Calculate detailed metrics
        total_questions = len(self.all_generated_questions)
        answered_questions = len(self.answered_questions)
        
        priority_breakdown = {}
        category_breakdown = {}
        
        for priority in QuestionPriority:
            priority_questions = [q for q in self.all_generated_questions.values() if q.priority == priority]
            priority_answered = [q for q in priority_questions if q.answered]
            priority_breakdown[priority.value] = {
                "total": len(priority_questions),
                "answered": len(priority_answered),
                "percentage": (len(priority_answered) / len(priority_questions) * 100) if priority_questions else 0
            }
        
        for category in QuestionCategory:
            category_questions = [q for q in self.all_generated_questions.values() if q.category == category]
            category_answered = [q for q in category_questions if q.answered]
            category_breakdown[category.value] = {
                "total": len(category_questions),
                "answered": len(category_answered),
                "percentage": (len(category_answered) / len(category_questions) * 100) if category_questions else 0
            }
        
        avg_quality_score = sum(q.answer_quality_score for q in self.answered_questions.values()) / len(self.answered_questions) if self.answered_questions else 0
        
        return {
            "session_info": {
                "initialized": self.session_initialized,
                "meeting_stage": self.meeting_stage,
                "rag_system_active": self.rag_initialized,
                "fastapi_base_url": self.fastapi_base_url
            },
            "client_profile": {
                "company": self.client_profile.company_name,
                "industry": self.client_profile.industry,
                "size": self.client_profile.company_size,
                "role": self.client_profile.role,
                "timeline": self.client_profile.timeline,
                "budget_range": self.client_profile.budget_range
            },
            "product_context": {
                "name": self.product_context.name,
                "vendor": self.product_context.vendor,
                "category": self.product_context.category,
                "pricing_model": self.product_context.pricing_model
            },
            "question_metrics": {
                "total_generated": total_questions,
                "answered": answered_questions,
                "remaining": len(self.question_stack),
                "answer_rate": (answered_questions / total_questions * 100) if total_questions else 0,
                "avg_quality_score": round(avg_quality_score, 2)
            },
            "priority_breakdown": priority_breakdown,
            "category_breakdown": category_breakdown,
            "conversation_metrics": {
                "total_exchanges": len(self.conversation_context),
                "duration_minutes": (time.time() - self.conversation_context[0]["timestamp"]) / 60 if self.conversation_context else 0
            },
            "meeting_assessment": {
                "should_end": self._should_end_meeting(),
                "completion_percentage": self._calculate_completion_percentage(),
                "meeting_quality": self._assess_meeting_quality()
            }
        }
    
    def _calculate_completion_percentage(self) -> float:
        """Calculate overall meeting completion percentage"""
        if not self.all_generated_questions:
            return 0.0
        
        # Weight by priority
        priority_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        
        total_weight = sum(priority_weights[q.priority.value] for q in self.all_generated_questions.values())
        answered_weight = sum(priority_weights[q.priority.value] for q in self.answered_questions.values())
        
        return (answered_weight / total_weight * 100) if total_weight else 0.0
    
    def _assess_meeting_quality(self) -> str:
        """Assess overall meeting quality"""
        completion_pct = self._calculate_completion_percentage()
        avg_quality = sum(q.answer_quality_score for q in self.answered_questions.values()) / len(self.answered_questions) if self.answered_questions else 0
        
        if completion_pct >= 80 and avg_quality >= 0.8:
            return "Excellent"
        elif completion_pct >= 60 and avg_quality >= 0.6:
            return "Good"
        elif completion_pct >= 40 and avg_quality >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def get_detailed_questions_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of all questions"""
        analysis = {
            "questions_by_priority": {},
            "questions_by_category": {},
            "unanswered_critical": [],
            "top_quality_answers": [],
            "areas_needing_attention": []
        }
        
        # Group by priority
        for priority in QuestionPriority:
            questions = [q for q in self.all_generated_questions.values() if q.priority == priority]
            analysis["questions_by_priority"][priority.value] = [
                {
                    "id": q.id,
                    "text": q.text,
                    "category": q.category.value,
                    "answered": q.answered,
                    "quality_score": q.answer_quality_score,
                    "business_rationale": q.business_rationale
                } for q in questions
            ]
        
        # Group by category
        for category in QuestionCategory:
            questions = [q for q in self.all_generated_questions.values() if q.category == category]
            analysis["questions_by_category"][category.value] = [
                {
                    "id": q.id,
                    "text": q.text,
                    "priority": q.priority.value,
                    "answered": q.answered,
                    "quality_score": q.answer_quality_score
                } for q in questions
            ]
        
        # Identify unanswered critical questions
        analysis["unanswered_critical"] = [
            {
                "text": q.text,
                "category": q.category.value,
                "business_rationale": q.business_rationale,
                "expected_answer_type": q.expected_answer_type
            }
            for q in self.all_generated_questions.values()
            if q.priority == QuestionPriority.CRITICAL and not q.answered
        ]
        
        # Top quality answers
        answered_questions = [q for q in self.all_generated_questions.values() if q.answered]
        answered_questions.sort(key=lambda x: x.answer_quality_score, reverse=True)
        analysis["top_quality_answers"] = [
            {
                "text": q.text,
                "category": q.category.value,
                "quality_score": q.answer_quality_score,
                "answer_summary": q.answer_summary[:200] + "..." if len(q.answer_summary) > 200 else q.answer_summary
            }
            for q in answered_questions[:5]
        ]
        
        # Areas needing attention
        category_scores = {}
        for category in QuestionCategory:
            category_questions = [q for q in self.answered_questions.values() if q.category == category]
            if category_questions:
                avg_score = sum(q.answer_quality_score for q in category_questions) / len(category_questions)
                category_scores[category.value] = avg_score
        
        analysis["areas_needing_attention"] = [
            {"category": cat, "avg_quality": score}
            for cat, score in sorted(category_scores.items(), key=lambda x: x[1])[:3]
            if score < 0.6
        ]
        
        return analysis
    
    def export_comprehensive_summary(self) -> Dict[str, Any]:
        """Export comprehensive meeting summary"""
        return {
            "session_overview": self.get_enhanced_session_status(),
            "detailed_analysis": self.get_detailed_questions_analysis(),
            "conversation_timeline": [
                {
                    "timestamp": entry.get("timestamp", 0),
                    "role": entry["role"],
                    "content": entry["content"][:100] + "..." if len(entry["content"]) > 100 else entry["content"],
                    "meeting_stage": entry.get("meeting_stage", "unknown")
                }
                for entry in self.conversation_context
            ],
            "client_context": {
                "profile": {
                    "company": self.client_profile.company_name,
                    "industry": self.client_profile.industry,
                    "size": self.client_profile.company_size,
                    "role": self.client_profile.role,
                    "pain_points": self.client_profile.pain_points,
                    "decision_criteria": self.client_profile.decision_criteria,
                    "technical_requirements": self.client_profile.technical_requirements,
                    "compliance_needs": self.client_profile.compliance_needs
                },
                "product_evaluated": {
                    "name": self.product_context.name,
                    "vendor": self.product_context.vendor,
                    "key_features": self.product_context.key_features,
                    "competitors": self.product_context.main_competitors,
                    "value_props": self.product_context.unique_value_props
                }
            },
            "recommendations": self._generate_meeting_recommendations()
        }
    
    def _generate_meeting_recommendations(self) -> List[str]:
        """Generate recommendations for the salesperson"""
        recommendations = []
        
        # Check unanswered critical questions
        unanswered_critical = [q for q in self.all_generated_questions.values() 
                              if q.priority == QuestionPriority.CRITICAL and not q.answered]
        if unanswered_critical:
            recommendations.append(f"Address {len(unanswered_critical)} unanswered critical questions in follow-up")
        
        # Check low quality answers
        low_quality_answers = [q for q in self.answered_questions.values() if q.answer_quality_score < 0.5]
        if low_quality_answers:
            recommendations.append(f"Provide more detailed responses for {len(low_quality_answers)} questions that received low satisfaction scores")
        
        # Check category gaps
        important_categories = [QuestionCategory.PRICING, QuestionCategory.COMPETITORS, QuestionCategory.ROI_VALUE]
        uncovered_important = [cat for cat in important_categories 
                              if not any(q.category == cat and q.answered for q in self.all_generated_questions.values())]
        if uncovered_important:
            recommendations.append(f"Focus on uncovered important areas: {', '.join([cat.value for cat in uncovered_important])}")
        
        # Meeting stage recommendations
        if self.meeting_stage in ["opening", "discovery"] and len(self.answered_questions) > 10:
            recommendations.append("Consider moving to presentation/demo phase based on discovery completion")
        
        if not recommendations:
            recommendations.append("Meeting conducted comprehensively - follow up with proposed next steps")
        
        return recommendations

# Enhanced example usage with improved error handling and timeouts
async def enhanced_demo():
    """Enhanced demonstration of the improved ClientSimulationAgent with timeout fixes"""

    fastapi_base_url = "http://localhost:8000"
    
    try:
        # Step 1: Call scrape-all with proper timeout handling
        print("=== Triggering Scrape-All (with extended timeout) ===")
        
        # Configure client with extended timeouts
        timeout_config = httpx.Timeout(
            connect=60.0,    # 30 seconds to connect
            read=3600.0,      # 60 minutes for scraping operations (can be long)
            write=60.0,      # 30 seconds to write
            pool=60.0        # 30 seconds for connection pool
        )
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            scrape_all_payload = {
                "user_company": "CloudTech Solutions",
                "user_product": "EnterpriseCloud Pro",
                "description": "Cloud Infrastructure Platform for industrial operations",
                "client_company": "MegaCorp Industries"
            }
            
            print("Sending scrape-all request (this may take several minutes)...")
            try:
                scrape_all_response = await client.post(f"{fastapi_base_url}/scrape-all", json=scrape_all_payload)
                print(f"Scrape-All Response: {scrape_all_response.json()}")
            except httpx.ReadTimeout:
                print("Scrape-all operation timed out. This might be normal for large scraping operations.")
                print("Continuing with simulation using available data...")
            except httpx.ConnectTimeout:
                print("Could not connect to FastAPI service. Please ensure it's running on localhost:8000")
                return
            except Exception as e:
                print(f"Error during scrape-all: {e}")
                print("Continuing with simulation...")

        # Step 2: Run the simulation with enhanced error handling
        print("\n=== Initializing Client Simulation Agent ===")
        agent = ClientSimulationAgent(fastapi_base_url=fastapi_base_url)
        
        success = await agent.initialize_session(
            user_company="CloudTech Solutions",
            user_product="EnterpriseCloud Pro",
            client_company="MegaCorp Industries",
            client_role="CTO",
            client_industry="Manufacturing",
            company_size="Enterprise",
            additional_context={
                "product_category": "Cloud Infrastructure Platform",
                "evaluation_stage": "vendor_selection",
                "budget_cycle": "Q4_planning"
            }
        )

        if success:
            print("=== Enhanced Session Initialized Successfully ===")
            status = agent.get_enhanced_session_status()
            print(f"Client: {status['client_profile']['company']} ({status['client_profile']['industry']})")
            print(f"Product: {status['product_context']['name']} by {status['product_context']['vendor']}")
            print(f"Generated {status['question_metrics']['total_generated']} contextual questions")

            # Simulate sophisticated conversation
            sales_responses = [
                "EnterpriseCloud Pro offers 99.99% uptime SLA with auto-failover across 15 global data centers, specifically designed for manufacturing operations with real-time data processing capabilities.",
                "Compared to AWS and Azure, we provide 40% better price-performance for manufacturing workloads, with dedicated support for IoT device integration and predictive maintenance analytics.",
                "Our typical enterprise manufacturing clients see ROI within 8-12 months, with average cost savings of $2.3M annually through improved operational efficiency and reduced downtime.",
                "Implementation typically takes 12-16 weeks for enterprise manufacturing, including full data migration, staff training, and integration with existing MES and ERP systems."
            ]

            print("\n=== Simulated Sales Conversation ===")
            for i, response in enumerate(sales_responses, 1):
                print(f"\n--- Exchange {i} ---")
                print(f"Salesperson: {response}")
                
                try:
                    client_response = await agent.simulate_client_response(response)
                    print(f"Client (CTO): {client_response}")
                    
                    current_status = agent.get_enhanced_session_status()
                    print(f"[Status: {current_status['meeting_assessment']['meeting_quality']} meeting, "
                          f"{current_status['question_metrics']['answer_rate']:.1f}% questions addressed, "
                          f"Stage: {current_status['session_info']['meeting_stage']}]")
                except Exception as e:
                    print(f"Error in client response: {e}")
                    print("Continuing with next exchange...")

            # Generate comprehensive analysis
            print("\n=== Comprehensive Meeting Analysis ===")
            try:
                analysis = agent.get_detailed_questions_analysis()

                print(f"\nUnanswered Critical Questions: {len(analysis['unanswered_critical'])}")
                for q in analysis['unanswered_critical'][:3]:
                    print(f"  - [{q['category']}] {q['text'][:80]}...")

                print(f"\nTop Quality Interactions:")
                for q in analysis['top_quality_answers'][:3]:
                    print(f"  - [{q['category']}] Score: {q['quality_score']:.2f} - {q['text'][:60]}...")

                if analysis['areas_needing_attention']:
                    print(f"\nAreas Needing Attention:")
                    for area in analysis['areas_needing_attention']:
                        print(f"  - {area['category']}: {area['avg_quality']:.2f} avg quality")

                # Export comprehensive summary
                summary = agent.export_comprehensive_summary()
                print(f"\n=== Meeting Recommendations ===")
                for rec in summary['recommendations']:
                    print(f"  • {rec}")

                print(f"\nMeeting Completion: {summary['session_overview']['meeting_assessment']['completion_percentage']:.1f}%")
                print(f"Overall Quality: {summary['session_overview']['meeting_assessment']['meeting_quality']}")
            
            except Exception as e:
                print(f"Error generating analysis: {e}")

            # Cleanup
            await agent.cleanup_rag()
            print("\n=== Session completed successfully ===")
        else:
            print("Failed to initialize enhanced session")
            
    except Exception as e:
        print(f"Critical error in demo: {e}")
        print("Please ensure:")
        print("1. FastAPI service is running on localhost:8000")
        print("2. RAG system is properly initialized")
        print("3. Network connectivity is available")

if __name__ == "__main__":
    asyncio.run(enhanced_demo())