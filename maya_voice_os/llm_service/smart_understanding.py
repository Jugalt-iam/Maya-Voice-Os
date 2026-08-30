"""
Smart Understanding Module for Aayna v2

Provides intelligent understanding of customer messages:
1. Intent detection (what they said)
2. Subtext inference (what they really mean)
3. Emotional state detection
4. Conversation stage tracking
5. Recommended action determination

This is an ADDITIVE module - it enhances existing intent detection without replacing it.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ============================================================================
# TYPES & ENUMS
# ============================================================================

class ConversationStage(Enum):
    """Sales conversation stages"""
    OPENING = "opening"
    DISCOVERY = "discovery"
    PRESENTATION = "presentation"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"


class EmotionalState(Enum):
    """Customer emotional states"""
    NEUTRAL = "neutral"
    INTERESTED = "interested"
    SKEPTICAL = "skeptical"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    CONFUSED = "confused"
    READY_TO_BUY = "ready_to_buy"


class RecommendedAction(Enum):
    """What Aayna should do next"""
    CONTINUE_DISCOVERY = "continue_discovery"
    PRESENT_VALUE = "present_value"
    HANDLE_OBJECTION = "handle_objection"
    CLOSE_NOW = "close_now"
    RE_ENGAGE = "re_engage"
    SCHEDULE_MEETING = "schedule_meeting"
    TRANSFER_HUMAN = "transfer_human"
    PLAY_CACHED_RESPONSE = "play_cached_response"


@dataclass
class Understanding:
    """Complete understanding of a customer message"""
    # Surface level
    surface_text: str
    detected_language: str = "en"
    
    # Deep understanding
    intent: str = "general_query"
    subtext: str = ""
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    stage: ConversationStage = ConversationStage.DISCOVERY
    
    # Actionable insights
    recommended_action: RecommendedAction = RecommendedAction.CONTINUE_DISCOVERY
    cached_response_key: Optional[str] = None  # If action is PLAY_CACHED_RESPONSE
    
    # Confidence
    confidence: float = 0.5
    
    # Buying signals
    buying_signals: List[str] = field(default_factory=list)
    objections: List[str] = field(default_factory=list)


# ============================================================================
# INTENT PATTERNS
# ============================================================================

INTENT_PATTERNS = {
    # Greetings
    "greeting": [
        r"\b(hello|hi|hey|namaste|namaskar|pranam)\b",
        r"\b(good\s*(morning|afternoon|evening))\b",
    ],
    
    # Price/Budget objections
    "objection_price": [
        r"\b(expensive|costly|too\s*much|budget|afford|price\s*high)\b",
        r"\b(mahanga|bahut\s*zyada|budget\s*nahi)\b",
    ],
    
    # Timing objections  
    "objection_timing": [
        r"\b(later|not\s*now|next\s*month|think\s*about|busy)\b",
        r"\b(baad\s*mein|abhi\s*nahi|sochna\s*hai|dekhte\s*hain)\b",
    ],
    
    # Authority objections
    "objection_authority": [
        r"\b(boss|manager|check\s*with|decision\s*maker|family)\b",
        r"\b(ghar\s*wale|senior|permission)\b",
    ],
    
    # Competitor mentions
    "competitor_comparison": [
        r"\b(vs|versus|compared\s*to|better\s*than|other\s*options)\b",
        r"\b(competitor|alternative|dusra\s*option)\b",
    ],
    
    # Ready to buy signals
    "ready_to_buy": [
        r"\b(ready|let'?s\s*do\s*it|proceed|sign\s*up|buy|purchase)\b",
        r"\b(book\s*kar|le\s*lo|ready\s*hoon|haan\s*chalega)\b",
    ],
    
    # Scheduling intent
    "scheduling": [
        r"\b(meet|call|visit|appointment|schedule|book\s*time)\b",
        r"\b(milna|call\s*karo|time\s*fix)\b",
    ],
    
    # Product inquiry
    "product_inquiry": [
        r"\b(feature|include|capability|does\s*it|what\s*can|how\s*does)\b",
        r"\b(kya\s*kar\s*sakta|kya\s*milega|features)\b",
    ],
    
    # Price inquiry
    "price_inquiry": [
        r"\b(cost|price|pricing|rate|how\s*much|kitna)\b",
    ],
    
    # Goodbye/closing
    "closing": [
        r"\b(bye|goodbye|thank\s*you|thanks|that'?s\s*all)\b",
        r"\b(bas|dhanyavaad|shukriya|alvida)\b",
    ],
    
    # Confirmation
    "confirmation": [
        r"^(yes|yeah|yup|haan|ji|bilkul|theek|okay|ok|sure)[\s\.!]*$",
    ],
    
    # Denial
    "denial": [
        r"^(no|nahi|nope|not\s*interested|don'?t\s*want)[\s\.!]*$",
    ],
}

# Subtext rules based on context
SUBTEXT_RULES = {
    "objection_price": {
        "first_mention": "Testing value proposition, not a real rejection",
        "competitor_mentioned": "Comparing with competitor, needs specific differentiation",
        "asked_payment_before": "Real budget concern, offer payment options",
        "engaged_5_plus_turns": "Interested but negotiating, highlight value and scarcity",
    },
    "objection_timing": {
        "default": "Hidden objection - need to surface real concern",
        "early_stage": "Not yet convinced of value",
        "late_stage": "Might be real timing issue, offer flexibility",
    },
}

# Cached response mapping
CACHED_RESPONSE_MAP = {
    "greeting": ["greetings__hello_generic", "greetings__namaste"],
    "confirmation": ["acknowledgments__ok_1", "acknowledgments__bilkul"],
    "objection_price": ["sales__good_choice", "aayna_pricing__pricing_value"],
    "closing": ["closings__thank_you_1", "closings__goodbye_1"],
}


# ============================================================================
# SMART UNDERSTANDING CLASS
# ============================================================================

class SmartUnderstanding:
    """
    Intelligent understanding layer for Aayna.
    
    Goes beyond simple intent classification to understand:
    - What the customer really means (subtext)
    - Where we are in the sales process (stage)
    - What emotional state they're in
    - What action to take next
    """
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.objection_count = 0
        self.turn_count = 0
        self.mentioned_competitor = False
        self.asked_about_payment = False
        self.product_info_shared = False
        self.active_objection: Optional[str] = None
        self.buying_signal_count = 0
        
    def understand(self, message: str, context: Optional[Dict[str, Any]] = None) -> Understanding:
        """
        Perform deep understanding of a customer message.
        
        Args:
            message: The customer's message text
            context: Optional conversation context from Redis
            
        Returns:
            Understanding object with all insights
        """
        context = context or {}
        self.turn_count += 1
        
        understanding = Understanding(surface_text=message)
        
        # 1. Detect language
        understanding.detected_language = self._detect_language(message)
        
        # 2. Detect intent
        understanding.intent = self._detect_intent(message)
        
        # 3. Detect emotional state
        understanding.emotional_state = self._detect_emotion(message, understanding.intent)
        
        # 4. Determine conversation stage
        understanding.stage = self._determine_stage(understanding.intent, context)
        
        # 5. Infer subtext
        understanding.subtext = self._infer_subtext(message, understanding.intent, context)
        
        # 6. Detect buying signals
        understanding.buying_signals = self._detect_buying_signals(message)
        self.buying_signal_count += len(understanding.buying_signals)
        
        # 7. Detect objections
        understanding.objections = self._detect_objections(message, understanding.intent)
        
        # 8. Determine recommended action
        understanding.recommended_action, understanding.cached_response_key = \
            self._determine_action(understanding)
        
        # 9. Set confidence
        understanding.confidence = self._calculate_confidence(understanding)
        
        # Log understanding
        logger.info(f"SmartUnderstanding: intent={understanding.intent}, "
                   f"stage={understanding.stage.value}, "
                   f"action={understanding.recommended_action.value}")
        
        return understanding
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text patterns"""
        hindi_chars = re.findall(r'[\u0900-\u097F]', text)
        if len(hindi_chars) > 3:
            return "hi"
        
        hindi_words = ["haan", "nahi", "kya", "hai", "hoon", "aap", "main", 
                       "kar", "ho", "toh", "mein", "ke", "ki", "ka"]
        text_lower = text.lower()
        hindi_word_count = sum(1 for word in hindi_words if word in text_lower.split())
        
        if hindi_word_count >= 2:
            return "hinglish"
        
        return "en"
    
    def _detect_intent(self, text: str) -> str:
        """Detect primary intent from text"""
        text_lower = text.lower().strip()
        
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Track state based on intent
                    if "objection" in intent:
                        self.objection_count += 1
                        self.active_objection = intent
                    if intent == "competitor_comparison":
                        self.mentioned_competitor = True
                    return intent
        
        return "general_query"
    
    def _detect_emotion(self, text: str, intent: str) -> EmotionalState:
        """Detect emotional state from text and intent"""
        text_lower = text.lower()
        
        # Frustration indicators
        if any(word in text_lower for word in ["frustrated", "annoyed", "angry", "upset", "problem"]):
            return EmotionalState.FRUSTRATED
        
        # Excitement indicators
        if any(word in text_lower for word in ["great", "amazing", "wow", "excited", "love"]):
            return EmotionalState.EXCITED
        
        # Confusion indicators
        if "?" in text and any(word in text_lower for word in ["what", "how", "why", "confused"]):
            return EmotionalState.CONFUSED
        
        # Skepticism from objection intents
        if "objection" in intent:
            return EmotionalState.SKEPTICAL
        
        # Interest from buying signals
        if intent == "ready_to_buy":
            return EmotionalState.READY_TO_BUY
        
        # Interest from product inquiry
        if intent in ["product_inquiry", "price_inquiry"]:
            return EmotionalState.INTERESTED
        
        return EmotionalState.NEUTRAL
    
    def _determine_stage(self, intent: str, context: Dict[str, Any]) -> ConversationStage:
        """Determine current conversation stage"""
        # Opening: First 2 turns
        if self.turn_count <= 2:
            return ConversationStage.OPENING
        
        # Check for objection handling
        if "objection" in intent:
            return ConversationStage.OBJECTION_HANDLING
        
        # Check for closing signals
        if intent == "ready_to_buy" or self.buying_signal_count >= 2:
            return ConversationStage.CLOSING
        
        # Check if product info was shared
        if self.product_info_shared:
            return ConversationStage.PRESENTATION
        
        # Default to discovery
        return ConversationStage.DISCOVERY
    
    def _infer_subtext(self, message: str, intent: str, context: Dict[str, Any]) -> str:
        """Infer the hidden meaning behind the message"""
        if intent not in SUBTEXT_RULES:
            return ""
        
        rules = SUBTEXT_RULES[intent]
        
        # Check specific conditions
        if self.objection_count == 1 and "first_mention" in rules:
            return rules["first_mention"]
        
        if self.mentioned_competitor and "competitor_mentioned" in rules:
            return rules["competitor_mentioned"]
        
        if self.asked_about_payment and "asked_payment_before" in rules:
            return rules["asked_payment_before"]
        
        if self.turn_count > 5 and "engaged_5_plus_turns" in rules:
            return rules["engaged_5_plus_turns"]
        
        if "default" in rules:
            return rules["default"]
        
        return ""
    
    def _detect_buying_signals(self, text: str) -> List[str]:
        """Detect buying signals in text"""
        signals = []
        text_lower = text.lower()
        
        buying_phrases = [
            ("pricing", "Asked about pricing"),
            ("payment", "Interested in payment options"),
            ("when can", "Asking about timeline"),
            ("how soon", "Urgency signal"),
            ("next step", "Ready to proceed"),
            ("let's do", "Ready to commit"),
            ("sign up", "Ready to commit"),
            ("deal", "Negotiating"),
        ]
        
        for phrase, signal in buying_phrases:
            if phrase in text_lower:
                signals.append(signal)
        
        return signals
    
    def _detect_objections(self, text: str, intent: str) -> List[str]:
        """Detect objections in text"""
        objections = []
        
        if "objection_price" in intent:
            objections.append("Price concern")
        if "objection_timing" in intent:
            objections.append("Timing concern")
        if "objection_authority" in intent:
            objections.append("Authority concern")
        
        return objections
    
    def _determine_action(self, understanding: Understanding) -> tuple:
        """Determine recommended action and optional cached response key"""
        intent = understanding.intent
        stage = understanding.stage
        emotion = understanding.emotional_state
        
        # Check if we should use cached response
        if intent in CACHED_RESPONSE_MAP:
            import random
            cached_options = CACHED_RESPONSE_MAP[intent]
            cached_key = random.choice(cached_options)
            
            # Use cached for simple intents
            if intent in ["greeting", "confirmation", "closing"]:
                return RecommendedAction.PLAY_CACHED_RESPONSE, cached_key
        
        # Close if ready
        if emotion == EmotionalState.READY_TO_BUY or self.buying_signal_count >= 3:
            return RecommendedAction.CLOSE_NOW, None
        
        # Handle objections
        if "objection" in intent:
            return RecommendedAction.HANDLE_OBJECTION, None
        
        # Schedule meeting on request
        if intent == "scheduling":
            return RecommendedAction.SCHEDULE_MEETING, None
        
        # Re-engage frustrated customers
        if emotion == EmotionalState.FRUSTRATED:
            return RecommendedAction.RE_ENGAGE, None
        
        # Present value after discovery
        if stage == ConversationStage.PRESENTATION:
            return RecommendedAction.PRESENT_VALUE, None
        
        # Default to discovery
        return RecommendedAction.CONTINUE_DISCOVERY, None
    
    def _calculate_confidence(self, understanding: Understanding) -> float:
        """Calculate confidence in the understanding"""
        base_confidence = 0.5
        
        # Increase for clear intent patterns
        if understanding.intent != "general_query":
            base_confidence += 0.2
        
        # Increase for consistent signals
        if len(understanding.buying_signals) > 0:
            base_confidence += 0.1
        
        # Decrease for mixed signals
        if len(understanding.objections) > 0 and len(understanding.buying_signals) > 0:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def reset(self):
        """Reset conversation state for new conversation"""
        self.conversation_history = []
        self.objection_count = 0
        self.turn_count = 0
        self.mentioned_competitor = False
        self.asked_about_payment = False  
        self.product_info_shared = False
        self.active_objection = None
        self.buying_signal_count = 0


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_smart_understanding: Optional[SmartUnderstanding] = None


def get_smart_understanding() -> SmartUnderstanding:
    """Get or create global SmartUnderstanding instance"""
    global _smart_understanding
    if _smart_understanding is None:
        _smart_understanding = SmartUnderstanding()
    return _smart_understanding


def reset_smart_understanding():
    """Reset the global instance for new conversation"""
    global _smart_understanding
    if _smart_understanding:
        _smart_understanding.reset()
