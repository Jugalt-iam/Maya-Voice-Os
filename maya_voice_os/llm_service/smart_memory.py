"""
Smart Memory Module for Aayna v2 (Phase 5)

Enhanced conversation state tracking using Redis:
1. Objections raised and handled
2. Buying signals detected
3. Emotional trajectory
4. Conversation stage progression
5. Information shared to customer
6. Cross-session memory

This is an ADDITIVE module - enhances Redis context without breaking existing flow.
"""

import logging
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# TYPES
# ============================================================================

class ConversationStage(Enum):
    """Sales conversation stages"""
    OPENING = "opening"
    DISCOVERY = "discovery"
    PRESENTATION = "presentation"
    OBJECTION_HANDLING = "objection_handling"
    NEGOTIATION = "negotiation"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"


class EmotionalTrend(Enum):
    """Customer emotional trajectory"""
    WARMING = "warming"      # Getting more positive
    COOLING = "cooling"      # Getting more negative
    STABLE = "stable"        # No significant change
    VOLATILE = "volatile"    # Fluctuating


@dataclass
class ObjectionRecord:
    """Record of an objection raised during conversation"""
    type: str                    # price, timing, authority, competitor, etc.
    text: str                    # Original customer statement
    timestamp: float             # When raised
    handled: bool = False        # Whether we addressed it
    handling_effective: bool = False  # Did handling work?
    rebuttal_used: Optional[str] = None


@dataclass
class BuyingSignal:
    """Record of a buying signal detected"""
    type: str                    # pricing_inquiry, urgency, next_steps, etc.
    text: str                    # Original statement
    timestamp: float
    strength: float = 0.5        # 0-1 strength indicator


@dataclass 
class InfoShared:
    """Information we've shared with the customer"""
    topic: str                   # pricing, features, comparison, etc.
    summary: str                 # What we told them
    timestamp: float
    customer_reaction: Optional[str] = None  # positive, neutral, negative


@dataclass
class SmartMemory:
    """
    Enhanced conversation state tracking.
    
    Stored in Redis alongside existing context.
    """
    conversation_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Stage tracking
    current_stage: str = "opening"
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Objection tracking
    objections: List[Dict[str, Any]] = field(default_factory=list)
    active_objection: Optional[str] = None
    objection_count: int = 0
    
    # Buying signals
    buying_signals: List[Dict[str, Any]] = field(default_factory=list)
    buying_signal_score: float = 0.0  # Cumulative score 0-10
    
    # Emotional tracking
    emotional_states: List[Dict[str, Any]] = field(default_factory=list)
    emotional_trend: str = "stable"
    
    # Information shared
    info_shared: List[Dict[str, Any]] = field(default_factory=list)
    topics_covered: List[str] = field(default_factory=list)
    
    # Turn tracking
    turn_count: int = 0
    customer_turns: int = 0
    agent_turns: int = 0
    
    # Engagement metrics
    avg_response_length: float = 0.0
    question_count: int = 0
    confirmation_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmartMemory":
        """Create from dictionary (Redis load)"""
        return cls(**data)


class SmartMemoryManager:
    """
    Manages SmartMemory state in Redis.
    
    Usage:
        manager = SmartMemoryManager(redis_client)
        memory = await manager.get_or_create(conversation_id)
        memory = await manager.record_objection(conversation_id, objection)
        await manager.save(memory)
    """
    
    REDIS_PREFIX = "smart_memory:"
    TTL_SECONDS = 86400 * 7  # 7 days
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        # In-process fallback store: without this, running with no Redis
        # configured meant every get_or_create() returned a blank object,
        # silently discarding state even within the same running process
        # (not just across restarts, which is what "no Redis" should mean).
        self._local_store: Dict[str, "SmartMemory"] = {}
    
    def set_redis_client(self, redis_client):
        """Set Redis client after initialization"""
        self.redis_client = redis_client
    
    async def get_or_create(self, conversation_id: str) -> SmartMemory:
        """Get existing memory or create new one"""
        if self.redis_client:
            try:
                key = f"{self.REDIS_PREFIX}{conversation_id}"
                data = await self.redis_client.get(key)
                if data:
                    return SmartMemory.from_dict(json.loads(data))
            except Exception as e:
                logger.warning(f"Failed to load smart memory: {e}")
        elif conversation_id in self._local_store:
            return self._local_store[conversation_id]
        
        return SmartMemory(conversation_id=conversation_id)
    
    async def save(self, memory: SmartMemory):
        """Save memory to Redis, or to the in-process fallback store if
        Redis isn't configured — so conversation state still persists
        across turns for the life of this process."""
        memory.updated_at = time.time()

        if not self.redis_client:
            self._local_store[memory.conversation_id] = memory
            return
        
        try:
            key = f"{self.REDIS_PREFIX}{memory.conversation_id}"
            await self.redis_client.setex(
                key,
                self.TTL_SECONDS,
                json.dumps(memory.to_dict())
            )
        except Exception as e:
            logger.warning(f"Failed to save smart memory, using in-process fallback instead: {e}")
            self._local_store[memory.conversation_id] = memory
    
    async def record_objection(
        self,
        conversation_id: str,
        objection_type: str,
        objection_text: str
    ) -> SmartMemory:
        """Record a new objection"""
        memory = await self.get_or_create(conversation_id)
        
        objection = ObjectionRecord(
            type=objection_type,
            text=objection_text,
            timestamp=time.time()
        )
        
        memory.objections.append(asdict(objection))
        memory.objection_count += 1
        memory.active_objection = objection_type
        
        # Update stage if not already in objection handling
        if memory.current_stage != "objection_handling":
            memory.stage_history.append({
                "from": memory.current_stage,
                "to": "objection_handling",
                "timestamp": time.time(),
                "trigger": f"objection:{objection_type}"
            })
            memory.current_stage = "objection_handling"
        
        await self.save(memory)
        logger.info(f"Recorded objection: {objection_type}")
        return memory
    
    async def mark_objection_handled(
        self,
        conversation_id: str,
        effective: bool = True
    ) -> SmartMemory:
        """Mark the active objection as handled"""
        memory = await self.get_or_create(conversation_id)
        
        if memory.objections:
            # Update last objection
            memory.objections[-1]["handled"] = True
            memory.objections[-1]["handling_effective"] = effective
        
        memory.active_objection = None
        
        # Move back to previous stage or negotiation
        if effective:
            new_stage = "negotiation" if memory.buying_signal_score > 5 else "presentation"
        else:
            new_stage = "discovery"  # Need to rebuild value
        
        memory.stage_history.append({
            "from": memory.current_stage,
            "to": new_stage,
            "timestamp": time.time(),
            "trigger": f"objection_handled:{'effective' if effective else 'ineffective'}"
        })
        memory.current_stage = new_stage
        
        await self.save(memory)
        return memory
    
    async def record_buying_signal(
        self,
        conversation_id: str,
        signal_type: str,
        signal_text: str,
        strength: float = 0.5
    ) -> SmartMemory:
        """Record a buying signal"""
        memory = await self.get_or_create(conversation_id)
        
        signal = BuyingSignal(
            type=signal_type,
            text=signal_text,
            timestamp=time.time(),
            strength=strength
        )
        
        memory.buying_signals.append(asdict(signal))
        memory.buying_signal_score = min(10.0, memory.buying_signal_score + strength * 2)
        
        # Check if ready to close
        if memory.buying_signal_score >= 7:
            memory.stage_history.append({
                "from": memory.current_stage,
                "to": "closing",
                "timestamp": time.time(),
                "trigger": f"buying_signal_score:{memory.buying_signal_score:.1f}"
            })
            memory.current_stage = "closing"
        
        await self.save(memory)
        logger.info(f"Recorded buying signal: {signal_type} (score now: {memory.buying_signal_score:.1f})")
        return memory
    
    async def record_emotional_state(
        self,
        conversation_id: str,
        emotion: str,
        intensity: float = 0.5
    ) -> SmartMemory:
        """Record emotional state and update trend"""
        memory = await self.get_or_create(conversation_id)
        
        memory.emotional_states.append({
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": time.time()
        })
        
        # Calculate trend from last 3 states
        if len(memory.emotional_states) >= 3:
            recent = memory.emotional_states[-3:]
            positive = ["interested", "excited", "happy", "ready_to_buy"]
            negative = ["frustrated", "skeptical", "confused", "angry"]
            
            trend_score = 0
            for state in recent:
                if state["emotion"] in positive:
                    trend_score += 1
                elif state["emotion"] in negative:
                    trend_score -= 1
            
            if trend_score >= 2:
                memory.emotional_trend = "warming"
            elif trend_score <= -2:
                memory.emotional_trend = "cooling"
            elif abs(trend_score) <= 1:
                memory.emotional_trend = "stable"
            else:
                memory.emotional_trend = "volatile"
        
        await self.save(memory)
        return memory
    
    async def record_info_shared(
        self,
        conversation_id: str,
        topic: str,
        summary: str,
        reaction: Optional[str] = None
    ) -> SmartMemory:
        """Record information shared with customer"""
        memory = await self.get_or_create(conversation_id)
        
        info = InfoShared(
            topic=topic,
            summary=summary,
            timestamp=time.time(),
            customer_reaction=reaction
        )
        
        memory.info_shared.append(asdict(info))
        if topic not in memory.topics_covered:
            memory.topics_covered.append(topic)
        
        await self.save(memory)
        return memory
    
    async def increment_turn(
        self,
        conversation_id: str,
        is_customer: bool = True,
        response_length: int = 0
    ) -> SmartMemory:
        """Track conversation turn"""
        memory = await self.get_or_create(conversation_id)
        
        memory.turn_count += 1
        if is_customer:
            memory.customer_turns += 1
        else:
            memory.agent_turns += 1
            # Update average response length
            if memory.agent_turns > 0:
                memory.avg_response_length = (
                    (memory.avg_response_length * (memory.agent_turns - 1) + response_length)
                    / memory.agent_turns
                )
        
        # Auto-advance stage based on turn count
        if memory.current_stage == "opening" and memory.turn_count >= 2:
            memory.stage_history.append({
                "from": "opening",
                "to": "discovery",
                "timestamp": time.time(),
                "trigger": "turn_count:2"
            })
            memory.current_stage = "discovery"
        
        await self.save(memory)
        return memory
    
    async def get_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary for LLM context injection"""
        memory = await self.get_or_create(conversation_id)
        
        return {
            "stage": memory.current_stage,
            "turn_count": memory.turn_count,
            "objection_count": memory.objection_count,
            "active_objection": memory.active_objection,
            "buying_signal_score": memory.buying_signal_score,
            "emotional_trend": memory.emotional_trend,
            "topics_covered": memory.topics_covered,
            "ready_to_close": memory.buying_signal_score >= 7,
            "needs_reengagement": memory.emotional_trend == "cooling"
        }


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_memory_manager: Optional[SmartMemoryManager] = None


def get_memory_manager() -> SmartMemoryManager:
    """Get or create global SmartMemoryManager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = SmartMemoryManager()
    return _memory_manager


def initialize_memory_manager(redis_client) -> SmartMemoryManager:
    """Initialize memory manager with Redis client"""
    global _memory_manager
    _memory_manager = SmartMemoryManager(redis_client)
    return _memory_manager
