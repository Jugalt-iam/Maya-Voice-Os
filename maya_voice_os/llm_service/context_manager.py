"""
Context Manager
Manages conversation context, memory, and state across multiple turns
Supports short-term and long-term memory with intelligent summarization
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import hashlib

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage"""
    SHORT_TERM = "short_term"  # Current conversation
    WORKING = "working"  # Active processing
    LONG_TERM = "long_term"  # Persistent across sessions
    EPISODIC = "episodic"  # Specific events/interactions
    SEMANTIC = "semantic"  # Facts and knowledge


class ContextScope(Enum):
    """Scope of context information"""
    TURN = "turn"  # Single turn
    CONVERSATION = "conversation"  # Current conversation
    SESSION = "session"  # Current session
    USER = "user"  # Across all user sessions
    GLOBAL = "global"  # System-wide


@dataclass
class MemoryItem:
    """A single memory item"""
    id: str
    content: str
    memory_type: MemoryType
    scope: ContextScope
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 to 1.0
    confidence: float = 1.0
    source: str = "user"  # user, system, external
    
    # Relationships
    related_items: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Lifecycle
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


@dataclass
class ConversationTurn:
    """A single conversation turn"""
    turn_id: str
    user_input: str
    bot_response: str
    timestamp: float = field(default_factory=time.time)
    
    # Context at time of turn
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    processing_time_ms: float = 0.0
    confidence: float = 1.0
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """A complete conversation session"""
    session_id: str
    user_id: str
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Conversation data
    turns: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Session metadata
    topic: Optional[str] = None
    sentiment: str = "neutral"
    language: str = "en"
    channel: str = "voice"
    
    # Status
    is_active: bool = True
    completion_reason: Optional[str] = None


@dataclass
class ContextConfig:
    """Context manager configuration"""
    # Memory limits
    max_short_term_items: int = 100
    max_working_memory_items: int = 20
    max_long_term_items: int = 1000
    
    # Retention settings
    short_term_retention_hours: int = 24
    working_memory_retention_minutes: int = 30
    long_term_retention_days: int = 365
    
    # Summarization
    enable_auto_summarization: bool = True
    summarization_threshold: int = 50  # turns
    summary_compression_ratio: float = 0.3
    
    # Context window
    max_context_tokens: int = 4000
    context_overlap_tokens: int = 200
    
    # Performance
    enable_memory_compression: bool = True
    cleanup_interval_minutes: int = 60


class ContextManager:
    """
    Context Manager
    Manages conversation context, memory, and state across multiple turns
    """
    
    def __init__(self, config: ContextConfig, llm_engine=None):
        self.config = config
        self.llm_engine = llm_engine
        
        # Memory storage
        self.memory_store: Dict[MemoryType, Dict[str, MemoryItem]] = {
            memory_type: {} for memory_type in MemoryType
        }
        
        # Active sessions
        self.sessions: Dict[str, ConversationSession] = {}
        
        # Context cache
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.memory_access_times: List[float] = []
        self.context_build_times: List[float] = []
        
        logger.info("Initialized Context Manager")
    
    async def initialize(self) -> bool:
        """Initialize the context manager"""
        try:
            # Start background cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Context Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Context Manager: {e}")
            return False
    
    async def create_session(
        self,
        session_id: str,
        user_id: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new conversation session"""
        
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists")
            return False
        
        try:
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id
            )
            
            # Initialize context
            if initial_context:
                session.context.update(initial_context)
            
            # Load user's long-term memory
            await self._load_user_memory(session)
            
            self.sessions[session_id] = session
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            return False
    
    async def add_turn(
        self,
        session_id: str,
        user_input: str,
        bot_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a conversation turn to the session"""
        
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        try:
            session = self.sessions[session_id]
            
            # Create turn
            turn = ConversationTurn(
                turn_id=f"{session_id}_turn_{len(session.turns)}",
                user_input=user_input,
                bot_response=bot_response,
                context_snapshot=session.context.copy()
            )
            
            # Add metadata
            if metadata:
                turn.processing_time_ms = metadata.get('processing_time_ms', 0.0)
                turn.confidence = metadata.get('confidence', 1.0)
                turn.intent = metadata.get('intent')
                turn.entities = metadata.get('entities', {})
            
            session.turns.append(turn)
            session.last_activity = time.time()
            
            # Update memory
            await self._update_memory_from_turn(session, turn)
            
            # Check for summarization
            if (self.config.enable_auto_summarization and 
                len(session.turns) % self.config.summarization_threshold == 0):
                await self._summarize_conversation(session)
            
            logger.debug(f"Added turn to session {session_id}: {len(session.turns)} total turns")
            return True
            
        except Exception as e:
            logger.error(f"Error adding turn to session {session_id}: {e}")
            return False
    
    async def get_context(
        self,
        session_id: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get current context for a session"""
        
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return {}
        
        start_time = time.time()
        
        try:
            session = self.sessions[session_id]
            max_tokens = max_tokens or self.config.max_context_tokens
            
            # Check cache
            cache_key = f"{session_id}_{max_tokens}_{len(session.turns)}"
            if cache_key in self.context_cache:
                return self.context_cache[cache_key]
            
            # Build context
            context = await self._build_context(session, max_tokens)
            
            # Cache result
            self.context_cache[cache_key] = context
            
            # Update performance tracking
            build_time = (time.time() - start_time) * 1000
            self.context_build_times.append(build_time)
            if len(self.context_build_times) > 100:
                self.context_build_times = self.context_build_times[-100:]
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context for session {session_id}: {e}")
            return {}
    
    async def _build_context(
        self,
        session: ConversationSession,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Build context for a session"""
        
        context = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "conversation_history": [],
            "current_context": session.context.copy(),
            "memory": {},
            "metadata": {
                "turn_count": len(session.turns),
                "session_duration": time.time() - session.start_time,
                "topic": session.topic,
                "sentiment": session.sentiment,
                "language": session.language
            }
        }
        
        # Add conversation history (recent turns first)
        token_count = 0
        for turn in reversed(session.turns):
            turn_tokens = self._estimate_tokens(turn.user_input + turn.bot_response)
            
            if token_count + turn_tokens > max_tokens - self.config.context_overlap_tokens:
                break
            
            context["conversation_history"].insert(0, {
                "user": turn.user_input,
                "assistant": turn.bot_response,
                "timestamp": turn.timestamp,
                "intent": turn.intent,
                "entities": turn.entities
            })
            
            token_count += turn_tokens
        
        # Add relevant memory
        context["memory"] = await self._get_relevant_memory(session, max_tokens - token_count)
        
        return context
    
    async def _get_relevant_memory(
        self,
        session: ConversationSession,
        available_tokens: int
    ) -> Dict[str, Any]:
        """Get relevant memory items for context"""
        
        memory_context = {
            "short_term": [],
            "working": [],
            "long_term": [],
            "facts": []
        }
        
        # Get recent short-term memory
        short_term_items = self._get_memory_items(
            MemoryType.SHORT_TERM,
            user_id=session.user_id,
            limit=10
        )
        
        for item in short_term_items:
            if self._estimate_tokens(item.content) < available_tokens:
                memory_context["short_term"].append({
                    "content": item.content,
                    "importance": item.importance,
                    "timestamp": item.timestamp
                })
                available_tokens -= self._estimate_tokens(item.content)
        
        # Get working memory
        working_items = self._get_memory_items(
            MemoryType.WORKING,
            session_id=session.session_id,
            limit=5
        )
        
        for item in working_items:
            if self._estimate_tokens(item.content) < available_tokens:
                memory_context["working"].append({
                    "content": item.content,
                    "importance": item.importance
                })
                available_tokens -= self._estimate_tokens(item.content)
        
        # Get important long-term memory
        long_term_items = self._get_memory_items(
            MemoryType.LONG_TERM,
            user_id=session.user_id,
            min_importance=0.7,
            limit=5
        )
        
        for item in long_term_items:
            if self._estimate_tokens(item.content) < available_tokens:
                memory_context["long_term"].append({
                    "content": item.content,
                    "importance": item.importance,
                    "tags": item.tags
                })
                available_tokens -= self._estimate_tokens(item.content)
        
        return memory_context
    
    def _get_memory_items(
        self,
        memory_type: MemoryType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Get memory items with filtering"""
        
        items = list(self.memory_store[memory_type].values())
        
        # Filter by user_id
        if user_id:
            items = [item for item in items if user_id in item.id or item.scope in [ContextScope.USER, ContextScope.GLOBAL]]
        
        # Filter by session_id
        if session_id:
            items = [item for item in items if session_id in item.id or item.scope in [ContextScope.SESSION, ContextScope.USER, ContextScope.GLOBAL]]
        
        # Filter by importance
        items = [item for item in items if item.importance >= min_importance]
        
        # Sort by importance and recency
        items.sort(key=lambda x: (x.importance, x.last_accessed), reverse=True)
        
        return items[:limit]
    
    async def _update_memory_from_turn(self, session: ConversationSession, turn: ConversationTurn):
        """Update memory based on conversation turn"""
        
        # Extract important information from turn
        important_info = await self._extract_important_info(turn)
        
        for info in important_info:
            await self.store_memory(
                content=info["content"],
                memory_type=info["type"],
                scope=info["scope"],
                importance=info["importance"],
                user_id=session.user_id,
                session_id=session.session_id,
                tags=info.get("tags", [])
            )
    
    async def _extract_important_info(self, turn: ConversationTurn) -> List[Dict[str, Any]]:
        """Extract important information from a conversation turn"""
        
        important_info = []
        
        # Extract entities as facts
        for entity_type, entity_value in turn.entities.items():
            important_info.append({
                "content": f"{entity_type}: {entity_value}",
                "type": MemoryType.SEMANTIC,
                "scope": ContextScope.USER,
                "importance": 0.7,
                "tags": [entity_type, "entity"]
            })
        
        # Store user preferences or important statements
        user_input_lower = turn.user_input.lower()
        
        # Preference indicators
        preference_indicators = ["i like", "i prefer", "i want", "i need", "my favorite"]
        for indicator in preference_indicators:
            if indicator in user_input_lower:
                important_info.append({
                    "content": turn.user_input,
                    "type": MemoryType.LONG_TERM,
                    "scope": ContextScope.USER,
                    "importance": 0.8,
                    "tags": ["preference", "user_statement"]
                })
                break
        
        # Personal information
        personal_indicators = ["my name is", "i am", "i work", "i live"]
        for indicator in personal_indicators:
            if indicator in user_input_lower:
                important_info.append({
                    "content": turn.user_input,
                    "type": MemoryType.LONG_TERM,
                    "scope": ContextScope.USER,
                    "importance": 0.9,
                    "tags": ["personal_info", "user_statement"]
                })
                break
        
        # Current conversation context
        if turn.intent:
            important_info.append({
                "content": f"User intent: {turn.intent}",
                "type": MemoryType.WORKING,
                "scope": ContextScope.CONVERSATION,
                "importance": 0.6,
                "tags": ["intent", "context"]
            })
        
        return important_info
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        scope: ContextScope,
        importance: float = 0.5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store a memory item"""
        
        # Generate memory ID
        memory_id = self._generate_memory_id(content, memory_type, scope, user_id, session_id)
        
        # Check if memory already exists
        if memory_id in self.memory_store[memory_type]:
            # Update existing memory
            existing_item = self.memory_store[memory_type][memory_id]
            existing_item.importance = max(existing_item.importance, importance)
            existing_item.access_count += 1
            existing_item.last_accessed = time.time()
            if tags:
                existing_item.tags.extend(tag for tag in tags if tag not in existing_item.tags)
            return memory_id
        
        # Create new memory item
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            scope=scope,
            importance=importance,
            tags=tags or []
        )
        
        # Set expiration based on memory type
        if memory_type == MemoryType.SHORT_TERM:
            memory_item.expires_at = time.time() + (self.config.short_term_retention_hours * 3600)
        elif memory_type == MemoryType.WORKING:
            memory_item.expires_at = time.time() + (self.config.working_memory_retention_minutes * 60)
        elif memory_type == MemoryType.LONG_TERM:
            memory_item.expires_at = time.time() + (self.config.long_term_retention_days * 86400)
        
        # Store memory
        self.memory_store[memory_type][memory_id] = memory_item
        
        # Check memory limits
        await self._enforce_memory_limits(memory_type)
        
        logger.debug(f"Stored {memory_type.value} memory: {content[:50]}...")
        return memory_id
    
    def _generate_memory_id(
        self,
        content: str,
        memory_type: MemoryType,
        scope: ContextScope,
        user_id: Optional[str],
        session_id: Optional[str]
    ) -> str:
        """Generate a unique memory ID"""
        
        # Create hash from content and context
        hash_input = f"{content}_{memory_type.value}_{scope.value}_{user_id}_{session_id}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"{memory_type.value}_{scope.value}_{hash_value}"
    
    async def _enforce_memory_limits(self, memory_type: MemoryType):
        """Enforce memory limits by removing least important items"""
        
        memory_store = self.memory_store[memory_type]
        
        # Get limit for this memory type
        if memory_type == MemoryType.SHORT_TERM:
            limit = self.config.max_short_term_items
        elif memory_type == MemoryType.WORKING:
            limit = self.config.max_working_memory_items
        elif memory_type == MemoryType.LONG_TERM:
            limit = self.config.max_long_term_items
        else:
            return  # No limit for other types
        
        if len(memory_store) <= limit:
            return
        
        # Sort by importance and access patterns
        items = list(memory_store.values())
        items.sort(key=lambda x: (x.importance, x.access_count, x.last_accessed))
        
        # Remove least important items
        items_to_remove = items[:len(items) - limit]
        
        for item in items_to_remove:
            del memory_store[item.id]
            logger.debug(f"Removed {memory_type.value} memory due to limit: {item.content[:30]}...")
    
    async def _load_user_memory(self, session: ConversationSession):
        """Load user's persistent memory into session context"""
        
        # Get user's long-term memory
        user_memories = self._get_memory_items(
            MemoryType.LONG_TERM,
            user_id=session.user_id,
            min_importance=0.5,
            limit=20
        )
        
        # Add to session context
        session.context["user_memories"] = [
            {
                "content": memory.content,
                "importance": memory.importance,
                "tags": memory.tags
            }
            for memory in user_memories
        ]
    
    async def _summarize_conversation(self, session: ConversationSession):
        """Summarize conversation and store as long-term memory"""
        
        if not self.llm_engine or len(session.turns) < 5:
            return
        
        try:
            # Get recent turns for summarization
            recent_turns = session.turns[-self.config.summarization_threshold:]
            
            # Create summarization prompt
            conversation_text = "\n".join([
                f"User: {turn.user_input}\nAssistant: {turn.bot_response}"
                for turn in recent_turns
            ])
            
            summary_prompt = f"""
Please summarize this conversation, focusing on:
1. Key topics discussed
2. Important user preferences or information
3. Decisions made or actions taken
4. Unresolved issues

Conversation:
{conversation_text}

Summary:
"""
            
            # Generate summary using LLM. self.llm_engine here is our own
            # llm-service/llm_router.py's LLMRouter instance (passed in at
            # ContextManager construction), not the original local_llm_engine
            # (a GGUF-model loader we don't ship). LLMRouter.chat() is
            # synchronous/blocking like homemath itself, so it's run in a
            # thread to avoid blocking the event loop during summarization.
            if self.llm_engine is None:
                logger.debug("No llm_engine configured, skipping summarization")
                return

            summary = await asyncio.to_thread(
                self.llm_engine.chat,
                [{"role": "user", "content": summary_prompt}],
                "You are a precise, neutral conversation summarizer. Output only the summary, no preamble.",
            )
            summary = (summary or "").strip()
            if not summary:
                logger.warning("Summarization returned empty text, skipping storage")
                return
            
            # Store summary as long-term memory
            await self.store_memory(
                content=f"Conversation summary: {summary}",
                memory_type=MemoryType.LONG_TERM,
                scope=ContextScope.USER,
                importance=0.8,
                user_id=session.user_id,
                session_id=session.session_id,
                tags=["summary", "conversation"]
            )
            
            logger.info(f"Summarized conversation for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    async def _cleanup_loop(self):
        """Background task to clean up expired memory and sessions"""
        logger.info("Started context cleanup loop")
        
        try:
            while True:
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                await self._cleanup_expired_memory()
                await self._cleanup_inactive_sessions()
                
        except asyncio.CancelledError:
            logger.info("Context cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_memory(self):
        """Clean up expired memory items"""
        current_time = time.time()
        
        for memory_type, memory_store in self.memory_store.items():
            expired_items = [
                item_id for item_id, item in memory_store.items()
                if item.expires_at and item.expires_at < current_time
            ]
            
            for item_id in expired_items:
                del memory_store[item_id]
            
            if expired_items:
                logger.debug(f"Cleaned up {len(expired_items)} expired {memory_type.value} memories")
    
    async def _cleanup_inactive_sessions(self):
        """Clean up inactive sessions"""
        current_time = time.time()
        inactive_threshold = 3600  # 1 hour
        
        inactive_sessions = [
            session_id for session_id, session in self.sessions.items()
            if not session.is_active or (current_time - session.last_activity) > inactive_threshold
        ]
        
        for session_id in inactive_sessions:
            await self.end_session(session_id, "inactive")
    
    async def end_session(self, session_id: str, reason: str = "manual"):
        """End a conversation session"""
        
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session.is_active = False
        session.completion_reason = reason
        
        # Final summarization if needed
        if len(session.turns) >= 5:
            await self._summarize_conversation(session)
        
        # Clear session from memory
        del self.sessions[session_id]
        
        # Clear related cache entries
        cache_keys_to_remove = [key for key in self.context_cache.keys() if key.startswith(session_id)]
        for key in cache_keys_to_remove:
            del self.context_cache[key]
        
        logger.info(f"Ended session {session_id}: {reason}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get context manager performance statistics"""
        
        memory_counts = {
            memory_type.value: len(memory_store)
            for memory_type, memory_store in self.memory_store.items()
        }
        
        return {
            "active_sessions": len(self.sessions),
            "memory_counts": memory_counts,
            "total_memory_items": sum(memory_counts.values()),
            "cache_size": len(self.context_cache),
            "avg_context_build_time_ms": (
                sum(self.context_build_times) / len(self.context_build_times)
                if self.context_build_times else 0.0
            ),
            "avg_memory_access_time_ms": (
                sum(self.memory_access_times) / len(self.memory_access_times)
                if self.memory_access_times else 0.0
            )
        }
    
    async def cleanup(self):
        """Clean up context manager"""
        try:
            # Cancel cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # End all sessions
            session_ids = list(self.sessions.keys())
            for session_id in session_ids:
                await self.end_session(session_id, "cleanup")
            
            # Clear memory
            for memory_store in self.memory_store.values():
                memory_store.clear()
            
            self.context_cache.clear()
            
            logger.info("Context manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during context manager cleanup: {e}")

