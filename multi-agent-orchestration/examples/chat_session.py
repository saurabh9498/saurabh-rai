"""
Chat Session Example

Demonstrates multi-turn conversations with the multi-agent system.
Shows conversation history management, context tracking, and agent interactions.
"""

import asyncio
import os
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import system components
from src.core.llm import LLMClient
from src.agents.orchestrator import Orchestrator
from src.utils import (
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    token_tracker,
)

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChatSession:
    """
    Manages a chat conversation session.
    
    Handles message history, context, and agent interactions.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    max_history: int = 20  # Maximum messages to keep in context
    
    def add_message(self, role: str, content: str, **metadata) -> Message:
        """Add a message to the session."""
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        
        # Trim history if needed
        if len(self.messages) > self.max_history:
            # Keep system message if present
            if self.messages[0].role == "system":
                self.messages = [self.messages[0]] + self.messages[-self.max_history+1:]
            else:
                self.messages = self.messages[-self.max_history:]
        
        return message
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Format messages for LLM API call."""
        api_messages = []
        
        # Add system prompt if set
        if self.system_prompt:
            api_messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        
        # Add conversation history
        for msg in self.messages:
            api_messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        return api_messages
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
    
    def summary(self) -> str:
        """Get a summary of the conversation."""
        return f"Session {self.session_id}: {len(self.messages)} messages"


class ChatBot:
    """
    Interactive chatbot using the multi-agent system.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
    ):
        self.client = LLMClient(provider="openai", model=model)
        self.model = model
        self.session = ChatSession(
            system_prompt=system_prompt or self._default_system_prompt()
        )
        self.logger = get_logger("ChatBot")
    
    def _default_system_prompt(self) -> str:
        return """You are a helpful AI assistant. You provide clear, accurate, 
and thoughtful responses. When you're not sure about something, you say so.
You maintain context from the conversation and refer back to previous 
messages when relevant."""
    
    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        """
        # Set request context for logging
        request_id = str(uuid.uuid4())[:8]
        set_request_context(request_id=request_id)
        
        try:
            # Add user message
            self.session.add_message("user", user_message)
            
            # Get response from LLM
            messages = self.session.get_messages_for_api()
            
            # Count tokens before call
            prompt_tokens = token_tracker.count_tokens(
                messages=messages,
                model=self.model,
            )
            
            self.logger.info(
                f"Chat request: {len(user_message)} chars, "
                f"{prompt_tokens} prompt tokens"
            )
            
            response = await self.client.chat(
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            
            # Add assistant response
            self.session.add_message("assistant", response)
            
            return response
            
        finally:
            clear_request_context()
    
    def reset(self) -> None:
        """Reset the conversation."""
        self.session.clear()
        self.logger.info("Conversation reset")


async def simple_chat_demo():
    """
    Demonstrate a simple multi-turn conversation.
    """
    print("\n" + "="*60)
    print("SIMPLE CHAT DEMO")
    print("="*60)
    
    bot = ChatBot()
    
    # Simulated conversation
    messages = [
        "Hi! What can you help me with?",
        "I'm working on a Python project. Can you suggest some best practices?",
        "What about testing? Can you elaborate on that?",
        "Thanks! Can you summarize the main points from our conversation?",
    ]
    
    for user_msg in messages:
        print(f"\n👤 User: {user_msg}")
        response = await bot.chat(user_msg)
        print(f"\n🤖 Assistant: {response}")
        print("-"*60)
    
    print(f"\n{bot.session.summary()}")


async def context_aware_chat_demo():
    """
    Demonstrate context awareness across turns.
    """
    print("\n" + "="*60)
    print("CONTEXT-AWARE CHAT DEMO")
    print("="*60)
    
    system_prompt = """You are a helpful math tutor. You explain concepts 
clearly and build on previous explanations. Always encourage the student 
and check their understanding."""
    
    bot = ChatBot(system_prompt=system_prompt)
    
    # Math tutoring conversation
    conversation = [
        "I'm trying to understand quadratic equations. Can you explain what they are?",
        "What does the 'a' coefficient do?",
        "So if a is negative, the parabola opens downward?",
        "Can you give me an example to solve?",
    ]
    
    for user_msg in conversation:
        print(f"\n📚 Student: {user_msg}")
        response = await bot.chat(user_msg)
        print(f"\n👩‍🏫 Tutor: {response}")
        print("-"*60)


async def multi_session_demo():
    """
    Demonstrate managing multiple chat sessions.
    """
    print("\n" + "="*60)
    print("MULTI-SESSION DEMO")
    print("="*60)
    
    # Create multiple chat sessions with different personas
    sessions = {
        "tech": ChatBot(
            system_prompt="You are a tech expert who explains technology in simple terms."
        ),
        "chef": ChatBot(
            system_prompt="You are a friendly chef who loves sharing recipes and cooking tips."
        ),
    }
    
    # Interleaved conversations
    interactions = [
        ("tech", "What is machine learning?"),
        ("chef", "How do I make a perfect omelette?"),
        ("tech", "How is it different from traditional programming?"),
        ("chef", "What cheese would you recommend?"),
    ]
    
    for session_key, user_msg in interactions:
        bot = sessions[session_key]
        persona = "🔧 Tech Expert" if session_key == "tech" else "👨‍🍳 Chef"
        
        print(f"\n[{persona}]")
        print(f"👤 User: {user_msg}")
        response = await bot.chat(user_msg)
        print(f"🤖: {response[:200]}..." if len(response) > 200 else f"🤖: {response}")
        print("-"*60)


async def interactive_chat():
    """
    Run an interactive chat session.
    
    This function provides a REPL-like interface for chatting.
    """
    print("\n" + "="*60)
    print("INTERACTIVE CHAT SESSION")
    print("="*60)
    print("\nType 'quit' to exit, 'reset' to clear history, 'stats' for usage stats.\n")
    
    bot = ChatBot()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == "reset":
                bot.reset()
                print("\n✨ Conversation reset!")
                continue
            
            if user_input.lower() == "stats":
                stats = token_tracker.get_stats()
                print(f"\n📊 Usage Stats:")
                print(f"   Total requests: {stats.total_requests}")
                print(f"   Total tokens: {stats.total_tokens:,}")
                print(f"   Total cost: ${stats.total_cost:.6f}")
                continue
            
            response = await bot.chat(user_input)
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def main():
    """Run all chat examples."""
    print("\n" + "#"*60)
    print("# MULTI-AGENT SYSTEM - CHAT SESSION EXAMPLES")
    print("#"*60)
    
    try:
        # Run demos
        await simple_chat_demo()
        await context_aware_chat_demo()
        await multi_session_demo()
        
        # Print final stats
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        
        stats = token_tracker.get_stats()
        print(f"\nSession Summary:")
        print(f"  Total requests: {stats.total_requests}")
        print(f"  Total tokens: {stats.total_tokens:,}")
        print(f"  Total cost: ${stats.total_cost:.6f}")
        print("="*60 + "\n")
        
        # Uncomment to run interactive mode
        # await interactive_chat()
        
    except Exception as e:
        logger.exception(f"Chat demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
