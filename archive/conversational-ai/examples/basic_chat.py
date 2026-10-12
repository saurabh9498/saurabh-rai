#!/usr/bin/env python3
"""
Basic Chat Example

This example demonstrates how to:
1. Initialize the conversational AI pipeline
2. Process text input through NLU
3. Generate responses using dialog management

Usage:
    python examples/basic_chat.py
    python examples/basic_chat.py --session-id my_session
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlu.pipeline import NLUPipeline
from src.dialog.state_tracker import StateTracker
from src.dialog.policy import DialogPolicy
from src.dialog.response_generator import ResponseGenerator
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChatBot:
    """Simple chatbot for text-based conversations."""
    
    def __init__(self, config_path: str = "configs/dialog_config.yaml"):
        """
        Initialize chatbot components.
        
        Args:
            config_path: Path to dialog configuration
        """
        self.config = load_config(config_path)
        
        # Initialize NLU pipeline
        logger.info("Loading NLU pipeline...")
        self.nlu = NLUPipeline()
        
        # Initialize dialog components
        logger.info("Initializing dialog manager...")
        self.state_tracker = StateTracker()
        self.policy = DialogPolicy(self.config)
        self.response_generator = ResponseGenerator()
        
        logger.info("Chatbot ready!")
    
    def process_message(self, text: str, session_id: str = "default") -> dict:
        """
        Process a user message and generate response.
        
        Args:
            text: User input text
            session_id: Session identifier for context
            
        Returns:
            Response dictionary with text and metadata
        """
        # Step 1: NLU - Understand the user's intent
        nlu_result = self.nlu.process(text)
        
        logger.debug(f"NLU Result: {nlu_result}")
        
        # Step 2: Update dialog state
        self.state_tracker.update(
            session_id=session_id,
            user_input=text,
            nlu_result=nlu_result
        )
        
        current_state = self.state_tracker.get_state(session_id)
        
        # Step 3: Determine next action using policy
        action = self.policy.get_action(current_state, nlu_result)
        
        # Step 4: Generate response
        response = self.response_generator.generate(
            action=action,
            state=current_state,
            nlu_result=nlu_result
        )
        
        # Update state with response
        self.state_tracker.add_turn(
            session_id=session_id,
            user_text=text,
            bot_text=response["text"],
            action=action
        )
        
        return {
            "text": response["text"],
            "intent": nlu_result.get("intent", {}).get("name"),
            "confidence": nlu_result.get("intent", {}).get("confidence"),
            "entities": nlu_result.get("entities", []),
            "sentiment": nlu_result.get("sentiment"),
            "action": action
        }
    
    def reset_session(self, session_id: str = "default"):
        """Reset conversation state for a session."""
        self.state_tracker.reset(session_id)
        logger.info(f"Session {session_id} reset")
    
    def get_history(self, session_id: str = "default") -> list:
        """Get conversation history for a session."""
        return self.state_tracker.get_history(session_id)


def run_interactive_chat(chatbot: ChatBot, session_id: str):
    """Run interactive chat loop."""
    
    print("\n" + "="*50)
    print("Conversational AI Assistant")
    print("="*50)
    print("Type your message and press Enter.")
    print("Commands: 'quit' to exit, 'reset' to start over")
    print("="*50 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'reset':
                chatbot.reset_session(session_id)
                print("\n[Session reset]\n")
                continue
            
            if user_input.lower() == 'history':
                history = chatbot.get_history(session_id)
                print("\n--- Conversation History ---")
                for turn in history:
                    print(f"User: {turn['user']}")
                    print(f"Bot:  {turn['bot']}")
                    print()
                continue
            
            if user_input.lower() == 'debug':
                # Toggle debug mode
                import logging
                current_level = logger.level
                if current_level == logging.DEBUG:
                    logger.setLevel(logging.INFO)
                    print("[Debug mode OFF]")
                else:
                    logger.setLevel(logging.DEBUG)
                    print("[Debug mode ON]")
                continue
            
            # Process message
            response = chatbot.process_message(user_input, session_id)
            
            # Display response
            print(f"\nBot: {response['text']}")
            
            # Show metadata if intent detected
            if response.get('intent'):
                intent_info = f"[Intent: {response['intent']}"
                if response.get('confidence'):
                    intent_info += f" ({response['confidence']:.0%})"
                intent_info += "]"
                print(f"     {intent_info}")
            
            # Show entities if found
            if response.get('entities'):
                entities_str = ", ".join(
                    f"{e['type']}={e['value']}" 
                    for e in response['entities']
                )
                print(f"     [Entities: {entities_str}]")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nSorry, something went wrong. Please try again.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chatbot example"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="interactive",
        help="Session ID for conversation context"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dialog_config.yaml",
        help="Path to dialog configuration"
    )
    parser.add_argument(
        "--single-message",
        type=str,
        default=None,
        help="Process a single message and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize chatbot
    print("Initializing chatbot...")
    chatbot = ChatBot(config_path=args.config)
    
    if args.single_message:
        # Single message mode
        response = chatbot.process_message(args.single_message, args.session_id)
        print(f"\nBot: {response['text']}")
        print(f"\nFull response: {response}")
    else:
        # Interactive mode
        run_interactive_chat(chatbot, args.session_id)


if __name__ == "__main__":
    main()
