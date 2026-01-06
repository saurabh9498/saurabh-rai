#!/usr/bin/env python3
"""
Voice Assistant Example

This example demonstrates how to:
1. Capture audio from microphone
2. Transcribe speech using Whisper ASR
3. Process through the conversational pipeline
4. Generate speech response using TTS

Usage:
    python examples/voice_assistant.py
    python examples/voice_assistant.py --asr-model small --tts-voice female
"""

import argparse
import sys
import time
import wave
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.whisper_asr import WhisperASR
from src.asr.vad import VoiceActivityDetector
from src.nlu.pipeline import NLUPipeline
from src.dialog.state_tracker import StateTracker
from src.dialog.policy import DialogPolicy
from src.dialog.response_generator import ResponseGenerator
from src.tts.synthesizer import TTSSynthesizer
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Check for audio dependencies
try:
    import pyaudio
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("PyAudio not available. Install with: pip install pyaudio")


class VoiceAssistant:
    """Full voice-enabled conversational assistant."""
    
    def __init__(
        self,
        asr_model: str = "base",
        tts_model: str = "tacotron2-DDC",
        tts_voice: str = "default",
        config_path: str = "configs/dialog_config.yaml"
    ):
        """
        Initialize voice assistant components.
        
        Args:
            asr_model: Whisper model size
            tts_model: TTS model name
            tts_voice: Voice profile
            config_path: Dialog configuration path
        """
        self.config = load_config(config_path)
        
        # Initialize ASR
        logger.info(f"Loading ASR model ({asr_model})...")
        self.asr = WhisperASR(model_size=asr_model)
        self.vad = VoiceActivityDetector()
        
        # Initialize NLU
        logger.info("Loading NLU pipeline...")
        self.nlu = NLUPipeline()
        
        # Initialize Dialog
        logger.info("Initializing dialog manager...")
        self.state_tracker = StateTracker()
        self.policy = DialogPolicy(self.config)
        self.response_generator = ResponseGenerator()
        
        # Initialize TTS
        logger.info(f"Loading TTS model ({tts_model})...")
        self.tts = TTSSynthesizer(model_name=tts_model, voice=tts_voice)
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        
        logger.info("Voice assistant ready!")
    
    def listen(self, timeout: float = 5.0, silence_threshold: float = 1.0) -> Optional[str]:
        """
        Listen for voice input and transcribe.
        
        Args:
            timeout: Maximum recording time in seconds
            silence_threshold: Silence duration to stop recording
            
        Returns:
            Transcribed text or None if no speech detected
        """
        if not AUDIO_AVAILABLE:
            # Fallback to text input
            return input("You (text): ").strip()
        
        print("🎤 Listening...")
        
        try:
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            silent_chunks = 0
            max_silent_chunks = int(silence_threshold * self.sample_rate / self.chunk_size)
            max_chunks = int(timeout * self.sample_rate / self.chunk_size)
            
            for i in range(max_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Convert to numpy for VAD
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Check for voice activity
                if self.vad.is_speech(audio_chunk, self.sample_rate):
                    silent_chunks = 0
                else:
                    silent_chunks += 1
                
                # Stop if silence detected after speech
                if len(frames) > 10 and silent_chunks > max_silent_chunks:
                    break
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Convert frames to audio array
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe
            print("📝 Transcribing...")
            result = self.asr.transcribe(audio_array, self.sample_rate)
            
            text = result.get("text", "").strip()
            
            if text:
                print(f"You said: {text}")
                return text
            else:
                print("(No speech detected)")
                return None
                
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            return None
    
    def speak(self, text: str, block: bool = True):
        """
        Convert text to speech and play.
        
        Args:
            text: Text to speak
            block: Whether to block until playback complete
        """
        if not text:
            return
        
        print(f"🔊 Speaking: {text}")
        
        try:
            # Generate audio
            audio = self.tts.synthesize(text)
            
            if AUDIO_AVAILABLE:
                # Play audio
                self.tts.play(audio, block=block)
            else:
                # Just print if audio not available
                logger.info("(Audio playback not available)")
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def process(self, text: str, session_id: str = "voice") -> str:
        """
        Process text through the conversational pipeline.
        
        Args:
            text: User input text
            session_id: Session identifier
            
        Returns:
            Response text
        """
        # NLU
        nlu_result = self.nlu.process(text)
        
        # Update state
        self.state_tracker.update(
            session_id=session_id,
            user_input=text,
            nlu_result=nlu_result
        )
        
        current_state = self.state_tracker.get_state(session_id)
        
        # Get action
        action = self.policy.get_action(current_state, nlu_result)
        
        # Generate response
        response = self.response_generator.generate(
            action=action,
            state=current_state,
            nlu_result=nlu_result
        )
        
        # Update state
        self.state_tracker.add_turn(
            session_id=session_id,
            user_text=text,
            bot_text=response["text"],
            action=action
        )
        
        return response["text"]
    
    def run_conversation_loop(self, session_id: str = "voice"):
        """Run continuous voice conversation loop."""
        
        print("\n" + "="*50)
        print("🎙️  Voice Assistant")
        print("="*50)
        print("Speak naturally. Say 'goodbye' or 'exit' to quit.")
        print("="*50 + "\n")
        
        # Initial greeting
        self.speak("Hello! How can I help you today?")
        
        while True:
            try:
                # Listen for input
                text = self.listen()
                
                if not text:
                    continue
                
                # Check for exit commands
                if any(word in text.lower() for word in ['goodbye', 'exit', 'quit', 'bye']):
                    self.speak("Goodbye! Have a great day!")
                    break
                
                # Process and respond
                response = self.process(text, session_id)
                self.speak(response)
                
                print()  # Blank line between turns
                
            except KeyboardInterrupt:
                print("\n")
                self.speak("Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                self.speak("Sorry, something went wrong. Please try again.")


def process_audio_file(
    assistant: VoiceAssistant,
    audio_path: str,
    output_path: Optional[str] = None
) -> dict:
    """
    Process an audio file through the pipeline.
    
    Args:
        assistant: Voice assistant instance
        audio_path: Path to input audio file
        output_path: Optional path to save response audio
        
    Returns:
        Processing results
    """
    import soundfile as sf
    
    # Load audio
    audio, sample_rate = sf.read(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Transcribe
    asr_result = assistant.asr.transcribe(audio, sample_rate)
    text = asr_result.get("text", "").strip()
    
    if not text:
        return {"error": "No speech detected"}
    
    # Process
    response = assistant.process(text)
    
    # Generate response audio
    response_audio = assistant.tts.synthesize(response)
    
    # Save if requested
    if output_path:
        sf.write(output_path, response_audio, 22050)
    
    return {
        "input_text": text,
        "response_text": response,
        "asr_confidence": asr_result.get("confidence"),
        "output_audio_path": output_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Voice-enabled conversational assistant"
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper ASR model size"
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default="tacotron2-DDC",
        help="TTS model name"
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default="default",
        help="TTS voice profile"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Process a single audio file instead of live input"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output audio file path"
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Use text input/output only (no audio)"
    )
    
    args = parser.parse_args()
    
    # Initialize assistant
    print("Initializing voice assistant...")
    assistant = VoiceAssistant(
        asr_model=args.asr_model,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice
    )
    
    if args.audio_file:
        # Process single file
        result = process_audio_file(
            assistant,
            args.audio_file,
            args.output
        )
        print(f"\nInput: {result.get('input_text')}")
        print(f"Response: {result.get('response_text')}")
        if args.output:
            print(f"Audio saved to: {args.output}")
    else:
        # Run interactive loop
        assistant.run_conversation_loop()


if __name__ == "__main__":
    main()
