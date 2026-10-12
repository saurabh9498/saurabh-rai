"""Integration tests for the full pipeline."""

import pytest
import numpy as np


class TestFullPipeline:
    """Test the complete voice assistant pipeline."""
    
    def test_text_to_text_pipeline(self):
        """Test text input -> NLU -> Dialog -> Response."""
        from src.nlu.pipeline import NLUPipeline
        from src.dialog.state_tracker import DialogStateTracker
        from src.dialog.response_generator import ResponseGenerator
        
        nlu = NLUPipeline(use_transformer_ner=False, enable_sentiment=False)
        dialog = DialogStateTracker()
        generator = ResponseGenerator()
        
        # Process user input
        text = "What time is it?"
        nlu_result = nlu.process(text)
        
        # Update dialog state
        state = dialog.update(
            session_id="test",
            user_utterance=text,
            intent=nlu_result.intent.intent,
            entities=[e.to_dict() for e in nlu_result.entities.entities],
        )
        
        # Generate response
        from src.dialog.policy import DialogPolicy
        policy = DialogPolicy()
        action = policy.select_action(state)
        response = generator.generate(action)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_audio_to_text(self):
        """Test audio -> ASR -> text."""
        from src.asr.whisper_asr import WhisperASR, WhisperConfig
        
        config = WhisperConfig(model_size="tiny")
        asr = WhisperASR(config)
        
        # Create test audio (1 second)
        audio = np.zeros(16000, dtype=np.float32)
        
        result = asr.transcribe(audio)
        
        assert result.text is not None
        assert result.duration == 1.0
