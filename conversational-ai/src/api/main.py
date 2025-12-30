"""
FastAPI Application

REST and WebSocket API for the Conversational AI platform.
Handles voice streaming, NLU requests, and dialog management.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import logging
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# Pydantic Models
class TextRequest(BaseModel):
    """Request for text-based interaction."""
    text: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    language: str = "en"


class TextResponse(BaseModel):
    """Response for text interaction."""
    text: str
    intent: str
    confidence: float
    entities: List[Dict[str, Any]]
    session_id: str
    audio_url: Optional[str] = None


class TTSRequest(BaseModel):
    """Request for text-to-speech."""
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = "default"
    speed: float = 1.0
    format: str = "wav"  # wav, mp3, opus


class ASRRequest(BaseModel):
    """Request for speech-to-text."""
    audio_base64: str
    language: Optional[str] = None
    format: str = "wav"


class NLURequest(BaseModel):
    """Request for NLU processing."""
    text: str
    context: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, str]


# Application state
class AppState:
    def __init__(self):
        self.asr = None
        self.nlu = None
        self.dialog_manager = None
        self.tts = None
        self.sessions: Dict[str, Any] = {}

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Conversational AI API...")
    
    try:
        # Initialize components (lazy loading)
        from ..asr.whisper_asr import WhisperASR, WhisperConfig
        from ..nlu.pipeline import NLUPipeline
        from ..dialog.state_tracker import DialogStateTracker
        from ..tts.synthesizer import TTSService
        
        app_state.asr = WhisperASR(WhisperConfig(model_size="base"))
        app_state.nlu = NLUPipeline()
        app_state.dialog_manager = DialogStateTracker()
        app_state.tts = TTSService()
        
        logger.info("All components initialized")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Conversational AI API...")
    app_state.sessions.clear()


# Create FastAPI app
app = FastAPI(
    title="Conversational AI API",
    description="Voice-enabled conversational AI with ASR, NLU, Dialog Management, and TTS",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "asr": "ready" if app_state.asr else "not_loaded",
            "nlu": "ready" if app_state.nlu else "not_loaded",
            "dialog": "ready" if app_state.dialog_manager else "not_loaded",
            "tts": "ready" if app_state.tts else "not_loaded",
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Conversational AI API", "docs": "/docs"}


# Text interaction endpoint
@app.post("/chat", response_model=TextResponse)
async def chat(request: TextRequest):
    """
    Process text input and return response.
    
    Full pipeline: NLU -> Dialog -> Response Generation -> (optional) TTS
    """
    start_time = time.time()
    
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # NLU processing
        if app_state.nlu:
            nlu_result = app_state.nlu.process(request.text)
            intent = nlu_result.intent.intent
            confidence = nlu_result.intent.confidence
            entities = [e.to_dict() for e in nlu_result.entities.entities]
        else:
            intent = "unknown"
            confidence = 0.0
            entities = []
        
        # Dialog management
        if app_state.dialog_manager:
            app_state.dialog_manager.update(
                session_id=session_id,
                user_utterance=request.text,
                intent=intent,
                entities=entities,
            )
            
            context = app_state.dialog_manager.get_context(session_id)
        else:
            context = {}
        
        # Generate response (simplified)
        response_text = _generate_response(intent, entities, context)
        
        # Update dialog with response
        if app_state.dialog_manager:
            state = app_state.dialog_manager.get_or_create_state(session_id)
            if state.history:
                state.history[-1].system_response = response_text
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Chat processed in {latency_ms:.1f}ms: intent={intent}")
        
        return TextResponse(
            text=response_text,
            intent=intent,
            confidence=confidence,
            entities=entities,
            session_id=session_id,
        )
    
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ASR endpoint
@app.post("/asr")
async def transcribe(request: ASRRequest):
    """
    Transcribe audio to text.
    """
    import base64
    
    if not app_state.asr:
        raise HTTPException(status_code=503, detail="ASR not available")
    
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Convert to numpy array (assuming 16-bit PCM)
        import numpy as np
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        result = app_state.asr.transcribe(audio, language=request.language)
        
        return {
            "text": result.text,
            "language": result.language,
            "confidence": result.language_confidence,
            "segments": [s.to_dict() if hasattr(s, 'to_dict') else {} for s in result.segments],
        }
    
    except Exception as e:
        logger.error(f"ASR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# TTS endpoint
@app.post("/tts")
async def synthesize(request: TTSRequest):
    """
    Synthesize speech from text.
    """
    if not app_state.tts:
        raise HTTPException(status_code=503, detail="TTS not available")
    
    try:
        result = app_state.tts.synthesize(
            text=request.text,
            speaker=request.voice if request.voice != "default" else None,
        )
        
        # Return audio as streaming response
        audio_bytes = result.to_wav_bytes()
        
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={
                "X-Duration": str(result.duration),
                "X-Processing-Time": str(result.processing_time),
            },
        )
    
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# NLU endpoint
@app.post("/nlu")
async def understand(request: NLURequest):
    """
    Process text with NLU pipeline.
    """
    if not app_state.nlu:
        raise HTTPException(status_code=503, detail="NLU not available")
    
    try:
        result = app_state.nlu.process(request.text)
        
        return {
            "intent": result.intent.to_dict(),
            "entities": result.entities.to_dict(),
            "sentiment": result.sentiment,
        }
    
    except Exception as e:
        logger.error(f"NLU failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket for streaming voice
@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming voice interaction.
    
    Protocol:
    - Client sends audio chunks (binary)
    - Server sends transcription and responses (JSON)
    """
    await websocket.accept()
    logger.info(f"Voice WebSocket connected: {session_id}")
    
    try:
        # Initialize streaming ASR
        from ..asr.streaming import StreamingASR, StreamingConfig
        
        streaming_config = StreamingConfig()
        
        if app_state.asr:
            streaming_asr = StreamingASR(asr=app_state.asr, config=streaming_config)
        else:
            streaming_asr = None
        
        audio_buffer = []
        
        while True:
            # Receive message
            message = await websocket.receive()
            
            if "bytes" in message:
                # Audio data
                audio_chunk = message["bytes"]
                audio_buffer.append(audio_chunk)
                
                # Process when we have enough audio
                if len(audio_buffer) >= 10:  # ~300ms of audio
                    combined = b"".join(audio_buffer)
                    audio_buffer.clear()
                    
                    # Transcribe
                    if app_state.asr:
                        import numpy as np
                        audio = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0
                        result = app_state.asr.transcribe(audio)
                        
                        if result.text.strip():
                            # Send transcription
                            await websocket.send_json({
                                "type": "transcription",
                                "text": result.text,
                                "is_final": True,
                            })
                            
                            # Process with NLU and dialog
                            response = await _process_voice_input(
                                result.text,
                                session_id,
                            )
                            
                            await websocket.send_json({
                                "type": "response",
                                **response,
                            })
            
            elif "text" in message:
                # Control message
                data = message["text"]
                if data == "end":
                    break
    
    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
        await websocket.close(code=1011)


async def _process_voice_input(text: str, session_id: str) -> Dict[str, Any]:
    """Process voice input through the full pipeline."""
    # NLU
    if app_state.nlu:
        nlu_result = app_state.nlu.process(text)
        intent = nlu_result.intent.intent
        entities = [e.to_dict() for e in nlu_result.entities.entities]
    else:
        intent = "unknown"
        entities = []
    
    # Dialog
    if app_state.dialog_manager:
        app_state.dialog_manager.update(
            session_id=session_id,
            user_utterance=text,
            intent=intent,
            entities=entities,
        )
        context = app_state.dialog_manager.get_context(session_id)
    else:
        context = {}
    
    # Response
    response_text = _generate_response(intent, entities, context)
    
    return {
        "text": response_text,
        "intent": intent,
        "entities": entities,
    }


def _generate_response(
    intent: str,
    entities: List[Dict],
    context: Dict[str, Any],
) -> str:
    """Generate response based on intent and context."""
    # Simplified response generation
    responses = {
        "greeting": "Hello! How can I help you today?",
        "goodbye": "Goodbye! Have a great day!",
        "help": "I can help you with weather, reminders, music, and more. What would you like to do?",
        "get_weather": "Let me check the weather for you. It looks like it will be sunny with a high of 72°F.",
        "set_timer": "I've set a timer for you.",
        "set_reminder": "I'll remind you about that.",
        "play_music": "Playing music for you now.",
        "fallback": "I'm not sure I understood. Could you rephrase that?",
        "unknown": "I'm sorry, I didn't quite catch that. Could you try again?",
    }
    
    # Check for missing slots
    missing = context.get("missing_slots", [])
    if missing:
        slot = missing[0]
        return f"What {slot.replace('_', ' ')} would you like?"
    
    return responses.get(intent, responses["fallback"])


# Run with: uvicorn src.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
