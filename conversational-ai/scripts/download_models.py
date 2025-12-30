#!/usr/bin/env python3
"""
Download and setup all required models for the Conversational AI Assistant.

This script downloads:
- Whisper ASR models (OpenAI)
- TTS models (Coqui TTS)
- NLU models (spaCy, Transformers)

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --asr-model base --tts-model tacotron2-DDC
    python scripts/download_models.py --components asr nlu  # Download specific components
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Configurations
# =============================================================================

WHISPER_MODELS = {
    "tiny": {"size": "39M", "vram": "~1GB", "speed": "~32x"},
    "base": {"size": "74M", "vram": "~1GB", "speed": "~16x"},
    "small": {"size": "244M", "vram": "~2GB", "speed": "~6x"},
    "medium": {"size": "769M", "vram": "~5GB", "speed": "~2x"},
    "large": {"size": "1550M", "vram": "~10GB", "speed": "~1x"},
    "large-v2": {"size": "1550M", "vram": "~10GB", "speed": "~1x"},
    "large-v3": {"size": "1550M", "vram": "~10GB", "speed": "~1x"},
}

TTS_MODELS = {
    "tacotron2-DDC": "tts_models/en/ljspeech/tacotron2-DDC",
    "tacotron2-DCA": "tts_models/en/ljspeech/tacotron2-DCA",
    "glow-tts": "tts_models/en/ljspeech/glow-tts",
    "speedy-speech": "tts_models/en/ljspeech/speedy-speech",
    "vits": "tts_models/en/ljspeech/vits",
    "fast-pitch": "tts_models/en/ljspeech/fast_pitch",
}

SPACY_MODELS = {
    "en_core_web_sm": {"size": "12MB", "accuracy": "basic"},
    "en_core_web_md": {"size": "40MB", "accuracy": "medium"},
    "en_core_web_lg": {"size": "560MB", "accuracy": "high"},
    "en_core_web_trf": {"size": "460MB", "accuracy": "highest (transformer)"},
}

TRANSFORMERS_MODELS = {
    "distilbert-base-uncased": "distilbert-base-uncased",
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base",
}


# =============================================================================
# Download Functions
# =============================================================================

def download_whisper_model(model_name: str = "base", cache_dir: Optional[str] = None) -> bool:
    """Download Whisper ASR model.
    
    Args:
        model_name: Model size (tiny, base, small, medium, large, large-v2, large-v3)
        cache_dir: Custom cache directory for models
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading Whisper model: {model_name}")
    
    if model_name not in WHISPER_MODELS:
        logger.error(f"Invalid model: {model_name}. Available: {list(WHISPER_MODELS.keys())}")
        return False
    
    model_info = WHISPER_MODELS[model_name]
    logger.info(f"  Size: {model_info['size']}, VRAM: {model_info['vram']}, Speed: {model_info['speed']}")
    
    try:
        import whisper
        
        # Set cache directory if specified
        if cache_dir:
            os.environ["XDG_CACHE_HOME"] = cache_dir
        
        # Download model
        logger.info(f"  Downloading (this may take a while)...")
        model = whisper.load_model(model_name)
        
        logger.info(f"  ✅ Whisper {model_name} model downloaded successfully!")
        return True
        
    except ImportError:
        logger.error("whisper package not installed. Run: pip install openai-whisper")
        return False
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {e}")
        return False


def download_tts_model(model_name: str = "tacotron2-DDC") -> bool:
    """Download Coqui TTS model.
    
    Args:
        model_name: TTS model name
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading TTS model: {model_name}")
    
    if model_name not in TTS_MODELS:
        logger.error(f"Invalid model: {model_name}. Available: {list(TTS_MODELS.keys())}")
        return False
    
    model_path = TTS_MODELS[model_name]
    
    try:
        from TTS.api import TTS
        
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Downloading (this may take a while)...")
        
        # This will download the model
        tts = TTS(model_path)
        
        logger.info(f"  ✅ TTS {model_name} model downloaded successfully!")
        return True
        
    except ImportError:
        logger.error("TTS package not installed. Run: pip install TTS")
        return False
    except Exception as e:
        logger.error(f"Failed to download TTS model: {e}")
        return False


def download_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """Download spaCy NLP model.
    
    Args:
        model_name: spaCy model name
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading spaCy model: {model_name}")
    
    if model_name not in SPACY_MODELS:
        logger.error(f"Invalid model: {model_name}. Available: {list(SPACY_MODELS.keys())}")
        return False
    
    model_info = SPACY_MODELS[model_name]
    logger.info(f"  Size: {model_info['size']}, Accuracy: {model_info['accuracy']}")
    
    try:
        # Use subprocess to download spaCy model
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"  ✅ spaCy {model_name} model downloaded successfully!")
            return True
        else:
            logger.error(f"  Failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download spaCy model: {e}")
        return False


def download_transformers_model(model_name: str = "distilbert-base-uncased") -> bool:
    """Download Hugging Face Transformers model.
    
    Args:
        model_name: Transformers model name
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading Transformers model: {model_name}")
    
    model_path = TRANSFORMERS_MODELS.get(model_name, model_name)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        logger.info(f"  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"  Downloading model...")
        model = AutoModel.from_pretrained(model_path)
        
        logger.info(f"  ✅ Transformers {model_name} model downloaded successfully!")
        return True
        
    except ImportError:
        logger.error("transformers package not installed. Run: pip install transformers")
        return False
    except Exception as e:
        logger.error(f"Failed to download Transformers model: {e}")
        return False


def download_faster_whisper_model(model_name: str = "base") -> bool:
    """Download Faster-Whisper model (CTranslate2 optimized).
    
    Args:
        model_name: Model size
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading Faster-Whisper model: {model_name}")
    
    try:
        from faster_whisper import WhisperModel
        
        logger.info(f"  Downloading CTranslate2 optimized model...")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        
        logger.info(f"  ✅ Faster-Whisper {model_name} model downloaded successfully!")
        return True
        
    except ImportError:
        logger.error("faster-whisper package not installed. Run: pip install faster-whisper")
        return False
    except Exception as e:
        logger.error(f"Failed to download Faster-Whisper model: {e}")
        return False


# =============================================================================
# Main Functions
# =============================================================================

def download_all_models(
    asr_model: str = "base",
    tts_model: str = "tacotron2-DDC",
    spacy_model: str = "en_core_web_sm",
    transformers_model: str = "distilbert-base-uncased",
    components: Optional[List[str]] = None,
    use_faster_whisper: bool = False,
) -> dict:
    """Download all required models.
    
    Args:
        asr_model: Whisper model size
        tts_model: TTS model name
        spacy_model: spaCy model name
        transformers_model: Transformers model name
        components: List of components to download (asr, tts, nlu). If None, download all.
        use_faster_whisper: Use Faster-Whisper instead of OpenAI Whisper
        
    Returns:
        Dictionary with download status for each component
    """
    results = {}
    
    # Determine which components to download
    if components is None:
        components = ["asr", "tts", "nlu"]
    
    logger.info("=" * 60)
    logger.info("Conversational AI - Model Download Script")
    logger.info("=" * 60)
    logger.info(f"Components to download: {components}")
    logger.info("")
    
    # Download ASR model
    if "asr" in components:
        logger.info("-" * 40)
        logger.info("ASR Models")
        logger.info("-" * 40)
        
        if use_faster_whisper:
            results["faster_whisper"] = download_faster_whisper_model(asr_model)
        else:
            results["whisper"] = download_whisper_model(asr_model)
        logger.info("")
    
    # Download TTS model
    if "tts" in components:
        logger.info("-" * 40)
        logger.info("TTS Models")
        logger.info("-" * 40)
        
        results["tts"] = download_tts_model(tts_model)
        logger.info("")
    
    # Download NLU models
    if "nlu" in components:
        logger.info("-" * 40)
        logger.info("NLU Models")
        logger.info("-" * 40)
        
        results["spacy"] = download_spacy_model(spacy_model)
        results["transformers"] = download_transformers_model(transformers_model)
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    
    for component, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"  {component}: {status}")
    
    total = len(results)
    successful = sum(results.values())
    logger.info("")
    logger.info(f"Total: {successful}/{total} models downloaded successfully")
    
    return results


def list_available_models():
    """Print all available models."""
    print("\n" + "=" * 60)
    print("Available Models")
    print("=" * 60)
    
    print("\nWhisper ASR Models:")
    print("-" * 40)
    for name, info in WHISPER_MODELS.items():
        print(f"  {name:12} - Size: {info['size']:8} VRAM: {info['vram']:8} Speed: {info['speed']}")
    
    print("\nTTS Models (Coqui):")
    print("-" * 40)
    for name, path in TTS_MODELS.items():
        print(f"  {name:20} - {path}")
    
    print("\nspaCy Models:")
    print("-" * 40)
    for name, info in SPACY_MODELS.items():
        print(f"  {name:20} - Size: {info['size']:8} Accuracy: {info['accuracy']}")
    
    print("\nTransformers Models:")
    print("-" * 40)
    for name, path in TRANSFORMERS_MODELS.items():
        print(f"  {name:30} - {path}")
    
    print("")


def verify_models(
    asr_model: str = "base",
    spacy_model: str = "en_core_web_sm",
) -> dict:
    """Verify that required models are installed.
    
    Returns:
        Dictionary with verification status
    """
    results = {}
    
    logger.info("Verifying installed models...")
    
    # Check Whisper
    try:
        import whisper
        whisper.load_model(asr_model)
        results["whisper"] = True
        logger.info(f"  ✅ Whisper {asr_model} is installed")
    except:
        results["whisper"] = False
        logger.info(f"  ❌ Whisper {asr_model} is NOT installed")
    
    # Check spaCy
    try:
        import spacy
        spacy.load(spacy_model)
        results["spacy"] = True
        logger.info(f"  ✅ spaCy {spacy_model} is installed")
    except:
        results["spacy"] = False
        logger.info(f"  ❌ spaCy {spacy_model} is NOT installed")
    
    # Check TTS
    try:
        from TTS.api import TTS
        # Just check if TTS is importable
        results["tts"] = True
        logger.info(f"  ✅ TTS package is installed")
    except:
        results["tts"] = False
        logger.info(f"  ❌ TTS package is NOT installed")
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download models for Conversational AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models with defaults
  python download_models.py
  
  # Download specific ASR model
  python download_models.py --asr-model small
  
  # Download only ASR and NLU components
  python download_models.py --components asr nlu
  
  # List all available models
  python download_models.py --list
  
  # Verify installed models
  python download_models.py --verify
        """
    )
    
    parser.add_argument(
        "--asr-model",
        default="base",
        choices=list(WHISPER_MODELS.keys()),
        help="Whisper ASR model size (default: base)"
    )
    
    parser.add_argument(
        "--tts-model",
        default="tacotron2-DDC",
        choices=list(TTS_MODELS.keys()),
        help="TTS model (default: tacotron2-DDC)"
    )
    
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        choices=list(SPACY_MODELS.keys()),
        help="spaCy model (default: en_core_web_sm)"
    )
    
    parser.add_argument(
        "--transformers-model",
        default="distilbert-base-uncased",
        help="Transformers model for intent classification"
    )
    
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["asr", "tts", "nlu"],
        help="Specific components to download"
    )
    
    parser.add_argument(
        "--use-faster-whisper",
        action="store_true",
        help="Use Faster-Whisper (CTranslate2) instead of OpenAI Whisper"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify installed models"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Custom cache directory for models"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        list_available_models()
        return
    
    # Verify models
    if args.verify:
        verify_models(args.asr_model, args.spacy_model)
        return
    
    # Download models
    results = download_all_models(
        asr_model=args.asr_model,
        tts_model=args.tts_model,
        spacy_model=args.spacy_model,
        transformers_model=args.transformers_model,
        components=args.components,
        use_faster_whisper=args.use_faster_whisper,
    )
    
    # Exit with error if any download failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
