"""
Speech-to-Text module.

Supports:
- OpenAI Whisper API (default, cloud)
- Groq Whisper API (fast cloud alternative)
- HuggingFace Whisper (local, requires GPU for speed)
"""

import os


def transcribe_audio(audio_path: str, provider: str = "OpenAI Whisper API") -> str:
    """Transcribe audio file to text using the selected provider."""

    if provider == "OpenAI Whisper API":
        return _transcribe_openai(audio_path)
    elif provider == "Groq Whisper":
        return _transcribe_groq(audio_path)
    elif provider == "HuggingFace Whisper (local)":
        return _transcribe_hf_local(audio_path)
    else:
        return _transcribe_openai(audio_path)


def _transcribe_openai(audio_path: str) -> str:
    """Use OpenAI Whisper API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return transcript.strip()
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    except Exception as e:
        raise RuntimeError(f"OpenAI Whisper error: {e}")


def _transcribe_groq(audio_path: str) -> str:
    """Use Groq's Whisper API (faster & cheaper alternative)."""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text"
            )
        return transcript.strip()
    except ImportError:
        raise RuntimeError("groq package not installed. Run: pip install groq")
    except Exception as e:
        raise RuntimeError(f"Groq Whisper error: {e}")


def _transcribe_hf_local(audio_path: str) -> str:
    """
    Use HuggingFace Whisper locally.
    Requires: pip install transformers torch torchaudio
    NOTE: Slow on CPU. Recommend GPU or cloud fallback.
    """
    try:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=device
        )
        result = pipe(audio_path)
        return result["text"].strip()
    except ImportError:
        raise RuntimeError(
            "transformers/torch not installed. Run: pip install transformers torch torchaudio"
        )
    except Exception as e:
        raise RuntimeError(f"HuggingFace Whisper error: {e}")
