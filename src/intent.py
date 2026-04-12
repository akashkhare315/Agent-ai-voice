"""
Intent Understanding module.

Uses an LLM to classify the user's intent and extract parameters
from transcribed text. Supports compound commands (multiple intents).

Supported intents:
  - create_file       : Create an empty or text file
  - write_code        : Generate code and save to file
  - summarize_text    : Summarize provided content
  - general_chat      : Conversational query
  - create_folder     : Create a directory
  - list_files        : List files in output directory
  - unknown           : Fallback
"""

import os
import json
import re


INTENT_SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled file and code agent.

Analyze the user's command and return a JSON object with:
{
  "intents": ["intent1", "intent2"],   // list, supports compound commands
  "intent": "primary_intent",          // primary/first intent
  "parameters": {
    "filename": "...",                 // target filename if mentioned
    "language": "...",                 // programming language if mentioned
    "content": "...",                  // content to write/summarize if provided
    "description": "...",             // what to generate/do
    "folder_name": "..."               // folder name if creating folder
  },
  "confidence": 0.95,
  "reasoning": "brief explanation"
}

Valid intents: create_file, write_code, summarize_text, general_chat, create_folder, list_files, unknown

Rules:
- Detect ALL intents if compound command (e.g. "summarize this and save it" = [summarize_text, create_file])
- Be generous with write_code: "create a python file", "make a script", "write a function" → write_code
- If no filename given, infer a sensible one based on context
- Infer language from context (mention of Python, JS, etc.) or file extension
- Return ONLY valid JSON, no markdown fences, no extra text
"""


def classify_intent(text: str, provider: str = "Anthropic Claude", context: str = "") -> dict:
    """Classify intent from transcribed text."""

    prompt = f"""Previous context: {context}

User command: {text}

Classify the intent(s) and extract parameters."""

    if provider == "Anthropic Claude":
        return _classify_anthropic(prompt)
    elif provider == "OpenAI GPT-4":
        return _classify_openai(prompt)
    elif provider == "Ollama (local)":
        return _classify_ollama(prompt)
    else:
        return _classify_anthropic(prompt)


def _parse_intent_json(raw: str) -> dict:
    """Robustly parse intent JSON from LLM output."""
    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw)
        # Normalize: ensure 'intents' list exists
        if "intent" in data and "intents" not in data:
            data["intents"] = [data["intent"]]
        elif "intents" not in data:
            data["intents"] = ["unknown"]
            data["intent"] = "unknown"
        if "intent" not in data:
            data["intent"] = data["intents"][0]
        if "parameters" not in data:
            data["parameters"] = {}
        return data
    except json.JSONDecodeError:
        return {
            "intent": "general_chat",
            "intents": ["general_chat"],
            "parameters": {"description": raw[:200]},
            "confidence": 0.3,
            "reasoning": "Failed to parse structured response"
        }


def _classify_anthropic(prompt: str) -> dict:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=INTENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return _parse_intent_json(response.content[0].text)
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
    except Exception as e:
        raise RuntimeError(f"Anthropic API error: {e}")


def _classify_openai(prompt: str) -> dict:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        return _parse_intent_json(response.choices[0].message.content)
    except Exception as e:
        raise RuntimeError(f"OpenAI error: {e}")


def _classify_ollama(prompt: str) -> dict:
    try:
        import requests
        payload = {
            "model": os.environ.get("OLLAMA_MODEL", "llama3"),
            "system": INTENT_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        return _parse_intent_json(r.json().get("response", "{}"))
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}. Make sure Ollama is running: ollama serve")
