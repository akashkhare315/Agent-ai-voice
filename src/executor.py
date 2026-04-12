"""
Tool Execution module.

Executes actions based on detected intent:
  - create_file    : Creates a new file in output/
  - write_code     : Generates code via LLM and saves to output/
  - summarize_text : Summarizes text via LLM
  - general_chat   : Conversational response
  - create_folder  : Creates a subfolder in output/
  - list_files     : Lists output/ contents
"""

import os
import json
import re
from pathlib import Path


CODE_GEN_SYSTEM = """You are an expert software engineer. 
Generate clean, well-commented, production-quality code.
Return ONLY the raw code with no markdown fences, no explanation outside comments.
Include a docstring at the top explaining what the code does."""

SUMMARIZE_SYSTEM = """You are a precise summarizer. 
Produce a concise, well-structured summary. 
Use bullet points for key points. Be factual and neutral."""

CHAT_SYSTEM = """You are a helpful, knowledgeable assistant integrated into a voice agent.
Be concise but complete. Respond in plain text."""


def execute_action(intent_data: dict, transcript: str, output_dir: Path, llm_provider: str) -> dict:
    """Route to the correct executor based on intent."""

    intents = intent_data.get("intents", [intent_data.get("intent", "unknown")])
    params = intent_data.get("parameters", {})

    # Handle compound commands — execute all intents
    if len(intents) > 1:
        return _execute_compound(intents, params, transcript, output_dir, llm_provider)

    intent = intents[0].lower()

    if intent == "write_code":
        return _execute_write_code(params, transcript, output_dir, llm_provider)
    elif intent == "create_file":
        return _execute_create_file(params, transcript, output_dir)
    elif intent == "summarize_text":
        return _execute_summarize(params, transcript, output_dir, llm_provider)
    elif intent == "create_folder":
        return _execute_create_folder(params, output_dir)
    elif intent == "list_files":
        return _execute_list_files(output_dir)
    elif intent == "general_chat":
        return _execute_chat(transcript, llm_provider)
    else:
        return _execute_chat(transcript, llm_provider)


def _execute_compound(intents, params, transcript, output_dir, llm_provider):
    """Execute multiple intents in sequence."""
    results = []
    combined_output = ""
    last_output = ""

    for intent in intents:
        # Pass previous output as content for chained operations
        if intent == "summarize_text" and last_output:
            params["content"] = last_output
        elif intent == "create_file" and last_output:
            params["content"] = last_output

        r = execute_action(
            {"intents": [intent], "intent": intent, "parameters": params},
            transcript, output_dir, llm_provider
        )
        results.append(f"[{intent}] {r.get('action_taken', '')}")
        combined_output += r.get("output", "") + "\n"
        last_output = r.get("output", "")

    return {
        "status": "success",
        "action_taken": " → ".join(results),
        "output": combined_output.strip()
    }


def _execute_write_code(params, transcript, output_dir, llm_provider):
    """Generate code and save to file."""
    description = params.get("description") or transcript
    language = params.get("language", "python")
    filename = params.get("filename") or _infer_filename(description, language)

    # Ensure safe filename inside output/
    filename = _safe_filename(filename, default_ext=_lang_ext(language))
    filepath = output_dir / filename

    prompt = f"""Generate {language} code for the following requirement:
{description}

Language: {language}
Filename will be: {filename}"""

    code = _llm_call(prompt, CODE_GEN_SYSTEM, llm_provider)
    code = _strip_fences(code)

    filepath.write_text(code, encoding="utf-8")

    return {
        "status": "success",
        "action_taken": f"Generated {language} code → output/{filename}",
        "output": f"```{language}\n{code[:800]}{'...' if len(code)>800 else ''}\n```",
        "filepath": str(filepath)
    }


def _execute_create_file(params, transcript, output_dir):
    """Create an empty or content-filled file."""
    filename = params.get("filename") or _infer_filename(transcript, "txt")
    filename = _safe_filename(filename, default_ext=".txt")
    content = params.get("content", "")
    filepath = output_dir / filename
    filepath.write_text(content, encoding="utf-8")

    return {
        "status": "success",
        "action_taken": f"Created file → output/{filename}",
        "output": f"File `output/{filename}` created successfully.\n{f'Content: {content[:200]}' if content else 'File is empty.'}"
    }


def _execute_summarize(params, transcript, output_dir, llm_provider):
    """Summarize provided text."""
    content = params.get("content") or transcript
    save_file = params.get("filename")

    summary = _llm_call(
        f"Summarize the following:\n\n{content}",
        SUMMARIZE_SYSTEM,
        llm_provider
    )

    output = summary
    action = "Generated summary"

    # Save if filename given or compound command
    if save_file:
        save_file = _safe_filename(save_file, default_ext=".txt")
        (output_dir / save_file).write_text(summary, encoding="utf-8")
        action += f" → output/{save_file}"

    return {
        "status": "success",
        "action_taken": action,
        "output": summary
    }


def _execute_create_folder(params, output_dir):
    """Create a subfolder inside output/."""
    folder_name = params.get("folder_name") or params.get("filename") or "new_folder"
    folder_name = re.sub(r'[^\w\-]', '_', folder_name)
    folder_path = output_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return {
        "status": "success",
        "action_taken": f"Created folder → output/{folder_name}/",
        "output": f"Folder `output/{folder_name}/` created."
    }


def _execute_list_files(output_dir):
    """List files in output directory."""
    files = list(output_dir.glob("**/*"))
    if not files:
        return {"status": "success", "action_taken": "Listed output/", "output": "output/ is empty."}
    listing = "\n".join(f"  {'📁' if f.is_dir() else '📄'} {f.relative_to(output_dir)}" for f in sorted(files))
    return {
        "status": "success",
        "action_taken": "Listed output/ directory",
        "output": f"output/ contents:\n{listing}"
    }


def _execute_chat(transcript, llm_provider):
    """General conversational response."""
    response = _llm_call(transcript, CHAT_SYSTEM, llm_provider)
    return {
        "status": "success",
        "action_taken": "General chat response",
        "output": response
    }


# ── LLM helpers ────────────────────────────────────────────────────────────

def _llm_call(prompt: str, system: str, provider: str) -> str:
    if provider == "Anthropic Claude":
        return _llm_anthropic(prompt, system)
    elif provider == "OpenAI GPT-4":
        return _llm_openai(prompt, system)
    elif provider == "Ollama (local)":
        return _llm_ollama(prompt, system)
    return _llm_anthropic(prompt, system)


def _llm_anthropic(prompt, system):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    r = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text


def _llm_openai(prompt, system):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=2048
    )
    return r.choices[0].message.content


def _llm_ollama(prompt, system):
    import requests
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": os.environ.get("OLLAMA_MODEL", "llama3"),
        "system": system,
        "prompt": prompt,
        "stream": False
    }, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")


# ── Utilities ──────────────────────────────────────────────────────────────

def _infer_filename(text: str, language: str = "txt") -> str:
    text = text.lower()
    # Extract meaningful words
    words = re.findall(r'\b[a-z]+\b', text)
    stop = {"a","an","the","create","make","write","generate","new","file","with","and","for","to","in"}
    meaningful = [w for w in words if w not in stop][:3]
    name = "_".join(meaningful) if meaningful else "output"
    return f"{name}.{_lang_ext(language).lstrip('.')}"


def _safe_filename(name: str, default_ext: str = ".txt") -> str:
    """Sanitize filename and ensure it stays within output/."""
    # Remove path traversal
    name = Path(name).name
    name = re.sub(r'[^\w\.\-]', '_', name)
    if "." not in name:
        name = name + default_ext
    return name


def _lang_ext(language: str) -> str:
    mapping = {
        "python": ".py", "javascript": ".js", "typescript": ".ts",
        "java": ".java", "c": ".c", "cpp": ".cpp", "c++": ".cpp",
        "go": ".go", "rust": ".rs", "ruby": ".rb", "php": ".php",
        "shell": ".sh", "bash": ".sh", "html": ".html", "css": ".css",
        "sql": ".sql", "r": ".r", "kotlin": ".kt", "swift": ".swift",
        "txt": ".txt", "markdown": ".md", "md": ".md", "json": ".json",
        "yaml": ".yaml", "toml": ".toml"
    }
    return mapping.get(language.lower(), ".py")


def _strip_fences(text: str) -> str:
    """Remove markdown code fences."""
    return re.sub(r"```[\w]*\n?|```", "", text).strip()
