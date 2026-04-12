import streamlit as st
import os
import json
import time
import tempfile
import datetime
from pathlib import Path

# Page config
st.set_page_config(
    page_title="VoiceAgent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from src.stt import transcribe_audio
from src.intent import classify_intent
from src.executor import execute_action
from src.memory import SessionMemory

# Initialize session memory
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
  <div class="header-title">
    <span class="accent">Voice</span>Agent
  </div>
  <div class="header-sub">Speak. Understand. Execute.</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ Configuration</div>', unsafe_allow_html=True)

    stt_provider = st.selectbox(
        "STT Provider",
        ["OpenAI Whisper API", "Groq Whisper", "HuggingFace Whisper (local)"],
        index=0,
        help="Choose how audio is transcribed to text"
    )

    llm_provider = st.selectbox(
        "LLM Provider",
        ["Anthropic Claude", "Ollama (local)", "OpenAI GPT-4"],
        index=0,
        help="Choose the model for intent understanding"
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-title">🛡 Safety</div>', unsafe_allow_html=True)
    human_in_loop = st.toggle("Human-in-the-Loop", value=True,
        help="Require confirmation before file operations")

    st.markdown("---")
    st.markdown('<div class="sidebar-title">📁 Output Folder</div>', unsafe_allow_html=True)
    st.code(str(OUTPUT_DIR.absolute()), language=None)

    # List output files
    files = list(OUTPUT_DIR.glob("*"))
    if files:
        st.markdown("**Created files:**")
        for f in sorted(files)[-5:]:
            st.markdown(f'<span class="file-chip">📄 {f.name}</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Session", use_container_width=True):
        st.session_state.history = []
        st.session_state.memory = SessionMemory()
        st.session_state.pending_action = None
        st.rerun()

# ── Main layout ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="section-title">🎙 Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Audio Source",
        ["Upload Audio File", "Record via Microphone (browser)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    audio_bytes = None

    if input_mode == "Upload Audio File":
        uploaded = st.file_uploader(
            "Drop an audio file here",
            type=["wav", "mp3", "m4a", "ogg", "webm"],
            label_visibility="collapsed"
        )
        if uploaded:
            st.audio(uploaded, format=uploaded.type)
            audio_bytes = uploaded.read()
            uploaded.seek(0)
            audio_bytes = uploaded.read()
    else:
        audio_input = st.audio_input("Click to record", label_visibility="collapsed")
        if audio_input:
            audio_bytes = audio_input.read()
            st.audio(audio_input)

    # Text override
    with st.expander("✍ Or type a command directly (skip audio)"):
        text_override = st.text_area(
            "Command text",
            placeholder="e.g. Create a Python file with a retry function",
            label_visibility="collapsed",
            height=80
        )

    run_button = st.button("▶  Process", type="primary", use_container_width=True)

    # ── Pipeline execution ─────────────────────────────────────────────────
    if run_button and (audio_bytes or text_override.strip()):
        result_box = st.container()

        with result_box:
            # Step 1: Transcription
            transcript = ""
            if text_override.strip():
                transcript = text_override.strip()
                st.markdown("""
                <div class="step-card">
                  <div class="step-label">STEP 1 · TRANSCRIPTION</div>
                  <div class="step-content">Using typed input directly.</div>
                </div>""", unsafe_allow_html=True)
            else:
                with st.spinner("🔊 Transcribing audio…"):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(audio_bytes)
                            tmp_path = tmp.name
                        transcript = transcribe_audio(tmp_path, provider=stt_provider)
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"STT Error: {e}")
                        st.stop()

            st.markdown(f"""
            <div class="step-card">
              <div class="step-label">STEP 1 · TRANSCRIPTION</div>
              <div class="transcript-text">"{transcript}"</div>
            </div>""", unsafe_allow_html=True)

            # Step 2: Intent classification
            with st.spinner("🧠 Classifying intent…"):
                try:
                    intent_data = classify_intent(
                        transcript,
                        provider=llm_provider,
                        context=st.session_state.memory.get_context()
                    )
                except Exception as e:
                    st.error(f"LLM Error: {e}")
                    st.stop()

            intents_html = " ".join([
                f'<span class="intent-badge intent-{i.lower().replace(" ","_")}">{i}</span>'
                for i in intent_data.get("intents", [intent_data.get("intent","unknown")])
            ])
            st.markdown(f"""
            <div class="step-card">
              <div class="step-label">STEP 2 · INTENT</div>
              <div style="margin-bottom:8px">{intents_html}</div>
              <div class="step-content"><b>Parameters:</b> {json.dumps(intent_data.get("parameters", {}), indent=None)}</div>
            </div>""", unsafe_allow_html=True)

            # Step 3: Human-in-the-loop confirmation
            file_intents = {"create_file", "write_code", "create_folder"}
            detected = set(i.lower().replace(" ", "_") for i in intent_data.get("intents", [intent_data.get("intent", "")]))
            needs_confirm = human_in_loop and bool(detected & file_intents)

            if needs_confirm:
                st.session_state.pending_action = {
                    "transcript": transcript,
                    "intent_data": intent_data,
                    "llm_provider": llm_provider
                }
                st.markdown("""
                <div class="confirm-box">
                  ⚠ This action will create or modify files in the <code>output/</code> folder.
                  Confirm below to proceed.
                </div>""", unsafe_allow_html=True)
            else:
                st.session_state.pending_action = {
                    "transcript": transcript,
                    "intent_data": intent_data,
                    "llm_provider": llm_provider,
                    "auto_approve": True
                }

    # ── Confirmation buttons ───────────────────────────────────────────────
    if st.session_state.pending_action and not st.session_state.pending_action.get("auto_approve"):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm & Execute", use_container_width=True, type="primary"):
                st.session_state.pending_action["auto_approve"] = True
                st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.pending_action = None
                st.warning("Action cancelled.")

    # ── Execute approved action ────────────────────────────────────────────
    if st.session_state.pending_action and st.session_state.pending_action.get("auto_approve"):
        pending = st.session_state.pending_action
        st.session_state.pending_action = None

        with st.spinner("⚙ Executing…"):
            try:
                exec_result = execute_action(
                    pending["intent_data"],
                    pending["transcript"],
                    output_dir=OUTPUT_DIR,
                    llm_provider=pending["llm_provider"]
                )
            except Exception as e:
                exec_result = {"status": "error", "message": str(e), "output": ""}

        status_cls = "success" if exec_result.get("status") == "success" else "error"
        status_icon = "✅" if exec_result.get("status") == "success" else "❌"

        st.markdown(f"""
        <div class="step-card">
          <div class="step-label">STEP 3 · ACTION</div>
          <div class="step-content">{exec_result.get('action_taken','')}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card {status_cls}">
          <div class="step-label">STEP 4 · RESULT {status_icon}</div>
          <div class="result-output">{exec_result.get('output','') or exec_result.get('message','')}</div>
        </div>""", unsafe_allow_html=True)

        # Save to session history
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "transcript": pending["transcript"],
            "intents": pending["intent_data"].get("intents", [pending["intent_data"].get("intent","")]),
            "action": exec_result.get("action_taken", ""),
            "status": exec_result.get("status", ""),
            "output": exec_result.get("output", "")[:300]
        })
        st.session_state.memory.add(pending["transcript"], exec_result)
        st.rerun()

with right_col:
    st.markdown('<div class="section-title">📜 Session History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="empty-state">
          No actions yet.<br>Upload audio or type a command to get started.
        </div>""", unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.history):
            intents_html = " ".join([
                f'<span class="intent-badge-sm">{i}</span>'
                for i in item["intents"]
            ])
            status_dot = "🟢" if item["status"] == "success" else "🔴"
            st.markdown(f"""
            <div class="history-card">
              <div class="history-header">
                <span class="history-time">{item['time']}</span>
                {intents_html}
                <span class="history-status">{status_dot}</span>
              </div>
              <div class="history-transcript">"{item['transcript'][:120]}{"..." if len(item['transcript'])>120 else ""}"</div>
              <div class="history-action">{item['action']}</div>
              {f'<div class="history-output">{item["output"][:200]}</div>' if item["output"] else ''}
            </div>""", unsafe_allow_html=True)
