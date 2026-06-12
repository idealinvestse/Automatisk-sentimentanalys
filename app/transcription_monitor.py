"""Non-blocking Transcription Monitor View for Fas 5.1.

Modern thread-safe UI with queue, progress, settings, start/pause/stop and log.
Import in dashboard.py and call render_transcription_view() when view == 'transcription'.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


def init_transcription_state() -> None:
    defaults = {
        "trans_pending_folder": "inputs/pending",
        "trans_output_dir": "outputs/transcripts",
        "trans_status": {"is_running": False, "current_file": None, "progress": 0.0, "processed": 0, "total": 0, "start_time": None},
        "trans_logs": [],
        "trans_settings": {"backend": "faster", "model": "KBLab/kb-whisper-large", "device": "auto", "language": "sv", "hotwords": "", "preprocess": True, "diarize": False, "num_speakers": None},
        "trans_pause": False,
        "trans_stop": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_trans_log(level: str, msg: str, file: str | None = None) -> None:
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), "level": level, "msg": msg, "file": file}
    logs = st.session_state.setdefault("trans_logs", [])
    logs.append(entry)
    if len(logs) > 200:
        st.session_state["trans_logs"] = logs[-150:]


def scan_pending(folder: str) -> list[Path]:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts], key=lambda x: x.stat().st_mtime, reverse=True)


def transcription_worker(files: list[Path], settings: dict, status: dict) -> None:
    add_trans_log("INFO", f"Startar batch på {len(files)} filer")
    status["is_running"] = True
    status["total"] = len(files)
    status["processed"] = 0
    status["start_time"] = datetime.now().isoformat()
    try:
        from src.transcription.factory import get_transcriber
        transcriber = get_transcriber(backend=settings.get("backend", "faster"), model_name=settings.get("model", "KBLab/kb-whisper-large"), device=settings.get("device", "auto"))
        real = True
    except Exception as e:
        add_trans_log("ERROR", f"Kunde inte ladda transcriber: {e}. Simulation mode.")
        real = False
    for i, fpath in enumerate(files):
        if st.session_state.get("trans_stop"):
            break
        while st.session_state.get("trans_pause"):
            time.sleep(0.3)
            if st.session_state.get("trans_stop"):
                break
        status["current_file"] = fpath.name
        status["progress"] = round(i / max(1, len(files)), 2)
        add_trans_log("INFO", f"Bearbetar {fpath.name}...", file=fpath.name)
        try:
            if real:
                tr_obj = transcriber.transcribe(audio_path=str(fpath), language=settings.get("language", "sv"), preprocess=settings.get("preprocess", True), diarize=settings.get("diarize", False))
                tr = tr_obj.to_dict() if hasattr(tr_obj, "to_dict") else {}
                add_trans_log("INFO", f"Klart: {fpath.name} – {len(tr.get('segments', []))} segment", file=fpath.name)
            else:
                time.sleep(1.2)
                add_trans_log("INFO", f"[SIM] Klart: {fpath.name}", file=fpath.name)
        except Exception as e:
            add_trans_log("ERROR", f"Fel på {fpath.name}: {e}", file=fpath.name)
        status["processed"] = i + 1
        status["progress"] = round((i + 1) / max(1, len(files)), 2)
    status["is_running"] = False
    status["current_file"] = None
    add_trans_log("INFO", "Batch slutförd")


def render_transcription_view() -> None:
    """Main render function for the transcription view."""
    st.header("🎙️ Transkriberingskö & Monitor (Fas 5.1)")
    init_transcription_state()
    status = st.session_state["trans_status"]
    pending = scan_pending(st.session_state["trans_pending_folder"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Väntande", len(pending))
    c2.metric("Status", "Pågår" if status["is_running"] else "Idle")
    c3.metric("Progress", f"{status['processed']}/{status['total']}")
    c4.metric("%", f"{int(status.get('progress', 0)*100)}%")

    if status["is_running"]:
        with st.status("Transkriberar i bakgrunden (UI responsiv)", expanded=True) as s:
            st.write(f"**Fil:** {status.get('current_file', '–')}")
            st.progress(float(status.get("progress", 0)))
            s.update(label="Pågår", state="running")
    else:
        st.info("Tryck Start för att köra i bakgrundstråd.")

    btn1, btn2, btn3, btn4 = st.columns(4)
    with btn1:
        if st.button("▶️ Start", type="primary", disabled=status["is_running"], use_container_width=True):
            if pending:
                st.session_state["trans_stop"] = False
                st.session_state["trans_pause"] = False
                t = threading.Thread(target=transcription_worker, args=(pending[:], st.session_state["trans_settings"].copy(), st.session_state["trans_status"]), daemon=True)
                t.start()
                add_trans_log("INFO", f"Startade tråd för {len(pending)} filer")
                st.rerun()
            else:
                st.warning("Inga filer i mappen.")
    with btn2:
        if st.button("⏸️ Paus", disabled=not status["is_running"], use_container_width=True):
            st.session_state["trans_pause"] = True
            add_trans_log("WARNING", "Paus")
            st.rerun()
    with btn3:
        if st.button("▶️ Återuppta", disabled=not st.session_state.get("trans_pause"), use_container_width=True):
            st.session_state["trans_pause"] = False
            add_trans_log("INFO", "Återupptagen")
            st.rerun()
    with btn4:
        if st.button("⏹️ Stopp", disabled=not status["is_running"], use_container_width=True):
            st.session_state["trans_stop"] = True
            add_trans_log("WARNING", "Stopp")
            st.rerun()

    with st.expander("⚙️ Inställningar", expanded=False):
        s = st.session_state["trans_settings"]
        s["backend"] = st.selectbox("Backend", ["faster", "transformers", "whisperx"], index=0)
        s["model"] = st.text_input("Modell", value=s.get("model", ""))
        s["device"] = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        s["preprocess"] = st.checkbox("Preprocess", value=s.get("preprocess", True))
        s["diarize"] = st.checkbox("Diarize", value=s.get("diarize", False))
        s["hotwords"] = st.text_area("Hotwords", value=s.get("hotwords", ""), height=50)

    st.session_state["trans_pending_folder"] = st.text_input("Väntemapp", value=st.session_state["trans_pending_folder"])

    st.subheader("Väntande filer")
    for f in pending[:5]:
        st.write(f"• {f.name}")

    with st.expander("📜 Logg", expanded=True):
        for entry in reversed(st.session_state.get("trans_logs", [])[-40:]):
            icon = {"INFO": "🟢", "WARNING": "🟡", "ERROR": "🔴"}.get(entry["level"], "⚪")
            st.markdown(f"{icon} {entry['ts']} – {entry['msg']}")

    if st.button("Tillbaka till Översikt"):
        st.session_state["view"] = "overview"
        st.rerun()
