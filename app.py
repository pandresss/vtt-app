#!/usr/bin/env python3
"""
VTT - Video to Text
A local web app for transcribing MP4 videos using OpenAI Whisper
with speaker diarization powered by pyannote.audio.
"""

import os
import sys
import uuid

# Ensure Homebrew binaries (ffmpeg, etc.) are on PATH regardless of how the app is launched
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
import threading
import whisper
from flask import Flask, request, jsonify, send_file, render_template_string

# ─── Fix pyannote.audio 3.4 / huggingface_hub 1.8+ incompatibility ───────────
# pyannote uses 'use_auth_token' internally; huggingface_hub 1.x removed it.
# Log in via hub, patch the top-level function, then also patch pyannote's
# own module-level import of hf_hub_download (which was bound at import time).
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
import huggingface_hub
huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)

_orig_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    kwargs.pop("use_auth_token", None)
    return _orig_hf_hub_download(*args, **kwargs)

huggingface_hub.hf_hub_download = _patched_hf_hub_download
import huggingface_hub.file_download
huggingface_hub.file_download.hf_hub_download = _patched_hf_hub_download

# PyTorch 2.6 changed torch.load default to weights_only=True, breaking pyannote
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # force False — lightning_fabric passes True explicitly
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# Also patch lightning_fabric's local reference
try:
    import lightning_fabric.utilities.cloud_io as _lf_io
    _lf_io.torch.load = _patched_torch_load
except Exception:
    pass

# Import pyannote now, then patch its local binding too
from pyannote.audio import Pipeline
import pyannote.audio.core.pipeline as _pyannote_pipeline_mod
_pyannote_pipeline_mod.hf_hub_download = _patched_hf_hub_download
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Config
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
OUTPUT_FOLDER = os.path.expanduser("~/VTT Data/transcripts")
DB_PATH = os.path.expanduser("~/VTT Data/vtt.db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─── Database setup ───────────────────────────────────────────────────────────
import sqlite3, re, json as _json
from datetime import datetime

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                filename    TEXT,
                duration_s  REAL,
                company     TEXT,
                role        TEXT,
                interviewer TEXT,
                round       TEXT,
                topics      TEXT,
                transcript  TEXT,
                transcript_path TEXT
            )
        """)
        conn.commit()
    print("Database ready at", DB_PATH, flush=True)

init_db()

def extract_metadata(text):
    """Extract company, role, interviewer, round from transcript using regex."""
    meta = {"company": None, "role": None, "interviewer": None, "round": None, "topics": []}

    # ── Interviewer name ──────────────────────────────────────────────────────
    # "I'm Sarah", "I am Matt", "my name is Jillian", "this is Rachel"
    for pat in [
        r"(?:my name is|I(?:'m| am)|this is)\s+([A-Z][a-z]+)",
        r"^SPEAKER_\w+:\s+(?:Hi|Hello)[^.]*?(?:I(?:'m| am)|my name is)\s+([A-Z][a-z]+)",
    ]:
        m = re.search(pat, text[:4000], re.IGNORECASE | re.MULTILINE)
        if m:
            name = m.group(1).strip()
            # Filter out common false positives
            if name.lower() not in ("paul", "sure", "great", "good", "here", "just", "looking"):
                meta["interviewer"] = name
                break

    # ── Company name ──────────────────────────────────────────────────────────
    for pat in [
        r"(?:here at|at|with|from|joining)\s+([A-Z][A-Za-z0-9\s&]{1,30}?)\s+(?:today|as|and|,|\.|to)",
        r"welcome to\s+([A-Z][A-Za-z0-9\s&]{1,30}?)[\s,\.]",
        r"team (?:here )?at\s+([A-Z][A-Za-z0-9\s&]{1,30}?)[\s,\.]",
        r"(?:company|organization) (?:is |called )?([A-Z][A-Za-z0-9\s&]{1,30}?)[\s,\.]",
        r"([A-Z][A-Za-z0-9]{2,})\s+(?:is hiring|is looking|team is)",
    ]:
        m = re.search(pat, text[:5000], re.MULTILINE)
        if m:
            company = m.group(1).strip()
            # Filter noise
            if company.lower() not in ("the", "our", "this", "your", "a", "an", "we", "i", "you"):
                meta["company"] = company
                break

    # ── Role ─────────────────────────────────────────────────────────────────
    for pat in [
        r"(?:interviewing for|applying for|chat about|discussing)\s+(?:the\s+)?([A-Za-z\s]{3,50}?)\s+(?:role|position|opening)",
        r"(?:the\s+)?([A-Za-z\s]{3,50}?)\s+(?:role|position)\s+(?:here|at|with)",
        r"(?:hiring|looking)\s+(?:a|an|for\s+a|for\s+an)\s+([A-Za-z\s]{3,50?}?)[\.,]",
    ]:
        m = re.search(pat, text[:5000], re.IGNORECASE)
        if m:
            role = m.group(1).strip()
            if len(role) > 3 and role.lower() not in ("the", "this", "our", "a", "an"):
                meta["role"] = role
                break

    # ── Round ─────────────────────────────────────────────────────────────────
    # Look across the whole transcript since round info can appear anywhere
    for pat in [
        r"(first|second|third|fourth|final)\s+(?:round|interview|call|step)",
        r"(?:round|interview)\s+(one|two|three|four|1|2|3|4)",
        r"(pre[\s\-]?screen|phone screen|initial call|intro call|onsite|final round)",
        r"(?:next step|moving forward|advance)[^.]{0,60}?(second|third|final|next round)",
        r"(hiring manager|panel|technical|take-?home)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            meta["round"] = m.group(1).strip().title()
            break

    # ── Topics ────────────────────────────────────────────────────────────────
    topic_keywords = [
        "AI", "machine learning", "automation", "workflow", "integration", "API",
        "CSM", "customer success", "onboarding", "renewal", "churn", "upsell",
        "EBR", "QBR", "technical", "python", "agent", "data", "pipeline",
        "implementation", "adoption", "stakeholder", "enterprise", "SaaS",
    ]
    meta["topics"] = [k for k in topic_keywords if k.lower() in text.lower()]

    print(f"[metadata] Extracted: {meta}", flush=True)
    return meta

def save_to_db(job_id, filename, transcript_text, transcript_path, duration_s=None):
    """Save completed transcription to DB, return record id and extracted metadata."""
    meta = extract_metadata(transcript_text)
    now = datetime.now().isoformat()
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO recordings
                (created_at, filename, duration_s, company, role, interviewer, round, topics, transcript, transcript_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, filename, duration_s, meta["company"], meta["role"],
              meta["interviewer"], meta["round"], _json.dumps(meta["topics"]),
              transcript_text, transcript_path))
        conn.commit()
        record_id = cur.lastrowid
    return record_id, meta
# ─────────────────────────────────────────────────────────────────────────────

# Track transcription jobs
jobs = {}

# Force flush prints immediately
# Use Apple Silicon GPU (MPS) if available, otherwise CPU
import torch as _torch
if _torch.backends.mps.is_available():
    DEVICE = _torch.device("mps")
    print("Apple Silicon GPU (MPS) detected — using for faster processing.", flush=True)
else:
    DEVICE = _torch.device("cpu")
    print("No GPU detected — using CPU.", flush=True)

print("Loading Whisper model (base)... this may take a moment on first run.", flush=True)
whisper_model = whisper.load_model("base")  # Whisper stays on CPU (sparse tensors unsupported on MPS)
print("Whisper model loaded!", flush=True)

diarization_pipeline = None
try:
    print("Loading speaker diarization model... (first run downloads ~300MB)", flush=True)
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization_pipeline.to(DEVICE)  # pyannote supports MPS for ~10x speedup
    print(f"Speaker diarization model loaded! (device: {DEVICE})", flush=True)
except Exception as e:
    import traceback
    print("WARNING: Could not load diarization model:", flush=True)
    traceback.print_exc()
    print("Speaker identification will be disabled. Transcription still works.", flush=True)


def assign_speakers_to_segments(diarization_result, whisper_segments):
    """Match Whisper transcript segments to speaker labels from diarization."""
    labeled_segments = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_mid = (seg_start + seg_end) / 2

        # Find the speaker active at the midpoint of this segment
        best_speaker = "Unknown"
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if turn.start <= seg_mid <= turn.end:
                best_speaker = speaker
                break

        labeled_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"].strip(),
            "speaker": best_speaker,
        })
    return labeled_segments


def transcribe_task(job_id, filepath, output_format, num_speakers):
    """Run transcription in a background thread."""
    try:
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["step"] = "Running speech-to-text..."

        # Step 1: Whisper transcription
        result = whisper_model.transcribe(filepath)
        segments = result["segments"]

        # Step 2: Speaker diarization (if available)
        use_diarization = diarization_pipeline is not None and num_speakers != 0
        labeled_segments = None

        print(f"[{job_id}] use_diarization={use_diarization}, num_speakers={num_speakers}, pipeline={diarization_pipeline is not None}", flush=True)
        if use_diarization:
            try:
                jobs[job_id]["step"] = "Identifying speakers..."
                print(f"[{job_id}] Starting diarization on {filepath}", flush=True)
                # Extract audio to wav first — pyannote handles wav more reliably than mov
                import subprocess, tempfile
                wav_path = filepath + ".wav"
                ffmpeg_bin = next(
                    (p for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]
                     if __import__('shutil').which(p) or __import__('os').path.isfile(p)),
                    "ffmpeg"
                )
                subprocess.run(
                    [ffmpeg_bin, "-y", "-i", filepath, "-ar", "16000", "-ac", "1", wav_path],
                    check=True, capture_output=True
                )
                print(f"[{job_id}] Audio extracted to {wav_path}, running pipeline...", flush=True)
                diarize_kwargs = {}
                if num_speakers and num_speakers > 0:
                    diarize_kwargs["num_speakers"] = num_speakers
                diarization_result = diarization_pipeline(wav_path, **diarize_kwargs)
                print(f"[{job_id}] Diarization complete, assigning speakers...", flush=True)
                labeled_segments = assign_speakers_to_segments(diarization_result, segments)
                print(f"[{job_id}] Got {len(labeled_segments)} labeled segments", flush=True)
                os.remove(wav_path)
            except Exception as e:
                import traceback
                print(f"[{job_id}] Diarization failed: {e}", flush=True)
                traceback.print_exc()
                labeled_segments = None

        # Step 3: Build output
        if labeled_segments:
            if output_format == "txt":
                output_text = generate_speaker_txt(labeled_segments)
                ext = "txt"
            elif output_format == "srt":
                output_text = generate_speaker_srt(labeled_segments)
                ext = "srt"
            elif output_format == "vtt":
                output_text = generate_speaker_vtt(labeled_segments)
                ext = "vtt"
            else:
                output_text = generate_speaker_txt(labeled_segments)
                ext = "txt"
            preview = generate_speaker_txt(labeled_segments)[:500]
        else:
            if output_format == "txt":
                output_text = result["text"].strip()
                ext = "txt"
            elif output_format == "srt":
                output_text = generate_srt(segments)
                ext = "srt"
            elif output_format == "vtt":
                output_text = generate_vtt(segments)
                ext = "vtt"
            else:
                output_text = result["text"].strip()
                ext = "txt"
            preview = result["text"].strip()[:500]

        # Save output
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        # Remove the job_id prefix from base_name if present
        if base_name.startswith(job_id + "_"):
            base_name = base_name[len(job_id) + 1:]
        output_filename = f"{base_name}.{ext}"
        output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_{output_filename}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        # Save to database and extract metadata
        jobs[job_id]["step"] = "Saving to database..."
        record_id, meta = save_to_db(
            job_id=job_id,
            filename=os.path.basename(filepath),
            transcript_text=output_text,
            transcript_path=output_path,
        )
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["output_path"] = output_path
        jobs[job_id]["output_filename"] = output_filename
        jobs[job_id]["text_preview"] = preview
        jobs[job_id]["record_id"] = record_id
        jobs[job_id]["meta"] = meta
        # Flag which fields need user input
        jobs[job_id]["needs_input"] = [
            k for k in ["company", "role", "interviewer", "round"]
            if not meta.get(k)
        ]

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


# ─── Speaker-aware formatters ────────────────────────────────────────────────

def generate_speaker_txt(segments):
    """Generate plain text with speaker labels, grouping consecutive same-speaker lines."""
    lines = []
    current_speaker = None
    current_text = []

    for seg in segments:
        if seg["speaker"] != current_speaker:
            if current_speaker is not None:
                lines.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = seg["speaker"]
            current_text = [seg["text"]]
        else:
            current_text.append(seg["text"])

    if current_speaker is not None:
        lines.append(f"{current_speaker}: {' '.join(current_text)}")

    return "\n\n".join(lines)


def generate_speaker_srt(segments):
    """Generate SRT with speaker labels."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n[{seg['speaker']}] {seg['text']}\n")
    return "\n".join(lines)


def generate_speaker_vtt(segments):
    """Generate WebVTT with speaker labels."""
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = format_timestamp_vtt(seg["start"])
        end = format_timestamp_vtt(seg["end"])
        lines.append(f"{start} --> {end}\n<v {seg['speaker']}>{seg['text']}\n")
    return "\n".join(lines)


# ─── Plain formatters (no speakers) ─────────────────────────────────────────

def generate_srt(segments):
    """Generate SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def generate_vtt(segments):
    """Generate WebVTT subtitle format."""
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = format_timestamp_vtt(seg["start"])
        end = format_timestamp_vtt(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def format_timestamp_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, diarization_available=diarization_pipeline is not None)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    output_format = request.form.get("format", "txt")
    num_speakers = int(request.form.get("speakers", "0"))
    job_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")
    file.save(filepath)

    jobs[job_id] = {
        "status": "uploading",
        "filename": file.filename,
        "format": output_format,
        "step": "Uploading...",
    }

    # Start transcription in background
    thread = threading.Thread(
        target=transcribe_task,
        args=(job_id, filepath, output_format, num_speakers),
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in jobs or jobs[job_id]["status"] != "complete":
        return jsonify({"error": "File not ready"}), 404
    return send_file(
        jobs[job_id]["output_path"],
        as_attachment=True,
        download_name=jobs[job_id]["output_filename"],
    )


@app.route("/save-meta/<job_id>", methods=["POST"])
def save_meta(job_id):
    """Save user-provided metadata corrections to the DB."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    data = request.json
    record_id = jobs[job_id].get("record_id")
    if not record_id:
        return jsonify({"error": "No DB record for this job"}), 400
    fields = {k: v for k, v in data.items() if k in ["company", "role", "interviewer", "round"]}
    if fields:
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        with get_db() as conn:
            conn.execute(f"UPDATE recordings SET {set_clause} WHERE id = ?",
                         list(fields.values()) + [record_id])
            conn.commit()
    return jsonify({"ok": True})


# ─── HTML Template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VTT — Video to Text</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 560px;
            padding: 2rem;
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: #fff;
        }

        .subtitle {
            color: #888;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }

        /* Drop zone */
        .dropzone {
            border: 2px dashed #333;
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: #161616;
            margin-bottom: 1.5rem;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: #4f8ff7;
            background: #1a1f2e;
        }

        .dropzone-icon {
            font-size: 2.5rem;
            margin-bottom: 0.75rem;
        }

        .dropzone-text {
            font-size: 1rem;
            color: #aaa;
        }

        .dropzone-text strong {
            color: #4f8ff7;
        }

        .dropzone-hint {
            font-size: 0.8rem;
            color: #555;
            margin-top: 0.5rem;
        }

        .file-selected {
            display: none;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            background: #1a1f2e;
            border: 1px solid #2a3a5c;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }

        .file-selected.show { display: flex; }

        .file-selected .file-icon { font-size: 1.3rem; }

        .file-selected .file-name {
            flex: 1;
            font-size: 0.9rem;
            color: #ccc;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .file-selected .file-remove {
            background: none;
            border: none;
            color: #666;
            cursor: pointer;
            font-size: 1.2rem;
            padding: 0.25rem;
        }

        .file-selected .file-remove:hover { color: #e55; }

        /* Options row */
        .options {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .options label {
            font-size: 0.85rem;
            color: #888;
        }

        .options select, .options input[type="number"] {
            background: #1a1a1a;
            color: #e0e0e0;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            font-size: 0.9rem;
            cursor: pointer;
        }

        .options input[type="number"] {
            width: 60px;
        }

        .speaker-options {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }

        .speaker-options label {
            font-size: 0.85rem;
            color: #888;
        }

        .speaker-options select {
            background: #1a1a1a;
            color: #e0e0e0;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            font-size: 0.9rem;
        }

        .badge {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-on {
            background: #1a3a2a;
            color: #4ade80;
            border: 1px solid #2a5a3a;
        }

        .badge-off {
            background: #2d1515;
            color: #f87171;
            border: 1px solid #5c2020;
        }

        /* Button */
        .btn {
            width: 100%;
            padding: 0.85rem;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            background: #4f8ff7;
            color: #fff;
            transition: background 0.2s;
        }

        .btn:hover { background: #3a7be0; }
        .btn:disabled { background: #333; color: #666; cursor: not-allowed; }

        /* Progress */
        .progress-area {
            display: none;
            margin-top: 1.5rem;
            text-align: center;
        }

        .progress-area.show { display: block; }

        .spinner {
            width: 40px; height: 40px;
            border: 3px solid #222;
            border-top: 3px solid #4f8ff7;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .progress-text {
            font-size: 0.9rem;
            color: #888;
        }

        /* Result */
        .result-area {
            display: none;
            margin-top: 1.5rem;
        }

        .result-area.show { display: block; }

        .result-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .result-header .checkmark {
            font-size: 1.3rem;
        }

        .result-header span {
            font-size: 1rem;
            color: #4ade80;
            font-weight: 600;
        }

        .result-preview {
            background: #161616;
            border: 1px solid #282828;
            border-radius: 8px;
            padding: 1rem;
            max-height: 250px;
            overflow-y: auto;
            font-size: 0.85rem;
            color: #bbb;
            line-height: 1.6;
            margin-bottom: 1rem;
            white-space: pre-wrap;
        }

        .btn-download {
            background: #22c55e;
        }

        .btn-download:hover { background: #16a34a; }

        .btn-new {
            background: transparent;
            border: 1px solid #333;
            color: #aaa;
            margin-top: 0.75rem;
        }

        .btn-new:hover { background: #1a1a1a; color: #fff; }

        /* Error */
        .error-msg {
            display: none;
            background: #2d1515;
            border: 1px solid #5c2020;
            border-radius: 8px;
            padding: 1rem;
            color: #f87171;
            font-size: 0.9rem;
            margin-top: 1rem;
        }

        .error-msg.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>VTT</h1>
        <p class="subtitle">Video to Text — drop a video and get a transcript</p>

        <div class="dropzone" id="dropzone">
            <div class="dropzone-icon">🎬</div>
            <div class="dropzone-text">Drag & drop your video here, or <strong>browse</strong></div>
            <div class="dropzone-hint">Supports MP4, MOV, AVI, MKV, WEBM</div>
        </div>
        <input type="file" id="fileInput" accept="video/*,.mp4,.mov,.avi,.mkv,.webm" hidden>

        <div class="file-selected" id="fileSelected">
            <span class="file-icon">🎥</span>
            <span class="file-name" id="fileName"></span>
            <button class="file-remove" id="fileRemove">&times;</button>
        </div>

        <div class="options">
            <label for="format">Output format:</label>
            <select id="format">
                <option value="txt">Plain Text (.txt)</option>
                <option value="srt">Subtitles (.srt)</option>
                <option value="vtt">WebVTT (.vtt)</option>
            </select>
        </div>

        <div class="speaker-options">
            <label for="speakers">Speakers:</label>
            <select id="speakers" {% if not diarization_available %}disabled{% endif %}>
                <option value="0" {% if not diarization_available %}selected{% endif %}>
                    {% if diarization_available %}Off — no speaker labels{% else %}Unavailable (no HF token){% endif %}
                </option>
                {% if diarization_available %}
                <option value="-1" selected>Auto-detect</option>
                <option value="2">2 speakers</option>
                <option value="3">3 speakers</option>
                <option value="4">4 speakers</option>
                <option value="5">5 speakers</option>
                {% endif %}
            </select>
            {% if diarization_available %}
            <span class="badge badge-on">ENABLED</span>
            {% else %}
            <span class="badge badge-off">DISABLED</span>
            {% endif %}
        </div>

        <button class="btn" id="transcribeBtn" disabled>Transcribe</button>

        <div class="progress-area" id="progressArea">
            <div class="spinner"></div>
            <div class="progress-text" id="progressText">Uploading video...</div>
        </div>

        <div class="result-area" id="resultArea">
            <div class="result-header">
                <span class="checkmark">✅</span>
                <span>Transcription complete</span>
            </div>
            <div class="result-preview" id="resultPreview"></div>

            <!-- Metadata form — shown after transcription -->
            <div id="metaForm" style="display:none; background:#161616; border:1px solid #2a2a2a; border-radius:10px; padding:1.25rem; margin-bottom:1rem;">
                <p style="font-size:0.85rem; color:#888; margin-bottom:1rem;">📁 Saved to database — fill in any missing details:</p>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;">
                    <div>
                        <label style="font-size:0.75rem; color:#666; display:block; margin-bottom:0.25rem;">Company</label>
                        <input id="metaCompany" type="text" placeholder="e.g. Atomi" style="width:100%; background:#1a1a1a; color:#e0e0e0; border:1px solid #333; border-radius:6px; padding:0.5rem; font-size:0.85rem;">
                    </div>
                    <div>
                        <label style="font-size:0.75rem; color:#666; display:block; margin-bottom:0.25rem;">Role</label>
                        <input id="metaRole" type="text" placeholder="e.g. Technical CSM" style="width:100%; background:#1a1a1a; color:#e0e0e0; border:1px solid #333; border-radius:6px; padding:0.5rem; font-size:0.85rem;">
                    </div>
                    <div>
                        <label style="font-size:0.75rem; color:#666; display:block; margin-bottom:0.25rem;">Interviewer</label>
                        <input id="metaInterviewer" type="text" placeholder="e.g. Jillian" style="width:100%; background:#1a1a1a; color:#e0e0e0; border:1px solid #333; border-radius:6px; padding:0.5rem; font-size:0.85rem;">
                    </div>
                    <div>
                        <label style="font-size:0.75rem; color:#666; display:block; margin-bottom:0.25rem;">Round</label>
                        <input id="metaRound" type="text" placeholder="e.g. first, second..." style="width:100%; background:#1a1a1a; color:#e0e0e0; border:1px solid #333; border-radius:6px; padding:0.5rem; font-size:0.85rem;">
                    </div>
                </div>
                <button id="saveMetaBtn" class="btn" style="margin-top:0.75rem; background:#4f8ff7; padding:0.6rem;">Save details</button>
                <span id="metaSaved" style="display:none; color:#4ade80; font-size:0.8rem; margin-left:0.75rem;">Saved ✓</span>
            </div>

            <button class="btn btn-download" id="downloadBtn">Download transcript</button>
            <button class="btn btn-new" id="newBtn">Transcribe another video</button>
        </div>

        <div class="error-msg" id="errorMsg"></div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const fileSelected = document.getElementById('fileSelected');
        const fileName = document.getElementById('fileName');
        const fileRemove = document.getElementById('fileRemove');
        const transcribeBtn = document.getElementById('transcribeBtn');
        const progressArea = document.getElementById('progressArea');
        const progressText = document.getElementById('progressText');
        const resultArea = document.getElementById('resultArea');
        const resultPreview = document.getElementById('resultPreview');
        const downloadBtn = document.getElementById('downloadBtn');
        const newBtn = document.getElementById('newBtn');
        const errorMsg = document.getElementById('errorMsg');
        const formatSelect = document.getElementById('format');
        const speakersSelect = document.getElementById('speakers');

        let selectedFile = null;
        let currentJobId = null;

        // Drag & drop
        dropzone.addEventListener('click', () => fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                selectFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                selectFile(fileInput.files[0]);
            }
        });

        function selectFile(file) {
            selectedFile = file;
            fileName.textContent = file.name;
            fileSelected.classList.add('show');
            dropzone.style.display = 'none';
            transcribeBtn.disabled = false;
            errorMsg.classList.remove('show');
        }

        fileRemove.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            fileSelected.classList.remove('show');
            dropzone.style.display = 'block';
            transcribeBtn.disabled = true;
        });

        // Transcribe
        transcribeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            transcribeBtn.disabled = true;
            progressArea.classList.add('show');
            resultArea.classList.remove('show');
            errorMsg.classList.remove('show');
            progressText.textContent = 'Uploading video...';

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('format', formatSelect.value);
            formData.append('speakers', speakersSelect.value);

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.error) throw new Error(data.error);

                currentJobId = data.job_id;
                progressText.textContent = 'Transcribing audio...';
                pollStatus(data.job_id);
            } catch (err) {
                showError(err.message);
            }
        });

        function pollStatus(jobId) {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch(`/status/${jobId}`);
                    const data = await res.json();

                    if (data.step) {
                        progressText.textContent = data.step;
                    }

                    if (data.status === 'complete') {
                        clearInterval(interval);
                        showResult(data);
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        showError(data.error || 'Transcription failed');
                    }
                } catch (err) {
                    clearInterval(interval);
                    showError('Lost connection to server');
                }
            }, 1500);
        }

        function showResult(data) {
            progressArea.classList.remove('show');
            resultArea.classList.add('show');
            resultPreview.textContent = data.text_preview || '';
            downloadBtn.onclick = () => {
                window.location.href = `/download/${currentJobId}`;
            };

            // Show metadata form, pre-fill anything auto-extracted
            const meta = data.meta || {};
            document.getElementById('metaCompany').value = meta.company || '';
            document.getElementById('metaRole').value = meta.role || '';
            document.getElementById('metaInterviewer').value = meta.interviewer || '';
            document.getElementById('metaRound').value = meta.round || '';
            document.getElementById('metaForm').style.display = 'block';
            document.getElementById('metaSaved').style.display = 'none';

            document.getElementById('saveMetaBtn').onclick = async () => {
                const payload = {
                    company: document.getElementById('metaCompany').value,
                    role: document.getElementById('metaRole').value,
                    interviewer: document.getElementById('metaInterviewer').value,
                    round: document.getElementById('metaRound').value,
                };
                await fetch(`/save-meta/${currentJobId}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                document.getElementById('metaSaved').style.display = 'inline';
            };
        }

        function showError(msg) {
            progressArea.classList.remove('show');
            errorMsg.textContent = msg;
            errorMsg.classList.add('show');
            transcribeBtn.disabled = false;
        }

        newBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            fileSelected.classList.remove('show');
            dropzone.style.display = 'block';
            transcribeBtn.disabled = true;
            resultArea.classList.remove('show');
            errorMsg.classList.remove('show');
            currentJobId = null;
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n✦ VTT — Video to Text")
    print("  Open http://localhost:8080 in your browser\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
