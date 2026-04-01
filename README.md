# VTT — Video to Text

A local web app for transcribing video files to text with speaker diarization powered by OpenAI Whisper and pyannote.audio.

## Features

- Drag & drop video upload (MP4, MOV, AVI, MKV, WEBM)
- Transcription via OpenAI Whisper (runs locally, no API cost)
- Speaker diarization — identifies who said what (SPEAKER_00, SPEAKER_01, etc.)
- Output formats: Plain Text, SRT subtitles, WebVTT
- Apple Silicon GPU acceleration (MPS) for faster diarization
- Runs entirely on your machine — no data leaves your computer

## Requirements

- Python 3.9+
- FFmpeg
- A [Hugging Face](https://huggingface.co) account with access to:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Setup

**1. Install FFmpeg**
```bash
brew install ffmpeg
```

**2. Install Python dependencies**
```bash
pip3 install openai-whisper flask pyannote.audio python-dotenv torch torchaudio
```

**3. Add your Hugging Face token**

Create a `.env` file in the project root:
```
HF_TOKEN=hf_your_token_here
```

Get a Read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Make sure you've accepted the terms on both pyannote model pages above.

**4. Run the app**
```bash
python3 app.py
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

## Usage

1. Drop a video file onto the upload zone
2. Choose output format (Plain Text, SRT, or WebVTT)
3. Set number of speakers (or leave on Auto-detect)
4. Click **Transcribe**
5. Download the transcript when complete

> **Tip:** If you know the number of speakers, set it explicitly — it produces cleaner results than auto-detect.

## Performance

- Whisper runs on CPU (sparse tensor operations not supported on MPS)
- Diarization runs on Apple Silicon GPU (MPS) automatically if available
- A 1-hour call takes roughly 15-20 min to transcribe + 2-3 min to diarize on M-series Mac

## Auto-start on Login

A LaunchAgent is included to run the app automatically on login:
```bash
launchctl load ~/Library/LaunchAgents/com.vtt.app.plist
```

Logs are written to `/tmp/vtt_app.log`.
