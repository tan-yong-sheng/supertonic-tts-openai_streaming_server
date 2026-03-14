# Supertonic TTS OpenAI-Compatible Server (Scaffold)

OpenAI-compatible Text-to-Speech API server targeting the Supertonic TTS backend.
This repo is a Phase 1 scaffold adapted from the Pocket-TTS server structure.

## Status

- Phase 1 scaffold complete: Flask app structure, routes, Modal wiring, Dockerfile.
- Phase 2 will implement Supertonic ONNX inference and voice state extraction.

## Quick Start (Local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

Server will start on `http://localhost:49112`.

## API

- `GET /health`
- `GET /v1/models`
- `GET /v1/voices`
- `POST /v1/audio/speech`

Example:

```bash
curl -sS http://localhost:49112/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"supertonic-2","input":"Hello","voice":"M1","response_format":"wav","lang":"en"}' \
  -o /tmp/tts.wav
```

## Modal Deployment

See [infra/modal/GUIDE.md](/workspaces/supertonic-tts-openai_streaming_server/infra/modal/GUIDE.md).

## Notes

- Supertonic inference is wired via the `supertonic` Python package.
- Default sample rate is 44.1kHz (Supertonic output).
- Custom voices support JSON styles or audio prompts when a voice extractor is configured.
- Streaming is opt-in via `stream_format: "audio"`; server-side chunking is used only when `chunk_chars` is explicitly provided.
- Model name is case-sensitive and currently only `supertonic-2` is accepted.

## Custom Voices (Audio Prompts)

To use audio prompts (WAV/MP3/FLAC/OGG), configure a voice style extractor that outputs
a Supertonic JSON style file. Set:

```bash
export SUPERTONIC_VOICE_STYLE_CACHE_DIR=/models/voice_styles
export SUPERTONIC_VOICE_EXTRACTOR_CMD="python /app/tools/extract_style.py --input {input} --output {output}"
export SUPERTONIC_VOICE_EXTRACTOR_TIMEOUT=120
```

If no extractor is configured, audio prompts will return an error.
