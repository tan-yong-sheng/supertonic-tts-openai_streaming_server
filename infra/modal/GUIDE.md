# Modal Deployment Guide (Scale-to-Zero)

This guide mirrors the Pocket-TTS Modal setup and is optimized for
occasional use with scale-to-zero and memory snapshots.

## 1) What You Get

- Single Modal deployment serving the Flask app.
- Scale-to-zero by default (`min_containers=0`).
- Volume-cached model weights to avoid repeated downloads.
- Optional voices volume for custom voice packs.
- Memory snapshots to reduce cold-start latency after the first deploy.

## 2) Files

- `infra/modal/modal_config.py` — Modal entrypoint
- `infra/modal/README.md` — quick entry point
- `app/*`, `templates/*`, `static/*` — Flask app

## 3) One-Time Setup

```bash
# Install Modal CLI (if not installed)
# https://modal.com/docs/guide

# Create a volume for model caching
modal volume create supertonic-tts-models
```

Optional voices volume (only if you want to store custom voices in a volume):

```bash
modal volume create supertonic-tts-voices
```

## 4) Optional Environment Variables

You can either export variables in your shell **or** create
`infra/modal/.env.modal` (auto-loaded by `modal_config.py`).

Set these in your shell before running Modal commands:

```bash
# Or copy the example file and edit it:
cp infra/modal/.env.modal.example infra/modal/.env.modal

# Override Modal app name
export SUPERTONIC_MODAL_APP=supertonic-tts

# Where to cache downloaded model files
export MODEL_VOLUME_NAME=supertonic-tts-models

# Optional: mount a voices volume at /voices
export VOICES_VOLUME_NAME=supertonic-tts-voices

# Optional: attach Modal secrets
export AUTH_SECRET_NAME=supertonic-tts-auth
export HF_SECRET_NAME=supertonic-tts-hf

# Optional: force a specific model path or variant
export SUPERTONIC_MODEL_PATH=/models/supertonic2
export SUPERTONIC_MODEL_NAME=supertonic-2
export SUPERTONIC_AUTO_DOWNLOAD=true

# Optional: override voices directory in the container
export SUPERTONIC_VOICES_DIR=

# Optional: source path to sync voices into the voices volume
export SUPERTONIC_VOICES_SRC=/app/voices

# Optional: enable cold-start timing logs
export SUPERTONIC_COLDSTART_LOG=true

# Optional: enable per-request timing logs
export SUPERTONIC_REQUEST_TIMING_LOG=true
export SUPERTONIC_REQUEST_TIMING_LOG_JSON=true

# Optional: disable the web UI at /
export SUPERTONIC_UI_ENABLED=false

# Optional: force HuggingFace offline mode (use after model is cached in volume)
export SUPERTONIC_HF_OFFLINE=true

# Optional: quality/speed defaults
export SUPERTONIC_DIFFUSION_STEPS=5
export SUPERTONIC_DEFAULT_SPEED=1.0
export SUPERTONIC_DEFAULT_LANG=en

# Optional: ONNX Runtime threading
export SUPERTONIC_INTRA_OP_THREADS=2
export SUPERTONIC_INTER_OP_THREADS=1

# Optional: voice style extraction for audio prompts
export SUPERTONIC_VOICE_STYLE_CACHE_DIR=/models/voice_styles
# Example: python /app/tools/extract_style.py --input {input} --output {output}
export SUPERTONIC_VOICE_EXTRACTOR_CMD=
export SUPERTONIC_VOICE_EXTRACTOR_TIMEOUT=120
```

### Auth tokens (optional, recommended via Modal Secret)

Use a Modal secret so tokens are injected at runtime and not baked into images.
Avoid putting `AUTHENTICATION_ALLOWED_TOKENS` directly in `infra/modal/.env.modal`.

```bash
modal secret create supertonic-tts-auth \
  AUTHENTICATION_ALLOWED_TOKENS=token1,token2
```

## 5) Download the Model to the Volume (One-Time)

```bash
modal run infra/modal/modal_config.py::download_models
```

This will initialize the app and cache model assets into the Modal volume.

## 6) Sync Voices to the Volume (Optional, One-Time)

If you set `VOICES_VOLUME_NAME`, you can pre-load custom voices into the
voices volume. By default, this copies from `/app/voices` in the image.

```bash
modal run infra/modal/modal_config.py::sync_voices
```

## 7) Deploy

```bash
modal deploy infra/modal/modal_config.py
```

## 8) Quick Tests

```bash
# Replace with your deployed URL
APP_URL="https://<your-app>.modal.run"

curl -sS "$APP_URL/health"

curl -sS "$APP_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello","voice":"M1","response_format":"wav"}' \
  -o /tmp/tts.wav
```

## 9) Notes

- Scale-to-zero is enabled (`min_containers=0`), so cold starts are expected.
- Memory snapshots reduce cold-start time after the first deploy.
- The voices directory defaults to `/app/voices` (bundled with the image).
- If `VOICES_VOLUME_NAME` is set, `/voices` is mounted and used instead (the image will skip bundling `voices/`).
