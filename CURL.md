# CURL Recipes

## Health Check

```bash
curl -sS http://localhost:49112/health
```

## Non-Streaming WAV (Playability-First)

```bash
curl "http://localhost:49112/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "You can reach the hotel front desk at (212) 555-0142 ext. 402 anytime.",
    "voice": "M1",
    "response_format": "wav",
    "speed": 2,
    "stream": false
  }' \
  --output speech.wav
```

## Streaming MP3 (Format-Correct, Buffered Stream)

```bash
curl "http://localhost:49112/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Streaming MP3 test.",
    "voice": "M1",
    "response_format": "mp3",
    "stream": true
  }' \
  --output speech.mp3
```

## Streaming WAV (Low Latency, Header Has Unknown Length)

```bash
curl "http://localhost:49112/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Streaming WAV test.",
    "voice": "M1",
    "response_format": "wav",
    "stream": true
  }' \
  --output speech_stream.wav
```

Note: Streaming WAV uses an unknown-length header. If your player rejects it, rewrite the header:

```bash
python3 - <<'PY'
import struct
p = "speech_stream.wav"
data = bytearray(open(p, "rb").read())
if data[:4] != b"RIFF":
    raise SystemExit("Not a WAV file")
data_size = len(data) - 44
riff_size = data_size + 36
data[4:8] = struct.pack("<I", riff_size)
data[40:44] = struct.pack("<I", data_size)
open("speech_stream_fixed.wav", "wb").write(data)
PY
```

## Measure Latency

```bash
curl -sS -w '\nTOTAL_S=%{time_total}\nTTFB_S=%{time_starttransfer}\n' \
  -o speech.wav \
  -X POST http://localhost:49112/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Latency test.","voice":"M1","response_format":"wav","speed":1.0,"stream":false}'
```

## Validate Output File

```bash
python3 tools/validate_audio.py speech.wav
```
