Source: https://platform.openai.com/docs/guides/text-to-speech?lang=curl
Source: https://platform.openai.com/docs/api-reference/audio/createSpeech

1. Non-streaming default (no stream_format).
curl -X POST http://localhost:49112/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world!",
    "voice": "alba",
    "response_format": "wav"
  }' \
  --output speech.wav
Expected: full file response; request completes after generation finishes.

2. Streaming raw audio via stream_format=audio.
curl -N -X POST http://localhost:49112/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world!",
    "voice": "alba",
    "response_format": "wav",
    "stream_format": "audio"
  }' \
  | ffplay -i -
Expected: audio begins playing before full synthesis completes.

3. stream_format=sse should be rejected with clear error and log.
curl -X POST http://localhost:49112/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world!",
    "voice": "alba",
    "response_format": "wav",
    "stream_format": "sse"
  }'
Expected: HTTP 400 with error message indicating SSE not supported. Server logs include invalid param details.

4. stream parameter should be rejected with clear error and log.
curl -X POST http://localhost:49112/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world!",
    "voice": "alba",
    "response_format": "wav",
    "stream": true
  }'
Expected: HTTP 400 with error message indicating stream is not supported. Server logs include invalid param details.
