# Supertonic-TTS OpenAI-Compatible Server - Implementation Plan

**Objective:** Replicate the pocket-tts-openai_streaming_server using Supertonic TTS backend, optimized for CPU deployment on Modal.com with lower resource requirements than the original pocket-tts implementation.

**Target:** OpenAI-compatible TTS API server with streaming support, suitable for CPU-only inference on Modal Starter ($30/month free credits).

---

## Executive Summary

### Current State
- **Source System:** pocket-tts-openai_streaming_server (PyTorch-based, ~500MB model, CPU-optimized)
- **Target System:** Supertonic-TTS (ONNX-based, smaller models, highly efficient CPU inference)
- **Deployment Platform:** Modal.com (serverless, CPU-based)

### Key Advantages of Supertonic Over Pocket-TTS
1. **Model Efficiency:** ONNX format with separate, smaller submodels vs monolithic PyTorch
2. **CPU Performance:** M4 Pro CPU benchmarks show 912–1,263 cps (RTF 0.012–0.015). For 5-step inference: 596–850 cps (RTF 0.018–0.023). WebGPU is faster but not relevant for CPU deployment.
3. **Lower Memory Footprint:** Modular ONNX models vs large PyTorch checkpoint
4. **Streaming Support:** Native streaming API with diffusion-based latent synthesis
5. **Multilingual:** Out-of-box support for EN, KO, ES, PT, FR
6. **Voice Management:** Style-based voice system with parametric control (style vectors)

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────┐
│        OpenAI-Compatible REST API Layer        │
│  (Flask/WSGI, /v1/audio/speech endpoint)       │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┴───────────┬──────────────┐
       │                       │              │
    ┌──▼──┐            ┌──────▼────┐   ┌─────▼──┐
    │Voice│            │   Text    │   │Streaming
    │Mgmt │            │Processing │   │Handler
    └──┬──┘            └──────┬────┘   └────┬────┘
       │                      │             │
       └──────────────┬───────┴─────────────┘
                      │
          ┌───────────▼──────────────┐
          │  Supertonic TTS Service  │
          │  (ONNX Runtime)          │
          │  - Text Encoder          │
          │  - Latent Denoiser       │
          │  - Voice Decoder         │
          │  - Vocoder               │
          └──────────┬───────────────┘
                     │
          ┌──────────▼───────────┐
          │  Audio Output        │
          │  (MP3/WAV/PCM/OPUS)  │
          └──────────────────────┘
```

### Component Mapping: Pocket-TTS → Supertonic-TTS

| Layer | Pocket-TTS | Supertonic | Notes |
|-------|-----------|-----------|-------|
| **Model Backend** | PyTorch (monolithic) | ONNX (modular) | Supertonic uses 4 ONNX sessions |
| **Voice Format** | Audio files | Audio files + Style vectors | Supertonic extracts voice state |
| **Streaming** | Custom chunking | Native streaming API | Supertonic has `generate_audio_stream()` |
| **Sample Rate** | 24kHz | 44.1kHz | Format conversion required |
| **Inference Mode** | torch.inference_mode() | ONNX Runtime sessions | No GPU patching needed |

---

## Implementation Phases

### Phase 0: Research & Discovery (COMPLETED)
✅ Analyzed supertonic-inc/supertonic architecture from DeepWiki + Tavily
✅ Mapped Supertonic ONNX pipeline (4-stage: encoder → denoiser → decoder → vocoder)
✅ Reviewed pocket-tts modal deployment structure
✅ Identified resource optimization opportunities

### Phase 1: Project Scaffold & Environment Setup

**Deliverables:**
- Clone pocket-tts structure and adapt for Supertonic
- Create new repository structure maintaining API compatibility
- Set up Supertonic ONNX model loading

**Files to Create/Modify:**
```
supertonic-tts-openai-server/
├── app/
│   ├── __init__.py               (Flask app factory - adapt from pocket-tts)
│   ├── config.py                 (new: Supertonic-specific config)
│   ├── logging_config.py         (reuse from pocket-tts)
│   ├── routes.py                 (adapt: map to Supertonic API)
│   └── services/
│       ├── tts.py                (NEW: Supertonic TTS wrapper)
│       ├── audio.py              (adapt: add 44.1kHz support)
│       ├── preprocess.py         (reuse: text preprocessing)
│       └── voice_manager.py      (NEW: voice state extraction)
├── infra/
│   └── modal/
│       ├── modal_config.py       (adapt from pocket-tts)
│       └── GUIDE.md              (new: deployment guide)
├── server.py                     (reuse from pocket-tts)
├── Dockerfile                    (new: optimized for Supertonic)
├── requirements.txt              (new: Supertonic dependencies)
└── README.md                     (new: Supertonic-specific docs)
```

**Key Adaptations:**
- Replace `pocket_tts` imports with Supertonic ONNX client
- Update config constants for Supertonic (44.1kHz, ONNX model paths)
- Adapt Modal volume structure for ONNX assets

**Dependencies:**
- `onnxruntime` (for ONNX Runtime inference)
- `transformers` (for tokenizer compatibility)
- `torch` (minimal, only for audio tensor operations)
- Keep existing: Flask, waitress, librosa, etc.

---

### Phase 2: Supertonic TTS Service Implementation

**Objective:** Create abstraction layer that maintains pocket-tts API surface while using Supertonic backend

**Files:**
- `app/services/tts.py` (SupertonicTTSService class)
- `app/services/voice_manager.py` (Voice state caching & extraction)

**Core Classes:**

```python
class SupertonicTTSService:
    """Supertonic ONNX-based TTS service."""

    def __init__(self):
        self.text_encoder_session = None
        self.latent_denoiser_session = None
        self.voice_decoder_session = None
        self.vocoder_session = None
        self.tokenizer = None
        self.voice_cache = {}

    def load_model(self, model_path: str = None) -> None:
        """Load ONNX models and tokenizer."""
        # Load 4 ONNX sessions from HF or local path
        # Initialize AutoTokenizer
        # Set device (CPU or CUDA if available)

    def get_voice_state(self, voice_audio_path: str) -> dict:
        """Extract voice state from audio file."""
        # Load audio, normalize
        # Run through model to extract style vectors
        # Cache result

    def generate_audio(self, text: str, voice_state: dict,
                      speed: float = 1.0, steps: int = 5) -> torch.Tensor:
        """Generate complete audio."""
        # Text tokenization
        # Duration prediction
        # Latent inference with diffusion steps
        # Vocoding to waveform

    def generate_audio_stream(self, text: str, voice_state: dict, ...)
        -> Iterator[torch.Tensor]:
        """Stream audio chunks (chunk-by-chunk)."""
        # Intelligent text chunking strategy
        # Generate per-chunk + maintain state
        # Yield audio tensors
```

**Voice Management:**
- Supertonic uses "style vectors" (style_ttl, style_dp) not raw audio embeddings
- Design voice file → style vector extraction pipeline
- Support both:
  1. Built-in voices (style vectors pre-computed)
  2. Custom voice files (extract style on-demand with caching)

**Key Differences from Pocket-TTS:**
- ONNX Runtime input/output shapes and inference
- 44.1kHz vs 24kHz sample rate (need resampling)
- Diffusion steps tuning (Supertonic uses 1-50, default varies)
- Tokenizer compatibility (use AutoTokenizer with Supertonic tokenizer)

---

### Phase 3: API Routes & Streaming Handler

**Files:**
- `app/routes.py` (adapt POST /v1/audio/speech endpoint)

**Endpoint Adaptations:**

**Request Parameters:**
```json
{
  "model": "supertonic-2",            // Supertonic model identifier
  "input": "Hello world",             // Text input
  "voice": "M1",                       // Supertonic voice ID (M1-M5, F1-F5, or custom file)
  "response_format": "mp3",           // mp3, wav, pcm, opus, flac
  "stream": false,                    // Enable streaming
  "speed": 1.0,                       // Speech rate (0.8-1.5 typical)
  "steps": 20                         // Diffusion steps (1-50, higher = better quality)
}
```

**Implementation Details:**
1. Parse request
2. Load/validate voice state
3. Generate audio (streaming or complete)
4. Encode to requested format
5. Return response (stream or complete)

**Streaming Strategy:**
- For long text: intelligently chunk on sentence/phrase boundaries
- Per-chunk generation with maintained voice state
- Continuous audio output with proper concatenation
- Optional: gap/silence between chunks (parametrize via config)

**Response Format Handling:**
- Reuse pocket-tts audio module (librosa-based)
- Ensure 44.1kHz input → any target format output
- MP3 default (size/compatibility)

---

### Phase 4: Modal Deployment Configuration

**Files:**
- `infra/modal/modal_config.py` (adapt from pocket-tts)
- `infra/modal/GUIDE.md` (new deployment guide)
- `Dockerfile` (new: Supertonic image)

**Resource Optimization for CPU:**

| Metric | Pocket-TTS | Supertonic Target | Notes |
|--------|-----------|------------------|-------|
| CPU | 2 cores | 1–2 cores (start at 2, tune down) | Match pocket-tts Modal config before downscaling |
| Memory | 4096 MB | 2048–4096 MB (start at 4GB, tune down) | Measure peak usage before lowering |
| Model Size | ~500 MB | ~300-400 MB (ONNX) | More efficient binary |
| Coldstart | ~10-15s | ≤10s after snapshot restore (target) | Snapshots reduce init work; storage load still applies |
| Per-request | ~200-500ms (100 chars) | M4 Pro CPU equiv: ~80–170ms (100 chars) | Derived from M4 Pro cps; Modal CPU will be slower |

**Modal Deployment Strategy:**
1. Two-volume setup:
   - **Models Volume** (persistent): ONNX models + HF cache
   - **Voices Volume** (optional): Custom voice files

2. Function setup:
   - `download_models()`: One-time ONNX download
   - `PocketTTSApp` class → rename/adapt for Supertonic
   - Warmup: synthesize sample text on container start
   - Memory snapshots: for faster container reuse

3. Environment variables:
   - Reuse pocket-tts env var structure
   - Add Supertonic-specific:
     - `SUPERTONIC_MODEL_PATH` (default: HF ONNX distribution)
     - `SUPERTONIC_DIFFUSION_STEPS` (quality/speed tradeoff)
     - `SUPERTONIC_DEFAULT_SPEED`

**Cost Estimates (Modal serverless pricing, Starter includes $30/month credit):**
- Serverless CPU: $0.0000131 / physical core / sec; memory: $0.00000222 / GiB / sec
- Sandboxes/Notebooks (if used): CPU $0.00003942 / core / sec; memory $0.00000672 / GiB / sec
- Example baseline (serverless): 1 core + 2 GiB ≈ $0.00001754 / sec
- Cost per request formula: `cost ≈ (audio_seconds × RTF) × per_second_cost`

---

### Feasibility on Modal CPU (Grounded)

- **Model speed:** Supertonic reports M4 Pro CPU throughput of 912–1,263 cps (RTF 0.012–0.015) and 5-step throughput of 596–850 cps (RTF 0.018–0.023). This indicates faster-than-real-time synthesis on a strong CPU, but Modal CPU performance must be benchmarked on target hardware before final SLOs.
- **Model storage and load:** Modal Volumes are a high-performance distributed filesystem designed for model weights and offer up to 2.5 GB/s bandwidth (actual throughput varies). Use Volumes to avoid repeated downloads and keep model files off the image.
- **Cold start mitigation:** Memory Snapshots can reduce cold-start latency 3–10x for initialization-heavy workloads when you warm up in `@modal.enter(snap=True)`, but snapshots do not speed up loading weights from storage.
- **Cost control:** The Starter plan includes $30/month in free credits. With serverless per-second pricing, CPU-only deployment is cost-feasible as long as RTF and concurrency are controlled and measured.

---

### Phase 5: Testing & Validation

**Unit Tests:**
- Voice state extraction
- Audio format conversion (resampling 44.1kHz)
- Streaming chunk concatenation
- Request validation & error handling

**Integration Tests:**
- Full API request → response cycle
- Streaming vs non-streaming modes
- Multiple voices & parameters
- Edge cases: empty text, long text, special characters

**Performance Benchmarks:**
- Coldstart time (container initialization)
- Warmup time (model loading + sample inference)
- Per-request latency (100, 500, 1000 character inputs)
- Characters-per-second and RTF on Modal (1 core vs 2 cores; 2-step vs 5-step)
- Memory usage during concurrent requests
- Audio quality (subjective listening tests)

**Deployment Test:**
- Deploy to Modal Starter with $30 credit
- Verify free tier works without exceeding limits
- Test concurrent requests (up to 10 containers)
- Monitor cost/usage via Modal dashboard

---

### Phase 6: Documentation & Launch

**Documentation:**
- Project README (quick start, API reference)
- Modal deployment guide (step-by-step)
- Architecture doc (internal design)
- API examples (curl, Python client, streaming)
- Troubleshooting guide

**Final Deliverables:**
- GitHub repository ready for deployment
- Docker image buildable locally
- Modal YAML configuration
- Example client code (OpenAI SDK compatible)

---

## Technical Deep Dives

### Voice State Extraction (Supertonic Specific)

Supertonic uses **style vectors** instead of raw embeddings:
- **style_ttl:** Acoustic characteristics (timbre, tone, pitch range)
- **style_dp:** Duration/prosody characteristics (speaking rate, rhythm)

**Implementation Approach:**
1. Load custom voice file (audio)
2. Process through Supertonic model to extract style vector
3. Cache vector in memory (keyed by voice filename)
4. Reuse cached vector for faster generation

**Pseudocode:**
```python
def get_voice_state(self, voice_audio_path: str) -> dict:
    if voice_audio_path in self.voice_cache:
        return self.voice_cache[voice_audio_path]

    # Load audio → resample to 44.1kHz
    audio = librosa.load(voice_audio_path, sr=44100)

    # Run inference to extract style
    # (exact method depends on Supertonic API)
    style_ttl, style_dp = self.model.extract_style(audio)

    state = {"style_ttl": style_ttl, "style_dp": style_dp}
    self.voice_cache[voice_audio_path] = state
    return state
```

### Audio Format Conversion (Supertonic Output)

Supertonic generates **44.1kHz float32 PCM** tensors.

**Conversion Pipeline:**
1. Torch tensor (44.1kHz, float32) → NumPy array
2. Clip/normalize to [-1.0, 1.0]
3. Use librosa/soundfile for format encoding:
   - **MP3:** Via ffmpeg (requires external binary) or pydub
   - **WAV:** Direct scipy.io.wavfile
   - **PCM:** Raw bytes (return as-is)
   - **OPUS:** Via opuslib
   - **FLAC:** Via soundfile

**Reuse pocket-tts module:** Already handles this via `app/services/audio.py`

---

### Streaming Chunking Strategy

**Problem:** Long text (2000+ chars) needs intelligent chunking for:
- Real-time response (don't buffer entire audio)
- Consistent prosody across chunks
- Minimal latency

**Algorithm:**

```python
def intelligent_chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text on sentence/clause boundaries to preserve prosody.
    Heuristic: prefer splits at: periods, question marks, exclamation marks
    Then: commas, semicolons
    Finally: word boundaries
    """
    # Implementation: regex-based sentence detection
    # Ensure no chunk exceeds chunk_size
    # Maintain semantic coherence
```

**Streaming Response:**
```
1. Receive request (e.g., 2000 character text)
2. Chunk intelligently into [500, 500, 500, 500]
3. For each chunk:
   a. Generate audio tensor
   b. Encode to response format (e.g., MP3 frame)
   c. Yield bytes to client
4. Client receives continuous audio stream
```

---

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **ONNX inference slower than expected** | Medium | HIGH | Benchmark early; fallback to Pocket-TTS if needed |
| **Voice state extraction accuracy** | Medium | MEDIUM | Collect listening test feedback; iterate on extraction |
| **Modal CPU coldstart > 30s** | Low | MEDIUM | Optimize dependencies; use memory snapshots |
| **Model size > free tier storage** | Low | MEDIUM | Use HF-hosted ONNX; lazy-load if needed |
| **Streaming audio quality issues** | Low | MEDIUM | Test with long-form content; tune chunk boundaries |
| **OpenAI SDK compatibility** | Low | LOW | Test with OpenAI client library during development |

---

## Success Criteria

✅ **Functional:**
- [ ] OpenAI-compatible `/v1/audio/speech` endpoint works
- [ ] All response formats (MP3, WAV, PCM, OPUS, FLAC) supported
- [ ] Streaming mode produces continuous, audible output
- [ ] Custom voices loadable & cacheable

✅ **Performance (on Modal CPU):**
- [ ] Baseline cps/RTF measured on Modal (1 and 2 cores; 2-step and 5-step)
- [ ] Coldstart: snapshot restore ≤ 10 seconds (target); first cold boot documented
- [ ] 100-character synthesis ≤ 500ms latency (provisional; adjust after baseline)
- [ ] 1000-character synthesis ≤ 3 seconds (streaming, provisional)
- [ ] Memory usage ≤ 2GB during inference (provisional; use 4GB if needed)
- [ ] Cost model documented using Modal per-second pricing and measured RTF

✅ **Code Quality:**
- [ ] 80%+ test coverage (unit + integration)
- [ ] No hardcoded secrets
- [ ] Comprehensive error handling
- [ ] Security review passed (no SSRF, injection, etc.)

✅ **Deployment:**
- [ ] Docker image builds successfully
- [ ] Modal deployment works end-to-end
- [ ] API accessible at `https://<app>.modal.run/v1/audio/speech`
- [ ] Documentation complete & tested

---

## Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1 (Scaffold) | 4-6 hours | Mostly file copy + config adaptation |
| Phase 2 (TTS Service) | 8-12 hours | ONNX integration + voice state extraction |
| Phase 3 (Routes) | 4-6 hours | Streaming handler + format conversion |
| Phase 4 (Modal) | 4-6 hours | Environment setup + deployment |
| Phase 5 (Testing) | 6-8 hours | Unit/integration/perf tests |
| Phase 6 (Docs) | 2-3 hours | README + deployment guide |
| **Total** | **28-41 hours** | Parallelizable; can overlap phases 2-4 |

---

## Next Steps

1. **Initiate Phase 1:** Clone pocket-tts scaffold; adapt config for Supertonic
2. **Research Supertonic ONNX API:** Verify exact ONNX session interface
3. **Download ONNX models locally:** Test ONNX Runtime session creation
4. **Prototype voice state extraction:** Validate accuracy on 2-3 custom voices
5. **Set up Modal account:** Verify free tier credits available
6. **Create GitHub repo:** Initialize git with Phase 1 structure
7. **Begin Phase 2:** TTS service implementation (can run in parallel with research)

---

## Reference Materials

### Supertonic Resources
- GitHub: https://github.com/supertone-inc/supertonic
- ONNX Models: https://huggingface.co/onnx-community/Supertonic-TTS-ONNX
- Benchmarks: https://github.com/supertone-inc/supertonic#benchmarks
- Architecture: DeepWiki (supertone-inc/supertonic) - Core Architecture section

### Pocket-TTS Reference
- Source: `/workspaces/supertonic-tts-openai_streaming_server/pocket-tts-openai_streaming_server/`
- Key files: `app/services/tts.py`, `infra/modal/modal_config.py`, `server.py`

### Modal.com Docs
- Python SDK: https://modal.com/docs/guide
- Web App (WSGI): https://modal.com/docs/guide/web
- Volumes: https://modal.com/docs/guide/volumes
- Environment: https://modal.com/docs/guide/environment-variables
- Pricing: https://modal.com/pricing

---

## Appendix: Supertonic vs Pocket-TTS Comparison

| Aspect | Pocket-TTS | Supertonic |
|--------|-----------|-----------|
| **Model Format** | PyTorch (.pt) | ONNX (.onnx) |
| **Voice System** | Audio files | Audio + style vectors |
| **Languages** | English focus | 5+ languages |
| **Sample Rate** | 24kHz | 44.1kHz |
| **Streaming** | Custom implementation | Native API |
| **CPU Performance** | Good (~100-200 cps) | Excellent (M4 Pro CPU 912–1,263 cps; 5-step 596–850 cps) |
| **Model Size** | ~500MB | ~300-400MB |
| **Inference Runtime** | PyTorch | ONNX Runtime |
| **Voice Cloning** | Supported | Limited (style extraction) |
| **Multilingual** | Via models | Native |

---

**Status:** Ready for implementation
**Created:** 2026-03-13
**Last Updated:** 2026-03-13
