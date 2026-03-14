"""
Flask routes for the OpenAI-compatible TTS API.
"""

import re
import threading
import time

from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

import json
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from app.config import Config
from app.logging_config import get_logger
from app.services.audio import (
    convert_audio,
    get_mime_type,
    tensor_to_pcm_bytes,
    validate_format,
    write_wav_header,
)
from app.services.preprocess import TextPreprocessor
from app.services.tts import get_tts_service

logger = get_logger('routes')

# Cold-start request logging (first request only)
_FIRST_REQUEST_LOGGED = False
_FIRST_REQUEST_LOCK = threading.Lock()

# Create blueprint
api = Blueprint('api', __name__)

# Create text preprocessor instance, some options changed from defaults
text_preprocessor = TextPreprocessor(
    remove_urls=False,
    remove_emails=False,
    remove_html=True,
    remove_hashtags=True,
    remove_mentions=False,
    remove_punctuation=False,
    remove_stopwords=False,
    remove_extra_whitespace=False,
)

_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_RE_CLAUSE_SPLIT = re.compile(r'(?<=[,;:])\s+')


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')

    model: StrictStr | None = None
    input: StrictStr = Field(..., min_length=1)
    voice: StrictStr | None = None
    response_format: StrictStr | None = None
    format: StrictStr | None = None
    stream_format: StrictStr | None = None
    speed: StrictFloat | StrictInt | None = None
    steps: StrictInt | None = None
    lang: StrictStr | None = None
    language: StrictStr | None = None
    max_chunk_length: StrictInt | None = None
    silence_duration: StrictFloat | StrictInt | None = None

    @field_validator('model')
    @classmethod
    def _validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in Config.ALLOWED_MODELS:
            raise ValueError(
                f"Unsupported model '{value}'. Allowed: {', '.join(Config.ALLOWED_MODELS)}."
            )
        return value

    @field_validator('voice', mode='before')
    @classmethod
    def _normalize_voice(cls, value):
        if value is None:
            return None
        if isinstance(value, dict):
            candidate = value.get('id') or value.get('name')
            if not candidate:
                raise ValueError("Voice object must include 'id' or 'name'.")
            return candidate
        if isinstance(value, str):
            return value
        raise ValueError("'voice' must be a string")


def _split_at_clauses(text: str, target: int, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts = [p.strip() for p in _RE_CLAUSE_SPLIT.split(text) if p.strip()]
    if not parts:
        return [text]

    chunks: list[str] = []
    current = ''

    for part in parts:
        if len(current) + len(part) + 1 <= target:
            current = f'{current} {part}'.strip() if current else part
            continue

        if current:
            chunks.append(current)
            current = ''

        if len(part) > max_chars:
            words = part.split()
            for word in words:
                candidate = word if not current else f'{current} {word}'
                if len(candidate) <= max_chars:
                    current = candidate
                    continue
                if current:
                    chunks.append(current)
                current = word
        else:
            current = part

    if current:
        chunks.append(current)

    return chunks


def _smart_chunk_text(text: str, target_chars: int, max_chars: int) -> list[str]:
    """Chunk text at sentence/clause boundaries to roughly target_chars."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    sentences = [s.strip() for s in _RE_SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ''

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= target_chars:
            current = f'{current} {sentence}'.strip() if current else sentence
            continue

        if current:
            chunks.append(current)
            current = ''

        if len(sentence) <= target_chars:
            current = sentence
        else:
            sub_chunks = _split_at_clauses(sentence, target_chars, max_chars)
            if not sub_chunks:
                continue
            chunks.extend(sub_chunks[:-1])
            current = sub_chunks[-1]

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def _default_param_suggestions() -> dict:
    model_default = Config.ALLOWED_MODELS[0] if Config.ALLOWED_MODELS else Config.MODEL_NAME
    return {
        'model': model_default,
        'voice': Config.DEFAULT_VOICE,
        'speed': Config.DEFAULT_SPEED,
        'steps': Config.DEFAULT_STEPS,
        'lang': Config.DEFAULT_LANG,
        'response_format': 'mp3',
    }


def _log_invalid_param(param: str, provided, request_id: str, extra: dict | None = None) -> None:
    payload = {
        'param': param,
        'provided': provided,
        'defaults': _default_param_suggestions(),
        'request_id': request_id,
    }
    if extra:
        payload.update(extra)
    logger.warning('Invalid request parameter: %s', payload)


def _handle_validation_error(exc: ValidationError, raw_data: dict | None, request_id: str):
    errors = exc.errors()
    if not errors:
        return _error_response('Invalid request', 400, request_id)

    first = errors[0]
    loc = first.get('loc', ())
    param = loc[0] if loc else None
    err_type = first.get('type', '')

    if param == 'model':
        provided = raw_data.get('model') if isinstance(raw_data, dict) else None
        _log_invalid_param(
            'model',
            provided,
            request_id,
            {'allowed_models': Config.ALLOWED_MODELS},
        )
        return _error_response(
            f"Unsupported model '{provided}'. Use '{Config.ALLOWED_MODELS[0]}' (case-sensitive).",
            401,
            request_id,
            param='model',
            error_type='invalid_request_error',
            extra={
                'allowed_models': Config.ALLOWED_MODELS,
                'hint': 'Use /v1/models to list available models',
                'suggested_defaults': _default_param_suggestions(),
            },
        )

    if param == 'input':
        if err_type in ('missing', 'string_too_short'):
            return _error_response("Missing 'input' text", 400, request_id, param='input')
        if err_type in ('string_type',):
            return _error_response("'input' must be a string", 400, request_id, param='input')

    if param == 'voice':
        return _error_response("'voice' must be a string", 400, request_id, param='voice')

    if param == 'stream_format':
        provided = raw_data.get('stream_format') if isinstance(raw_data, dict) else None
        _log_invalid_param(
            'stream_format',
            provided,
            request_id,
            {'allowed_stream_formats': ('audio', 'sse')},
        )
        return _error_response(
            "'stream_format' must be a string", 400, request_id, param='stream_format'
        )

    if param == 'response_format':
        return _error_response(
            "'response_format' must be a string", 400, request_id, param='response_format'
        )

    if param == 'format':
        return _error_response("'format' must be a string", 400, request_id, param='format')

    if param in ('lang', 'language'):
        return _error_response("'lang' must be a string", 400, request_id, param='lang')

    if param == 'max_chunk_length':
        return _error_response(
            "'max_chunk_length' must be an integer", 400, request_id, param='max_chunk_length'
        )

    if param == 'silence_duration':
        return _error_response(
            "'silence_duration' must be a number", 400, request_id, param='silence_duration'
        )

    if param == 'speed':
        return _error_response("'speed' must be a number", 400, request_id, param='speed')

    if param == 'steps':
        return _error_response("'steps' must be an integer", 400, request_id, param='steps')

    return _error_response(first.get('msg', 'Invalid request'), 400, request_id, param=param)

@api.route('/')
def home():
    """Serve the web interface."""
    from app.config import Config

    if not Config.UI_ENABLED:
        return (
            jsonify(
                {
                    'service': 'supertonic-tts',
                    'status': 'ok',
                    'endpoints': {
                        'health': '/health',
                        'models': '/v1/models',
                        'voices': '/v1/voices',
                        'speech': '/v1/audio/speech',
                    },
                }
            ),
            200,
        )

    return render_template('index.html', is_docker=Config.IS_DOCKER)


@api.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for container orchestration.

    Returns service status and basic model info.
    """
    tts = get_tts_service()

    # Validate a default voice quickly
    voice_valid, voice_msg = tts.validate_voice(Config.DEFAULT_VOICE)

    return jsonify(
        {
            'status': 'healthy' if tts.is_loaded else 'unhealthy',
            'model_loaded': tts.is_loaded,
            'device': tts.device if tts.is_loaded else None,
            'sample_rate': tts.sample_rate if tts.is_loaded else None,
            'voices_dir': tts.voices_dir,
            'default_voice': Config.DEFAULT_VOICE,
            'default_speed': Config.DEFAULT_SPEED,
            'default_steps': Config.DEFAULT_STEPS,
            'voice_check': {'valid': voice_valid, 'message': voice_msg},
        }
    ), 200 if tts.is_loaded else 503


@api.route('/v1/voices', methods=['GET'])
def list_voices():
    """
    List available voices.

    Returns OpenAI-compatible voice list format.
    """
    tts = get_tts_service()
    voices = tts.list_voices()

    return jsonify(
        {
            'object': 'list',
            'data': [
                {
                    'id': v['id'],
                    'name': v['name'],
                    'object': 'voice',
                    'type': v.get('type', 'builtin'),
                }
                for v in voices
            ],
        }
    )


@api.route('/v1/models', methods=['GET'])
def list_models():
    """
    List available models.

    Returns OpenAI-compatible model list format.
    """
    models = Config.ALLOWED_MODELS
    return jsonify(
        {
            'object': 'list',
            'data': [
                {
                    'id': model_id,
                    'object': 'model',
                    'owned_by': 'supertonic',
                }
                for model_id in models
            ],
        }
    )


@api.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible speech generation endpoint.

    Request body:
        model: string (optional) - Model name (case-sensitive, only "supertonic-2")
        input: string (required) - Text to synthesize
        voice: string (optional) - Voice ID or path
        response_format: string (optional) - Audio format
        format: string (optional) - Alias for response_format
        stream_format: string (optional) - "audio" (raw bytes) or "sse" (not supported)
        speed: number (optional) - Speech rate multiplier
        steps: integer (optional) - Diffusion steps (quality vs speed)
        lang: string (optional) - Language code (e.g., "en")
        max_chunk_length: integer (optional) - Supertonic chunk length
        silence_duration: number (optional) - Silence between chunks (seconds)

    Returns:
        Audio file or streaming audio response
    """
    from flask import current_app

    request_start = time.monotonic()
    request_id = request.headers.get(Config.REQUEST_ID_HEADER) or uuid.uuid4().hex
    data = request.get_json(silent=True)

    if not data:
        return _error_response('Missing JSON body', 400, request_id)

    if isinstance(data, dict) and 'stream' in data:
        _log_invalid_param(
            'stream',
            data.get('stream'),
            request_id,
            {'hint': "Use 'stream_format' set to 'audio' for streaming."},
        )
        return _error_response(
            "'stream' is not supported; use 'stream_format' instead",
            400,
            request_id,
            param='stream',
            error_type='invalid_request_error',
            extra={'hint': "Set 'stream_format' to 'audio' for raw audio streaming."},
        )

    try:
        payload = SpeechRequest.model_validate(data)
    except ValidationError as exc:
        return _handle_validation_error(exc, data, request_id)

    text = payload.input
    if len(text) > Config.MAX_INPUT_CHARS:
        return _error_response(
            f"'input' exceeds max length of {Config.MAX_INPUT_CHARS} characters",
            413,
            request_id,
            param='input',
        )

    voice = payload.voice or Config.DEFAULT_VOICE
    stream_format = payload.stream_format

    response_format = payload.response_format
    if response_format is None:
        response_format = payload.format or 'mp3'
    target_format = validate_format(response_format)

    lang = payload.lang or payload.language or Config.DEFAULT_LANG
    max_chunk_length = (
        payload.max_chunk_length
        if payload.max_chunk_length is not None
        else Config.DEFAULT_MAX_CHUNK_LENGTH
    )
    silence_duration = (
        payload.silence_duration
        if payload.silence_duration is not None
        else Config.DEFAULT_SILENCE_DURATION
    )
    speed = payload.speed if payload.speed is not None else Config.DEFAULT_SPEED
    steps = payload.steps if payload.steps is not None else Config.DEFAULT_STEPS

    tts = get_tts_service()

    # Validate voice first
    is_valid, msg = tts.validate_voice(voice)
    if not is_valid:
        available = [v['id'] for v in tts.list_voices()]
        _log_invalid_param(
            'voice',
            voice,
            request_id,
            {'available_voices': available[:10]},
        )
        return _error_response(
            f"Voice '{voice}' not found",
            401,
            request_id,
            param='voice',
            error_type='invalid_request_error',
            extra={
                'available_voices': available[:10],  # Limit to first 10
                'hint': 'Use /v1/voices to see all available voices',
                'suggested_defaults': _default_param_suggestions(),
            },
        )

    try:
        voice_state = tts.get_voice_state(voice)

        # Check if streaming should be used
        if stream_format is not None:
            normalized_format = stream_format.strip().lower()
            allowed_formats = ('audio', 'sse')
            if normalized_format not in allowed_formats:
                _log_invalid_param(
                    'stream_format',
                    stream_format,
                    request_id,
                    {'allowed_stream_formats': allowed_formats},
                )
                return _error_response(
                    "Invalid 'stream_format'. Use 'audio' for streaming.",
                    400,
                    request_id,
                    param='stream_format',
                    error_type='invalid_request_error',
                    extra={
                        'allowed_stream_formats': allowed_formats,
                        'supported_stream_formats': ('audio',),
                    },
                )
            if normalized_format == 'sse':
                _log_invalid_param(
                    'stream_format',
                    stream_format,
                    request_id,
                    {'reason': 'sse_not_supported', 'supported_stream_formats': ('audio',)},
                )
                return _error_response(
                    "stream_format 'sse' is not supported. Use 'audio' for streaming.",
                    400,
                    request_id,
                    param='stream_format',
                    error_type='invalid_request_error',
                    extra={'supported_stream_formats': ('audio',)},
                )
            use_streaming = True
        else:
            use_streaming = False

        # Streaming is supported for all formats; for non-PCM/WAV we stream the
        # fully encoded file buffer (less latency-friendly but format-correct).
        # Check if text preprocessing should be used
        use_text_preprocess = current_app.config.get('TEXT_PREPROCESS_DEFAULT', False)
        # Preprocess text
        if use_text_preprocess:
            # logger.info(f'Preprocessing text: {text}')
            text = text_preprocessor.process(text)
            # logger.info(f'Preprocessed text: {text}')
        chunk_target = Config.CHUNK_TARGET_CHARS
        chunk_max = Config.CHUNK_MAX_CHARS
        if chunk_max < chunk_target:
            chunk_max = chunk_target

        if chunk_target <= 0 or chunk_max <= 0:
            chunks = [text]
        else:
            chunks = _smart_chunk_text(text, chunk_target, chunk_max)
        if len(chunks) > 1:
            logger.info(
                'Chunking input into %s segments (target=%s max=%s chars)',
                len(chunks),
                chunk_target,
                chunk_max,
            )
        if use_streaming:
            pre_response_s = time.monotonic() - request_start
            if len(chunks) > 1:
                return _stream_audio_chunks(
                    tts,
                    voice_state,
                    chunks,
                    target_format,
                    request_start,
                    pre_response_s,
                    request_id,
                    speed,
                    steps,
                    lang,
                    max_chunk_length,
                    silence_duration,
                )
            return _stream_audio(
                tts,
                voice_state,
                text,
                target_format,
                request_start,
                pre_response_s,
                request_id,
                speed,
                steps,
                lang,
                max_chunk_length,
                silence_duration,
            )
        if len(chunks) > 1:
            return _generate_file_chunked(
                tts,
                voice_state,
                chunks,
                target_format,
                request_start,
                request_id,
                speed,
                steps,
                lang,
                max_chunk_length,
                silence_duration,
            )
        return _generate_file(
            tts,
            voice_state,
            text,
            target_format,
            request_start,
            request_id,
            speed,
            steps,
            lang,
            max_chunk_length,
            silence_duration,
        )

    except ValueError as e:
        logger.warning(f'Voice loading failed: {e}')
        return _error_response(str(e), 400, request_id, param='voice')
    except Exception as e:
        logger.exception('Generation failed')
        return _error_response('Generation failed', 500, request_id)


def _emit_timing_log(event: str, metrics: dict) -> None:
    payload = {'event': event, **metrics}
    if Config.REQUEST_TIMING_LOG_JSON:
        logger.info(json.dumps(payload, separators=(',', ':')))
    else:
        logger.info('%s: %s', event, payload)


def _log_first_request(metrics: dict) -> None:
    if not Config.COLDSTART_LOG:
        return
    global _FIRST_REQUEST_LOGGED
    with _FIRST_REQUEST_LOCK:
        if _FIRST_REQUEST_LOGGED:
            return
        _FIRST_REQUEST_LOGGED = True
    _emit_timing_log('first_request_timing', metrics)


def _error_response(
    message: str,
    status_code: int,
    request_id: str,
    extra: dict | None = None,
    param: str | None = None,
    code: str | None = None,
    error_type: str | None = None,
):
    if error_type is None:
        if status_code == 401:
            error_type = 'authentication_error'
        elif status_code in (400, 413, 422):
            error_type = 'invalid_request_error'
        elif status_code >= 500:
            error_type = 'server_error'
        else:
            error_type = 'unknown_error'

    error_payload = {
        'message': message,
        'type': error_type,
        'param': param,
        'code': code,
    }
    if extra:
        error_payload['details'] = extra

    payload = {'error': error_payload}
    response = jsonify(payload)
    response.status_code = status_code
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    return response


def _generate_file(
    tts,
    voice_state,
    text: str,
    fmt: str,
    request_start: float,
    request_id: str,
    speed: float,
    steps: int,
    lang: str,
    max_chunk_length: int,
    silence_duration: float,
):
    """Generate complete audio and return as file."""
    t0 = time.time()
    audio_tensor = tts.generate_audio(
        voice_state,
        text,
        speed=speed,
        steps=steps,
        lang=lang,
        max_chunk_length=max_chunk_length,
        silence_duration=silence_duration,
    )
    generation_time = time.time() - t0

    logger.info(f'Generated {len(text)} chars in {generation_time:.2f}s')

    convert_t0 = time.time()
    audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
    convert_time = time.time() - convert_t0
    mimetype = get_mime_type(fmt)
    total_s = time.monotonic() - request_start
    _log_first_request(
        {
            'mode': 'non_stream',
            'format': fmt,
            'text_len': len(text),
            'generation_s': round(generation_time, 4),
            'total_s': round(total_s, 4),
            'request_id': request_id,
        }
    )
    if Config.REQUEST_TIMING_LOG:
        _emit_timing_log(
            'request_timing',
            {
                'mode': 'non_stream',
                'format': fmt,
                'text_len': len(text),
                'generation_s': round(generation_time, 4),
                'convert_s': round(convert_time, 4),
                'total_s': round(total_s, 4),
                'request_id': request_id,
            },
        )

    response = send_file(
        audio_buffer, mimetype=mimetype, as_attachment=True, download_name=f'speech.{fmt}'
    )
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    return response


def _generate_file_chunked(
    tts,
    voice_state,
    chunks: list[str],
    fmt: str,
    request_start: float,
    request_id: str,
    speed: float,
    steps: int,
    lang: str,
    max_chunk_length: int,
    silence_duration: float,
):
    """Generate complete audio from chunked text and return as file."""
    import torch

    t0 = time.time()
    tensors = []
    total_chars = 0
    for chunk in chunks:
        total_chars += len(chunk)
        tensors.append(
            tts.generate_audio(
                voice_state,
                chunk,
                speed=speed,
                steps=steps,
                lang=lang,
                max_chunk_length=max_chunk_length,
                silence_duration=silence_duration,
            )
        )
    generation_time = time.time() - t0

    processed = []
    for tensor in tensors:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        processed.append(tensor)
    audio_tensor = processed[0] if len(processed) == 1 else torch.cat(processed, dim=1)

    convert_t0 = time.time()
    audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
    convert_time = time.time() - convert_t0
    total_s = time.monotonic() - request_start
    _log_first_request(
        {
            'mode': 'non_stream',
            'format': fmt,
            'text_len': total_chars,
            'chunks': len(chunks),
            'generation_s': round(generation_time, 4),
            'total_s': round(total_s, 4),
            'request_id': request_id,
        }
    )
    if Config.REQUEST_TIMING_LOG:
        _emit_timing_log(
            'request_timing',
            {
                'mode': 'non_stream',
                'format': fmt,
                'text_len': total_chars,
                'chunks': len(chunks),
                'generation_s': round(generation_time, 4),
                'convert_s': round(convert_time, 4),
                'total_s': round(total_s, 4),
                'request_id': request_id,
            },
        )

    response = send_file(
        audio_buffer, mimetype=get_mime_type(fmt), as_attachment=True, download_name=f'speech.{fmt}'
    )
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    return response


def _stream_encoded_buffer(
    audio_buffer,
    fmt: str,
    request_start: float,
    pre_response_s: float,
    request_id: str,
    text_len: int,
    extra_metrics: dict | None = None,
):
    """Stream an already-encoded audio buffer in chunks."""
    audio_buffer.seek(0)
    first_log = {'done': False}
    counters = {'bytes': 0, 'chunks': 0}

    def _maybe_log_first_bytes():
        if first_log['done']:
            return
        first_log['done'] = True
        first_bytes_s = time.monotonic() - request_start
        metrics = {
            'mode': 'stream',
            'format': fmt,
            'text_len': text_len,
            'pre_response_s': round(pre_response_s, 4),
            'first_bytes_s': round(first_bytes_s, 4),
            'request_id': request_id,
        }
        _log_first_request(metrics)
        if Config.TTFA_LOG:
            _emit_timing_log('ttfa', metrics)

    def stream_buffer():
        stream_start = time.monotonic()
        try:
            while True:
                chunk = audio_buffer.read(64 * 1024)
                if not chunk:
                    break
                _maybe_log_first_bytes()
                counters['chunks'] += 1
                counters['bytes'] += len(chunk)
                yield chunk
        except Exception:
            logger.exception('Streaming encoded audio failed mid-stream')
        finally:
            if Config.REQUEST_TIMING_LOG:
                total_s = time.monotonic() - stream_start
                payload = {
                    'mode': 'stream',
                    'format': fmt,
                    'text_len': text_len,
                    'pre_response_s': round(pre_response_s, 4),
                    'total_s': round(total_s, 4),
                    'chunks': counters['chunks'],
                    'bytes': counters['bytes'],
                    'request_id': request_id,
                }
                if extra_metrics:
                    payload.update(extra_metrics)
                _emit_timing_log('request_timing', payload)

    response = Response(stream_with_context(stream_buffer()), mimetype=get_mime_type(fmt))
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


def _stream_audio(
    tts,
    voice_state,
    text: str,
    fmt: str,
    request_start: float,
    pre_response_s: float,
    request_id: str,
    speed: float,
    steps: int,
    lang: str,
    max_chunk_length: int,
    silence_duration: float,
):
    """Stream audio chunks."""
    if fmt not in ('pcm', 'wav'):
        t0 = time.time()
        audio_tensor = tts.generate_audio(
            voice_state,
            text,
            speed=speed,
            steps=steps,
            lang=lang,
            max_chunk_length=max_chunk_length,
            silence_duration=silence_duration,
        )
        generation_time = time.time() - t0
        logger.info(f'Generated {len(text)} chars in {generation_time:.2f}s')
        convert_t0 = time.time()
        audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
        convert_time = time.time() - convert_t0
        extra_metrics = {
            'generation_s': round(generation_time, 4),
            'convert_s': round(convert_time, 4),
        }
        return _stream_encoded_buffer(
            audio_buffer,
            fmt,
            request_start,
            pre_response_s,
            request_id,
            len(text),
            extra_metrics,
        )

    # PCM/WAV streaming emits raw PCM bytes, optionally wrapped in a WAV header.
    stream_fmt = fmt

    first_log = {'done': False}
    counters = {'bytes': 0, 'chunks': 0}

    try:
        stream = tts.generate_audio_stream(
            voice_state,
            text,
            speed=speed,
            steps=steps,
            lang=lang,
            max_chunk_length=max_chunk_length,
            silence_duration=silence_duration,
        )
        first_chunk = next(stream, None)
    except Exception:
        logger.exception('Streaming generation failed before first chunk')
        return _error_response('Streaming generation failed', 500, request_id)

    if first_chunk is None:
        logger.error('Streaming generation produced no audio chunks')
        return _error_response('Streaming generation produced no audio', 500, request_id)

    def _maybe_log_first_bytes():
        if first_log['done']:
            return
        first_log['done'] = True
        first_bytes_s = time.monotonic() - request_start
        metrics = {
            'mode': 'stream',
            'format': stream_fmt,
            'text_len': len(text),
            'pre_response_s': round(pre_response_s, 4),
            'first_bytes_s': round(first_bytes_s, 4),
            'request_id': request_id,
        }
        _log_first_request(metrics)
        if Config.TTFA_LOG:
            _emit_timing_log('ttfa', metrics)

    def stream_with_header():
        stream_start = time.monotonic()
        try:
            # Yield WAV header first if streaming as WAV
            if stream_fmt == 'wav':
                _maybe_log_first_bytes()
                header = write_wav_header(tts.sample_rate, num_channels=1, bits_per_sample=16)
                counters['bytes'] += len(header)
                yield header

            _maybe_log_first_bytes()
            chunk_bytes = tensor_to_pcm_bytes(first_chunk)
            counters['chunks'] += 1
            counters['bytes'] += len(chunk_bytes)
            yield chunk_bytes

            for chunk_tensor in stream:
                _maybe_log_first_bytes()
                chunk_bytes = tensor_to_pcm_bytes(chunk_tensor)
                counters['chunks'] += 1
                counters['bytes'] += len(chunk_bytes)
                yield chunk_bytes
        except Exception:
            logger.exception('Streaming generation failed mid-stream')
        finally:
            if Config.REQUEST_TIMING_LOG:
                total_s = time.monotonic() - stream_start
                _emit_timing_log(
                    'request_timing',
                    {
                        'mode': 'stream',
                        'format': stream_fmt,
                        'text_len': len(text),
                        'pre_response_s': round(pre_response_s, 4),
                        'total_s': round(total_s, 4),
                        'chunks': counters['chunks'],
                        'bytes': counters['bytes'],
                        'request_id': request_id,
                    },
                )

    mimetype = get_mime_type(stream_fmt)

    response = Response(stream_with_context(stream_with_header()), mimetype=mimetype)
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


def _stream_audio_chunks(
    tts,
    voice_state,
    chunks: list[str],
    fmt: str,
    request_start: float,
    pre_response_s: float,
    request_id: str,
    speed: float,
    steps: int,
    lang: str,
    max_chunk_length: int,
    silence_duration: float,
):
    """Stream audio chunks from chunked text input."""
    if fmt not in ('pcm', 'wav'):
        import torch

        t0 = time.time()
        tensors = []
        total_chars = 0
        for chunk_text in chunks:
            total_chars += len(chunk_text)
            tensors.append(
                tts.generate_audio(
                    voice_state,
                    chunk_text,
                    speed=speed,
                    steps=steps,
                    lang=lang,
                    max_chunk_length=max_chunk_length,
                    silence_duration=silence_duration,
                )
            )
        generation_time = time.time() - t0

        processed = []
        for tensor in tensors:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            processed.append(tensor)
        audio_tensor = processed[0] if len(processed) == 1 else torch.cat(processed, dim=1)

        convert_t0 = time.time()
        audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
        convert_time = time.time() - convert_t0
        extra_metrics = {
            'generation_s': round(generation_time, 4),
            'convert_s': round(convert_time, 4),
        }
        return _stream_encoded_buffer(
            audio_buffer,
            fmt,
            request_start,
            pre_response_s,
            request_id,
            total_chars,
            extra_metrics,
        )

    # PCM/WAV streaming emits raw PCM bytes, optionally wrapped in a WAV header.
    stream_fmt = fmt

    first_log = {'done': False}
    counters = {'bytes': 0, 'chunks': 0}

    try:
        stream = tts.generate_audio_stream(
            voice_state,
            chunks[0],
            speed=speed,
            steps=steps,
            lang=lang,
            max_chunk_length=max_chunk_length,
            silence_duration=silence_duration,
        )
        first_chunk = next(stream, None)
    except Exception:
        logger.exception('Streaming generation failed before first chunk')
        return _error_response('Streaming generation failed', 500, request_id)

    if first_chunk is None:
        logger.error('Streaming generation produced no audio chunks')
        return _error_response('Streaming generation produced no audio', 500, request_id)

    def _maybe_log_first_bytes():
        if first_log['done']:
            return
        first_log['done'] = True
        first_bytes_s = time.monotonic() - request_start
        metrics = {
            'mode': 'stream',
            'format': stream_fmt,
            'text_len': sum(len(chunk) for chunk in chunks),
            'chunks': len(chunks),
            'pre_response_s': round(pre_response_s, 4),
            'first_bytes_s': round(first_bytes_s, 4),
            'request_id': request_id,
        }
        _log_first_request(metrics)
        if Config.TTFA_LOG:
            _emit_timing_log('ttfa', metrics)

    def stream_with_header():
        stream_start = time.monotonic()
        try:
            if stream_fmt == 'wav':
                _maybe_log_first_bytes()
                header = write_wav_header(tts.sample_rate, num_channels=1, bits_per_sample=16)
                counters['bytes'] += len(header)
                yield header

            _maybe_log_first_bytes()
            chunk_bytes = tensor_to_pcm_bytes(first_chunk)
            counters['chunks'] += 1
            counters['bytes'] += len(chunk_bytes)
            yield chunk_bytes

            for chunk_tensor in stream:
                _maybe_log_first_bytes()
                chunk_bytes = tensor_to_pcm_bytes(chunk_tensor)
                counters['chunks'] += 1
                counters['bytes'] += len(chunk_bytes)
                yield chunk_bytes

            for chunk_text in chunks[1:]:
                for chunk_tensor in tts.generate_audio_stream(
                    voice_state,
                    chunk_text,
                    speed=speed,
                    steps=steps,
                    lang=lang,
                    max_chunk_length=max_chunk_length,
                    silence_duration=silence_duration,
                ):
                    _maybe_log_first_bytes()
                    chunk_bytes = tensor_to_pcm_bytes(chunk_tensor)
                    counters['chunks'] += 1
                    counters['bytes'] += len(chunk_bytes)
                    yield chunk_bytes
        except Exception:
            logger.exception('Streaming generation failed mid-stream')
        finally:
            if Config.REQUEST_TIMING_LOG:
                total_s = time.monotonic() - stream_start
                _emit_timing_log(
                    'request_timing',
                    {
                        'mode': 'stream',
                        'format': stream_fmt,
                        'text_len': sum(len(chunk) for chunk in chunks),
                        'chunks': len(chunks),
                        'pre_response_s': round(pre_response_s, 4),
                        'total_s': round(total_s, 4),
                        'chunks_out': counters['chunks'],
                        'bytes': counters['bytes'],
                        'request_id': request_id,
                    },
                )

    mimetype = get_mime_type(stream_fmt)

    response = Response(stream_with_context(stream_with_header()), mimetype=mimetype)
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Accel-Buffering'] = 'no'
    return response
