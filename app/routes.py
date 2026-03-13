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


@api.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible speech generation endpoint.

    Request body:
        model: string (optional) - OpenAI model name (e.g., tts-1, tts-1-hd)
        input: string (required) - Text to synthesize
        voice: string (optional) - Voice ID or path
        response_format: string (optional) - Audio format
        format: string (optional) - Alias for response_format
        stream: boolean (optional) - Enable streaming
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

    model_name = data.get('model')
    if model_name is not None and not isinstance(model_name, str):
        return _error_response("'model' must be a string", 400, request_id, param='model')

    text = data.get('input')
    if not text:
        return _error_response("Missing 'input' text", 400, request_id, param='input')
    if not isinstance(text, str):
        return _error_response("'input' must be a string", 400, request_id, param='input')
    if len(text) > Config.MAX_INPUT_CHARS:
        return _error_response(
            f"'input' exceeds max length of {Config.MAX_INPUT_CHARS} characters",
            413,
            request_id,
            param='input',
        )

    voice = data.get('voice', Config.DEFAULT_VOICE)
    if isinstance(voice, dict):
        voice = voice.get('id') or voice.get('name')
    if not isinstance(voice, str):
        return _error_response("'voice' must be a string", 400, request_id, param='voice')
    stream_request = data.get('stream')
    if stream_request is not None and not isinstance(stream_request, bool):
        return _error_response("'stream' must be a boolean", 400, request_id, param='stream')

    response_format = data.get('response_format')
    if response_format is None:
        response_format = data.get('format', 'mp3')
    if not isinstance(response_format, str):
        return _error_response(
            "'response_format' must be a string", 400, request_id, param='response_format'
        )
    target_format = validate_format(response_format)

    lang = data.get('lang', data.get('language', Config.DEFAULT_LANG))
    if not isinstance(lang, str):
        return _error_response("'lang' must be a string", 400, request_id, param='lang')

    max_chunk_length = data.get('max_chunk_length', Config.DEFAULT_MAX_CHUNK_LENGTH)
    if not isinstance(max_chunk_length, int):
        return _error_response(
            "'max_chunk_length' must be an integer", 400, request_id, param='max_chunk_length'
        )

    silence_duration = data.get('silence_duration', Config.DEFAULT_SILENCE_DURATION)
    if not isinstance(silence_duration, (int, float)):
        return _error_response(
            "'silence_duration' must be a number", 400, request_id, param='silence_duration'
        )

    speed = data.get('speed', Config.DEFAULT_SPEED)
    if not isinstance(speed, (int, float)):
        return _error_response("'speed' must be a number", 400, request_id, param='speed')

    steps = data.get('steps', Config.DEFAULT_STEPS)
    if not isinstance(steps, int):
        return _error_response("'steps' must be an integer", 400, request_id, param='steps')

    if model_name:
        preset = Config.MODEL_PRESETS.get(model_name.lower())
        if preset and 'steps' not in data:
            steps = preset.get('steps', steps)

    tts = get_tts_service()

    # Validate voice first
    is_valid, msg = tts.validate_voice(voice)
    if not is_valid:
        available = [v['id'] for v in tts.list_voices()]
        return _error_response(
            f"Voice '{voice}' not found",
            400,
            request_id,
            param='voice',
            extra={
                'available_voices': available[:10],  # Limit to first 10
                'hint': 'Use /v1/voices to see all available voices',
            },
        )

    try:
        voice_state = tts.get_voice_state(voice)

        # Check if streaming should be used
        if isinstance(stream_request, bool):
            use_streaming = stream_request
        else:
            use_streaming = current_app.config.get('STREAM_DEFAULT', False)

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
