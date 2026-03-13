"""
Configuration management for SupertonicTTS OpenAI Server.
Loads settings from environment variables with sensible defaults.
"""

import os
import sys
from pathlib import Path


def _int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name, '').strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name, '').strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _csv_env(name: str) -> list[str] | None:
    value = os.environ.get(name, '')
    if not value:
        return None
    tokens = [item.strip() for item in value.split(',') if item.strip()]
    return tokens or None


def get_base_path() -> Path:
    """Get the base path for the application, handling PyInstaller frozen state."""
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS)
        return Path(sys.executable).parent
    return Path(__file__).parent.parent


class Config:
    """Application configuration loaded from environment variables."""

    BASE_PATH = get_base_path()
    IS_FROZEN = getattr(sys, 'frozen', False)

    HOST = os.environ.get('SUPERTONIC_HOST', '0.0.0.0')
    PORT = int(os.environ.get('SUPERTONIC_PORT', '49112'))

    MODEL_NAME = os.environ.get('SUPERTONIC_MODEL_NAME', 'supertonic-2')
    MODEL_PATH = os.environ.get('SUPERTONIC_MODEL_PATH', None)
    AUTO_DOWNLOAD = os.environ.get('SUPERTONIC_AUTO_DOWNLOAD', 'true').lower() == 'true'
    DEFAULT_VOICE = os.environ.get('SUPERTONIC_DEFAULT_VOICE', 'M1')
    DEFAULT_SPEED = _float_env('SUPERTONIC_DEFAULT_SPEED', 1.0) or 1.0
    DEFAULT_STEPS = _int_env('SUPERTONIC_DIFFUSION_STEPS', 5) or 5
    DEFAULT_LANG = os.environ.get('SUPERTONIC_DEFAULT_LANG', 'en')
    SAMPLE_RATE = _int_env('SUPERTONIC_SAMPLE_RATE', 44100) or 44100
    DEFAULT_MAX_CHUNK_LENGTH = _int_env('SUPERTONIC_MAX_CHUNK_LENGTH', 300) or 300
    DEFAULT_SILENCE_DURATION = _float_env('SUPERTONIC_SILENCE_DURATION', 0.3) or 0.3
    VOICE_STYLE_CACHE_DIR = os.environ.get(
        'SUPERTONIC_VOICE_STYLE_CACHE_DIR', str(BASE_PATH / 'voice_styles')
    )
    VOICE_EXTRACTOR_CMD = os.environ.get('SUPERTONIC_VOICE_EXTRACTOR_CMD', '').strip() or None
    VOICE_EXTRACTOR_TIMEOUT = _int_env('SUPERTONIC_VOICE_EXTRACTOR_TIMEOUT', 120) or 120

    VOICES_DIR = os.environ.get('SUPERTONIC_VOICES_DIR', None)

    STREAM_DEFAULT = os.environ.get('SUPERTONIC_STREAM_DEFAULT', 'false').lower() == 'true'
    TEXT_PREPROCESS_DEFAULT = (
        os.environ.get('SUPERTONIC_TEXT_PREPROCESS_DEFAULT', 'false').lower() == 'true'
    )

    @staticmethod
    def _is_docker() -> bool:
        if os.path.exists('/.dockerenv'):
            return True
        try:
            with open('/proc/1/cgroup') as f:
                return any('docker' in line or 'containerd' in line for line in f)
        except (FileNotFoundError, PermissionError):
            return False

    IS_DOCKER = _is_docker.__func__()

    LOG_LEVEL = os.environ.get('SUPERTONIC_LOG_LEVEL', 'INFO')
    LOG_DIR = os.environ.get('SUPERTONIC_LOG_DIR', str(BASE_PATH / 'logs'))
    LOG_FILE = os.environ.get('SUPERTONIC_LOG_FILE', 'supertonic_tts.log')
    LOG_MAX_BYTES = int(os.environ.get('SUPERTONIC_LOG_MAX_BYTES', str(10 * 1024 * 1024)))
    LOG_BACKUP_COUNT = int(os.environ.get('SUPERTONIC_LOG_BACKUP_COUNT', '5'))
    COLDSTART_LOG = os.environ.get('SUPERTONIC_COLDSTART_LOG', 'false').lower() == 'true'
    REQUEST_TIMING_LOG = (
        os.environ.get('SUPERTONIC_REQUEST_TIMING_LOG', 'false').lower() == 'true'
    )
    REQUEST_TIMING_LOG_JSON = (
        os.environ.get('SUPERTONIC_REQUEST_TIMING_LOG_JSON', 'false').lower() == 'true'
    )
    TTFA_LOG = os.environ.get('SUPERTONIC_TTFA_LOG', 'false').lower() == 'true'
    UI_ENABLED = os.environ.get('SUPERTONIC_UI_ENABLED', 'true').lower() == 'true'
    MAX_INPUT_CHARS = _int_env('SUPERTONIC_MAX_INPUT_CHARS', 4096) or 4096
    REQUEST_ID_HEADER = os.environ.get('SUPERTONIC_REQUEST_ID_HEADER', 'X-Request-ID')

    TORCH_NUM_THREADS = _int_env('SUPERTONIC_TORCH_THREADS')
    TORCH_NUM_INTEROP_THREADS = _int_env('SUPERTONIC_TORCH_INTEROP_THREADS')
    INTRA_OP_THREADS = _int_env('SUPERTONIC_INTRA_OP_THREADS')
    INTER_OP_THREADS = _int_env('SUPERTONIC_INTER_OP_THREADS')

    AUTHENTICATION_ALLOWED_TOKENS = _csv_env('AUTHENTICATION_ALLOWED_TOKENS')

    CHUNK_TARGET_CHARS = _int_env('SUPERTONIC_CHUNK_TARGET_CHARS', 100) or 100
    CHUNK_MAX_CHARS = _int_env(
        'SUPERTONIC_CHUNK_MAX_CHARS', _int_env('SUPERTONIC_CHUNK_CHARS', 150) or 150
    ) or 150
    CHUNK_CHARS_ALLOW_OVERRIDE = (
        os.environ.get('SUPERTONIC_CHUNK_CHARS_ALLOW_OVERRIDE', 'false').lower() == 'true'
    )

    @classmethod
    def is_auth_enabled(cls) -> bool:
        return bool(cls.AUTHENTICATION_ALLOWED_TOKENS)

    @classmethod
    def is_valid_token(cls, token: str) -> bool:
        if not cls.AUTHENTICATION_ALLOWED_TOKENS:
            return True
        return token in cls.AUTHENTICATION_ALLOWED_TOKENS

    BUILTIN_VOICES = [
        'M1',
        'M2',
        'M3',
        'M4',
        'M5',
        'F1',
        'F2',
        'F3',
        'F4',
        'F5',
    ]

    MODEL_PRESETS = {
        'tts-1': {'steps': 5},
        'tts-1-hd': {'steps': 10},
        'supertonic-tts-1': {'steps': 5},
        'supertonic-tts-1-hd': {'steps': 10},
    }

    VOICE_STYLE_EXTENSIONS = ('.json',)
    VOICE_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg')
    VOICE_EXTENSIONS = VOICE_STYLE_EXTENSIONS + VOICE_AUDIO_EXTENSIONS

    @classmethod
    def get_bundle_paths(cls) -> tuple[str | None, str | None]:
        if cls.IS_FROZEN:
            voices_dir = cls.BASE_PATH / 'voices'
            model_path = cls.BASE_PATH / 'models'
            return (
                str(voices_dir) if voices_dir.is_dir() else None,
                str(model_path) if model_path.is_dir() else None,
            )
        return None, None

    @classmethod
    def get_template_folder(cls) -> str:
        return str(cls.BASE_PATH / 'templates')

    @classmethod
    def get_static_folder(cls) -> str:
        return str(cls.BASE_PATH / 'static')
