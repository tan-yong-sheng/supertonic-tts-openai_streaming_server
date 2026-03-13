"""
Voice state extraction and caching for Supertonic.
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
from pathlib import Path

from app.config import Config
from app.logging_config import get_logger

logger = get_logger('voice_manager')


class VoiceManager:
    """
    Phase 1 scaffold for voice state extraction.

    Supertonic uses style vectors (style_ttl/style_dp). Implementation is
    provided in Phase 2.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}
        self._tts = None

    def bind_tts(self, tts) -> None:
        self._tts = tts

    def get_or_create(self, voice_key: str):
        if voice_key in self._cache:
            return self._cache[voice_key]

        if self._tts is None:
            raise RuntimeError('Supertonic TTS is not initialized')

        path = Path(voice_key)
        suffix = path.suffix.lower()
        if suffix in Config.VOICE_STYLE_EXTENSIONS and path.is_file():
            logger.info('Loading voice style from JSON: %s', voice_key)
            state = self._tts.get_voice_style_from_path(path)
        elif suffix in Config.VOICE_AUDIO_EXTENSIONS and path.is_file():
            style_path = self._extract_style_from_audio(path)
            logger.info('Loading extracted voice style: %s', style_path)
            state = self._tts.get_voice_style_from_path(style_path)
        else:
            logger.info('Loading built-in voice style: %s', voice_key)
            state = self._tts.get_voice_style(voice_key)

        self._cache[voice_key] = state
        return state

    def _extract_style_from_audio(self, audio_path: Path) -> Path:
        if not Config.VOICE_EXTRACTOR_CMD:
            raise RuntimeError(
                'Audio voice files require SUPERTONIC_VOICE_EXTRACTOR_CMD to be configured.'
            )

        cache_dir = Path(Config.VOICE_STYLE_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = self._hash_file(audio_path)
        style_path = cache_dir / f'{audio_path.stem}-{fingerprint}.json'

        if style_path.is_file():
            return style_path

        cmd_template = Config.VOICE_EXTRACTOR_CMD
        formatted = cmd_template.format(input=str(audio_path), output=str(style_path))
        cmd = shlex.split(formatted)

        logger.info('Extracting voice style via command: %s', cmd)
        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=Config.VOICE_EXTRACTOR_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Voice style extraction timed out') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('Voice style extraction command failed') from exc

        if not style_path.is_file():
            raise RuntimeError('Voice style extractor did not create output JSON')

        return style_path

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()[:16]
