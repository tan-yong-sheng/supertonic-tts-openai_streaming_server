"""
Supertonic TTS service - model loading, voice management, and audio generation.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from app.config import Config
from app.logging_config import get_logger
from app.services.voice_manager import VoiceManager

logger = get_logger('tts')


class SupertonicTTSService:
    """
    Service class for Supertonic ONNX-based TTS operations.

    Phase 1 scaffold: interfaces and structure only. Model inference is
    implemented in Phase 2.
    """

    def __init__(self) -> None:
        self.model = None
        self.voice_cache: dict[str, dict] = {}
        self.voices_dir: str | None = None
        self._model_loaded = False
        self._voice_manager = VoiceManager()
        self._device = 'cpu'
        self._voice_style_names: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def sample_rate(self) -> int:
        if self.model is not None and hasattr(self.model, 'sample_rate'):
            return int(self.model.sample_rate)
        return Config.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    def load_model(self, model_path: str | None = None) -> None:
        """Load the Supertonic TTS pipeline."""
        from supertonic import TTS

        effective_path = model_path or Config.MODEL_PATH
        if effective_path:
            logger.info('Configured model path: %s', effective_path)

        auto_download = Config.AUTO_DOWNLOAD
        if os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1':
            auto_download = False

        logger.info('Loading Supertonic model: %s', Config.MODEL_NAME)
        t0 = time.time()

        self.model = TTS(
            model=Config.MODEL_NAME,
            model_dir=effective_path,
            auto_download=auto_download,
            intra_op_num_threads=Config.INTRA_OP_THREADS,
            inter_op_num_threads=Config.INTER_OP_THREADS,
        )

        self._model_loaded = True
        self._device = 'cpu'
        self._voice_manager.bind_tts(self.model)
        self._voice_style_names = list(getattr(self.model, 'voice_style_names', []))

        load_time = time.time() - t0
        logger.info('Model loaded in %.2fs. Sample rate: %s', load_time, self.sample_rate)

    def set_voices_dir(self, voices_dir: str | None) -> None:
        if voices_dir and os.path.isdir(voices_dir):
            self.voices_dir = voices_dir
            logger.info('Voices directory set to: %s', voices_dir)
        elif voices_dir:
            logger.warning('Voices directory not found: %s', voices_dir)
            self.voices_dir = None
        else:
            self.voices_dir = None

    def list_voices(self) -> list[dict]:
        if self._voice_style_names:
            builtin = self._voice_style_names
        else:
            builtin = Config.BUILTIN_VOICES
        voices = [{'id': v, 'name': v, 'type': 'builtin'} for v in builtin]
        voices.extend(self._list_custom_voices())
        return voices

    def validate_voice(self, voice_id_or_path: str) -> tuple[bool, str | None]:
        try:
            self._resolve_voice_path(voice_id_or_path)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def get_voice_state(self, voice_id_or_path: str) -> dict:
        if not self.is_loaded:
            raise RuntimeError('Model not loaded. Call load_model() first.')

        resolved_key = self._resolve_voice_path(voice_id_or_path)
        if resolved_key in self.voice_cache:
            logger.debug('Using cached voice state for: %s', resolved_key)
            return self.voice_cache[resolved_key]

        logger.info('Preparing voice state for: %s', resolved_key)
        t0 = time.time()
        state = self._voice_manager.get_or_create(resolved_key)
        self.voice_cache[resolved_key] = state
        logger.info('Voice state ready in %.2fs: %s', time.time() - t0, resolved_key)
        return state

    def generate_audio(
        self,
        voice_state: dict,
        text: str,
        speed: float | None = None,
        steps: int | None = None,
        lang: str | None = None,
        max_chunk_length: int | None = None,
        silence_duration: float | None = None,
    ) -> torch.Tensor:
        if not self.is_loaded:
            raise RuntimeError('Model not loaded. Call load_model() first.')
        use_speed = speed if speed is not None else Config.DEFAULT_SPEED
        use_steps = steps if steps is not None else Config.DEFAULT_STEPS
        use_lang = lang or Config.DEFAULT_LANG

        wav, _duration = self.model.synthesize(
            text=text,
            voice_style=voice_state,
            speed=use_speed,
            total_steps=use_steps,
            lang=use_lang,
            max_chunk_length=(
                max_chunk_length
                if max_chunk_length is not None
                else Config.DEFAULT_MAX_CHUNK_LENGTH
            ),
            silence_duration=(
                silence_duration
                if silence_duration is not None
                else Config.DEFAULT_SILENCE_DURATION
            ),
        )

        tensor = torch.from_numpy(np.asarray(wav)).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def generate_audio_stream(
        self,
        voice_state: dict,
        text: str,
        speed: float | None = None,
        steps: int | None = None,
        lang: str | None = None,
        max_chunk_length: int | None = None,
        silence_duration: float | None = None,
    ) -> Iterator[torch.Tensor]:
        if not self.is_loaded:
            raise RuntimeError('Model not loaded. Call load_model() first.')
        yield self.generate_audio(
            voice_state,
            text,
            speed=speed,
            steps=steps,
            lang=lang,
            max_chunk_length=max_chunk_length,
            silence_duration=silence_duration,
        )

    def _resolve_voice_path(self, voice_id_or_path: str) -> str:
        if voice_id_or_path.startswith(('http://', 'https://')):
            raise ValueError(
                f'URL scheme not allowed for security reasons: {voice_id_or_path[:50]}. '
                "Use 'hf://' or provide a local file path."
            )

        if voice_id_or_path.startswith('hf://'):
            return voice_id_or_path

        if voice_id_or_path in self._voice_style_names or voice_id_or_path in Config.BUILTIN_VOICES:
            return voice_id_or_path

        if self.voices_dir:
            candidate = Path(self.voices_dir) / voice_id_or_path
            if candidate.is_file() and candidate.suffix.lower() in Config.VOICE_EXTENSIONS:
                if candidate.suffix.lower() in Config.VOICE_AUDIO_EXTENSIONS:
                    if not Config.VOICE_EXTRACTOR_CMD:
                        raise ValueError(
                            'Audio voice files require SUPERTONIC_VOICE_EXTRACTOR_CMD.'
                        )
                return str(candidate)

            for ext in Config.VOICE_EXTENSIONS:
                with_ext = candidate.with_suffix(ext)
                if with_ext.is_file():
                    if with_ext.suffix.lower() in Config.VOICE_AUDIO_EXTENSIONS:
                        if not Config.VOICE_EXTRACTOR_CMD:
                            raise ValueError(
                                'Audio voice files require SUPERTONIC_VOICE_EXTRACTOR_CMD.'
                            )
                    return str(with_ext)

        if Path(voice_id_or_path).is_file():
            resolved = Path(voice_id_or_path).resolve()
            if resolved.suffix.lower() in Config.VOICE_AUDIO_EXTENSIONS:
                if not Config.VOICE_EXTRACTOR_CMD:
                    raise ValueError('Audio voice files require SUPERTONIC_VOICE_EXTRACTOR_CMD.')
                return str(resolved)
            if resolved.suffix.lower() not in Config.VOICE_STYLE_EXTENSIONS:
                raise ValueError('Custom voices must be JSON style files or audio prompts.')
            return str(resolved)

        raise ValueError(f"Voice '{voice_id_or_path}' not found")

    def _list_custom_voices(self) -> list[dict]:
        if not self.voices_dir:
            return []
        voices_dir = Path(self.voices_dir)
        if not voices_dir.is_dir():
            return []
        voices: list[dict] = []
        for path in voices_dir.iterdir():
            if path.is_file() and path.suffix.lower() in Config.VOICE_EXTENSIONS:
                vtype = 'custom_style' if path.suffix.lower() == '.json' else 'custom_audio'
                voices.append({'id': path.name, 'name': path.stem, 'type': vtype})
        return sorted(voices, key=lambda v: v['name'].lower())


_tts_service: SupertonicTTSService | None = None


def get_tts_service() -> SupertonicTTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = SupertonicTTSService()
    return _tts_service
