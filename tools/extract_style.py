#!/usr/bin/env python3
"""
Extract a Supertonic voice style JSON from an audio prompt.

This script is a best-effort wrapper that attempts to use any voice-style
extraction API exposed by the installed `supertonic` package.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np


def _serialize(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, 'dict'):
        return obj.dict()
    if hasattr(obj, '__dict__'):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    return str(obj)


def _call_save_style(tts, style, output: Path) -> bool:
    if not hasattr(tts, 'save_voice_style'):
        return False
    save_fn = getattr(tts, 'save_voice_style')
    try:
        save_fn(style, output)
        return True
    except Exception:
        pass
    try:
        save_fn(voice_style=style, path=output)
        return True
    except Exception:
        return False


def _extract_style(tts, audio_path: Path):
    candidates = [
        'extract_voice_style',
        'get_voice_style_from_audio',
        'get_voice_style_from_file',
        'get_voice_style_from_path',
    ]
    for name in candidates:
        if not hasattr(tts, name):
            continue
        fn = getattr(tts, name)
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) == 1:
                return fn(audio_path)
            return fn(path=audio_path)
        except Exception:
            continue
    raise RuntimeError(
        'No voice-style extraction API found in supertonic. '
        'Provide your own extractor or upgrade the supertonic package.'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract Supertonic voice style JSON')
    parser.add_argument('--input', required=True, help='Path to input audio file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    audio_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not audio_path.is_file():
        raise SystemExit(f'Input file not found: {audio_path}')

    from supertonic import TTS

    model_name = os.environ.get('SUPERTONIC_MODEL_NAME', 'supertonic-2')
    model_path = os.environ.get('SUPERTONIC_MODEL_PATH')
    auto_download = os.environ.get('SUPERTONIC_AUTO_DOWNLOAD', 'true').lower() == 'true'

    tts = TTS(
        model=model_name,
        model_dir=model_path,
        auto_download=auto_download,
        intra_op_num_threads=os.environ.get('SUPERTONIC_INTRA_OP_THREADS'),
        inter_op_num_threads=os.environ.get('SUPERTONIC_INTER_OP_THREADS'),
    )

    style = _extract_style(tts, audio_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _call_save_style(tts, style, output_path):
        return

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(_serialize(style), f)


if __name__ == '__main__':
    main()
