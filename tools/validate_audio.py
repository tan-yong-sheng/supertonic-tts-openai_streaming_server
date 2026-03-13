"""Validate audio outputs produced by the local Supertonic server."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def _print_header(path: Path) -> bytes:
    with path.open("rb") as f:
        head = f.read(12)
    print(f"header={head!r}")
    return head


def _validate_wav(path: Path) -> bool:
    data = path.read_bytes()
    if len(data) < 44:
        print("wav_error=too_short")
        return False
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        print("wav_error=missing_riff_wave")
        return False
    riff_size = struct.unpack("<I", data[4:8])[0]
    data_size = struct.unpack("<I", data[40:44])[0]
    expected_riff = len(data) - 8
    expected_data = len(data) - 44
    print(f"wav_riff_size={riff_size} wav_data_size={data_size}")
    print(f"wav_expected_riff={expected_riff} wav_expected_data={expected_data}")
    if riff_size != expected_riff or data_size != expected_data:
        print("wav_error=size_mismatch")
        return False
    return True


def _validate_mp3(path: Path) -> bool:
    head = path.read_bytes()[:3]
    if head == b"ID3" or (len(head) == 3 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return True
    print("mp3_error=missing_id3_or_frame_sync")
    return False


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 tools/validate_audio.py <path>")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"not_found={path}")
        return 2

    print(f"file={path} size={path.stat().st_size}")
    head = _print_header(path)

    if head.startswith(b"{"):
        try:
            print("json_error=detected")
            print(json.loads(path.read_text(errors="replace")))
        except Exception:
            print("json_error=unparseable")
        return 1

    if path.suffix.lower() == ".wav":
        ok = _validate_wav(path)
        print("wav_ok" if ok else "wav_bad")
        return 0 if ok else 1
    if path.suffix.lower() == ".mp3":
        ok = _validate_mp3(path)
        print("mp3_ok" if ok else "mp3_bad")
        return 0 if ok else 1

    print("unknown_format")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
