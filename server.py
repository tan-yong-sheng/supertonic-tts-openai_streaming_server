#!/usr/bin/env python3
"""
SupertonicTTS OpenAI-Compatible Server

OpenAI-compatible TTS API server using the Supertonic backend.
Supports streaming, custom voices, and CPU-only deployment.

Usage:
    python server.py [OPTIONS]

    SUPERTONIC_PORT=8080 python server.py
"""

import argparse
import os
import sys

from app import create_app, init_tts_service
from app.config import Config
from app.logging_config import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SupertonicTTS OpenAI-Compatible Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with defaults
    python server.py

    # Custom port and voices directory
    python server.py --port 8080 --voices-dir ./my_voices

    # Enable streaming by default
    python server.py --stream

Environment Variables:
    SUPERTONIC_HOST            Server host (default: 0.0.0.0)
    SUPERTONIC_PORT            Server port (default: 49112)
    SUPERTONIC_MODEL_PATH      Path to model directory
    SUPERTONIC_VOICES_DIR      Path to voices directory
    SUPERTONIC_STREAM_DEFAULT  Enable streaming by default
    SUPERTONIC_TEXT_PREPROCESS_DEFAULT Enable text preprocessing by default
    SUPERTONIC_LOG_DIR         Log directory path
        """,
    )

    parser.add_argument(
        '--host', type=str, default=Config.HOST, help=f'Host to bind to (default: {Config.HOST})'
    )
    parser.add_argument(
        '--port', type=int, default=Config.PORT, help=f'Port to listen on (default: {Config.PORT})'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=Config.MODEL_PATH,
        dest='model_path',
        help='Path to model directory or variant name',
    )
    parser.add_argument(
        '--voices-dir',
        type=str,
        default=Config.VOICES_DIR,
        dest='voices_dir',
        help='Directory containing voice files',
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        default=Config.STREAM_DEFAULT,
        help='Enable streaming by default for all requests',
    )
    parser.add_argument(
        '--text-preprocess',
        action='store_true',
        default=Config.TEXT_PREPROCESS_DEFAULT,
        help='Enable text preprocessing for all requests',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=Config.LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        dest='log_level',
        help='Logging level',
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    os.environ.setdefault('SUPERTONIC_LOG_LEVEL', args.log_level)

    app = create_app(
        {'STREAM_DEFAULT': args.stream, 'TEXT_PREPROCESS_DEFAULT': args.text_preprocess}
    )

    logger = get_logger()

    try:
        init_tts_service(model_path=args.model_path, voices_dir=args.voices_dir)
    except Exception as exc:
        logger.error('Failed to initialize TTS service: %s', exc)
        sys.exit(1)

    try:
        from waitress import serve

        logger.info('Starting SupertonicTTS server on http://%s:%s', args.host, args.port)
        logger.info('Press Ctrl+C to stop')

        serve(app, host=args.host, port=args.port, threads=4, url_scheme='http')

    except ImportError:
        logger.warning('Waitress not installed, falling back to Flask dev server')
        logger.warning('Install waitress for production: pip install waitress')
        app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
