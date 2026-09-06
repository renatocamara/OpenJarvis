#!/usr/bin/env python3

import json
import sys
import time

from openjarvis.speech.voice_io import record_until_silence
from openjarvis.speech.faster_whisper import FasterWhisperBackend


def main():
    print("[STT] Loading Faster-Whisper...", flush=True)

    stt = FasterWhisperBackend(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    start = time.perf_counter()

    if not stt.health():
        raise RuntimeError(
            f"Could not load Faster-Whisper: {stt.last_error()}"
        )

    elapsed = time.perf_counter() - start
    print(f"JARVIS_STT_READY={elapsed:.2f}", flush=True)

    try:
        for line in sys.stdin:
            line = line.strip()

            if not line:
                continue

            try:
                request = json.loads(line)

                if request.get("action") != "listen":
                    print(
                        "JARVIS_STT_ERROR=Unknown action",
                        flush=True,
                    )
                    continue

                total_start = time.perf_counter()

                print("[LISTENING] Fale agora...", flush=True)

                record_start = time.perf_counter()

                audio = record_until_silence(
                    sample_rate=16000,
                    silence_seconds=1.5,
                    startup_silence_seconds=5.0,
                    max_seconds=15.0,
                )

                record_elapsed = time.perf_counter() - record_start

                print("[TRANSCRIBING] Processando...", flush=True)

                transcribe_start = time.perf_counter()

                result = stt.transcribe(audio)

                transcribe_elapsed = (
                    time.perf_counter() - transcribe_start
                )

                print()
                print("=== TRANSCRIPTION ===")
                print("Text:", result.text)
                print("Language:", result.language)
                print("Confidence:", result.confidence)

                print(
                    f"JARVIS_TRANSCRIPT={result.text}",
                    flush=True,
                )

                print(
                    f"JARVIS_STT_RECORD={record_elapsed:.2f}",
                    flush=True,
                )

                print(
                    f"JARVIS_STT_TRANSCRIBE={transcribe_elapsed:.2f}",
                    flush=True,
                )

                total_elapsed = time.perf_counter() - total_start

                print(
                    f"JARVIS_STT_DONE={total_elapsed:.2f}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"JARVIS_STT_ERROR={type(exc).__name__}: {exc}",
                    flush=True,
                )

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
