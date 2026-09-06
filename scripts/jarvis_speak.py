#!/usr/bin/env python3

import json
import sys
import time

from openjarvis.speech.kokoro_tts import KokoroTTSBackend
from openjarvis.speech.voice_io import play_wav


VOICE = "am_adam"


def main():
    print("[TTS] Loading Kokoro...", flush=True)

    tts = KokoroTTSBackend(device="cpu")

    # Warm-up: carrega modelo e pipeline uma única vez.
    start = time.perf_counter()

    tts.synthesize(
        "Ready.",
        voice_id=VOICE,
        speed=1.0,
        output_format="wav",
    )

    elapsed = time.perf_counter() - start
    print(f"JARVIS_TTS_READY={elapsed:.2f}", flush=True)

    try:
        for line in sys.stdin:
            line = line.strip()

            if not line:
                continue

            try:
                request = json.loads(line)
                text = str(request.get("text", "")).strip()

                if not text:
                    print("JARVIS_TTS_DONE=0.00", flush=True)
                    continue

                total_start = time.perf_counter()
                synth_start = time.perf_counter()

                result = tts.synthesize(
                    text,
                    voice_id=VOICE,
                    speed=1.0,
                    output_format="wav",
                )

                synth_elapsed = time.perf_counter() - synth_start

                print(
                    f"JARVIS_TTS_SYNTH={synth_elapsed:.2f}",
                    flush=True,
                )

                if result.audio:
                    play_wav(result.audio)

                total_elapsed = time.perf_counter() - total_start

                print(
                    f"JARVIS_TTS_DONE={total_elapsed:.2f}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"JARVIS_TTS_ERROR={type(exc).__name__}: {exc}",
                    flush=True,
                )

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
