#!/usr/bin/env python3

import sys

from openjarvis.speech.kokoro_tts import KokoroTTSBackend
from openjarvis.speech.voice_io import play_wav


VOICE = "am_adam"


def main():
    text = sys.stdin.read().strip()

    if not text:
        return

    print("[SPEAKING]", flush=True)

    tts = KokoroTTSBackend(device="cpu")

    result = tts.synthesize(
        text,
        voice_id=VOICE,
        speed=1.0,
        output_format="wav",
    )

    if result.audio:
        play_wav(result.audio)


if __name__ == "__main__":
    main()
