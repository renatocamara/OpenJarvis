#!/usr/bin/env python3

from openjarvis.speech.voice_io import record_until_silence
from openjarvis.speech.faster_whisper import FasterWhisperBackend


def main():
    print("[LISTENING] Fale agora...", flush=True)

    audio = record_until_silence(
        sample_rate=16000,
        silence_seconds=1.5,
        startup_silence_seconds=5.0,
        max_seconds=15.0,
    )

    print("[TRANSCRIBING] Processando...", flush=True)

    stt = FasterWhisperBackend(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    result = stt.transcribe(audio)

    print()
    print("=== TRANSCRIPTION ===")
    print("Text:", result.text)
    print("Language:", result.language)
    print("Confidence:", result.confidence)

    # Esta linha será usada depois pelo wake-word service.
    print(f"JARVIS_TRANSCRIPT={result.text}")


if __name__ == "__main__":
    main()
