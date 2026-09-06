#!/usr/bin/env python3

import queue
import time

import sounddevice as sd
from openwakeword.model import Model


RATE = 16000
BLOCK = 1280
THRESHOLD = 0.50
COOLDOWN_SECONDS = 2.0


def find_rdp_source():
    for index, device in enumerate(sd.query_devices()):
        if (
            device["max_input_channels"] > 0
            and "RDPSource" in device["name"]
        ):
            return index

    raise RuntimeError("RDPSource microphone not found")


def main():
    device_id = find_rdp_source()
    device_name = sd.query_devices(device_id)["name"]

    print("=== JARVIS WAKE WORD SERVICE ===")
    print(f"Microphone : {device_id} - {device_name}")
    print(f"Sample rate: {RATE} Hz")
    print(f"Threshold  : {THRESHOLD}")
    print()

    print("Loading Hey Jarvis model...")

    model = Model(
        wakeword_models=["hey_jarvis"],
        inference_framework="tflite",
    )

    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")

        audio_queue.put(indata[:, 0].copy())

    print()
    print('[STANDBY] Waiting for "Hey Jarvis"...')
    print("Ctrl+C to stop.")
    print()

    last_detection = 0.0

    try:
        with sd.InputStream(
            device=device_id,
            samplerate=RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK,
            callback=callback,
        ):
            while True:
                audio = audio_queue.get()

                prediction = model.predict(audio)
                score = float(prediction.get("hey_jarvis", 0.0))

                now = time.time()

                if (
                    score >= THRESHOLD
                    and now - last_detection >= COOLDOWN_SECONDS
                ):
                    last_detection = now

                    print()
                    print("====================================")
                    print(">>> JARVIS ACTIVATED <<<")
                    print(f"Score: {score:.3f}")
                    print("====================================")
                    print()

                    model.reset()

                    print('[STANDBY] Waiting for "Hey Jarvis"...')

    except KeyboardInterrupt:
        print()
        print("Wake word service stopped.")


if __name__ == "__main__":
    main()
