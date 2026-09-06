#!/usr/bin/env python3

import queue
import subprocess
from pathlib import Path

import sounddevice as sd
from openwakeword.model import Model


RATE = 16000
BLOCK = 1280
THRESHOLD = 0.50

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENJARVIS_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
LISTENER_SCRIPT = REPO_ROOT / "scripts" / "jarvis_listen.py"


def find_rdp_source():
    for index, device in enumerate(sd.query_devices()):
        if (
            device["max_input_channels"] > 0
            and "RDPSource" in device["name"]
        ):
            return index

    raise RuntimeError("RDPSource microphone not found")


def wait_for_wake(model, device_id):
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")

        audio_queue.put(indata[:, 0].copy())

    print('[STANDBY] Waiting for "Hey Jarvis"...')

    # Importante:
    # este stream existe SOMENTE enquanto esperamos o wake word.
    # Ao retornar desta função, o microfone é liberado.
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

            if score >= THRESHOLD:
                return score


def listen_for_command():
    print()
    print("[ACTIVATED] Jarvis is listening...")
    print()

    process = subprocess.Popen(
        [
            str(OPENJARVIS_PYTHON),
            str(LISTENER_SCRIPT),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    transcript = None

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

        if line.startswith("JARVIS_TRANSCRIPT="):
            transcript = line.split("=", 1)[1].strip()

    return_code = process.wait()

    if return_code != 0:
        print()
        print(f"[ERROR] Listener exited with code {return_code}")
        return None

    return transcript


def main():
    device_id = find_rdp_source()
    device_name = sd.query_devices(device_id)["name"]

    print("=== JARVIS ALWAYS-LISTENING SERVICE ===")
    print(f"Microphone : {device_id} - {device_name}")
    print(f"Sample rate: {RATE} Hz")
    print(f"Threshold  : {THRESHOLD}")
    print()

    print("Loading Hey Jarvis model...")

    model = Model(
        wakeword_models=["hey_jarvis"],
        inference_framework="tflite",
    )

    print()
    print("Ctrl+C to stop.")
    print()

    try:
        while True:
            score = wait_for_wake(model, device_id)

            print()
            print("====================================")
            print(">>> JARVIS ACTIVATED <<<")
            print(f"Wake score: {score:.3f}")
            print("====================================")

            # Limpa o histórico do detector antes da próxima rodada.
            model.reset()

            # Neste momento wait_for_wake() já fechou o InputStream,
            # portanto jarvis_listen.py pode usar o microfone.
            transcript = listen_for_command()

            if transcript:
                print()
                print("====================================")
                print(f"COMMAND: {transcript}")
                print("====================================")
            else:
                print()
                print("[NO COMMAND DETECTED]")

            print()

    except KeyboardInterrupt:
        print()
        print("Jarvis always-listening service stopped.")


if __name__ == "__main__":
    main()
