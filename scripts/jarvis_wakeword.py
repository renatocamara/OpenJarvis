#!/usr/bin/env python3

import json
import queue
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import sounddevice as sd
from openwakeword.model import Model


RATE = 16000
BLOCK = 1280
THRESHOLD = 0.50

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
API_MODEL = "qwen3.5:9b"

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENJARVIS_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
LISTENER_SCRIPT = REPO_ROOT / "scripts" / "jarvis_listen.py"
SPEAKER_SCRIPT = REPO_ROOT / "scripts" / "jarvis_speak.py"


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


def ask_jarvis(command):
    payload = {
        "model": API_MODEL,
        "messages": [
            {
                "role": "user",
                "content": command,
            }
        ],
        "stream": False,
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[API ERROR] HTTP {exc.code}")
        print(body)
        return None

    except urllib.error.URLError as exc:
        print(f"[API ERROR] Cannot reach OpenJarvis: {exc}")
        return None

    except TimeoutError:
        print("[API ERROR] OpenJarvis request timed out.")
        return None

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("[API ERROR] Unexpected response:")
        print(json.dumps(data, indent=2))
        return None

    if not answer:
        return None

    return answer.strip()



def speak_response(text):
    process = subprocess.run(
        [
            str(OPENJARVIS_PYTHON),
            str(SPEAKER_SCRIPT),
        ],
        cwd=str(REPO_ROOT),
        input=text,
        text=True,
    )

    if process.returncode != 0:
        print(f"[TTS ERROR] Speaker exited with code {process.returncode}")


def main():
    device_id = find_rdp_source()
    device_name = sd.query_devices(device_id)["name"]

    print("=== JARVIS ALWAYS-LISTENING SERVICE ===")
    print(f"Microphone : {device_id} - {device_name}")
    print(f"Sample rate: {RATE} Hz")
    print(f"Threshold  : {THRESHOLD}")
    print(f"API        : {API_URL}")
    print(f"Model      : {API_MODEL}")
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

            model.reset()

            transcript = listen_for_command()

            if not transcript:
                print()
                print("[NO COMMAND DETECTED]")
                print()
                continue

            print()
            print("====================================")
            print(f"COMMAND: {transcript}")
            print("====================================")
            print()

            print("[THINKING] Sending command to OpenJarvis...")

            answer = ask_jarvis(transcript)

            if answer:
                print()
                print("====================================")
                print("JARVIS RESPONSE:")
                print(answer)
                print("====================================")
                print()

                speak_response(answer)
            else:
                print()
                print("[NO RESPONSE FROM JARVIS]")

            print()

    except KeyboardInterrupt:
        print()
        print("Jarvis always-listening service stopped.")


if __name__ == "__main__":
    main()
