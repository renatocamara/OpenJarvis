#!/usr/bin/env python3

import json
import queue
import subprocess
import time
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


def start_stt_process():
    print("[STT] Starting persistent speech recognition engine...")

    process = subprocess.Popen(
        [
            str(OPENJARVIS_PYTHON),
            str(LISTENER_SCRIPT),
        ],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    while True:
        line = process.stdout.readline()

        if not line:
            if process.poll() is not None:
                raise RuntimeError(
                    f"STT process exited with code {process.returncode}"
                )
            continue

        line = line.strip()

        if line:
            print(line)

        if line.startswith("JARVIS_STT_READY="):
            break

    print("[STT] Speech recognition engine ready.")
    print()

    return process


def stop_stt_process(process):
    if process is None:
        return

    if process.poll() is None:
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def listen_for_command(process):
    print()
    print("[ACTIVATED] Jarvis is listening...")
    print()

    if process.poll() is not None:
        print("[STT ERROR] Speech recognition engine is not running.")
        return None

    assert process.stdin is not None
    assert process.stdout is not None

    request = json.dumps({"action": "listen"})

    process.stdin.write(request + "\n")
    process.stdin.flush()

    transcript = None

    while True:
        line = process.stdout.readline()

        if not line:
            if process.poll() is not None:
                print(
                    f"[STT ERROR] Speech recognition engine exited "
                    f"with code {process.returncode}"
                )
                return None
            continue

        print(line, end="")

        stripped = line.strip()

        if stripped.startswith("JARVIS_TRANSCRIPT="):
            transcript = stripped.split("=", 1)[1].strip()

        elif stripped.startswith("JARVIS_STT_DONE="):
            return transcript

        elif stripped.startswith("JARVIS_STT_ERROR="):
            print(f"[STT ERROR] {stripped.split('=', 1)[1]}")
            return None


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
        api_start = time.perf_counter()

        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        api_elapsed = time.perf_counter() - api_start
        print(f"JARVIS_API_TIME={api_elapsed:.2f}", flush=True)

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



def start_tts_process():
    print("[TTS] Starting persistent voice engine...")

    process = subprocess.Popen(
        [
            str(OPENJARVIS_PYTHON),
            str(SPEAKER_SCRIPT),
        ],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    while True:
        line = process.stdout.readline()

        if not line:
            if process.poll() is not None:
                raise RuntimeError(
                    f"TTS process exited with code {process.returncode}"
                )
            continue

        line = line.strip()

        if line:
            print(line)

        if line.startswith("JARVIS_TTS_READY="):
            break

    print("[TTS] Voice engine ready.")
    print()

    return process


def stop_tts_process(process):
    if process is None:
        return

    if process.poll() is None:
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def speak_response(process, text):
    if process.poll() is not None:
        print("[TTS ERROR] Voice engine is not running.")
        return

    assert process.stdin is not None
    assert process.stdout is not None

    print("[SPEAKING]")

    request = json.dumps(
        {"text": text},
        ensure_ascii=False,
    )

    process.stdin.write(request + "\n")
    process.stdin.flush()

    while True:
        line = process.stdout.readline()

        if not line:
            if process.poll() is not None:
                print(
                    f"[TTS ERROR] Voice engine exited "
                    f"with code {process.returncode}"
                )
                return
            continue

        line = line.strip()

        if line.startswith("JARVIS_TTS_SYNTH="):
            seconds = line.split("=", 1)[1]
            print(f"[TTS] Synthesized in {seconds}s")

        elif line.startswith("JARVIS_TTS_DONE="):
            seconds = line.split("=", 1)[1]
            print(f"[TTS] Speech completed in {seconds}s")
            return

        elif line.startswith("JARVIS_TTS_ERROR="):
            print(f"[TTS ERROR] {line.split('=', 1)[1]}")
            return

        elif line:
            print(line)


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
    stt_process = start_stt_process()
    tts_process = start_tts_process()

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

            transcript = listen_for_command(stt_process)

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

                speak_response(tts_process, answer)
            else:
                print()
                print("[NO RESPONSE FROM JARVIS]")

            print()

    except KeyboardInterrupt:
        print()
        print("Jarvis always-listening service stopped.")

    finally:
        stop_stt_process(stt_process)
        stop_tts_process(tts_process)


if __name__ == "__main__":
    main()
