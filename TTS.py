# speak_sync_utils.py

import os
import subprocess
import base64
import json
import pyttsx3


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore"))
    return result.stdout.decode(errors="ignore")

def audio_file_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_json_transcript(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_facial_expression_and_animation(text: str):
    return {
        "facialExpression": "smile",
        "animation": "Angry"
    }

def synthesize_speech(text, output_path, voice_name="David"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    for voice in engine.getProperty('voices'):
        if voice_name.lower() in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.save_to_file(text, output_path)
    engine.runAndWait()

def generate_speak_and_sync_payload(text: str, message_id: str = "from_whisper"):
    base_path = f"audios/message_{message_id}"
    os.makedirs("audios", exist_ok=True)

    synthesize_speech(text, f"{base_path}.mp3")
    run_command(f"ffmpeg -y -i {base_path}.mp3 {base_path}.wav")
    run_command(rf"C:\Users\famil\Desktop\ghaith\Stage_2\TALAN\PItchy\Simulation_model\app\bin\Rhubarb\rhubarb.exe -f json -o {base_path}.json {base_path}.wav -r phonetic")

    audio_b64 = audio_file_to_base64(f"{base_path}.wav")
    lipsync_json = read_json_transcript(f"{base_path}.json")
    expression_data = get_facial_expression_and_animation(text)

    return {
        "text": text,
        "audio": audio_b64,
        "lipsync": lipsync_json,
        "facialExpression": expression_data["facialExpression"],
        "animation": expression_data["animation"]
    }
