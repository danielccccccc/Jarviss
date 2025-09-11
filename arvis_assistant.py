import os
import argparse
import tempfile
import wave
import sys
import sounddevice as sd
import whisper
import pyttsx3
import google.generativeai as genai

# -----------------------------
# Globals
# -----------------------------
whisper_model = None  # loaded at startup
tts_engine = pyttsx3.init()

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Audio Recording
# -----------------------------
def record_audio(device_index: int, samplerate: int = 16000, channels: int = 1, duration: int = 10) -> str:
    """Record audio from microphone into a temporary WAV file."""
    print("Listening for speech... (pause to stop)")

    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype="int16",
        device=device_index
    )
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav_path = f.name
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(samplerate)
            wf.writeframes(recording.tobytes())

    return wav_path

# -----------------------------
# Whisper Transcription
# -----------------------------
def load_whisper_model():
    """Load Whisper model from local cache if available."""
    print("Loading Whisper model...")
    model = whisper.load_model("base", download_root=os.path.expanduser("~/.cache/whisper"))
    return model

def transcribe_whisper_local(audio_path: str) -> str:
    """Run transcription using Whisper model."""
    print("Transcribing with local Whisper...")
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# -----------------------------
# Gemini Response
# -----------------------------
def ask_gemini(prompt: str) -> str:
    """Send user input to Gemini and return the response."""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# -----------------------------
# Text-to-Speech
# -----------------------------
def speak(text: str):
    """Speak text aloud using pyttsx3."""
    print(f"[Jarvis]: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# -----------------------------
# Wake Word Detection (Simulated)
# -----------------------------
def detect_wake_word() -> bool:
    """Simple simulated wake word detector."""
    try:
        user_input = input("Type 'Jarvis' to simulate wake word: ").strip().lower()
        return user_input == "jarvis"
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

# -----------------------------
# Main Assistant Logic
# -----------------------------
def main():
    global whisper_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="Microphone device index")
    args = parser.parse_args()

    print("\n------------------------------------------------------------")
    print("Jarvis is listening. Say the wake word: 'Jarvis'. Ctrl+C to exit.")
    print("------------------------------------------------------------")

    # Load Whisper once
    whisper_model = load_whisper_model()

    while True:
        if detect_wake_word():
            print("Wake word detected! 🔊")
            wav_path = record_audio(args.device)
            try:
                # Step 1: Transcribe speech
                user_text = transcribe_whisper_local(wav_path)
                print(f"You said: {user_text}")

                # Step 2: Get Gemini response
                reply = ask_gemini(user_text)
                print(f"[Jarvis Answer]: {reply}")

                # Step 3: Speak it
                speak(reply)
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    main()
