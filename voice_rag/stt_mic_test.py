import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

DURATION_SEC = 6          # speak for 6 seconds
SAMPLE_RATE = 16000       # Whisper likes 16k
CHANNELS = 1
WAV_PATH = "mic_input.wav"

# Whisper model: "tiny", "base", "small", "medium"
MODEL_SIZE = "small"      # start with small; medium is slower/heavier


def record_audio() -> None:
    """Record microphone audio for a fixed duration and save to a wav file."""
    print(f"\nRecording for {DURATION_SEC} seconds... Speak Marathi now.")
    audio = sd.rec(
        int(DURATION_SEC * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    sf.write(WAV_PATH, audio, SAMPLE_RATE)
    print(f"Saved: {WAV_PATH}")


def transcribe_audio() -> None:
    """Run Whisper on the recorded clip and print the transcript."""
    print("Loading Whisper model (first run will download weights)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    print("Transcribing...")
    segments, info = model.transcribe(
        WAV_PATH,
        language="mr",          # force Marathi; use None for auto
        vad_filter=True,
    )

    text_parts = [seg.text for seg in segments]
    final_text = " ".join(text_parts).strip()
    print("\n==============================")
    print("Detected language:", info.language, "| Prob:", round(info.language_probability, 3))
    print("TRANSCRIPT (Marathi):")
    print(final_text if final_text else "[EMPTY]")
    print("==============================\n")


if __name__ == "__main__":
    try:
        record_audio()
        transcribe_audio()
    except Exception as e:
        print("\nERROR:", e)
        print(
            "\nIf this is a microphone/device error, run `python mic_devices.py` "
            "or record via Windows Voice Recorder as a fallback."
        )
        sys.exit(1)
