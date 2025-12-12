import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

from voice_rag.rag.qa import answer_question_marathi

# -------------------
# Audio recording cfg
# -------------------
DURATION_SEC = 6
SAMPLE_RATE = 16000
CHANNELS = 1
WAV_PATH = "mic_input.wav"

# Whisper model sizes: tiny/base/small/medium
WHISPER_SIZE = "small"

def record_audio():
    print(f"\n🎙️ Recording for {DURATION_SEC} seconds... Speak now (Marathi/English).")
    audio = sd.rec(
        int(DURATION_SEC * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    sf.write(WAV_PATH, audio, SAMPLE_RATE)
    print(f"✅ Saved audio: {WAV_PATH}")

def stt_transcribe() -> str:
    print("🧠 Loading Whisper (first run downloads weights)...")
    model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    print("🔎 Transcribing...")
    segments, info = model.transcribe(
        WAV_PATH,
        language=None,      # auto-detect, supports Marathi+English
        vad_filter=True,
    )

    text = " ".join([s.text for s in segments]).strip()

    print("\n------------------------------")
    print(f"Detected language: {info.language} (prob={info.language_probability:.3f})")
    print("Transcript:")
    print(text if text else "[EMPTY]")
    print("------------------------------\n")

    return text

def main():
    print("\n=== Voice → STT → RAG → Marathi Answer (Terminal) ===")
    print("Pre-req: Knowledge base must already be built (Chroma store exists).")
    print("Tip: Ask questions that are answerable from your uploaded PDFs.\n")

    while True:
        user = input("Press ENTER to speak (or type 'q' to quit): ").strip().lower()
        if user == "q":
            print("Bye.")
            break

        try:
            record_audio()
            query_text = stt_transcribe()

            if not query_text:
                print("⚠️ No speech detected. Try again.\n")
                continue

            print("🧩 Running RAG (answer will be in Marathi)...")
            answer = answer_question_marathi(query_text)

            print("\n✅ Marathi Answer:")
            print(answer)
            print("\n============================================\n")

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print("\n❌ ERROR:", e)
            print("If this is KB-related, ensure you built the vector store from PDFs first.\n")
            break

if __name__ == "__main__":
    main()
