"""
Speaker-wise Verbatim Transcription
# Transcribing each diarized speaker segment using Faster-Whisper.
"""
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "processed_audio"
DIARIZATION_DIR = PROJECT_ROOT / "output" / "diarization"
OUTPUT_DIR = PROJECT_ROOT / "output" / "transcriptions"
LOG_DIR = PROJECT_ROOT / "output" / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "transcription.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
WHISPER_MODEL = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

if os.name == "nt":
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

def load_model():
    print("="*60)
    print("Module 4")
    print("Speaker-wise Verbatim Transcription")
    print("="*60)
    print("\nLoading Faster-Whisper...")
    logger.info("Loading Whisper model")
    model = WhisperModel(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    print("Whisper Loaded")
    logger.info("Whisper loaded")
    return model              

def load_audio(audio_path: Path):
    audio, sample_rate = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sample_rate

def load_diarization(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        diarization = json.load(f)
    diarization.sort(
        key=lambda x: x["start"]
    )
    return diarization

def extract_segment(audio, sample_rate, start, end,
    padding=0.20,
):
    start = max(
        0,
        start - padding,
    )
    end = min(
        len(audio)/sample_rate,
        end + padding,
    )
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment = audio[start_sample:end_sample]
    return segment   

def transcribe_segment(
    model,
    audio_segment,
    sample_rate,
):
    if len(audio_segment) < sample_rate * 0.30:
        return ""
    try:
        segments, info = model.transcribe(
            audio_segment,
            language="en",
            beam_size=5,
            best_of=5,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
        )
        text = " ".join(
            seg.text.strip()
            for seg in segments
        ).strip()
        if text == "":
            return "[unintelligible]"
        return text

    except Exception as e:
        logger.exception(e)
        return "[unintelligible]"

def process_audio_file(
    model,
    audio_path,
    diarization_path,
):
    print(f"\nProcessing {audio_path.name}")
    audio, sample_rate = load_audio(audio_path)
    diarization = load_diarization(diarization_path)
    transcription = []
    start = time.time()
    for segment in diarization:
        audio_chunk = extract_segment(
            audio,
            sample_rate,
            segment["start"],
            segment["end"],
        )
        text = transcribe_segment(
            model,
            audio_chunk,
            sample_rate,
        )
        transcription.append({
            "segment_id": segment["segment_id"],
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "duration": segment["duration"],
            "overlap": segment["overlap"],
            "text": text
        })
    elapsed = round(
        time.time() - start,
        2,
    )
    print(
        f"{len(transcription)} segments transcribed "
        f"({elapsed}s)"
    )
    logger.info(
        f"{audio_path.name} "
        f"{len(transcription)} segments "
        f"{elapsed}s"
    )
    return transcription

def save_transcription(
    audio_path,
    transcription,
):
    output_file = (
        OUTPUT_DIR /
        f"{audio_path.stem}.json"
    )
    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            transcription,
            f,
            indent=4,
            ensure_ascii=False,
        )

def main():
    model = load_model()
    audio_files = sorted(
        AUDIO_DIR.glob("*.wav")
    )
    success = 0
    failed = 0
    overall_start = time.time()
    for audio_file in audio_files:
        diarization_file = (DIARIZATION_DIR /f"{audio_file.stem}.json")
        if not diarization_file.exists():
            print( f"Missing diarization: "f"{audio_file.name}")
            failed += 1
            continue
        try:
            transcription = process_audio_file(model,audio_file,diarization_file,)
            save_transcription(audio_file,transcription,)
            success += 1
        except Exception as e:
            logger.exception(e)
            print(
                f" Failed {audio_file.name}"
            )
            failed += 1
    total = round(
        time.time() - overall_start,
        2,
    )
    print("task over")
    print(f"Successful : {success}")
    print(f"Failed     : {failed}")
    print(f"Elapsed    : {total} sec")

if __name__ == "__main__":
    main()