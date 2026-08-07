"""
Speaker Diarization
"""
import os
import logging
from pathlib import Path
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "processed_audio"
OUTPUT_DIR = PROJECT_ROOT / "output"
DIARIZATION_DIR = OUTPUT_DIR / "diarization"
STATS_DIR = OUTPUT_DIR / "diarization_stats"
LOG_DIR = OUTPUT_DIR / "logs"
DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "diarization.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "EXPECTED_SPEAKERS": 2,
    "MIN_SPEAKERS": 2,
    "MAX_SPEAKERS": 2,
}

load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found inside .env")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("=" * 60)
print("Module 3")
print("Speaker Diarization")
print("=" * 60)
print(f"\nUsing device : {DEVICE}")
logger.info(f"Using device : {DEVICE}")

print("\nLoading Community-1 Pipeline...")
logger.info("Loading Community-1")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN,
)
pipeline.to(DEVICE)
print("✓ Pipeline loaded successfully.\n")
logger.info("Pipeline loaded successfully.")

import soundfile as sf
def load_audio(audio_path: Path):
    print(f"\nReading : {audio_path.name}")
    waveform, sample_rate = sf.read(audio_path)
    waveform = torch.tensor(waveform).float()
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    waveform = waveform.unsqueeze(0)

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz but found {sample_rate}"
        )
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    print(f"Duration : {waveform.shape[1]/sample_rate:.2f} sec")
    print(f"Channels : {waveform.shape[0]}")
    print(f"Sample Rate: {sample_rate}")
    return waveform, sample_rate

def diarize_audio(
    waveform: torch.Tensor,
    sample_rate: int
):
    print("Running speaker diarization...")
    logger.info("Running diarization")
    diarization = pipeline(
        {
            "waveform": waveform,
            "sample_rate": sample_rate,
        },
        min_speakers=CONFIG["MIN_SPEAKERS"],
        max_speakers=CONFIG["MAX_SPEAKERS"],
    )
    print(" Diarization complete")
    logger.info("Diarization completed")
    annotation = diarization.speaker_diarization
    return annotation

def process_audio(audio_path: Path):
    waveform, sample_rate = load_audio(audio_path)
    annotation = diarize_audio(
        waveform,
        sample_rate,
    )
    segments, stats = extract_segments(annotation)
    overlap_count = detect_overlap(segments)
    segments = merge_segments(segments)
    stats = defaultdict(
        lambda:{
            "turns":0,
            "speaking_time":0
        }
    )
    for seg in segments:
        stats[seg["speaker"]]["turns"] += 1
        stats[seg["speaker"]]["speaking_time"] += seg["duration"]
    validate_diarization(
        stats,
        overlap_count,
    )
    return (
        segments,
        stats,
        overlap_count,
    )
from collections import defaultdict

def extract_segments(annotation):
    print("\nExtracting speaker turns...")
    segments = []
    stats = defaultdict(
        lambda: {
            "turns": 0,
            "speaking_time": 0.0,
        }
    )
    segment_id = 1

    for segment, _, speaker in annotation.itertracks(yield_label=True):
        duration = round(segment.end - segment.start, 3)
        segments.append(
            {
                "segment_id": segment_id,
                "speaker": speaker,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "duration": duration,
                "overlap": False,
            }
        )
        stats[speaker]["turns"] += 1
        stats[speaker]["speaking_time"] += duration
        segment_id += 1
    print(f"Detected {len(stats)} speakers")
    print(f"Detected {len(segments)} speech segments")
    return segments, stats

def detect_overlap(segments):
    overlap_count = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[j]["start"] >= segments[i]["end"]:
                break
            if segments[i]["speaker"] == segments[j]["speaker"]:
                continue
            segments[i]["overlap"] = True
            segments[j]["overlap"] = True
            overlap_count += 1
    return overlap_count
def merge_segments(
    segments,
    max_gap=0.40,
    min_duration=0.30
):
    if len(segments) == 0:
        return segments
    merged = []
    current = segments[0].copy()
    for nxt in segments[1:]:
        gap = nxt["start"] - current["end"]
        if (
            nxt["speaker"] == current["speaker"]
            and gap <= max_gap
        ):
            current["end"] = nxt["end"]
            current["duration"] = round(
                current["end"] - current["start"],
                3,
            )
            current["overlap"] = (
                current["overlap"]
                or nxt["overlap"]
            )
        else:
            if current["duration"] >= min_duration:
                merged.append(current)
            current = nxt.copy()
    if current["duration"] >= min_duration:
        merged.append(current)
    for i, seg in enumerate(merged, start=1):
        seg["segment_id"] = i
    return merged

def validate_diarization(stats, overlap_count):
    speakers = len(stats)
    print("\n" + "=" * 50)
    print("Validation")
    print("=" * 50)
    print(f"Expected Speakers : {CONFIG['EXPECTED_SPEAKERS']}")
    print(f"Detected Speakers : {speakers}")
    if speakers != CONFIG["EXPECTED_SPEAKERS"]:
        print("\n WARNING")
        print("Speaker clustering may have failed.")
        logger.warning(
            f"Expected {CONFIG['EXPECTED_SPEAKERS']} speakers "
            f"but found {speakers}"
        )
    print()
    for speaker in sorted(stats):
        print(speaker)
        print(
            f"  Turns         : {stats[speaker]['turns']}"
        )
        print(
            f"  Speaking Time : "
            f"{stats[speaker]['speaking_time']:.2f} sec"
        )
        print()
    print(f"Overlap Segments : {overlap_count}")
    print("=" * 50)

import json
def save_outputs(
    audio_path,
    segments,
    stats
):
    diarization_file = (
        DIARIZATION_DIR /
        f"{audio_path.stem}.json"
    )
    with open(
        diarization_file,
        "w",
        encoding="utf8"
    ) as f:
        json.dump(
            segments,
            f,
            indent=4
        )
    total = sum(
        x["speaking_time"]
        for x in stats.values()
    )
    summary = {
        "audio_file":audio_path.name,
        "speakers":{}
    }
    for speaker in sorted(stats):
        summary["speakers"][speaker]={
            "turns":
            stats[speaker]["turns"],
            "speaking_time":
            round(
                stats[speaker]["speaking_time"],
                2
            ),
            "conversation_share":
            round(
                stats[speaker]["speaking_time"]/total*100,
                2
            )
        }
    stats_file = (
        STATS_DIR /
        f"{audio_path.stem}_stats.json"
    )
    with open(
        stats_file,
        "w",
        encoding="utf8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

def main():
    files = sorted(
        INPUT_DIR.glob("*.wav")
    )
    success = 0
    failed = 0
    for audio in files:
        try:
            print("\n"+"="*60)
            segments,stats,overlap=process_audio(audio)
            save_outputs(
                audio,
                segments,
                stats
            )
            success+=1
            print(f" Saved {audio.name}")
        except Exception as e:
            failed+=1
            logger.exception(e)
            print(f" Failed {audio.name}")
            print(e)
    print("\n")
    print("="*60)
    print("Module 3 Completed")
    print("="*60)
    print(f"Successful : {success}")
    print(f"Failed : {failed}")

if __name__=="__main__":
    main()