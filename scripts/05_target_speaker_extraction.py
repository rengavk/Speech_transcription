"""
Target Speaker Identification
-------
This automatically identify the target speaker from the speaker-wise transcription.
-------
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTION_DIR = (
    PROJECT_ROOT /
    "output" /
    "transcriptions"
)
OUTPUT_DIR = (
    PROJECT_ROOT /
    "output" /
    "target_speaker"
)
LOG_DIR = (
    PROJECT_ROOT /
    "output" /
    "logs"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)
logging.basicConfig(
    filename=LOG_DIR / "target_speaker.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG = {

    "AUTO_SELECT": True,
    "ALLOW_MANUAL_OVERRIDE": True,
    "WEIGHTS": {
        "speaking_time":0.50,
        "word_count":0.20,
        "turns":0.20,
        "avg_turn_duration":0.10
    }
}
def load_transcription(json_file):
    with open(
        json_file,
        "r",
        encoding="utf8"
    ) as f:
        return json.load(f)
    
def compute_statistics(transcription):
    stats = defaultdict(
        lambda:{
            "turns":0,
            "speaking_time":0,
            "word_count":0,
            "avg_turn_duration":0,
            "longest_turn":0
        }
    )

    for seg in transcription:
        speaker = seg["speaker"]
        duration = seg["duration"]
        text = seg["text"]
        words = len(text.split())
        stats[speaker]["turns"] += 1
        stats[speaker]["speaking_time"] += duration
        stats[speaker]["word_count"] += words
        stats[speaker]["longest_turn"] = max(
            stats[speaker]["longest_turn"],
            duration
        )

    for speaker in stats:
        stats[speaker]["avg_turn_duration"] = round(
            stats[speaker]["speaking_time"]
            /
            stats[speaker]["turns"], 3
        )
    return stats

def print_statistics(stats):
    print("\n")
    print("="*60)
    print("Speaker Statistics")
    print("="*60)
    for speaker in sorted(stats):
        print(f"\n{speaker}")
        print( f"Turns : {stats[speaker]['turns']}")
        print( f"Speaking Time : {stats[speaker]['speaking_time']:.2f}" )
        print(f"Words : {stats[speaker]['word_count']}")
        print(f"Average Turn : {stats[speaker]['avg_turn_duration']:.2f}" )
        print( f"Longest Turn : {stats[speaker]['longest_turn']:.2f}")

def normalize_statistics(stats):
    metrics = [
        "speaking_time",
        "word_count",
        "turns",
        "avg_turn_duration"
    ]
    normalized = {}
    for metric in metrics:
        values = [
            stats[s][metric]
            for s in stats
        ]
        maximum = max(values)
        if maximum == 0:maximum = 1
        for speaker in stats:
            normalized.setdefault(speaker, {})
            normalized[speaker][metric] = (
                stats[speaker][metric] / maximum
            )
    return normalized

def compute_scores( stats, normalized):
    scores = {}
    weights = CONFIG["WEIGHTS"]
    for speaker in stats:
        score = (
            normalized[speaker]["speaking_time"] * weights["speaking_time"]
            +
            normalized[speaker]["word_count"]* weights["word_count"]
            +
            normalized[speaker]["turns"]* weights["turns"]
            +
            normalized[speaker]["avg_turn_duration"] * weights["avg_turn_duration"]
        )
        scores[speaker] = round(score, 4)
    return scores

def compute_confidence(scores):
    ordered = sorted(
        scores.values(),
        reverse=True
    )
    if len(ordered) == 1:
        return 1.0
    confidence = ordered[0] - ordered[1]
    confidence = min(
        max(confidence, 0),
        1
    )
    return round(confidence, 3)

def auto_select_target(scores):
    target = max(
        scores,
        key=scores.get
    )
    confidence = compute_confidence(scores)
    return target, confidence

def manual_override(target,stats):
    if not CONFIG["ALLOW_MANUAL_OVERRIDE"]:
        return target
    print("\n")
    print("=" * 60)
    print("Automatic Target Speaker")
    print("=" * 60)
    print(f"Suggested : {target}")
    print("\nSpeakers")
    speakers = sorted(stats)
    for i, speaker in enumerate(speakers, start=1):
        print(f"{i}. {speaker}")
    choice = input(
        "\nPress ENTER to accept "
        "or type speaker number: "
    ).strip()
    if choice == "":
        return target
    try:
        index = int(choice) - 1
        if 0 <= index < len(speakers):
            return speakers[index]
    except Exception:
        pass
    print("Invalid selection. Using automatic choice.")
    return target

def print_decision(scores,target,confidence):
    print("\n")
    print("=" * 60)
    print("Target Speaker Decision")
    print("=" * 60)
    for speaker in sorted(scores):
        print(
            f"{speaker:<12}"
            f" Score : {scores[speaker]:.3f}"
        )
    print("\n")
    print(f"Selected Target : {target}")
    print(f"Confidence      : {confidence:.2%}")
    print("=" * 60)

def save_decision(
    audio_name,
    stats,
    scores,
    target,
    confidence,
    method="automatic"
):
    decision = {
        "audio_file": audio_name,
        "target_speaker": target,
        "selection_method": method,
        "confidence": confidence,
        "speaker_statistics": {},
        "speaker_scores": scores

    }

    for speaker in stats:
        decision["speaker_statistics"][speaker] = {
            "turns": stats[speaker]["turns"],
            "speaking_time": round(
                stats[speaker]["speaking_time"],
                2
            ),
            "word_count": stats[speaker]["word_count"],
            "avg_turn_duration": round(
                stats[speaker]["avg_turn_duration"],
                2
            ),
            "longest_turn": round(
                stats[speaker]["longest_turn"],
                2
            )
        }
    output_file = (
        OUTPUT_DIR /
        f"{Path(audio_name).stem}_decision.json"
    )
    with open(
        output_file,
        "w",
        encoding="utf8"
    ) as f:
        json.dump(
            decision,
            f,
            indent=4
        )

def save_target_transcript(
    audio_name,
    transcription,
    target
):
    target_segments = []
    transcript_text = []
    segment_no = 1
    for seg in transcription:
        if seg["speaker"] != target:
            continue
        new_seg = seg.copy()
        new_seg["segment_id"] = segment_no
        new_seg["speaker"] = "TARGET"
        target_segments.append(new_seg)
        transcript_text.append(seg["text"])
        segment_no += 1
    output = {
        "audio_file": audio_name,
        "target_speaker": target,
        "segments": target_segments,
        "full_transcript": " ".join(transcript_text)

    }
    output_file = (
        OUTPUT_DIR /
        f"{Path(audio_name).stem}_target.json"
    )
    with open(
        output_file,
        "w",
        encoding="utf8"
    ) as f:
        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

def process_file(json_file):
    print(f"\nProcessing {json_file.name}")
    transcription = load_transcription(json_file)
    stats = compute_statistics(transcription)
    print_statistics(stats)
    normalized = normalize_statistics(stats)
    scores = compute_scores(
        stats,
        normalized
    )
    target, confidence = auto_select_target(scores)
    method = "automatic"
    if CONFIG["ALLOW_MANUAL_OVERRIDE"]:
        chosen = manual_override(
            target,
            stats
        )
        if chosen != target:
            target = chosen
            method = "manual"
    print_decision(
        scores,
        target,
        confidence
    )
    save_decision(
        json_file.name,
        stats,
        scores,
        target,
        confidence,
        method
    )
    save_target_transcript(
        json_file.name,
        transcription,
        target
    )
    logger.info(
        f"{json_file.name} -> {target}"
    )

def main():
    print("="*60)
    print("Module 5")
    print("Target Speaker Identification")
    print("="*60)
    files = sorted(TRANSCRIPTION_DIR.glob("*.json"))
    success = 0
    failed = 0
    for file in files:
        try:
            process_file(file)
            success += 1
        except Exception as e:
            logger.exception(e)
            print(e)
            failed += 1

    print("\n")
    print("="*60)
    print("Module 5 Completed")
    print("="*60)
    print(f"Successful : {success}")
    print(f"Failed     : {failed}")

if __name__ == "__main__":
    main()