import json
from pathlib import Path

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering

# ======================================================
# Paths
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = PROJECT_ROOT / "data" / "processed_audio"
TRANSCRIPT_DIR = PROJECT_ROOT / "output" / "transcripts"

OUTPUT_DIR = PROJECT_ROOT / "output" / "speaker_segments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# Load Speaker Encoder
# ======================================================

print("=" * 60)
print("Loading Resemblyzer Voice Encoder...")
print("=" * 60)

encoder = VoiceEncoder()

print("Voice Encoder Loaded\n")

# ======================================================
# Process Every Audio File
# ======================================================

audio_files = sorted(AUDIO_DIR.glob("*.wav"))

for audio_path in audio_files:

    print(f"Processing {audio_path.name}")

    transcript_file = TRANSCRIPT_DIR / f"{audio_path.stem}.json"

    if not transcript_file.exists():
        print("Transcript missing - Skipping")
        continue

    # --------------------------------------------------

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    wav, sr = sf.read(audio_path)

    embeddings = []
    valid_segments = []

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    for seg in transcript:

        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)

        clip = wav[start:end]

        # Ignore tiny clips
        if len(clip) < sr:
            continue

        processed = preprocess_wav(clip, source_sr=sr)

        embedding = encoder.embed_utterance(processed)

        embeddings.append(embedding)
        valid_segments.append(seg)

    if len(embeddings) == 0:
        print("No usable speech segments.\n")
        continue

    embeddings = np.array(embeddings)

    # --------------------------------------------------
    # Estimate Number of Speakers
    # --------------------------------------------------

    max_speakers = min(5, len(embeddings))

    if max_speakers < 2:
        labels = np.zeros(len(embeddings), dtype=int)

    else:

        clustering = AgglomerativeClustering(
            n_clusters=2
        )

        labels = clustering.fit_predict(embeddings)

    # --------------------------------------------------
    # Save Output
    # --------------------------------------------------

    output = []

    for seg, label in zip(valid_segments, labels):

        output.append({

            "speaker": f"Speaker_{label}",

            "start": seg["start"],

            "end": seg["end"],

            "text": seg["text"]

        })

    output_file = OUTPUT_DIR / f"{audio_path.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Saved -> {output_file}\n")

print("=" * 60)
print("MODULE 4 COMPLETE")
print("=" * 60)