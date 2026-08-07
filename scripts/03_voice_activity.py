from pathlib import Path
import json
import torch 
from tqdm import tqdm

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

INPUT_FOLDER = Path("C:/my folder/Speechtranscription/data/processed_audio")
OUTPUT_FOLDER = Path("C:/my folder/Speechtranscription/output/speech_segments")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load VAD Model
# --------------------------------------------------

print("Loading Silero VAD Model...")

model = load_silero_vad()

print("Model Loaded Successfully.\n")

# --------------------------------------------------
# Process Each Audio File
# --------------------------------------------------

for audio_path in tqdm(sorted(INPUT_FOLDER.glob("*.wav"))):

    print(f"\nProcessing: {audio_path.name}")

    # Load Audio
    wav = read_audio(str(audio_path))

    # Detect Speech
    speech = get_speech_timestamps(
        wav,
        model,
        return_seconds=True
    )

    # Save JSON
    output_file = OUTPUT_FOLDER / f"{audio_path.stem}_segments.json"

    with open(output_file, "w") as f:
        json.dump(speech, f, indent=4)

print("\nDone!")