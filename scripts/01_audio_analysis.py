from pathlib import Path
import librosa
import pandas as pd

audio_folder = Path("C:/my folder/Speechtranscription/data/raw_audio")

rows = []

for audio_file in audio_folder.glob("*.mp3"):

    y, sr = librosa.load(audio_file, sr=None, mono=False)

    duration = librosa.get_duration(y=y, sr=sr)

    if len(y.shape) == 1:
        channels = 1
    else:
        channels = y.shape[0]

    rows.append({
        "File": audio_file.name,
        "Duration (min)": round(duration/60,2),
        "Sample Rate": sr,
        "Channels": channels
    })

df = pd.DataFrame(rows)

print(df)

df.to_csv("../output/evaluation/audio_statistics.csv", index=False)