from pathlib import Path
import librosa
import soundfile as sf
import noisereduce as nr

input_folder = Path("C:/my folder/Speechtranscription/data/raw_audio")
output_folder = Path("C:/my folder/Speechtranscription/data/processed_audio")

output_folder.mkdir(exist_ok=True)

for audio in input_folder.glob("*.mp3"):

    y, sr = librosa.load(audio, sr=16000, mono=True)

    reduced_noise = nr.reduce_noise(
        y=y,
        sr=sr
    )

    sf.write(
        output_folder / (audio.stem + ".wav"),
        reduced_noise,
        sr
    )

print("Finished preprocessing.")