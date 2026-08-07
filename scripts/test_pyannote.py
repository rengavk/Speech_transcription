import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

token = os.getenv("HF_TOKEN")

print("Token Loaded:", token[:10] + "...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=token
)

print("Model Loaded Successfully!")