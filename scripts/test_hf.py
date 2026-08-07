'''''
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")

api = HfApi()

user = api.whoami(token)

print(user)

'''''

from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os

load_dotenv()

hf_hub_download(
    repo_id="pyannote/speaker-diarization-community-1",
    filename="config.yaml",
    token=os.getenv("HF_TOKEN"),
)

print("SUCCESS")