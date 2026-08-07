# Speech Transcription System for Multi-Speaker Conversations 
# Project Overview
This project presents an end-to-end automated speech transcription pipeline for generating verbatim transcripts from two-speaker conversational audio. The system performs audio preprocessing, speaker diarization, speaker-wise transcription, target speaker identification, and final transcript generation while preserving natural speech phenomena such as filler words, repetitions, hesitations, incomplete sentences, and unintelligible speech.
The project was developed as part of the Verbatim Transcription of Speech from Audio Dialogues assignment.

# Features
* Audio cleaning and preprocessing
* Speaker diarization using Pyannote Community-1
* Speaker-wise transcription using Faster-Whisper
* Automatic target speaker identification
* Generation of final verbatim transcript

# Technologies Used
Python 3.11
PyTorch
Pyannote Audio
Faster-Whisper
SoundFile
NumPy
Hugging Face
FFmpeg
