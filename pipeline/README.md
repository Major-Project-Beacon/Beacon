# BEACON Fact Extraction Pipeline

This module contains the speech-processing and fact-extraction pipeline developed for **BEACON**.

The pipeline converts a meeting video into a speaker-labelled transcript and then extracts important factual information from the meeting using the BEACON fact-extraction model.

---

## 1. Pipeline Overview

```text
Meeting Video
      │
      ▼
Audio Extraction
      │
      ▼
16 kHz Mono WAV
      │
      ▼
Speech-to-Text
(Faster-Whisper)
      │
      ▼
Speaker Diarization
(pyannote)
      │
      ▼
Speaker-Labelled Transcript
      │
      ▼
BEACON Fact Extraction
(FLAN-T5 + LoRA)
      │
      ▼
Extracted Facts
      │
      ▼
Evaluation
(Precision / Recall / F1)
```

### Pipeline Components

| Component | File | Purpose |
|---|---|---|
| Audio extraction | `audio.py` | Extracts audio from meeting video |
| Transcription | `transcription.py` | Converts speech into text |
| Speaker diarization | `diarization.py` | Identifies different speaker clusters |
| Speaker transcript | `speaker_transcript.py` | Combines transcript and speaker information |
| Fact extraction | `fact_extraction.py` | Extracts important facts from the meeting |
| Evaluation | `evaluate_model.py` | Evaluates the fact-extraction model |

---

# 2. Requirements

## Hardware

The pipeline is designed to run with GPU acceleration.

Recommended:

- NVIDIA GPU
- CUDA-compatible PyTorch installation
- At least 8 GB system RAM
- Sufficient disk space for Whisper and pyannote models

The development environment was tested using an NVIDIA RTX 3050 Laptop GPU.

CPU execution may work for some components but will be considerably slower.

---

# 3. Software Requirements

Install:

- Python 3.12+
- Git
- FFmpeg
- NVIDIA GPU drivers
- CUDA-compatible PyTorch

The project was developed and tested on Windows.

---

# 4. Clone the Repository

Clone the BEACON repository:

```powershell
git clone https://github.com/Major-Project-Beacon/Beacon.git
```

Move into the repository:

```powershell
cd Beacon
```

Switch to the fact-extraction branch:

```powershell
git switch fact-extraction-module
```

The pipeline is located at:

```text
Beacon/
└── pipeline/
```

---

# 5. Create a Python Environment

It is recommended to use a dedicated virtual environment.

Example:

```powershell
python -m venv whisper_env
```

Activate it:

```powershell
.\whisper_env\Scripts\Activate.ps1
```

On successful activation, the terminal should show something similar to:

```text
(whisper_env) PS D:\...\Beacon>
```

---

# 6. Install Dependencies

From the `pipeline` directory:

```powershell
cd pipeline
```

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

The main dependencies include:

```text
faster-whisper
pyannote.audio
torch
transformers
peft
python-dotenv
```

---

# 7. Install FFmpeg

FFmpeg is required for audio extraction.

Verify that FFmpeg is available:

```powershell
ffmpeg -version
```

If the command is not recognized, install FFmpeg and add its `bin` directory to the system PATH.

Example:

```text
C:\ffmpeg\bin
```

After modifying PATH, restart the terminal and run:

```powershell
ffmpeg -version
```

again.

---

# 8. Hugging Face Authentication

Speaker diarization uses the pyannote speaker-diarization model.

Each developer should use **their own Hugging Face token**.

Do not share personal Hugging Face tokens with other team members.

## Create `.env`

Inside:

```text
pipeline/
```

create:

```text
.env
```

Add:

```text
HF_TOKEN=your_huggingface_token
```

Example:

```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

Do not commit this file.

The repository contains:

```text
.env.example
```

instead.

Copy `.env.example` to `.env` and replace the placeholder with your own token.

---

# 9. Hugging Face Model Access

The Hugging Face account used for the token must have access to the pyannote diarization model used by the pipeline.

The diarization script uses:

```text
pyannote/speaker-diarization-community-1
```

If Hugging Face requests model access or authentication, complete the required steps using your own Hugging Face account.

---

# 10. Model Setup

The fact-extraction pipeline uses:

```text
google/flan-t5-base
```

with the BEACON V2.1 LoRA adapter.

The trained adapter is **not included in this repository**.

The adapter used during development was stored locally at:

```text
D:\BEACON\models\beacon_flan_t5_v2_1_refined_lora
```

The adapter must be available locally before running:

```text
fact_extraction.py
```

and:

```text
evaluate_model.py
```

### Important

The current scripts contain a local model path.

If the model is stored elsewhere, update:

```python
ADAPTER_PATH
```

in:

```text
fact_extraction.py
```

and:

```text
evaluate_model.py
```

to point to the local BEACON V2.1 adapter.

Do not commit private model files or large model checkpoints to this repository unless the team explicitly decides to use Git LFS or another model-storage solution.

---

# 11. Input Meeting Video

The first pipeline component requires a meeting video.

The video can be any supported video format that FFmpeg can read.

Examples:

```text
.mp4
.mkv
.mov
.avi
```

Place the video somewhere accessible on your local machine.

Before running the pipeline, update the input video path in:

```text
audio.py
```

---

# 12. Running the Pipeline

The components should be executed in the following order.

## Step 1: Extract Audio

Run:

```powershell
python audio.py
```

This extracts the meeting audio as:

```text
16 kHz
Mono
PCM WAV
```

The output is written to the pipeline output directory.

Expected structure:

```text
pipeline/
└── output/
    └── <meeting>.wav
```

---

# 13. Step 2: Transcription

Update the audio path in:

```text
transcription.py
```

Then run:

```powershell
python transcription.py
```

The script uses **Faster-Whisper** to generate the meeting transcript.

The output contains timestamped transcript segments.

Example:

```text
[12.40 - 15.80] We should prioritize the performance issue.
[15.90 - 18.20] I agree with that.
```

Output:

```text
pipeline/output/transcript.txt
```

---

# 14. Step 3: Speaker Diarization

Make sure your `.env` contains:

```text
HF_TOKEN=your_token
```

Then run:

```powershell
python diarization.py
```

The script uses:

```text
pyannote/speaker-diarization-community-1
```

to identify speaker clusters.

Example output:

```text
[12.40 - 15.80] SPEAKER_02
[15.90 - 18.20] SPEAKER_05
```

The labels such as `SPEAKER_02` and `SPEAKER_05` represent diarization clusters. They are not actual person names.

Output:

```text
pipeline/output/diarization.txt
```

---

# 15. Step 4: Create Speaker-Labelled Transcript

Run:

```powershell
python speaker_transcript.py
```

This combines:

```text
transcript.txt
```

and:

```text
diarization.txt
```

using timestamp overlap.

The resulting output contains:

```text
Timestamp
Speaker
Transcript
```

Example:

```text
[12.40 - 15.80] SPEAKER_02: We should prioritize the performance issue.
[15.90 - 18.20] SPEAKER_05: I agree with that.
```

Output:

```text
pipeline/output/speaker_transcript.txt
```

This is the primary input to the BEACON fact-extraction stage.

---

# 16. Step 5: Fact Extraction

Make sure the BEACON V2.1 adapter path is correctly configured in:

```text
fact_extraction.py
```

Then run:

```powershell
python fact_extraction.py
```

The model analyzes the speaker-labelled meeting transcript and extracts important factual information.

The extraction focuses on information such as:

- decisions
- requirements
- commitments
- deadlines
- restrictions
- permissions
- financial conditions
- risks
- security
- performance
- availability
- responsibilities
- important actions

The model also attempts to ignore:

- greetings
- casual conversation
- irrelevant discussion
- repeated information
- unsupported statements

Output:

```text
pipeline/output/facts.json
```

Each extracted fact contains information linking it back to the original meeting context.

---

# 17. Step 6: Model Evaluation

The fact-extraction model can be evaluated against the BEACON test dataset.

Make sure:

1. The BEACON V2.1 adapter is available.
2. The test dataset is available locally.
3. The `TEST_FILE` path in `evaluate_model.py` points to the test dataset.

Then run:

```powershell
python evaluate_model.py
```

The evaluation reports:

```text
Precision
Recall
F1 Score
Exact Match
Numeric Preservation
No Extra Numbers
```

Example output format:

```text
=======================================================
BEACON V2.1 EVALUATION
=======================================================
Test records          : 60
Precision             : XX.XX%
Recall                : XX.XX%
F1 Score              : XX.XX%
Exact Match           : XX.XX%
Numeric Preservation  : XX.XX%
No Extra Numbers      : XX.XX%
=======================================================
```

Detailed results are saved to:

```text
pipeline/output/evaluation/v2_1_evaluation.json
```

---

# 18. Complete Execution Order

For a new meeting, execute:

```powershell
python audio.py
python transcription.py
python diarization.py
python speaker_transcript.py
python fact_extraction.py
```

The complete flow is:

```text
Video
  ↓
audio.py
  ↓
Audio WAV
  ↓
transcription.py
  ↓
transcript.txt
  ↓
diarization.py
  ↓
diarization.txt
  ↓
speaker_transcript.py
  ↓
speaker_transcript.txt
  ↓
fact_extraction.py
  ↓
facts.json
```

---

# 19. Output Directory

Generated files are stored under:

```text
pipeline/output/
```

Typical output:

```text
output/
├── <meeting>.wav
├── transcript.txt
├── diarization.txt
├── speaker_transcript.txt
├── facts.json
└── evaluation/
    └── v2_1_evaluation.json
```

The `output/` directory is excluded from Git.

---

# 20. Security

Never commit:

```text
.env
```

Never commit:

```text
HF_TOKEN
```

Never place an actual Hugging Face token inside Python source code.

Never share a personal Hugging Face token with teammates.

Each team member should create their own:

```text
pipeline/.env
```

using:

```text
HF_TOKEN=<their own token>
```

The repository only contains:

```text
.env.example
```

---

# 21. Git Files

The pipeline intentionally excludes:

```text
.env
output/
__pycache__/
*.pyc
```

These rules are defined in:

```text
pipeline/.gitignore
```

The repository contains:

```text
.env.example
```

so teammates know which environment variables are required.

---

# 22. Troubleshooting

## `HF_TOKEN environment variable is not set`

Check that:

```text
pipeline/.env
```

exists and contains:

```text
HF_TOKEN=your_token
```

Make sure there are no accidental spaces or incorrect filenames such as:

```text
.env.txt
```

## `ffmpeg is not recognized`

Verify:

```powershell
ffmpeg -version
```

If it fails, install FFmpeg and add its `bin` directory to PATH.

## CUDA is not available

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```text
True
```

If it returns `False`, verify:

- NVIDIA drivers
- PyTorch CUDA installation
- Python environment
- GPU availability

## Model adapter cannot be found

Check the path configured in:

```text
fact_extraction.py
```

and:

```text
evaluate_model.py
```

The path must point to the local BEACON V2.1 adapter.

## `faster_whisper` cannot be imported

Make sure the correct virtual environment is activated:

```powershell
.\whisper_env\Scripts\Activate.ps1
```

Then:

```powershell
pip install -r pipeline/requirements.txt
```

---

# 23. Development Notes

### Speech-to-Text

```text
Faster-Whisper
```

### Speaker Diarization

```text
pyannote/speaker-diarization-community-1
```

### Fact Extraction

```text
google/flan-t5-base
+
BEACON V2.1 LoRA adapter
```

### Evaluation

```text
Precision
Recall
F1
Exact Match
Numeric Preservation
```

---

# 24. Current Scope

This module is responsible for the **speech-processing and meeting fact-extraction side of BEACON**.

Current scope:

```text
Video
→ Audio
→ Transcription
→ Speaker Diarization
→ Speaker-Labelled Transcript
→ Fact Extraction
→ Evaluation
```

Downstream compliance analysis and contract-related processing are separate modules.

---

# 25. Contributor Guidelines

Before modifying the pipeline:

1. Create or switch to your feature branch.
2. Do not commit `.env`.
3. Do not commit generated meeting outputs.
4. Do not commit large model checkpoints unless explicitly approved.
5. Test the complete pipeline after modifying a component.
6. Keep the existing output format compatible with downstream modules.
7. Document major changes in the commit message.

---

# 26. Pipeline Status

The following components have been implemented and tested:

- [x] Audio extraction
- [x] Speech-to-text
- [x] Speaker diarization
- [x] Speaker-labelled transcript
- [x] BEACON fact extraction
- [x] Model evaluation
- [x] Environment-based Hugging Face authentication
- [x] Git-safe secret handling

---

## Repository

BEACON is maintained by the project team under the GitHub organization:

```text
Major-Project-Beacon
```

Pipeline branch:

```text
fact-extraction-module
```