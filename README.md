# STTTS Translator

A desktop Hindi–Marathi speech translator. It records speech, transcribes it locally with Faster Whisper, translates the text, and reads the translation aloud with XTTS v2.

## What it does

- Records from your microphone or listens in live mode.
- Transcribes Hindi or Marathi speech with the local Whisper `small` model.
- Translates between Hindi and Marathi using Google Translate through `deep-translator`.
- Speaks the translated text with XTTS v2 and a voice reference clip you provide.
- Includes simple audio-based emotion detection and playback controls.

Automatic language detection is a lightweight Hindi/Marathi heuristic. The app currently maps Marathi speech synthesis to XTTS's Hindi language setting, so Marathi pronunciation may be imperfect.

## Requirements

- Windows 10 or later
- Python 3.10 (Coqui TTS 0.22 supports Python 3.9–3.11)
- A working microphone and speakers
- Internet access for model downloads and translation
- A local WAV voice reference clip for XTTS

Create the environment and install dependencies from PowerShell:

```powershell
py -3.10 -m venv tts_env
.\tts_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place your own voice reference WAV in the project folder as `speaker.wav`. To use a different path, set `TTS_SPEAKER_WAV` before starting the app:

```powershell
$env:TTS_SPEAKER_WAV = 'C:\path\to\your\speaker.wav'
```

Then start the app with:

```powershell
.\run_app.bat
```

On first launch, Faster Whisper and XTTS download their model files. Model loading and speech generation can take time, especially without a supported CUDA GPU.

## Privacy

Recorded audio is processed locally by Whisper, then temporary input/output audio is written under `temp/`. The transcribed text is sent to Google Translate by `deep-translator`. XTTS runs locally after its model files are downloaded. Do not use sensitive speech unless you are comfortable sending its transcript to Google for translation.

Audio files, the Python environment, generated model/build files, and the old local notes file are excluded from Git. Only add recordings or voice samples if you have permission to publish them. Use a voice reference only with the speaker's consent.

## Model terms

The XTTS v2 model has its own [Coqui Public Model License](https://coqui.ai/cpml), separate from this project. Review the model terms before using it. This repository does not currently include a license for its application code.
