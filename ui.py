import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import customtkinter as ctk
import tkinter as tk
import threading, os, time, gc
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
import numpy as np
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from TTS.api import TTS
import torch


ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('blue')

# ---------------- APP ----------------

app = ctk.CTk()
app.title('STTTS Translator')
app.geometry('1100x720')
app.minsize(980,640)

app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(1, weight=1)

# ---------------- VARIABLES ----------------

status_var=tk.StringVar(value='Loading models...')
mode_var=tk.StringVar(value='Mode: Auto Detect')
emotion_var=tk.StringVar(value='😐 Neutral')
level_var=tk.DoubleVar(value=0)

source_lang='hi'
target_lang='mr'

auto_mode=True
recording=False
live_mode=False
processing=False

last_output_file='temp/output.wav'
speaker_wav=os.environ.get('TTS_SPEAKER_WAV')
current_emotion='neutral'

tts_lock = threading.Lock()

# ---------------- TEMP ----------------

if not os.path.exists("temp"):
    os.makedirs("temp")

# ---------------- MODELS ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"
compute = "float16" if device == "cuda" else "int8"

whisper = WhisperModel(
    'small',
    device=device,
    compute_type=compute
)

tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')

status_var.set(f'Ready ({device.upper()})')

# ---------------- UI HELPERS ----------------

def set_busy(busy):
    state='disabled' if busy else 'normal'

    for btn in [record_btn, replay_btn, switch_btn]:
        btn.configure(state=state)

    if not live_mode:
        live_btn.configure(state=state)

# ---------------- EMOTION ----------------

def apply_emotion(path, emo):
    try:
        sr,data=wavfile.read(path)

        if data.ndim>1:
            data=data[:,0]

        data=data.astype(np.float32)

        speed={
            'happy':1.08,
            'angry':1.12,
            'sad':0.90,
            'fear':1.05,
            'neutral':1.0
        }.get(emo,1.0)

        gain={
            'happy':1.08,
            'angry':1.18,
            'sad':0.88,
            'fear':0.95,
            'neutral':1.0
        }.get(emo,1.0)

        idx=np.arange(0,len(data),speed)

        out=np.interp(
            idx,
            np.arange(len(data)),
            data
        ) * gain

        wavfile.write(
            path,
            sr,
            np.clip(out,-32768,32767).astype(np.int16)
        )

    except:
        pass

# ---------------- AUDIO PLAYBACK ----------------

def play_audio(path):
    try:
        sr, data = wavfile.read(path)

        sd.play(data, sr)
        sd.wait()

    except Exception as e:
        print(e)
        status_var.set("Playback failed")

# ---------------- TTS ----------------

def speak_text(text, lang):
    global last_output_file

    with tts_lock:

        try:
            text = text[:220]

            code='hi' if lang=='mr' else lang
            voice_reference = speaker_wav or 'temp/input.wav'
            if not os.path.isfile(voice_reference):
                raise FileNotFoundError(
                    'Record a voice sample first or set TTS_SPEAKER_WAV.'
                )

            tts.tts_to_file(
                text=text,
                speaker_wav=voice_reference,
                language=code,
                file_path=last_output_file
            )

            apply_emotion(last_output_file,current_emotion)

            play_audio(last_output_file)

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except:
            status_var.set('Speech failed')

# ---------------- REPLAY ----------------

replay_playing=False

def replay():
    global replay_playing

    if replay_playing:
        return

    if os.path.exists(last_output_file):

        def _play():
            global replay_playing

            replay_playing=True

            replay_btn.configure(state='disabled')

            try:
                play_audio(last_output_file)

            finally:
                replay_playing=False
                replay_btn.configure(state='normal')

        threading.Thread(
            target=_play,
            daemon=True
        ).start()

# ---------------- LANGUAGE ----------------

def auto_detect_language(text, guess=None):
    global source_lang,target_lang

    t=text.lower()

    mr=['आहे','मला','तू','काय','नाही','मी','आणि']
    hi=['है','मुझे','तुम','क्या','नहीं','मैं','और']

    ms=sum(w in t for w in mr)
    hs=sum(w in t for w in hi)

    lang=guess if guess in ['hi','mr'] else None

    if ms>hs:
        lang='mr'

    elif hs>ms:
        lang='hi'

    elif lang is None:
        lang=source_lang

    source_lang=lang
    target_lang='mr' if lang=='hi' else 'hi'

    mode_var.set(
        'Mode: Hindi → Marathi'
        if lang=='hi'
        else 'Mode: Marathi → Hindi'
    )

def switch_mode():
    global source_lang,target_lang,auto_mode

    auto_mode=False
    auto_switch.deselect()

    if source_lang=='hi':
        source_lang,target_lang='mr','hi'
        mode_var.set('Mode: Marathi → Hindi')

    else:
        source_lang,target_lang='hi','mr'
        mode_var.set('Mode: Hindi → Marathi')

def toggle_auto():
    global auto_mode

    auto_mode=bool(auto_switch.get())

    if auto_mode:
        mode_var.set('Mode: Auto Detect')

# ---------------- EMOTION DETECTION ----------------

def detect_emotion(audio,fs):
    global current_emotion

    s=audio.flatten()

    e=np.mean(np.abs(s))
    z=np.sum(np.diff(np.sign(s))!=0)/max(len(s),1)
    d=len(s)/fs

    if e>0.18 and z>0.12:
        current_emotion='angry'
        emotion_var.set('😡 Angry')

    elif e>0.12:
        current_emotion='happy'
        emotion_var.set('😊 Happy')

    elif e<0.04 and d>2:
        current_emotion='sad'
        emotion_var.set('😢 Sad')

    elif z>0.15:
        current_emotion='fear'
        emotion_var.set('😨 Fear')

    else:
        current_emotion='neutral'
        emotion_var.set('😐 Neutral')

# ---------------- MAIN PROCESS ----------------

def process_audio(audio,fs):

    global processing

    if processing:
        return

    processing=True

    try:

        input_box.configure(state='normal')
        output_box.configure(state='normal')

        input_box.delete('1.0','end')
        output_box.delete('1.0','end')

        peak=np.max(np.abs(audio))

        if peak>0:
            audio=audio/peak*0.95

        write(
            'temp/input.wav',
            fs,
            (audio*32767).astype(np.int16)
        )

        detect_emotion(audio,fs)

        status_var.set('Transcribing...')

        lang_arg=None if auto_mode else source_lang

        segs,info=whisper.transcribe(
            'temp/input.wav',
            language=lang_arg,
            beam_size=2
        )

        text=' '.join(seg.text for seg in segs).strip()

        if auto_mode:
            auto_detect_language(
                text,
                getattr(info,'language',None)
            )

        input_box.insert('end',text)

        status_var.set('Translating...')

        translated=GoogleTranslator(
            source=source_lang,
            target=target_lang
        ).translate(text)

        translated = translated[:220]

        output_box.insert('end',translated)

        input_box.configure(state='disabled')
        output_box.configure(state='disabled')

        status_var.set('Speaking...')

        speak_text(translated,target_lang)

        status_var.set('Ready')

    except Exception as e:
        print(e)
        status_var.set('Error')

    finally:
        processing=False

# ---------------- RECORDING ----------------

def manual_record():

    global recording

    fs=16000
    frames=[]

    recording=True

    record_btn.configure(text='Stop Recording')

    with sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype='float32'
    ) as stream:

        status_var.set('Recording...')

        while recording:

            data,_=stream.read(1024)

            frames.append(data.copy())

            level_var.set(
                min(
                    float(np.mean(np.abs(data))*120),
                    100
                )
            )

    record_btn.configure(text='Start Recording')

    level_var.set(0)

    if frames:
        set_busy(True)

        process_audio(
            np.concatenate(frames,axis=0),
            fs
        )

        set_busy(False)

def toggle_record():
    global recording

    if not recording:

        threading.Thread(
            target=manual_record,
            daemon=True
        ).start()

    else:
        recording=False

# ---------------- LIVE MODE ----------------

def live_loop():

    global live_mode

    fs=16000
    chunk=1024

    silence_needed=25

    with sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype='float32'
    ) as stream:

        while live_mode:

            status_var.set('Live Listening...')

            frames=[]
            started=False
            silent=0

            while live_mode:

                data,_=stream.read(chunk)

                level_var.set(
                    min(
                        float(np.mean(np.abs(data))*120),
                        100
                    )
                )

                vol=float(np.mean(np.abs(data)))

                if vol>0.02:

                    started=True
                    frames.append(data.copy())
                    silent=0

                elif started:

                    frames.append(data.copy())
                    silent+=1

                    if silent>=silence_needed:
                        break

            if frames and live_mode:

                set_busy(True)

                process_audio(
                    np.concatenate(frames,axis=0),
                    fs
                )

                set_busy(False)

    level_var.set(0)

    status_var.set('Live Stopped')

def start_live():

    global live_mode

    if not live_mode:

        live_mode=True

        live_btn.configure(state='disabled')

        threading.Thread(
            target=live_loop,
            daemon=True
        ).start()

def stop_live():
    global live_mode

    live_mode=False

    live_btn.configure(state='normal')

# ---------------- UI ----------------

header=ctk.CTkFrame(app, corner_radius=0, height=70)

header.grid(
    row=0,
    column=0,
    columnspan=2,
    sticky='ew'
)

header.grid_columnconfigure(0, weight=1)

ctk.CTkLabel(
    header,
    text='STTTS Translator',
    font=ctk.CTkFont(size=28,weight='bold')
).grid(
    row=0,
    column=0,
    padx=24,
    pady=18,
    sticky='w'
)

ctk.CTkLabel(
    header,
    textvariable=status_var,
    width=180
).grid(
    row=0,
    column=1,
    padx=24,
    pady=18,
    sticky='e'
)

# ---------------- SIDEBAR ----------------

sidebar=ctk.CTkFrame(app,width=260)

sidebar.grid(
    row=1,
    column=0,
    sticky='nsw',
    padx=(16,8),
    pady=16
)

sidebar.grid_propagate(False)

ctk.CTkLabel(
    sidebar,
    textvariable=mode_var,
    font=ctk.CTkFont(size=18,weight='bold')
).pack(
    anchor='w',
    padx=18,
    pady=(18,8)
)

card=ctk.CTkFrame(sidebar)
card.pack(fill='x',padx=18,pady=8)

ctk.CTkLabel(card,text='Emotion').pack(anchor='w',padx=12,pady=(10,0))

ctk.CTkLabel(
    card,
    textvariable=emotion_var,
    font=ctk.CTkFont(size=22,weight='bold')
).pack(anchor='w',padx=12,pady=(2,8))

ctk.CTkLabel(card,text='Mic Level').pack(anchor='w',padx=12)

level_bar=ctk.CTkProgressBar(card,variable=level_var)

level_bar.pack(fill='x',padx=12,pady=(4,12))

level_bar.set(0)

record_btn=ctk.CTkButton(
    sidebar,
    text='Start Recording',
    height=42,
    command=toggle_record
)

record_btn.pack(fill='x',padx=18,pady=(14,8))

live_btn=ctk.CTkButton(
    sidebar,
    text='Live Mode',
    height=42,
    command=start_live
)

live_btn.pack(fill='x',padx=18,pady=8)

stop_btn=ctk.CTkButton(
    sidebar,
    text='Stop Live',
    height=42,
    command=stop_live,
    fg_color='#444444',
    hover_color='#555555'
)

stop_btn.pack(fill='x',padx=18,pady=8)

replay_btn=ctk.CTkButton(
    sidebar,
    text='Replay Output',
    height=42,
    command=replay
)

replay_btn.pack(fill='x',padx=18,pady=8)

switch_btn=ctk.CTkButton(
    sidebar,
    text='Switch Language',
    height=42,
    command=switch_mode
)

switch_btn.pack(fill='x',padx=18,pady=8)

auto_switch=ctk.CTkSwitch(
    sidebar,
    text='Auto Detect ON',
    command=toggle_auto
)

auto_switch.select()

auto_switch.pack(
    anchor='w',
    padx=20,
    pady=(18,8)
)

# ---------------- MAIN AREA ----------------

main=ctk.CTkFrame(app)

main.grid(
    row=1,
    column=1,
    sticky='nsew',
    padx=(8,16),
    pady=16
)

main.grid_columnconfigure(0,weight=1)
main.grid_rowconfigure(1,weight=2)
main.grid_rowconfigure(3,weight=2)

ctk.CTkLabel(
    main,
    text='Recognized Text',
    font=ctk.CTkFont(size=16,weight='bold')
).grid(
    row=0,
    column=0,
    sticky='w',
    padx=18,
    pady=(18,8)
)

input_box=ctk.CTkTextbox(
    main,
    corner_radius=12,
    font=ctk.CTkFont(size=28)
)

input_box.configure(state='disabled')

input_box.grid(
    row=1,
    column=0,
    sticky='nsew',
    padx=22,
    pady=(0,16)
)

ctk.CTkLabel(
    main,
    text='Translated Text',
    font=ctk.CTkFont(size=16,weight='bold')
).grid(
    row=2,
    column=0,
    sticky='w',
    padx=18,
    pady=(4,8)
)

output_box=ctk.CTkTextbox(
    main,
    corner_radius=12,
    font=ctk.CTkFont(size=28)
)

output_box.configure(state='disabled')

output_box.grid(
    row=3,
    column=0,
    sticky='nsew',
    padx=22,
    pady=(0,22)
)

# ---------------- START ----------------

app.mainloop()
