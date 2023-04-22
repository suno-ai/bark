import gradio as gr
from bark.generation import SUPPORTED_LANGS
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
import os
from datetime import datetime


def generate_text_to_speech(text_prompt, selected_speaker, text_temp, waveform_temp):
    audio_array = generate_audio(text_prompt, selected_speaker, text_temp, waveform_temp)

    now = datetime.now()
    date_str = now.strftime("%m-%d-%Y")
    time_str = now.strftime("%H-%M-%S")

    outputs_folder = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    sub_folder = os.path.join(outputs_folder, date_str)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    file_name = f"audio_{time_str}.wav"
    file_path = os.path.join(sub_folder, file_name)
    write_wav(file_path, SAMPLE_RATE, audio_array)

    return file_path


speakers_list = []

for lang, code in SUPPORTED_LANGS:
    for n in range(10):
        speakers_list.append(f"{code}_speaker_{n}")

input_text = gr.Textbox(label="Input Text", lines=4, placeholder="Enter text here...")
text_temp = gr.Slider(
    0.1,
    1.0,
    value=0.7,
    label="Generation Temperature",
    info="1.0 more diverse, 0.1 more conservative",
)
waveform_temp = gr.Slider(
    0.1, 1.0, value=0.7, label="Waveform temperature", info="1.0 more diverse, 0.1 more conservative"
)
output_audio = gr.Audio(label="Generated Audio", type="filepath")
speaker = gr.Dropdown(speakers_list, value=speakers_list[0], label="Acoustic Prompt")


interface = gr.Interface(
    fn=generate_text_to_speech,
    inputs=[input_text, speaker, text_temp, waveform_temp],
    outputs=output_audio,
    title="Text-to-Speech using Bark",
    description="A simple Bark TTS Web UI.",
)

interface.launch()
