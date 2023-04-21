from bark import SAMPLE_RATE, generate_audio
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav
import gradio as gr
import os 

voicePath = "bark/assets/prompts";
npz_files = [file for file in os.listdir(voicePath) if file.endswith(".npz")]
npz_files = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]
npz_files.insert(0, "auto")

def start(prompt, voice):
    if voice == "auto":
        audio_array = generate_audio(prompt)
    else:
        audio_array = generate_audio(prompt, history_prompt=voice)
    audio_array = generate_audio(prompt, history_prompt=voice)
    write_wav("audio.wav", SAMPLE_RATE, audio_array)
    return "audio.wav"

with gr.Blocks() as demo:
    gr.Title="Bark TTS WebUI",
    gr.Markdown("Bark TTS WebUI")
    with gr.Row():

        prompt = gr.Textbox(label="Prompt", placeholder="Enter your text here", lines=4)

    with gr.Row():
        voice = gr.Dropdown(npz_files, value=["auto"], label="Voice", info="Select the voice")
        launch_button = gr.Button("Launch")   
  
    with gr.Column(visible=True):
        output = gr.Audio(label="Result")
        
    launch_button.click(
        start,
        [prompt, voice],
        [output],
    )        

demo.launch()