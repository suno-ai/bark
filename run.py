from bark import SAMPLE_RATE, generate_audio
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', default='Hello world [laughs]', help='The text prompt to use')
parser.add_argument('--voice', default='auto', help='The voice to use')
args = parser.parse_args()

text_prompt = '"""\n' + args.prompt + '\n"""'
voice = args.voice

if voice == "auto":
    audio_array = generate_audio(text_prompt)
else:
    audio_array = generate_audio(text_prompt, history_prompt=voice)

Audio(audio_array, rate=SAMPLE_RATE)

write_wav("audio.wav", SAMPLE_RATE, audio_array)