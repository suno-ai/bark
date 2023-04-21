import argparse
import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
import os
import datetime
import soundfile as sf
import re

SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]


ALLOWED_PROMPTS = set()

ALLOWED_PROMPTS = {"announcer"}
for _, lang in SUPPORTED_LANGS:
    for n in range(10):
        ALLOWED_PROMPTS.add(f"{lang}_speaker_{n}")
    for n in range(10):
        ALLOWED_PROMPTS.add(f"speaker_{n}")

def estimate_spoken_time(text, wpm=150, time_limit=14):
    # Remove text within square brackets
    text_without_brackets = re.sub(r'\[.*?\]', '', text)
    
    words = text_without_brackets.split()
    word_count = len(words)
    time_in_seconds = (word_count / wpm) * 60
    
    if time_in_seconds > time_limit:
        return True, time_in_seconds
    else:
        return False, time_in_seconds

def save_audio_to_file(filename, audio_array, sample_rate=24000, format='WAV', subtype='PCM_16', output_dir=None):

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    i = 1
    name, ext = os.path.splitext(filepath)
    while os.path.exists(filepath):
        filepath = f"{name}_{i}{ext}"
        i += 1

    sf.write(filepath, audio_array, sample_rate, format=format, subtype=subtype)
    print(f"Saved audio to {filepath}")


def gen_and_save_audio(text_prompt,  history_prompt=None, text_temp=0.7, waveform_temp=0.7, filename="", output_dir="bark_samples"):
    def generate_unique_filename(base_filename):
        name, ext = os.path.splitext(base_filename)
        unique_filename = base_filename
        counter = 1
        while os.path.exists(unique_filename):
            unique_filename = f"{name}_{counter}{ext}"
            counter += 1
        return unique_filename

    longer_than_14_seconds, estimated_time = estimate_spoken_time(text_prompt)
    print(f"Estimated time: {estimated_time:.2f} seconds.")
    if longer_than_14_seconds:
        print(f"Text Prompt could be too long, might want to try a shorter one if you get a bad result.")
    print(f"Generating: {text_prompt}")
    if args.history_prompt:
        print(f"Using speaker: {history_prompt}")
  
    else:
        print(f"No speaker. Randomly generating a speaker.")
 
    audio_array = generate_audio(text_prompt, history_prompt, text_temp=text_temp,
                                                      waveform_temp=waveform_temp)

    if not filename:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        truncated_text = text_prompt.replace("WOMAN:", "").replace("MAN:", "")[:15].strip().replace(" ", "_")
        filename = f"{truncated_text}-history_prompt-{history_prompt}-text_temp-{text_temp}-waveform_temp-{waveform_temp}-{date_str}.wav"
        filename = generate_unique_filename(filename)

    save_audio_to_file(filename, audio_array, SAMPLE_RATE, output_dir=output_dir)




def print_speakers_list():
    print("Available history prompts:")
    print("\nNon-specific speakers:")
    print(" announcer")
    print(" speaker_0 to speaker_9")
    print("\nLanguage-specific speakers:")
    for language, lang_code in SUPPORTED_LANGS:
        speakers = ", ".join([f"{lang_code}_speaker_{n}" for n in range(10)])
        print(f"\n  {language}({lang_code}):\n{speakers}")



# If there's no text_prompt passed on the command line, process this list instead.
text_prompts = []

text_prompt = """
    In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.
"""
text_prompts.append(text_prompt)

text_prompt = """
    A common mistake that people make when trying to design something completely foolproof is to underestimate the ingenuity of complete fools.
"""
text_prompts.append(text_prompt)


def main(args):
    if args.list_speakers:
        print_speakers_list()
    else:
        if args.text_prompt:
            text_prompts_to_process = [args.text_prompt]
        else:
            print("No text prompt provided. Using default prompts defined in this file.")
            text_prompts_to_process = text_prompts
        if args.history_prompt: 
            history_prompt = args.history_prompt
        else:
            history_prompt = None
        text_temp = args.text_temp if args.text_temp else 0.7
        waveform_temp = args.waveform_temp if args.waveform_temp else 0.7
        filename = args.filename if args.filename else ""
        output_dir = args.output_dir if args.output_dir else "bark_samples"

        print("Loading Bark models...")
        
        if args.use_smaller_models:
            print("Using smaller models.")
            preload_models(use_smaller_models=True)
        else:
            preload_models()

        print("Models loaded.")

        for prompt in text_prompts_to_process:
            gen_and_save_audio(prompt, history_prompt, text_temp, waveform_temp, filename, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Generate and save audio.
        install this first: pip install soundfile
        Example: python bark_speak.py --text_prompt "It is a mistake to think you can solve any major problems just with potatoes." --history_prompt en_speaker_3
        """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--text_prompt", help="Text prompt. If not provided, a set of default prompts will be used defined in this file.")
    parser.add_argument("--history_prompt", help="Optional. Choose a speaker from the list of languages: " + ", ".join([lang[0] for lang in SUPPORTED_LANGS]) + ". Use --list_speakers to see all available options.")
    parser.add_argument("--text_temp", type=float, help="Text temperature. Default is 0.7.")
    parser.add_argument("--waveform_temp", type=float, help="Waveform temperature. Default is 0.7.")
    parser.add_argument("--filename", help="Output filename. If not provided, a unique filename will be generated based on the text prompt and other parameters.")
    parser.add_argument("--output_dir", help="Output directory. Default is 'bark_samples'.")
    parser.add_argument("--list_speakers", action="store_true", help="List all preset speaker options instead of generating audio.")
    parser.add_argument("--use_smaller_models", action="store_true", help="Use for GPUS with less than 10GB of memory, or for more speed.")

    args = parser.parse_args()
    main(args)
