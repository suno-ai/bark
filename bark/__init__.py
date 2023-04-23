from typing import Tuple

from .api import (
    generate_audio,
    save_as_prompt,
)
from .generation import SAMPLE_RATE, preload_models


def generate_and_save_audio(text: str, output_file: str) -> None:
    """
    Generate audio from the given text and save it to the specified output file.

    Args:
        text (str): The input text to be converted to speech.
        output_file (str): The path to the output file where the generated audio will be saved.

    Returns:
        None
    """
    # Preload models for faster generation
    preload_models()

    # Generate audio from text
    audio = generate_audio(text)

    # Save audio to a file
    save_as_prompt(audio, output_file)


if __name__ == "__main__":
    # Example usage
    input_text = "Hello, world!"
    output_path = "output.wav"
    generate_and_save_audio(input_text, output_path)
