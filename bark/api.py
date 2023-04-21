from typing import Optional

import numpy as np

from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic


def text_to_semantic(
    text: str,
    history_prompt: Optional[str] = None,
    temp: float = 0.7,
    silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[str] = None,
    temp: float = 0.7,
    silent: bool = False,
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy audio array at sample frequency 24khz
    """
    x_coarse_gen = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(x_fine_gen)
    return audio_arr


def generate_audio(
    text: str,
    history_prompt: Optional[str] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy audio array at sample frequency 24khz
    """
    x_semantic = text_to_semantic(
        text, history_prompt=history_prompt, temp=text_temp, silent=silent,
    )
    audio_arr = semantic_to_waveform(
        x_semantic, history_prompt=history_prompt, temp=waveform_temp, silent=silent,
    )
    return audio_arr
