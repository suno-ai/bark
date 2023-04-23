from typing import Optional, Tuple, Dict, Union

import numpy as np

from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic


def text_to_semantic(
    text: str,
    history_prompt: Optional[str] = None,
    temp: float = 0.7,
    silent: bool = False,
) -> np.ndarray:
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[str] = None,
    temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
) -> Union[Tuple[Dict[str, np.ndarray], np.ndarray], np.ndarray]:
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def save_as_prompt(filepath: str, full_generation: Dict[str, np.ndarray]) -> None:
    assert filepath.endswith(".npz"), "Filepath must end with .npz"
    assert isinstance(full_generation, dict), "full_generation must be a dictionary"
    assert "semantic_prompt" in full_generation, "semantic_prompt key is missing"
    assert "coarse_prompt" in full_generation, "coarse_prompt key is missing"
    assert "fine_prompt" in full_generation, "fine_prompt key is missing"
    np.savez(filepath, **full_generation)


def generate_audio(
    text: str,
    history_prompt: Optional[str] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
) -> Union[Tuple[Dict[str, np.ndarray], np.ndarray], np.ndarray]:
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
    )
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    audio_arr = out
    return audio_arr
