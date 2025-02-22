import os
import io
from pydub import AudioSegment
from scipy.io.wavfile import write as write_wav
from .api import generate_audio
from .generation import SAMPLE_RATE, preload_models
from tqdm import tqdm
# Preload Bark models once
preload_models()


def numpy_array_to_audiosegment(np_array, sample_rate):
    """Convert numpy audio array to pydub AudioSegment."""
    audio_buffer = io.BytesIO()
    write_wav(audio_buffer, sample_rate, (np_array * 32767).astype("int16"))
    audio_buffer.seek(0)
    return AudioSegment.from_file(audio_buffer, format="wav")


def parse_srt(srt_file_path):
    """Parse SRT file and extract subtitle entries."""
    subtitle_entries = []
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    subtitle_blocks = content.strip().split("\n\n")

    for block in subtitle_blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        idx = lines[0].strip()
        time_range = lines[1].strip()
        text = " ".join(lines[2:]).strip()

        if " --> " not in time_range:
            continue

        start_time, end_time = time_range.split(" --> ")
        subtitle_entries.append({
            "idx": idx,
            "start_time": time_to_seconds(start_time),
            "end_time": time_to_seconds(end_time),
            "text": text
        })

    return subtitle_entries


def time_to_seconds(time_str):
    """Convert timestamp (HH:MM:SS,MS) to total seconds."""
    hours, minutes, seconds = time_str.split(":")
    seconds, milliseconds = seconds.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def generate_silence(duration_ms, frame_rate=SAMPLE_RATE):
    """Generate silence of given duration in milliseconds."""
    return AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)


def srt_to_audio(srt_file_path: str, output_dir: str, history_prompt: str = "v2/en_speaker_6", chunk_size: int = 100):
    """
    Convert SRT subtitles to audio using Bark.

    Args:
        srt_file_path (str): Path to the input SRT file.
        output_dir (str): Directory to save output audio files.
        history_prompt (str): Voice prompt for Bark generation.
        chunk_size (int): Number of subtitles per audio chunk. or after how many dialogues the audio should be saved.
    """
    if not os.path.exists(srt_file_path):
        raise FileNotFoundError(f"SRT file not found: {srt_file_path}")

    os.makedirs(output_dir, exist_ok=True)
    subtitles = parse_srt(srt_file_path)

    final_audio = AudioSegment.empty()
    previous_end_time = 0
    part_number = 1

    for idx, entry in enumerate(tqdm(subtitles, desc="Generating Audio")):
        text = entry["text"]
        start_time = entry["start_time"]
        end_time = entry["end_time"]

        # Add silence for gaps
        silence_duration = max(0, (start_time - previous_end_time) * 1000)
        final_audio += generate_silence(silence_duration)

        # Generate audio for subtitle
        if text.strip():
            audio_np = generate_audio(text, history_prompt=history_prompt)
            subtitle_audio = numpy_array_to_audiosegment(audio_np, SAMPLE_RATE)
            final_audio += subtitle_audio

        previous_end_time = end_time

        # Save chunk
        if (idx + 1) % chunk_size == 0:
            output_path = os.path.join(output_dir, f"output_part_{part_number}.wav")
            final_audio.export(output_path, format="wav")
            print(f"✅ Saved: {output_path}")
            final_audio = AudioSegment.empty()
            part_number += 1

    # Save remaining audio
    if len(final_audio) > 0:
        output_path = os.path.join(output_dir, f"output_part_{part_number}.wav")
        final_audio.export(output_path, format="wav")
        print(f"✅ Saved final part: {output_path}")

    print("🎉 Audio generation complete!")
