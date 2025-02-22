> Notice: Bark is Suno's open-source text-to-speech+ model. If you are looking for our text-to-music models, please visit us on our [web page](https://suno.ai) and join our community on [Discord](https://suno.ai/discord).

# 🐶 Bark

[![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE?style=flat&compact=True)](https://suno.ai/discord)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/FM.svg?style=social&label=@suno_ai_)](https://twitter.com/suno_ai_)

> 🔗 [Examples](https://suno.ai/examples/bark-v0) • [Suno Studio Waitlist](https://suno-ai.typeform.com/suno-studio) • [Updates](#-updates) • [How to Use](#-usage-in-python) • [Installation](#-installation) • [FAQ](#-faq)

[//]: br "vertical spaces around image"

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/235310676-a4b3b511-90ec-4edf-8153-7ccf14905d73.png" width="500"></img>
</p>
<br>

Bark is a transformer-based text-to-audio model created by [Suno](https://suno.ai). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. To support the research community, we are providing access to pretrained model checkpoints, which are ready for inference and available for commercial use.

## ⚠ Disclaimer

Bark was developed for research purposes. It is not a conventional text-to-speech model but instead a fully generative text-to-audio model, which can deviate in unexpected ways from provided prompts. Suno does not take responsibility for any output generated. Use at your own risk, and please act responsibly.

## 📖 Quick Index

- [🚀 Updates](#-updates)
- [💻 Installation](#-installation)
- [🐍 Usage](#-usage-in-python)
- [🌀 Live Examples](https://suno.ai/examples/bark-v0)
- [❓ FAQ](#-faq)

## 🎧 Demos

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/suno/bark)
[![Open on Replicate](https://img.shields.io/badge/®️-Open%20on%20Replicate-blue.svg)](https://replicate.com/suno-ai/bark)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)

## 🚀 Updates

**2023.05.01**

- ©️ Bark is now licensed under the MIT License, meaning it's now available for commercial use!
- ⚡ 2x speed-up on GPU. 10x speed-up on CPU. We also added an option for a smaller version of Bark, which offers additional speed-up with the trade-off of slightly lower quality.
- 📕 [Long-form generation](notebooks/long_form_generation.ipynb), voice consistency enhancements and other examples are now documented in a new [notebooks](./notebooks) section.
- 👥 We created a [voice prompt library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). We hope this resource helps you find useful prompts for your use cases! You can also join us on [Discord](https://suno.ai/discord), where the community actively shares useful prompts in the **#audio-prompts** channel.
- 💬 Growing community support and access to new features here:

  [![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE)](https://suno.ai/discord)

- 💾 You can now use Bark with GPUs that have low VRAM (<4GB).

**2023.04.20**

- 🐶 Bark release!

## 🐍 Usage in Python

<details open>
  <summary><h3>🪑 Basics</h3></summary>

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs]
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

[pizza1.webm](https://user-images.githubusercontent.com/34592747/cfa98e54-721c-4b9c-b962-688e09db684f.webm)

</details>

<details open>
  <summary><h3>🌎 Foreign Language</h3></summary>
<br>
Bark supports various languages out-of-the-box and automatically determines language from input text. When prompted with code-switched text, Bark will attempt to employ the native accent for the respective languages. English quality is best for the time being, and we expect other languages to further improve with scaling. 
<br>
<br>

```python

text_prompt = """
    추석은 내가 가장 좋아하는 명절이다. 나는 며칠 동안 휴식을 취하고 친구 및 가족과 시간을 보낼 수 있습니다.
"""
audio_array = generate_audio(text_prompt)
```

[suno_korean.webm](https://user-images.githubusercontent.com/32879321/235313033-dc4477b9-2da0-4b94-9c8b-a8c2d8f5bb5e.webm)

_Note: since Bark recognizes languages automatically from input text, it is possible to use, for example, a german history prompt with english text. This usually leads to english audio with a german accent._

```python
text_prompt = """
    Der Dreißigjährige Krieg (1618-1648) war ein verheerender Konflikt, der Europa stark geprägt hat.
    This is a beginning of the history. If you want to hear more, please continue.
"""
audio_array = generate_audio(text_prompt)
```

[suno_german_accent.webm](https://user-images.githubusercontent.com/34592747/3f96ab3e-02ec-49cb-97a6-cf5af0b3524a.webm)

</details>

<details open>
  <summary><h3>🎶 Music</h3></summary>
Bark can generate all types of audio, and, in principle, doesn't see a difference between speech and music. Sometimes Bark chooses to generate text as music, but you can help it out by adding music notes around your lyrics.
<br>
<br>

```python
text_prompt = """
    ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
"""
audio_array = generate_audio(text_prompt)
```

[lion.webm](https://user-images.githubusercontent.com/5068315/230684766-97f5ea23-ad99-473c-924b-66b6fab24289.webm)

</details>

<details open>
<summary><h3>🎤 Voice Presets</h3></summary>
  
Bark supports 100+ speaker presets across [supported languages](#supported-languages). You can browse the library of supported voice presets [HERE](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c), or in the [code](bark/assets/prompts). The community also often shares presets in [Discord](https://discord.gg/J2B2vsjKuE).

> Bark tries to match the tone, pitch, emotion and prosody of a given preset, but does not currently support custom voice cloning. The model also attempts to preserve music, ambient noise, etc.

```python
text_prompt = """
    I have a silky smooth voice, and today I will tell you about
    the exercise regimen of the common sloth.
"""
audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_1")
```

[sloth.webm](https://user-images.githubusercontent.com/5068315/230684883-a344c619-a560-4ff5-8b99-b4463a34487b.webm)

</details>

### 📼 Generating Audio from SRT Files

The `srt_to_audio` function allows you to convert SRT subtitle files into audio using Bark. This is useful for dubbing videos or generating narrated content.

```python
from bark import srt_to_audio

# Path to your SRT file

# Generate audio from SRT
audio_array = srt_to_audio(
    srt_file_path="path/to/subtitles.srt", # Path to your SRT file
    output_dir="path/to/output" # Output directory
)
```

<details>
<summary> DEMO SRT: looks something like this </summary>

```srt
1
00:00:01,599 --> 00:00:06,600
Hello and welcome to CubicIn and this

2
00:00:06,600 --> 00:00:10,599
platform helps students and teachers in

3
00:00:10,599 --> 00:00:12,440
making quizzes.

4
00:00:12,440 --> 00:00:16,800
This quiz is AI based, so it is

5
00:00:16,800 --> 00:00:20,199
unique every time and it

6
00:00:20,199 --> 00:00:24,240
basically makes quizzes by following the structure of the syllabus and government and or
```

</details>

## `output`:

[srt_to_audio.webm](https://private-user-images.githubusercontent.com/56386987/415886704-c362c5d2-44df-4a90-af85-245cf8acf48f.webm?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDAyMDg1ODksIm5iZiI6MTc0MDIwODI4OSwicGF0aCI6Ii81NjM4Njk4Ny80MTU4ODY3MDQtYzM2MmM1ZDItNDRkZi00YTkwLWFmODUtMjQ1Y2Y4YWNmNDhmLndlYm0_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMjIyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDIyMlQwNzExMjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lYWExMTIxNDBjODEyNWUwZDRkYzBiZTJlN2VkY2M0YjllYTE1ZmMzYjcyMzFiNjMwMjZlZWJlM2EzZGNhN2Q5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.stUjq3t3qMMTPc77p1txGQmgFBVXSiLxVxM-3aGsvG0)

## Parameters:

- **`srt_path`** (_str_): Path to the input `.srt` file.
- **`output_dir`** (_str_): Path to the output directory to store `.wav` file.
- **`history_prompt`** (_str_, _optional_): Voice prompt for Bark generation.
- **`chunk_size`** (_int_, _optional_): Number of subtitles per audio chunk. or after how many dialogues the audio should be saved.

This will generate an audio file that narrates the subtitles from your `.srt` file in the chosen voice preset.

### 📃 Generating Longer Audio

By default, `generate_audio` works well with around 13 seconds of spoken text. For an example of how to do long-form generation, see 👉 **[Notebook](notebooks/long_form_generation.ipynb)** 👈

<details>
<summary>Click to toggle example long-form generations (from the example notebook)</summary>

[dialog.webm](https://user-images.githubusercontent.com/2565833/235463539-f57608da-e4cb-4062-8771-148e29512b01.webm)

[longform_advanced.webm](https://user-images.githubusercontent.com/2565833/235463547-1c0d8744-269b-43fe-9630-897ea5731652.webm)

[longform_basic.webm](https://user-images.githubusercontent.com/2565833/235463559-87efe9f8-a2db-4d59-b764-57db83f95270.webm)

</details>

## Command line

```commandline
python -m bark --text "Hello, my name is Suno." --output_filename "example.wav"
```

## 💻 Installation

_‼️ CAUTION ‼️ Do NOT use `pip install bark`. It installs a different package, which is not managed by Suno._

```bash
pip install git+https://github.com/suno-ai/bark.git
```

or

```bash
git clone https://github.com/suno-ai/bark
cd bark && pip install .
```

## 🤗 Transformers Usage

Bark is available in the 🤗 Transformers library from version 4.31.0 onwards, requiring minimal dependencies
and additional packages. Steps to get started:

1. First install the 🤗 [Transformers library](https://github.com/huggingface/transformers) from main:

```
pip install git+https://github.com/huggingface/transformers.git
```

2. Run the following Python code to generate speech samples:

```py
from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()
```

3. Listen to the audio samples either in an ipynb notebook:

```py
from IPython.display import Audio

sample_rate = model.generation_config.sample_rate
Audio(audio_array, rate=sample_rate)
```

Or save them as a `.wav` file using a third-party library, e.g. `scipy`:

```py
import scipy

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
```

For more details on using the Bark model for inference using the 🤗 Transformers library, refer to the
[Bark docs](https://huggingface.co/docs/transformers/main/en/model_doc/bark) or the hands-on
[Google Colab](https://colab.research.google.com/drive/1dWWkZzvu7L9Bunq9zvD-W02RFUXoW-Pd?usp=sharing).

## 🛠️ Hardware and Inference Speed

Bark has been tested and works on both CPU and GPU (`pytorch 2.0+`, CUDA 11.7 and CUDA 12.0).

On enterprise GPUs and PyTorch nightly, Bark can generate audio in roughly real-time. On older GPUs, default colab, or CPU, inference time might be significantly slower. For older GPUs or CPU you might want to consider using smaller models. Details can be found in out tutorial sections here.

The full version of Bark requires around 12GB of VRAM to hold everything on GPU at the same time.
To use a smaller version of the models, which should fit into 8GB VRAM, set the environment flag `SUNO_USE_SMALL_MODELS=True`.

If you don't have hardware available or if you want to play with bigger versions of our models, you can also sign up for early access to our model playground [here](https://suno-ai.typeform.com/suno-studio).

## ⚙️ Details

Bark is fully generative text-to-audio model devolved for research and demo purposes. It follows a GPT style architecture similar to [AudioLM](https://arxiv.org/abs/2209.03143) and [Vall-E](https://arxiv.org/abs/2301.02111) and a quantized Audio representation from [EnCodec](https://github.com/facebookresearch/encodec). It is not a conventional TTS model, but instead a fully generative text-to-audio model capable of deviating in unexpected ways from any given script. Different to previous approaches, the input text prompt is converted directly to audio without the intermediate use of phonemes. It can therefore generalize to arbitrary instructions beyond speech such as music lyrics, sound effects or other non-speech sounds.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://suno.ai/discord)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- CAPITALIZATION for emphasis of a word
- `[MAN]` and `[WOMAN]` to bias Bark toward male and female speakers, respectively

### Supported Languages

| Language                 | Status |
| ------------------------ | :----: |
| English (en)             |   ✅   |
| German (de)              |   ✅   |
| Spanish (es)             |   ✅   |
| French (fr)              |   ✅   |
| Hindi (hi)               |   ✅   |
| Italian (it)             |   ✅   |
| Japanese (ja)            |   ✅   |
| Korean (ko)              |   ✅   |
| Polish (pl)              |   ✅   |
| Portuguese (pt)          |   ✅   |
| Russian (ru)             |   ✅   |
| Turkish (tr)             |   ✅   |
| Chinese, simplified (zh) |   ✅   |

Requests for future language support [here](https://github.com/suno-ai/bark/discussions/111) or in the **#forums** channel on [Discord](https://suno.ai/discord).

## 🙏 Appreciation

- [nanoGPT](https://github.com/karpathy/nanoGPT) for a dead-simple and blazing fast implementation of GPT-style models
- [EnCodec](https://github.com/facebookresearch/encodec) for a state-of-the-art implementation of a fantastic audio codec
- [AudioLM](https://github.com/lucidrains/audiolm-pytorch) for related training and inference code
- [Vall-E](https://arxiv.org/abs/2301.02111), [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of Bark

## © License

Bark is licensed under the MIT License.

## 📱 Community

- [Twitter](https://twitter.com/suno_ai_)
- [Discord](https://suno.ai/discord)

## 🎧 Suno Studio (Early Access)

We’re developing a playground for our models, including Bark.

If you are interested, you can sign up for early access [here](https://suno-ai.typeform.com/suno-studio).

## ❓ FAQ

#### How do I specify where models are downloaded and cached?

- Bark uses Hugging Face to download and store models. You can see find more info [here](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome).

#### Bark's generations sometimes differ from my prompts. What's happening?

- Bark is a GPT-style model. As such, it may take some creative liberties in its generations, resulting in higher-variance model outputs than traditional text-to-speech approaches.

#### What voices are supported by Bark?

- Bark supports 100+ speaker presets across [supported languages](#supported-languages). You can browse the library of speaker presets [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). The community also shares presets in [Discord](https://suno.ai/discord). Bark also supports generating unique random voices that fit the input text. Bark does not currently support custom voice cloning.

#### Why is the output limited to ~13-14 seconds?

- Bark is a GPT-style model, and its architecture/context window is optimized to output generations with roughly this length.

#### How much VRAM do I need?

- The full version of Bark requires around 12Gb of memory to hold everything on GPU at the same time. However, even smaller cards down to ~2Gb work with some additional settings. Simply add the following code snippet before your generation:

```python
import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
```

#### My generated audio sounds like a 1980s phone call. What's happening?

- Bark generates audio from scratch. It is not meant to create only high-fidelity, studio-quality speech. Rather, outputs could be anything from perfect speech to multiple people arguing at a baseball game recorded with bad microphones.
