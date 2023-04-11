# 🐶 Bark

<a href="http://www.repostatus.org/#active"><img src="http://www.repostatus.org/badges/latest/active.svg" /></a>
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/OnusFM.svg?style=social&label=Follow%20%40OnusFM)](https://twitter.com/OnusFM)
[![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE?compact=true&style=flat)](https://discord.gg/J2B2vsjKuE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mnpdHViSiu4VIDGYA3k8aNgXmgOGWgU-?usp=sharing)

[Examples](https://suno-ai.notion.site/Bark-Examples-5edae8b02a604b54a42244ba45ebc2e2) | [Model Card](./model-card.md)

Bark is a transformer-based text-to-audio model created by [Suno](https://suno.ai). It can generate highly realistic multilingual speech, other audio, including music and background noise, and speaker emotions like laughing, sighing and crying. To support the community we give access to pretrained model checkpoints ready for inference.

<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/230698495-cbb1ced9-c911-4c9a-941d-a1a4a1286ac6.png" width="500"></img>
</p>

## 🤖 Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mnpdHViSiu4VIDGYA3k8aNgXmgOGWgU-?usp=sharing)

```python
from bark import SAMPLE_RATE, generate_audio
from IPython.display import Audio

text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)
Audio(audio_array, rate=SAMPLE_RATE)
```

[pizza.webm](https://user-images.githubusercontent.com/5068315/230490503-417e688d-5115-4eee-9550-b46a2b465ee3.webm)

### 🌎 Foreign Language

Bark supports various languages out-of-the-box and automatically determines language from input text. Code-switched text will even realistically use the same voice and add an accent.

```python
text_prompt = """
    Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. 
    But I suppose your english isn't terrible.
"""
audio_array = generate_audio(text_prompt)
```

[miguel.webm](https://user-images.githubusercontent.com/5068315/230684752-10baadfe-1e7c-46a2-8323-43282aef2c8c.webm)

### 🎶 Music

Bark can generate all types of audio, and in principle doesn't see a difference between speech and music. Sometimes it chooses to generate text as music, but you can help it out by adding notes around your lyrics.
```python
text_prompt = """
    ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
"""
audio_array = generate_audio(text_prompt)
```

[lion.webm](https://user-images.githubusercontent.com/5068315/230684766-97f5ea23-ad99-473c-924b-66b6fab24289.webm)

### 👥 Speaker Prompts

You can provide certain speaker prompts such as NARRATOR, MAN, WOMAN, etc. (Note that these are not always respected, especially if a conflicting audio history prompt is given.)

```python
text_prompt = """
    WOMAN: I would like an oatmilk latte please.
    MAN: Wow, that's expensive!
"""
audio_array = generate_audio(text_prompt)
```

[latte.webm](https://user-images.githubusercontent.com/5068315/230684864-12d101a1-a726-471d-9d56-d18b108efcb8.webm)

### 🎤 Voice/Audio Cloning

Bark has the capability to fully clone voices as well pick up music, ambience, etc. from input clips. However, to avoid misuse of this technology we limit the audio history prompts to a limited set of Suno-provided, fully synthetic options to choose from. 

```python
text_prompt = """
    I have a silky smooth voice, and today I will tell you about 
    the exercise regimen of the common sloth.
"""
audio_array = generate_audio(text_prompt, history_prompt="speech_0")
```

[sloth.webm](https://user-images.githubusercontent.com/5068315/230684883-a344c619-a560-4ff5-8b99-b4463a34487b.webm)

## 💻 Installation

```
pip install git+https://github.com/suno-ai/bark.git
```

or

```
git clone https://github.com/suno-ai/bark
cd bark && pip install . 
```

## 🛠️ Hardware and Inference Speed

Bark has been tested and works on both CPU and GPU (`pytorch 2.0+`, CUDA 11.7 and CUDA 12.0).
Running Bark requires running >100M parameter transformer models.
On modern GPUs and PyTorch nightly, Bark can generate audio in roughly realtime. On older GPUs, default colab, or CPU, inference time might be 10-100x slower. 

If you don't have new hardware available or if you want to play with bigger versions of our models, you can also sign up for early access to our Studio [here](https://3os84zs17th.typeform.com/suno-studio).

## ⚙️ Details

Similar to [Vall-E](https://arxiv.org/abs/2301.02111) and some other amazing work in the field, Bark uses GPT-style 
models to generate audio from scratch. Different from Vall-E, the initial text prompt is embedded into high-level semantic tokens without the use of phonemes. It can therefore generalize to arbitrary instructions beyond speech that occur in the training data, such as music lyrics, sound effects or other non-speech sounds. A subsequent second model is used to convert the generated semantic tokens into audio codec tokens to generate the full waveform. To enable the community to use Bark via public code we used the fantastic 
[EnCodec codec](https://github.com/facebookresearch/encodec) from Facebook to act as an audio representation.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://discord.gg/DqEx7FGbFP)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- capitalization for emphasis of a word
- `MAN/WOMAN:` for bias towards speaker


**Supported Languages**

| Language | Status |
| --- | --- |
| Chinese (Mandarin) | ✅ |
| English  | ✅ |
| French | ✅ |
| German | ✅ |
| Hindi  | ✅ |
| Italian | ✅ |
| Japanese | ✅ |
| Korean | ✅ |
| Polish | ✅ |
| Portuguese | ✅ |
| Russian | ✅ |
| Spanish | ✅ |
| Turkish | ✅ |
| Arabic  | Coming soon! |
| Bengali | Coming soon! |
| Telugu | Coming soon! |

## 🙏 Appreciation

- [nanoGPT](https://github.com/karpathy/nanoGPT) for a dead-simple and blazing fast implementation of gpt-style models
- [EnCodec](https://github.com/facebookresearch/encodec) for a state-of-the-art implementation of a fantastic audio codec
- [AudioLM](https://github.com/lucidrains/audiolm-pytorch) for very related training and inference code
- [Vall-E](https://arxiv.org/abs/2301.02111), [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of Bark

## © License

Bark is licensed under a non-commercial CC-BY 4.0 NC. The Suno models themselves may be used commercially. However, this version of Bark uses `EnCodec` as a neural codec backend, which is licensed under a [non-commercial license](https://github.com/facebookresearch/encodec/blob/main/LICENSE).

Please contact us at `bark@suno.ai` if you need access to a larger version of the model and/or a version of the model you can use commercially.  

## 📱 Community

- [Twitter](https://twitter.com/OnusFM)
- [Discord](https://discord.gg/J2B2vsjKuE)

## 🎧 Suno Studio (Early Access)

We’re developing a web interface for our models, including Bark. 

You can sign up for early access [here](https://3os84zs17th.typeform.com/suno-studio).
