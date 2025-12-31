# 🐶 Bark - Text to Speech Made Simple

> **What is Bark?** Turn any text into realistic speech audio. Type words, get speech. It's that simple.

[![Discord](https://dcbadge.vercel.app/api/server/J2B2vsjKuE?style=flat&compact=True)](https://suno.ai/discord)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/FM.svg?style=social&label=@suno_ai_)](https://twitter.com/suno_ai_)

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/235310676-a4b3b511-90ec-4edf-8153-7ccf14905d73.png" width="500"></img>
</p>
<br>

## 🚀 Quick Start - 3 Steps to Get Running

### Step 1: Install Bark

Copy and paste this command into your terminal:

```bash
pip install git+https://github.com/suno-ai/bark.git
```

**⚠️ WARNING:** Do NOT use `pip install bark` - that's a different package!

### Step 2: Create a Simple Test File

Create a new file called `test_bark.py` and paste this code:

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# Download models (only needed first time)
preload_models()

# Your text here
text = "Hello, my name is Bark. I can turn text into speech!"

# Generate audio
audio = generate_audio(text)

# Save to file
write_wav("output.wav", SAMPLE_RATE, audio)

print("Done! Check output.wav")
```

### Step 3: Run It

```bash
python test_bark.py
```

That's it! You'll get an audio file called `output.wav` with your text spoken out loud.

---

## 🎯 One-Line Command to Test

Want to test even faster? Just run this:

```bash
python -m bark --text "Hello, this is a test" --output_filename "test.wav"
```

---

## 💡 Try These Demos (No Installation Needed)

- [🤗 Try in Browser](https://huggingface.co/spaces/suno/bark) - No setup required
- [🔬 Google Colab](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing) - Run in your browser
- [Listen to Examples](https://suno.ai/examples/bark-v0)

---

## 🎨 Cool Things You Can Do

### Make It Laugh or Sigh

```python
text = "Hello [laughs] this is amazing! [sighs] But I'm a bit tired."
audio = generate_audio(text)
write_wav("emotions.wav", SAMPLE_RATE, audio)
```

### Different Voices

```python
# Use a different speaker voice
text = "I have a different voice now!"
audio = generate_audio(text, history_prompt="v2/en_speaker_6")
write_wav("different_voice.wav", SAMPLE_RATE, audio)
```

Browse 100+ voice options [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c).

### Generate Music

```python
text = "♪ In the jungle, the mighty jungle, the lion sleeps tonight ♪"
audio = generate_audio(text)
write_wav("music.wav", SAMPLE_RATE, audio)
```

### Other Languages (Auto-detected)

```python
# Korean
text = "안녕하세요, 저는 Bark입니다"
audio = generate_audio(text)

# Spanish
text = "Hola, mi nombre es Bark"
audio = generate_audio(text)

# French, German, Japanese, and more work too!
```

**Supported Languages:** English, German, Spanish, French, Hindi, Italian, Japanese, Korean, Polish, Portuguese, Russian, Turkish, Chinese

---

## 🛠️ Common Issues & Fixes

### "Not enough memory" or "CUDA out of memory"

Add this at the top of your Python script:

```python
import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
```

### Audio is only ~13 seconds long

Bark is designed for short audio clips. For longer audio, check out the [long-form generation notebook](notebooks/long_form_generation.ipynb).

### Where are the models downloaded?

Models are stored using Hugging Face cache. Default location:
- Linux/Mac: `~/.cache/huggingface/hub`
- Windows: `C:\Users\YourName\.cache\huggingface\hub`

---

## 📚 More Examples

### Full Basic Example

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# First time only - downloads models
preload_models()

# Generate with emotions and pauses
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs]
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# Save it
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
```

### All the Special Commands

Add these to your text for special effects:

- `[laughter]` or `[laughs]` - Make it laugh
- `[sighs]` - Make it sigh
- `[music]` - Add music
- `[gasps]` - Gasping sound
- `[clears throat]` - Throat clearing
- `—` or `...` - Pauses/hesitation
- `♪ lyrics ♪` - Sing lyrics
- `CAPITALIZATION` - Emphasize a word
- `[MAN]` or `[WOMAN]` - Hint at gender

---

## 💻 System Requirements

**Minimum:**
- 2GB GPU VRAM (with small models enabled)
- Or just use CPU (slower but works)

**Recommended:**
- 12GB GPU VRAM for best quality and speed
- CUDA 11.7+ or CUDA 12.0+

**No GPU?** It still works on CPU, just slower. Use the small models trick above.

---

## 🤝 Community & Help

- [Discord Community](https://suno.ai/discord) - Get help, share voices, tips
- [Twitter](https://twitter.com/suno_ai_) - Latest updates
- [Report Issues](https://github.com/suno-ai/bark/issues)

---

## ⚠️ Important Notes

- Bark is for **research and creative use**
- Audio quality varies - it's not always perfect studio quality
- Sometimes it gets creative and doesn't match your text exactly
- Free for commercial use (MIT License)
- Does NOT support custom voice cloning

---

## 🎓 Advanced Usage

### Using with Hugging Face Transformers

If you prefer using the Transformers library:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

Then:

```python
from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

inputs = processor("Hello, my dog is cute", voice_preset="v2/en_speaker_6")
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
```

[See full Transformers documentation](https://huggingface.co/docs/transformers/main/en/model_doc/bark)

---

## 📖 Learn More

- [Long-form Generation Guide](notebooks/long_form_generation.ipynb) - Create longer audio
- [Voice Library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) - Browse all voices
- [Model Details](model-card.md) - Technical architecture info

---

## 📄 License

MIT License - Free for commercial use!

---

**Made by [Suno](https://suno.ai) - Check out our other AI audio projects!**
