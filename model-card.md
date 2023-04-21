# Model Card: Bark

This is the official codebase for running the text to audio model, from Suno.ai.

The following is additional information about the models released here. 

## Model Details

Bark is a series of three transformer models that turn text into audio.
### Text to semantic tokens
 - Input: text, tokenized with [BERT tokenizer from Hugging Face](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer)
 - Output: semantic tokens that encode the audio to be generated

### Semantic to coarse tokens
 - Input: semantic tokens
 - Output: tokens from the first two codebooks of the [EnCodec Codec](https://github.com/facebookresearch/encodec) from facebook

### Coarse to fine tokens
 - Input: the first two codebooks from EnCodec
 - Output: 8 codebooks from EnCodec

### Architecture
|           Model           | Parameters | Attention  | Output Vocab size |  
|:-------------------------:|:----------:|------------|:-----------------:|
|  Text to semantic tokens  |    80 M    | Causal     |       10,000      |
| Semantic to coarse tokens |    80 M    | Causal     |     2x 1,024      |
|   Coarse to fine tokens   |    80 M    | Non-causal |     6x 1,024      |


### Release date
April 2023

## Broader Implications
We anticipate that this model's text to audio capabilities can be used to improve accessbility tools in a variety of languages. 
Straightforward improvements will allow models to run faster than realtime, rendering them useful for applications such as virtual assistants. 
 
While we hope that this release will enable users to express their creativity and build applications that are a force
for good, we acknowledge that any text to audio model has the potential for dual use. While it is not straightforward
to voice clone known people with Bark, they can still be used for nefarious purposes. To further reduce the chances of unintended use of Bark, 
we also release a simple classifier to detect Bark-generated audio with high accuracy (see notebooks section of the main repository). 
