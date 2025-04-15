# Seminar 12: Audio Generation

This seminar folder contains materials for the twelfth seminar in our Deep Generative Models course. In this session, we explore techniques in generating audio content, focusing on two major approaches: text-to-speech (TTS) synthesis using Coqui TTS and text-to-music generation with MusicGen.

## Seminar Overview

- **Topic**: Audio Generation with Coqui TTS and MusicGen  
- **Objective**:
  - **Understanding TTS Systems**: Explore the end-to-end process that converts text into high-fidelity speech, including text normalization, phonetic analysis, acoustic modeling, and vocoding.
  - **Coqui TTS Inference & Training**: Learn how to run inference using pre-trained Coqui TTS models, utilize custom speaker configurations, and examine the training pipeline with the GlowTTS architecture using the LJSpeech dataset.
  - **Music Generation with MusicGen**: Investigate how an auto-regressive Transformer model can generate music from text prompts, leveraging neural audio codecs and residual vector quantization for efficient, high-quality output.
  - **Practical Implementation**: Set up the environment, execute command-line utilities, and interact with audio outputs directly within a Jupyter Notebook.

## Key Concepts

1. **Coqui TTS for Speech Generation**  
   - **Pipeline Overview**: Modern TTS systems first preprocess text (normalize, expand abbreviations, apply Grapheme-to-Phoneme conversion), then perform linguistic/phonetic analysis, and finally predict acoustic representations which are transformed into audio waveforms using vocoders.
   - **Inference Examples**:
     - Execute commands to list pre-trained models and synthesize speech (e.g., generating "hello world" with a specific model).
     - Demonstrate speaker identification and customization using options like `--list_speaker_idxs` and external speaker inputs via `--speaker_wav`.
   - **Training Workflow**:
     - Prepare the LJSpeech dataset by downloading and extracting the archive.
     - Initialize the GlowTTS model with detailed configuration parameters (batch size, epochs, phoneme settings, etc.).
     - Set up audio processing pipelines and tokenization for effective model training.
     - Optionally, perform a one-batch test (overfitting) to debug the training loop before full-scale model training.

2. **MusicGen for Music Generation**  
   - **Model Architecture**:
     - **Text Encoder**: Encodes input text prompts (e.g., "Romantic ballad with a gentle acoustic guitar...") to guide music generation.
     - **Auto-Regressive Decoder**: Predicts a sequence of discrete audio tokens representing compressed audio.
     - **Neural Codec Decoder (EnCodec)**: Decodes the token sequence to reconstruct a high-fidelity audio waveform.
   - **Generation Modes**:
     - **Unconditional Generation**: Produces audio without textual input, using the model's learned priors.
     - **Text-Conditional Generation**: Conditions output on text prompts, utilizing classifier-free guidance (with parameters like `guidance_scale` to balance alignment and audio quality).
     - **Audio-Prompted Generation**: Extends an input audio clip using a combination of audio and textual conditioning, providing creative continuity.
   - **Key Techniques**:
     - Use of residual vector quantization (Residual VQ) to compress audio efficiently.
     - The transformer’s attention mechanisms (self-attention and cross-attention) play a pivotal role in generating coherent and musically aligned audio.

## Useful Links

### Papers

- **Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search (Kim, 2020)**: [arXiv:2005.11129](https://arxiv.org/abs/2005.11129)
- **Simple and Controllable Music Generation (Copet et al., 2023)**: [arXiv:2306.05284](https://arxiv.org/abs/2306.05284)

### Blogs & Tutorials

- **Glow-TTS (Tae, 2022)**: [https://jaketae.github.io/study/glowtts/](https://jaketae.github.io/study/glowtts/)
- **MusicGen from Meta AI — Model Architecture, Vector Quantization and Model Conditining Explained (Sankar, 2023)**: [Blog on Medium](https://medium.com/@AIBites/musicgen-from-meta-ai-model-architecture-vector-quantization-and-model-conditining-explained-f9a030382f7d)
- **Neural Audio Codec - Encodec and SoundStream (Shvecov, 2023)**: [YouTube Video](https://youtu.be/L6wiWYCCGFM)
- **Hugging Face Transformers API: MusicGen**: [https://huggingface.co/docs/transformers/en/model_doc/musicgen](https://huggingface.co/docs/transformers/en/model_doc/musicgen)
- **MusicGen in 🤗 Transformers (Gandhi, 2024)**: [https://github.com/sanchit-gandhi/notebooks/blob/main/MusicGen.ipynb](https://github.com/sanchit-gandhi/notebooks/blob/main/MusicGen.ipynb)
- **CoquiTTS Tutorial: Train your first 🐸 TTS model 💫**: [https://github.com/coqui-ai/TTS/blob/dev/notebooks/Tutorial_2_train_your_first_TTS_model.ipynb](https://github.com/coqui-ai/TTS/blob/dev/notebooks/Tutorial_2_train_your_first_TTS_model.ipynb)

### Code & Docs

- **Coqui TTS GitHub Repository**: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- **MusicGen Documentation**: [https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
- **LJSpeech Dataset Archive**: [https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2)
- **Transformers Documentation (Hugging Face)**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
