#!/usr/bin/env python3
"""Design a voice from a description — `llmctl voice design`.

Runs Qwen3-TTS VoiceDesign (PyTorch) once to speak a sample sentence in a voice
the description asks for, and writes it as a reference recording. The speech
backend (tts_server.py, llama.cpp) then clones that recording for every request.
Designing is slow — about 1.5x real time here, bound by kernel launches — but it
happens once per voice; speaking is fast.

Describing the speaker as a native speaker of the language matters: the preset
voices of the CustomVoice model are English and Chinese speakers and keep their
accent in German, while a designed "deutscher Muttersprachler" does not, and
neither does a clone of it.

    tts_voice_design.py MODEL_DIR OUT.wav "description" [--language German] [--text "..."]
"""
import argparse
import warnings

SAMPLES = {
    "German": "Guten Tag. Ich lese Ihnen gern etwas vor, ob kurze Ansagen oder längere Texte. "
              "Heute ist es draußen grau, aber freundlich.",
    "English": "Hello. I am happy to read to you, whether short announcements or longer texts. "
               "It is grey outside today, but friendly.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("description")
    p.add_argument("--language", default="German")
    p.add_argument("--text")
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    import soundfile as sf
    import torch
    # MIOpen searches a new convolution kernel for every new length and falls back
    # to a slow one meanwhile (9x real time instead of 1.6x); PyTorch's own is steady.
    torch.backends.cudnn.enabled = False
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained(a.model, device_map="cuda:0", dtype=torch.bfloat16,
                                      attn_implementation="sdpa")
    text = a.text or SAMPLES.get(a.language, SAMPLES["English"])
    wavs, sr = m.generate_voice_design(text=text, instruct=a.description, language=a.language)
    sf.write(a.out, wavs[0], sr)
    print(f"{a.out}: {len(wavs[0]) / sr:.1f} s")


if __name__ == "__main__":
    main()
