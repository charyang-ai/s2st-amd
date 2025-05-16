from __future__ import annotations

import os
import pathlib

import gradio as gr
import numpy as np
import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from seamless_communication.inference import Translator

from lang_list import (
    ASR_TARGET_LANGUAGE_NAMES,
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
)

HF_TOKEN = "add key here"
CHECKPOINTS_PATH = pathlib.Path("/home/charyang/speech/demo/seamless-m4t-v2-large").resolve()
# if not CHECKPOINTS_PATH.exists():
#     snapshot_download(
#         repo_id="facebook/seamless-m4t-v2-large", repo_type="model", local_dir=CHECKPOINTS_PATH, token=HF_TOKEN
#     )
asset_store.env_resolvers.clear()
asset_store.env_resolvers.append(lambda: "demo")
demo_metadata = [
    {
        "name": "seamlessM4T_v2_large@demo",
        "checkpoint": f"{CHECKPOINTS_PATH}/seamlessM4T_v2_large.pt",
        "char_tokenizer": f"{CHECKPOINTS_PATH}/spm_char_lang38_tc.model",
    },
    {
        "name": "vocoder_v2@demo",
        "checkpoint": f"{CHECKPOINTS_PATH}/vocoder_v2.pt",
    },
]
asset_store.metadata_providers.append(InProcAssetMetadataProvider(demo_metadata))

DESCRIPTION = """\
# Computex 2025 AMD
## Live Translator Powered by AMD Radeon w9070 Platform

"""

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "English"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    dtype = torch.float16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("mps")
    dtype = torch.float32
else:
    device = torch.device("cpu")
    dtype = torch.float32

translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",
    vocoder_name_or_card="vocoder_v2",
    device=device,
    dtype=dtype,
    apply_mintox=True,
)


def preprocess_audio(input_audio: str) -> None:
    arr, org_sr = torchaudio.load(input_audio)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    torchaudio.save(input_audio, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))


def run_s2st(
    input_audio: str, source_language: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    preprocess_audio(input_audio)
    print("input_audio:",type(input_audio))
    print("input_audio:",input_audio)
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    out_texts, out_audios = translator.predict(
        input=input_audio,
        task_str="S2ST",
        src_lang=source_language_code,
        tgt_lang=target_language_code,
    )
    print("out_audios:",type(out_audios))
    out_text = str(out_texts[0])
    out_wav = out_audios.audio_wavs[0].cpu().detach().numpy()
    print("out_wav:",type(out_wav))
    return (int(AUDIO_SAMPLE_RATE), out_wav), out_text



with gr.Blocks() as demo_s2st:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_audio = gr.Audio(label="Input speech",sources="microphone", type="filepath")
                btn = gr.Button("Translate")
                source_language = gr.Dropdown(
                    label="Source language",
                    choices=ASR_TARGET_LANGUAGE_NAMES,
                    value="Mandarin Chinese",
                )
                target_language = gr.Dropdown(
                    label="Target language",
                    choices=S2ST_TARGET_LANGUAGE_NAMES,
                    value=DEFAULT_TARGET_LANGUAGE,
                )

        with gr.Column():
            with gr.Group():
                output_audio = gr.Audio(
                    label="Translated speech",
                    autoplay=True,
                    streaming=True,
                    type="numpy",
                )
                output_text = gr.Textbox(label="Translated text")

        input_audio.stop_recording(
                    fn=run_s2st,
                    inputs=[input_audio, source_language, target_language],
                    outputs=[output_audio, output_text],
                    api_name="s2st",
                )
    # btn.click(

    # btn.click(
    #     fn=run_s2st,
    #     inputs=[input_audio, source_language, target_language],
    #     outputs=[output_audio, output_text],
    #     api_name="s2st",
    # )


with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.Tab(label="Speech-to-Speech Tranlation"):
            demo_s2st.render()


if __name__ == "__main__":
    demo.queue(max_size=50).launch(share=True)
