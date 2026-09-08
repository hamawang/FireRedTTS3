import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import os
import re
import torch
import torchaudio
import numpy as np
import gradio as gr
from functools import partial
from argparse import ArgumentParser
from fireredtts3.core import FireRedTTS3Instruct  # noqa: E402


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
MAX_EDIT_SECONDS = 20.0
MAX_TEXT_CHARS = 400


def _load_audio(path: str, max_seconds: float):
    if not path:
        raise gr.Error("Please provide an audio file first.")
    audio, audio_sr = torchaudio.load(path)
    audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1]/audio_sr > max_seconds:
        audio = audio[:, :int(audio_sr * max_seconds)]
        gr.Info(f"Audio truncated to the first {max_seconds:.0f}s.")
    return audio, audio_sr


def _to_gradio_audio(audio: torch.Tensor, sr: int):
    x = audio.detach().float().cpu().numpy()
    if x.ndim > 1:
        x = x[0]
    x = np.clip(x, -1.0, 1.0)
    return sr, (x * 32767.0).astype(np.int16)


def _check_text(text: str, what: str = "Text"):
    text = (text or "").strip()
    if not text:
        raise gr.Error(f"{what} must not be empty.")
    if len(text) > MAX_TEXT_CHARS:
        gr.Info(f"{what} truncated to {MAX_TEXT_CHARS} characters.")
        text = text[:MAX_TEXT_CHARS]
    return text


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def voice_design(
    tts: FireRedTTS3Instruct,
    instruction,
    text,
    inference_cfg=1.2,
    seed=2,
    do_tn=True,
):
    instruction = _check_text(instruction, "Voice description")
    text = _check_text(text, "Text to synthesize")

    gen_audio, gen_sr, gen_text = tts.generate_voice_design(
        instruction=instruction,
        text=text,
        n_timesteps=10,
        inference_cfg=float(inference_cfg),
        seed=int(seed),
        do_tn=bool(do_tn),
    )
    return _to_gradio_audio(gen_audio, gen_sr), (gen_text or "").strip()


def semantic_edit(
    tts: FireRedTTS3Instruct,
    audio_in,
    instruction,
    inference_cfg=1.2,
    seed=1234,
):
    instruction = _check_text(instruction, "Edit instruction")
    wav, sr = _load_audio(audio_in, MAX_EDIT_SECONDS)
    gen_audio, gen_sr, gen_text = tts.generate_semantic_edit(
        instruction=instruction,
        audio_in=wav,
        audio_in_sr=sr,
        n_timesteps=10,
        inference_cfg=float(inference_cfg),
        seed=int(seed),
    )
    return _to_gradio_audio(gen_audio, gen_sr), (gen_text or "").strip()


def compose_acoustic_instruction(attribute: str, value: float) -> str:
    """Acoustic edits only accept the templates the model was trained on."""
    if attribute == "Speed":
        return f"adjust the speed to {value:.1f}x"
    if attribute == "Volume":
        return f"adjust the volume to {value:.1f}"
    steps = int(round(value))
    return f"shift the pitch by {steps} step{'' if abs(steps) == 1 else 's'}"


def acoustic_edit(
    tts: FireRedTTS3Instruct,
    audio_in,
    attribute="Speed",
    value=0.8,
    inference_cfg=1.2,
    seed=1234,
):
    wav, sr = _load_audio(audio_in, MAX_EDIT_SECONDS)
    instruction = compose_acoustic_instruction(attribute, float(value))
    gen_audio, gen_sr = tts.generate_acoustic_edit(
        instruction=instruction,
        audio_in=wav,
        audio_in_sr=sr,
        n_timesteps=10,
        inference_cfg=float(inference_cfg),
        seed=int(seed),
    )
    return _to_gradio_audio(gen_audio, gen_sr), instruction


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def create_demo_interface(tts: FireRedTTS3Instruct):
    with gr.Blocks() as interface:
        gr.Markdown("""# 🔥 FireRedTTS3 - Instruction-Guided Voice Design and Speech Editing.""")
        
        with gr.Tabs():
            # --- Voice Design
            with gr.Tab("🎨 Voice Design"):
                with gr.Row():
                    with gr.Column():
                        design_instruction = gr.Textbox(
                            label="Voice description",
                            placeholder="e.g. A young woman with a gentle voice, speaking slowly…",
                            lines=3,
                        )
                        design_text = gr.Textbox(
                            label="Text to synthesize", lines=4,
                            placeholder="Type the text you want spoken…",
                        )
                        with gr.Accordion("Advanced options", open=False):
                            with gr.Row():   
                                design_cfg = gr.Number(value=1.2, precision=1, label="CFG strength")
                                design_seed = gr.Number(value=2, precision=0, label="Seed")
                            design_tn = gr.Checkbox(value=True, label="Text normalization")
                        design_btn = gr.Button("Design voice", variant="primary")
                    with gr.Column():
                        design_out = gr.Audio(label="Generated speech", type="numpy")
                        design_plan = gr.Textbox(
                            label="Voice-attribute plan (model chain-of-thought)", lines=4
                        )

            # --- Speech Edit
            with gr.Tab("✂️ Speech Editing"):
                with gr.Tabs():
                    # --- Semantic
                    with gr.Tab("Semantic (content)"):
                        gr.Markdown(
                            "Insert, delete or substitute words in an existing recording "
                            "while keeping the original voice. The model transcribes the "
                            "audio itself — just say what to change."
                        )
                        with gr.Row():
                            with gr.Column():
                                sem_audio = gr.Audio(
                                    label="Input speech (≤ 20 s)",
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                )
                                sem_instruction = gr.Textbox(
                                    label="Edit instruction",
                                    placeholder="e.g. Replace 'positive' with 'optimistic'.",
                                    lines=2,
                                )
                                with gr.Accordion("Advanced options", open=False):
                                    with gr.Row():
                                        sem_cfg = gr.Number(value=1.2, precision=1, label="CFG strength")
                                        sem_seed = gr.Number(value=1234, precision=0, label="Seed")
                                sem_btn = gr.Button("Apply edit", variant="primary")
                            with gr.Column():
                                sem_out = gr.Audio(label="Edited speech", type="numpy")
                                sem_text = gr.Textbox(label="Edited transcript", lines=3)
                    # --- Acoustic
                    with gr.Tab("Acoustic (speed / pitch / volume)"):
                        gr.Markdown(
                            "Re-render the same utterance with a different speaking rate, "
                            "pitch or loudness. These edits follow fixed instruction "
                            "templates the model was trained on."
                        )
                        with gr.Row():
                            with gr.Column():
                                aco_audio = gr.Audio(
                                    label="Input speech (≤ 20 s)",
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                )
                                aco_attr = gr.Radio(
                                    ["Speed", "Pitch", "Volume"],
                                    value="Speed",
                                    label="Attribute",
                                )
                                aco_value = gr.Slider(
                                    0.5, 2.0, value=0.8, step=0.1,
                                    label="Speed (×)",
                                )
                                with gr.Accordion("Advanced options", open=False):
                                    with gr.Row():
                                        aco_cfg = gr.Number(value=1.2, precision=1, label="CFG strength")
                                        aco_seed = gr.Number(value=1234, precision=0, label="Seed")
                                aco_btn = gr.Button("Apply edit", variant="primary")
                            with gr.Column():
                                aco_out = gr.Audio(label="Edited speech", type="numpy")
                                aco_instruction = gr.Textbox(
                                    label="Instruction sent to the model", lines=1
                                )
        
        def _attr_changed(attribute, current):
            lo, hi, step, label = {
                "Speed": (0.5, 2.0, 0.1, "Speed (×)"),
                "Volume": (0.3, 2.0, 0.1, "Volume (×)"),
                "Pitch": (-6, 6, 1, "Pitch shift (semitone steps)"),
            }[attribute]
            try:
                value = min(max(float(current), lo), hi)
            except (TypeError, ValueError):
                value = lo
            if attribute == "Pitch":
                value = int(round(value)) or 1
            return gr.update(minimum=lo, maximum=hi, step=step, value=value, label=label)

        aco_attr.change(_attr_changed, inputs=[aco_attr, aco_value], outputs=[aco_value])
        design_btn.click(
            partial(voice_design, tts),
            inputs=[design_instruction, design_text, design_cfg, design_seed, design_tn],
            outputs=[design_out, design_plan],
            api_name="voice_design",
        )
        sem_btn.click(
            partial(semantic_edit, tts),
            inputs=[sem_audio, sem_instruction, sem_cfg, sem_seed],
            outputs=[sem_out, sem_text],
            api_name="semantic_edit",
        )
        aco_btn.click(
            partial(acoustic_edit, tts),
            inputs=[aco_audio, aco_attr, aco_value, aco_cfg, aco_seed],
            outputs=[aco_out, aco_instruction],
            api_name="acoustic_edit",
        )

    return interface


if __name__ == "__main__":
    parser = ArgumentParser(description="FireRedTTS3-Instruct Demo Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root-path", type=str, default=os.environ.get("GRADIO_ROOT_PATH", ""))    # For debugging
    args = parser.parse_args()

    # Initialize App
    tts = FireRedTTS3Instruct(
        'pretrained_models',
        use_fasttext=True,
        use_llm_tn=False,
        use_wetext=True,
    )
    interface = create_demo_interface(tts)

    launch_kwargs = dict(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
        allowed_paths=[
            os.path.abspath("assets"),
        ],
    )
    if args.root_path:
        launch_kwargs["root_path"] = args.root_path
    interface.queue(default_concurrency_limit=1).launch(**launch_kwargs)