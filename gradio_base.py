import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import os
import logging
import torchaudio
import gradio as gr
from argparse import ArgumentParser
from fireredtts3.core import FireRedTTS3
from fireredtts3.utils.text_normalize import (
    build_wetext_normalizer,
    build_llm_normalizer,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



_HAS_DOTENV = os.path.exists(".env")


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
MULTI_LINGUAL_CASES = [
    ("Chinese",    "Chinese (中文)",        "今日体测数据：静息心率62.4次/分，体脂率18.7%，深蹲最大负重95.3公斤。"),
    ("English",    "English (英文)",        "The conference starts at 9:30 AM and ends at 5:15 PM on 12/10/2023."),
    ("Cantonese",  "Cantonese (粤语)",      "我哋聽日夜晚八點喺茶餐廳食飯，順便傾下嗰個合作計劃，埋單大概兩百蚊。"),
    ("Japanese",   "Japanese (日文)",       "会議は2024年5月10日午前10時に始まります。参加費用は¥1,500です。"),
    ("Korean",     "Korean (韩文)",         "회의는 2023년 10월 12일 오전 10시에 시작되며, 참가비는 25,000원입니다."),
    ("Spanish",    "Spanish (西班牙语)",    "La reunión es el 15 de octubre a las 10:30 de la mañana, la entrada cuesta 25 dólares."),
    ("French",     "French (法语)",         "La réunion aura lieu le 15 octobre à 10h30, le vol AF123 partira à 18h45."),
    ("Russian",    "Russian (俄语)",        "Встреча состоится 12 октября в 10:30 утра, проезд стоит 150 рублей."),
    ("Arabic",     "Arabic (阿拉伯语)",     "سيُعقد الاجتماع في 12 أكتوبر في العاشرة صباحاً، وتكلفة الدخول 15 ريالاً."),
    ("Turkish",    "Turkish (土耳其语)",    "Toplantı 12 Ekim'de saat 10:30'da yapılacak, giriş ücreti 150 liradır."),
    ("Indonesian", "Indonesian (印尼语)",   "Rapat akan diadakan pada 12 Oktober pukul 10:30, biaya masuknya Rp 150.000."),
    ("Portuguese", "Portuguese (葡萄牙语)", "A reunião será em 12 de outubro às 10h30, o ingresso custa R$ 25,00."),
    ("Italian",    "Italian (意大利语)",    "La riunione è fissata per il 12 ottobre alle 10:30, il biglietto costa 25 euro."),
    ("Dutch",      "Dutch (荷兰语)",        "De vergadering is op 12 oktober om 10:30, de toegang kost € 25,00."),
    ("Vietnamese", "Vietnamese (越南语)",   "Cuộc họp sẽ diễn ra vào lúc 10:30 ngày 12 tháng 10, vé vào cửa là 150.000 đồng."),
    ("German",     "German (德语)",         "Die Besprechung findet am 12. Oktober um 10:30 Uhr statt, der Eintritt kostet 25 €."),
    ("Ukrainian",  "Ukrainian (乌克兰语)",  "Зустріч відбудеться 12 жовтня о 10:30, вхід коштує 150 гривень."),
    ("Thai",       "Thai (泰语)",           "การประชุมจะจัดขึ้นในวันที่ 12 ตุลาคม เวลา 10:30 น. ค่าเข้าชม 150 บาท"),
    ("Polish",     "Polish (波兰语)",       "Spotkanie odbędzie się 12 października o 10:30, wstęp kosztuje 25 złotych."),
    ("Romanian",   "Romanian (罗马尼亚语)",  "Întâlnirea va avea loc pe 12 octombrie la ora 10:30, biletul costă 25 de lei."),
    ("Greek",      "Greek (希腊语)",        "Η συνάντηση θα γίνει στις 12 Οκτωβρίου στις 10:30, το εισιτήριο κοστίζει 15 ευρώ."),
    ("Czech",      "Czech (捷克语)",        "Schůzka se uskuteční 12. října v 10:30, vstupenka stojí 150 korun."),
    ("Finnish",    "Finnish (芬兰语)",      "Kokous pidetään 12. lokakuuta klo 10:30, pääsylippu maksaa 15 euroa."),
    ("Hindi",      "Hindi (印地语)",        "बैठक 12 अक्टूबर को सुबह 10:30 बजे होगी, प्रवेश शुल्क ₹150 है।"),
]
MULTI_DIALECT_CASES = [
    ("ZH_Anhui",   "Anhui (安徽话)",      "今个天挺好个，我们一块去街上逛逛吧，买点衣裳。"),
    ("ZH_Fujian",  "Fujian (福建话)",     "今仔日天气真好，咱来去公园散步，顺便买寡物件。"),
    ("ZH_Gansu",   "Gansu (甘肃话)",      "今个天气好得很，咱两个一搭去公园转一圈，吃碗牛肉面。"),
    ("ZH_Guizhou", "Guizhou (贵州话)",    "今天天气好得很，我们两个一路去公园耍，买点洋芋。"),
    ("ZH_Hebei",   "Hebei (河北话)",      "今儿个天儿不错，咱俩一块儿去公园遛达遛达。"),
    ("ZH_Henan",   "Henan (河南话)",      "今儿个天气真中，咱俩去公园耍耍吧，吃碗烩面。"),
    ("ZH_Hubei",   "Hubei (湖北话)",      "今天天气蛮好的，我们一起去公园逛哈子，买点热干面。"),
    ("ZH_Hunan",   "Hunan (湖南话)",      "今天天气蛮好的，我们一起去公园耍咯，呷碗米粉。"),
    ("ZH_Jiangxi", "Jiangxi (江西方言)",  "今日天气蛮好，我们一起去公园玩啰，恰碗瓦罐汤。"),
    ("ZH_Liaoning","Liaoning (辽宁话)",   "今儿个天儿挺好，咱俩一块儿去公园溜达溜达，撮顿烧烤。"),
    ("ZH_Minnan",  "Minnan (闽南话)",     "今呀日天气真好，咱来去公园行行，买淡薄物件。"),
    ("ZH_Ningxia", "Ningxia (宁夏话)",    "今个天气好的很，我们两个一起到公园转上一转，吃碗拉面。"),
    ("ZH_Shaanxi", "Shaanxi (陕西方言)",  "今儿个天气美得很，咱俩去公园谝一阵子，咥碗羊肉泡馍。"),
    ("ZH_Shandong","Shandong (山东话)",   "今儿这天儿真不孬，咱俩一块儿去公园逛逛，摊个煎饼。"),
    ("ZH_Shanghai","Shanghai (上海话)",   "今朝天气老好个，阿拉一道去公园白相相，吃只小笼包。"),
    ("ZH_Shanxi",  "Shanxi (山西话)",     "今儿个天气不错，咱俩去公园转圪转圪，吃碗刀削面。"),
    ("ZH_Sichuan", "Sichuan (四川话)",    "今天天气安逸得很，我们两个一路去公园耍哈，吃顿火锅。"),
    ("ZH_Tianjin", "Tianjin (天津话)",    "今儿个天儿倍儿好，咱俩去公园遛遛，来碗狗不理。"),
    ("ZH_Wenzhou", "Wenzhou (温州话)",    "该日天气几好，我伲一淘去公园走宕，吃份灯盏糕。"),
    ("ZH_Wu",      "Wu (吴语)",           "今朝天气蛮好个，阿拉一淘去公园白相，吃碗阳春面。"),
    ("ZH_Yunnan",  "Yunnan (云南方言)",   "今天天气板扎得很，我们两转去公园玩玩，甩碗过桥米线。"),
]
TAG2CASE = {
    case[0]: case
    for case in MULTI_LINGUAL_CASES+MULTI_DIALECT_CASES
}
GR_LANG_CHOICES = (
    [(display_name, lang_tag) for lang_tag, display_name, _ in MULTI_LINGUAL_CASES]
    + [("─────dialects─────", "_SEPARATOR_")]
    + [(display_name, lang_tag) for lang_tag, display_name, _ in MULTI_DIALECT_CASES]
)
GR_FRONTEND_CHOICES = [
    ("wetext (local, zh/en only)", "wetext"),
    ("llm_tn (LLM-based, all languages)", "llm_tn"),
    ("None (no normalization)", "none"),
]

# --------------------------------------------------------------------------- #
# Wrap around TTS3
# --------------------------------------------------------------------------- #
class FireRedDemoWrapper:
    # NOTE FireRedTTS3 should be loaded in advance
    def __init__(self, tts: FireRedTTS3):
        self.tts = tts
        # Original TN tool
        self.cached_wetext_normalizer = tts.wetext_normalizer
        self.cached_llm_normalizer = tts.llm_normalizer
        self.cached_llm_normalizer_args = (
            os.environ.get("LLM_TN_API_URL", None),
            os.environ.get("LLM_TN_API_KEY", None),
            os.environ.get("LLM_TN_MODEL", None)
        )
    
    def update_frontend(self, use_wetext: bool, use_llm_tn: bool, api_url: str, api_key: str, llm_tn_model: str):
        # Wetext
        if use_wetext:
            if self.cached_wetext_normalizer is None:
                self.cached_wetext_normalizer = build_wetext_normalizer()
            self.tts.wetext_normalizer = self.cached_wetext_normalizer
        else:
            self.tts.wetext_normalizer = None
        # LLM 
        if use_llm_tn:
            self.custom_llm_normalizer_args = (
                (api_url or "").strip() or self.cached_llm_normalizer_args[0],
                (api_key or "").strip() or self.cached_llm_normalizer_args[1],
                (llm_tn_model or "").strip() or self.cached_llm_normalizer_args[2],
            )
            if self.custom_llm_normalizer_args != self.cached_llm_normalizer_args or self.cached_llm_normalizer is None:
                self.custom_llm_normalizer = build_llm_normalizer(*self.custom_llm_normalizer_args)
            self.tts.llm_normalizer = self.custom_llm_normalizer
        else:
            self.tts.llm_normalizer = None

    def generate(
        self, 
        lang_tag: str, text: str, prompt_audio_path: str, prompt_text: str, # Input
        use_wetext: bool, use_llm_tn: bool, api_url: str, api_key: str, llm_tn_model: str,  # TN
        seed: int, cfg: float
    ):
        # Update TN
        self.update_frontend(use_wetext, use_llm_tn, api_url, api_key, llm_tn_model)
        # Prepare input
        prompt_audio, prompt_audio_sr = torchaudio.load(prompt_audio_path)
        prompt_audio = prompt_audio.mean(dim=0, keepdim=True)
        logger.info(
            f"[{lang_tag}] use_wetext: {use_wetext} use_llm_tn: {use_llm_tn} api_url: {api_url} api_key: {api_key} llm_tn_model: {llm_tn_model}"
            f"prompt_text: '{prompt_text}' generating: '{text}' with seed: {seed} cfg: {cfg}"
        )
        gen_audio, gen_audio_sr = self.tts.generate(
            language=lang_tag,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            prompt_audio_sr=prompt_audio_sr,
            text=text,
            do_clean=True,
            do_tn=(use_wetext or use_llm_tn),
            do_split=True,
            seed=seed,
            inference_cfg=cfg,
        )
        wav_np = gen_audio.squeeze(0).detach().cpu().numpy()
        return int(gen_audio_sr), wav_np


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def create_demo_interface(demo: FireRedDemoWrapper):
    with gr.Blocks() as interface:
        gr.Markdown(
            "# 🔥 FireRedTTS3 — Multilingual & Dialect TTS\n"
            "Zero-shot voice cloning with a reference (prompt) audio. "
            "Supports **24 languages** and **21 Chinese dialects**."
        )
        with gr.Accordion("🔑 LLM API Configuration (for llm_tn)", open=not _HAS_DOTENV):
            if _HAS_DOTENV:
                gr.Markdown(
                    "`.env` found; llm_tn will use `LLM_TN_API_URL` / "
                    "`LLM_TN_API_KEY` / `LLM_TN_MODEL` from it. "
                    "Override below (leave blank to use `.env`)."
                )
            else:
                gr.Markdown(
                    "No `.env` detected. If you enable **llm_tn**, "
                    "please provide your LLM API details below."
                )
            api_url = gr.Textbox(
                value="", label="LLM API URL",
                placeholder=os.environ.get("LLM_TN_API_URL", "https://your-api/chat/completions"),
            )
            api_key = gr.Textbox(
                value="", label="LLM API Key", type="password",
                placeholder=os.environ.get("LLM_TN_API_KEY", "sk-..."),
            )
            api_model = gr.Textbox(
                value="", label="LLM Model",
                placeholder=os.environ.get("LLM_TN_MODEL", "deepseek-v4-flash"),
            )
        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=GR_LANG_CHOICES,
                    value="Chinese",
                    label="🌐 Language / Dialect",
                )
                prompt_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="🎤 Prompt Audio (upload or click below for default)",
                )
                prompt_text = gr.Textbox(
                    label="📝 Prompt Transcript",
                    lines=2,
                    info="Transcript of the prompt audio",
                )
            with gr.Column(scale=1):
                text = gr.Textbox(
                    value=TAG2CASE['Chinese'][2],
                    label="✍️ Text to Synthesize",
                    lines=4,
                    info="Auto-filled with a sample when switching language",
                )
                frontend = gr.Dropdown(
                    choices=GR_FRONTEND_CHOICES,
                    value='wetext',
                    label="⚙️ Text Normalization Frontend",
                )
                seed = gr.Number(
                    precision=0,
                    value=1234,
                    label="Random seed",
                )
                cfg = gr.Number(
                    precision=1,
                    value=2.0,
                    label="CFG (guidance scale)",
                    info="Higher → closer to prompt; lower → more variation",
                )
        with gr.Row():
            with gr.Column(scale=1):
                run_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg")
                audio_output = gr.Audio(label="🔊 Generated Audio", type="numpy")
                gr.Markdown(
                    "---\n"
                    "**💡 Tips**\n"
                    "- Switching language auto-fills a default sample text.\n"
                    "- **llm_tn** supports all languages (requires LLM API).\n"
                    "- **wetext** is local, only for zh/en.\n"
                    "- **None** skips text normalization entirely."
                )
        # Language selector callback
        def _on_language_change_callback(lang_tag: str):
            if lang_tag == "_SEPARATOR_":
                raise gr.Error('Select a valid language!')
            case_info = TAG2CASE[lang_tag]
            text = case_info[2]
            return (gr.update(), gr.update(value=text))
        language.change(
            fn=_on_language_change_callback, 
            inputs=[language], outputs=[language, text],
        )
        # Generate button callback
        def _check_tts_input(
            text: str, prompt_audio: str, prompt_text: str,
            use_llm_tn: str, api_url: str, api_key: str, api_model: str,
        ):
            if text is None or text.strip() == '':
                raise gr.Error('Check text input!')
            if prompt_audio is None or (not os.path.exists(prompt_audio)):
                raise gr.Error('Check prompt audio!')
            if prompt_text is None or prompt_text.strip() == '':
                raise gr.Error('Check prompt text!')
            if use_llm_tn and not _HAS_DOTENV:
                if not (api_url and api_url.strip()) or not (api_key and api_key.strip()):
                    raise gr.Error("llm_tn requires LLM API credentials. Please fill in API URL and Key above.")

        def _generate(
            language: str, 
            text: str, prompt_audio: str, prompt_text: str,
            frontend: str, api_url: str, api_key: str, api_model: str,
            seed: int, cfg: float,
        ):
            use_wetext = (frontend == "wetext")
            use_llm_tn = (frontend == "llm_tn")

            # Check validity
            _check_tts_input(
                text, prompt_audio, prompt_text,
                use_llm_tn, api_url, api_key, api_model,
            )

            sr, wav = demo.generate(
                lang_tag=language,
                text=text,
                prompt_audio_path=prompt_audio,
                prompt_text=prompt_text,
                use_wetext=use_wetext,
                use_llm_tn=use_llm_tn,
                api_url=api_url,
                api_key=api_key,
                llm_tn_model=api_model,
                seed=int(seed),
                cfg=float(cfg),
            )
            return (sr, wav)

        run_btn.click(
            fn=_generate,
            inputs=[
                language, 
                text, prompt_audio, prompt_text, 
                frontend, api_url, api_key, api_model,
                seed, cfg,
            ],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )

    return interface


if __name__ == "__main__":
    parser = ArgumentParser(description="FireRedTTS3-Instruct Demo Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root-path", type=str, default=os.environ.get("GRADIO_ROOT_PATH", ""))    # For debugging
    args = parser.parse_args()

    # Initialize App
    tts = FireRedTTS3(
        'pretrained_models',
        use_wetext=False,
        use_llm_tn=False,
    )
    demo_app = FireRedDemoWrapper(tts)
    interface = create_demo_interface(demo_app)

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
