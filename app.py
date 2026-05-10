"""
app.py  —  Multimodal Emotion Recognition Demo
"""

import os, sys, warnings, traceback, shutil, tempfile
import numpy as np
import torch
import cv2
import gradio as gr
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset     import get_val_transforms
from models      import DINOFERModel, MultimodalFusionModel
from audio_model import AudioFERModel, _load_waveform

try:
    import imageio_ffmpeg as _iio_ff
    _ff_dir = tempfile.mkdtemp(prefix="ffmpeg_shim_")
    _ff_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    _ff_dst = os.path.join(_ff_dir, _ff_name)
    if not os.path.exists(_ff_dst):
        shutil.copy2(_iio_ff.get_ffmpeg_exe(), _ff_dst)
    os.environ["PATH"] = _ff_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

EMOTIONS = ["Surprise", "Fear",  "Disgust", "Happy", "Sad",  "Anger", "Neutral"]
EMOJIS   = ["😲",       "😨",    "🤢",      "😄",    "😢",   "😡",    "😐"]

device    = torch.device("cpu")
transform = get_val_transforms(224)
CKPT      = os.path.join(os.path.dirname(__file__), "checkpoints")

def _ckpt(model, fname):
    p = os.path.join(CKPT, fname)
    if os.path.exists(p):
        model.load_state_dict(torch.load(p, map_location=device))
    return model.to(device).eval()

print("Loading models ...")
face_model   = _ckpt(DINOFERModel(model_variant="vits16", freeze=True),  "best_dino.pt")
audio_model  = _ckpt(AudioFERModel(freeze_backbone=True),                "best_audio.pt")
fusion_model = _ckpt(MultimodalFusionModel(face_dim=384, audio_dim=768), "best_fusion.pt")
print("Ready.")

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@torch.no_grad()
def _run_face(pil_img):
    t    = transform(pil_img).unsqueeze(0).to(device)
    feat = face_model.backbone(pixel_values=t).last_hidden_state[:, 0, :]
    prob = torch.softmax(face_model.head(feat), -1).squeeze(0).cpu().tolist()
    return prob, feat

@torch.no_grad()
def _run_audio(wav_path):
    wave = _load_waveform(wav_path, 16000).unsqueeze(0).to(device)
    feat = audio_model.extract_features(wave)
    prob = torch.softmax(audio_model.head(feat), -1).squeeze(0).cpu().tolist()
    return prob, feat

@torch.no_grad()
def _run_fusion(ff, af):
    return torch.softmax(fusion_model(ff, af), -1).squeeze(0).cpu().tolist()

def _conf(probs):
    return {f"{EMOJIS[i]}  {EMOTIONS[i]}": round(probs[i], 4) for i in range(7)}

def _label(probs):
    i = int(np.argmax(probs))
    return f"{EMOJIS[i]}  {EMOTIONS[i]}   {probs[i]*100:.1f}%"

def _crop(arr):
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    dets = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(dets) > 0:
        x, y, w, h = max(dets, key=lambda b: b[2] * b[3])
        pad = int(0.15 * min(w, h))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(arr.shape[1], x + w + pad), min(arr.shape[0], y + h + pad)
        return Image.fromarray(arr[y1:y2, x1:x2])
    return Image.fromarray(arr)


def analyse_face(img):
    if img is None:
        return "No image provided.", {}
    try:
        crop = _crop(np.ascontiguousarray(img))
        probs, _ = _run_face(crop)
        return _label(probs), _conf(probs)
    except Exception:
        traceback.print_exc()
        return "Error processing image.", {}


def analyse_multimodal(video_path):
    if video_path is None:
        return "", "", "", {}
    face_probs = audio_probs = None
    face_feat  = audio_feat  = None
    try:
        cap = cv2.VideoCapture(video_path)
        best_crop, best_area, n = None, 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            n += 1
            if n % 5 != 0: continue
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(dets) > 0:
                x, y, w, h = max(dets, key=lambda b: b[2] * b[3])
                if w * h > best_area:
                    best_area = w * h
                    pad = int(0.15 * min(w, h))
                    x1, y1 = max(0, x-pad), max(0, y-pad)
                    x2, y2 = min(rgb.shape[1], x+w+pad), min(rgb.shape[0], y+h+pad)
                    best_crop = Image.fromarray(rgb[y1:y2, x1:x2])
        cap.release()
        if best_crop is None and n > 0:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
            ret, frame = cap.read(); cap.release()
            if ret: best_crop = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if best_crop is not None:
            face_probs, face_feat = _run_face(best_crop)
    except Exception:
        traceback.print_exc()

    audio_tmp = None
    try:
        import av, soundfile as sf
        container = av.open(video_path)
        if len(container.streams.audio) > 0:
            chunks, src_sr = [], container.streams.audio[0].sample_rate
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()
                if arr.dtype != np.float32: arr = arr.astype(np.float32) / 32768.0
                chunks.append(arr)
            container.close()
            if chunks:
                raw = np.concatenate(chunks, axis=1).mean(axis=0)
                wav = torch.from_numpy(raw)
                if src_sr != 16000:
                    import torchaudio
                    wav = torchaudio.functional.resample(wav, src_sr, 16000)
                audio_tmp = tempfile.mktemp(suffix=".wav")
                sf.write(audio_tmp, wav.numpy(), 16000)
        else:
            container.close()
    except Exception:
        traceback.print_exc()

    if audio_tmp and os.path.exists(audio_tmp):
        try:
            audio_probs, audio_feat = _run_audio(audio_tmp)
        except Exception:
            traceback.print_exc()
        finally:
            try: os.remove(audio_tmp)
            except: pass

    face_out  = _label(face_probs)  if face_probs  else "No face detected"
    audio_out = _label(audio_probs) if audio_probs else "No audio detected"

    if face_feat is not None and audio_feat is not None:
        fp = _run_fusion(face_feat, audio_feat)
        return face_out, audio_out, _label(fp), _conf(fp)
    elif face_probs:
        return face_out, audio_out, face_out, _conf(face_probs)
    elif audio_probs:
        return face_out, audio_out, audio_out, _conf(audio_probs)
    return face_out, audio_out, "Analysis incomplete.", {}


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#F0F4FF",
    body_background_fill_dark="#F0F4FF",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#E5E7EB",
    block_border_color_dark="#E5E7EB",
    block_border_width="1px",
    block_shadow="0 1px 4px rgba(0,0,0,0.05)",
    block_shadow_dark="0 1px 4px rgba(0,0,0,0.05)",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#E5E7EB",
    input_border_color_dark="#E5E7EB",
    border_color_primary="#E5E7EB",
    border_color_primary_dark="#E5E7EB",
    panel_background_fill="#FFFFFF",
    panel_background_fill_dark="#FFFFFF",
    body_text_color="#111827",
    body_text_color_dark="#111827",
    body_text_color_subdued="#6B7280",
    body_text_color_subdued_dark="#6B7280",
    button_primary_background_fill="#0F766E",
    button_primary_background_fill_dark="#0F766E",
    button_primary_background_fill_hover="#0D6B63",
    button_primary_background_fill_hover_dark="#0D6B63",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#F9FAFB",
    button_secondary_background_fill_dark="#F9FAFB",
    button_secondary_text_color="#374151",
    button_secondary_text_color_dark="#374151",
    color_accent_soft="#CCFBF1",
    color_accent_soft_dark="#CCFBF1",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html { color-scheme: light !important; }
footer { display: none !important; }

body, .gradio-container, .main {
    background: #F0F4FF !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #111827 !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ── Hero ─────────────────────────────────────────────────── */
.hero {
    background: #FFFFFF;
    border-bottom: 1px solid #E5E7EB;
    padding: 18px 40px 14px;
    text-align: center;
}
.hero-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin: 0 0 4px;
}
.hero-title .hl { color: #0F766E; }
.hero-sub {
    font-size: 0.75rem;
    color: #6B7280;
    font-weight: 400;
    margin: 0 0 12px;
}
.arch {
    display: inline-flex;
    align-items: center;
    gap: 14px;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 8px 18px;
}
.arch-node { text-align: center; }
.arch-node .tag {
    display: block;
    font-size: 0.50rem;
    letter-spacing: 1.5px;
    font-weight: 700;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 1px;
}
.arch-node .name {
    display: block;
    font-size: 0.75rem;
    font-weight: 700;
    color: #111827;
}
.arch-node .dim {
    display: block;
    font-size: 0.52rem;
    color: #9CA3AF;
    margin-top: 1px;
}
.arch-op { font-size: 0.9rem; font-weight: 300; color: #D1D5DB; }

/* ── Tab nav ────────────────────────────────────────────── */
.tab-nav, [role="tablist"] {
    background: #FFFFFF !important;
    border-bottom: 2px solid #E5E7EB !important;
    padding: 0 28px !important;
    margin: 0 !important;
}
.tab-nav button, [role="tab"] {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    color: #9CA3AF !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 13px 28px !important;
}
.tab-nav button.selected, [role="tab"][aria-selected="true"] {
    color: #111827 !important;
    border-bottom: 2px solid #0F766E !important;
}

/* ── Tab panel padding ──────────────────────────────────── */
[role="tabpanel"] {
    padding: 18px 28px !important;
    background: #F0F4FF !important;
}

/* ── Blocks ────────────────────────────────────────────── */
.block {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    overflow: hidden !important;
}

/* ── Labels ────────────────────────────────────────────── */
label > span, .block-label span {
    font-size: 0.60rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}

/* ── Inputs ────────────────────────────────────────────── */
textarea, input {
    background: #FFFFFF !important;
    color: #111827 !important;
    border-color: #E5E7EB !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Result text boxes ─────────────────────────────────── */
.result-main textarea {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
    text-align: center !important;
    padding: 14px !important;
    letter-spacing: 0.3px !important;
    line-height: 1.4 !important;
}
.result-branch textarea {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    text-align: center !important;
    padding: 8px !important;
}

/* ── Confidence bars ───────────────────────────────────── */
.label-wrap span {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ── Section note ─────────────────────────────────────── */
.snote {
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #9CA3AF;
    display: block;
    margin-bottom: 12px;
}

/* ── Analyse button ────────────────────────────────────── */
.analyse-btn button {
    background: #0F766E !important;
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    text-transform: none !important;
    padding: 11px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(15,118,110,0.22) !important;
    transition: background 0.15s !important;
    margin-top: 8px !important;
}
.analyse-btn button:hover {
    background: #0D6B63 !important;
    box-shadow: 0 4px 14px rgba(15,118,110,0.28) !important;
}

div[data-testid="image"] > div,
div[data-testid="video"] > div {
    background: #FFFFFF !important;
}

"""

HERO_HTML = """
<div class="hero">
  <h1 class="hero-title">
    <span class="hl">Multimodal</span> Emotion Recognition
  </h1>
  <p class="hero-sub">Face &amp; Voice &nbsp;&middot;&nbsp; Deep Learning</p>
  <div class="arch">
    <div class="arch-node">
      <span class="tag">Visual</span>
      <span class="name">DINO ViT-S/16</span>
      <span class="dim">RAF-DB &middot; 384-d</span>
    </div>
    <span class="arch-op">+</span>
    <div class="arch-node">
      <span class="tag">Speech</span>
      <span class="name">Wav2Vec2-base</span>
      <span class="dim">RAVDESS &middot; 768-d</span>
    </div>
    <span class="arch-op">&rarr;</span>
    <div class="arch-node">
      <span class="tag">Fusion</span>
      <span class="name">Late-Fusion MLP</span>
      <span class="dim">7 emotions</span>
    </div>
  </div>
</div>
"""

# Inject a <style> block early and run per-frame inline-style overrides on tab buttons
FORCE_LIGHT_JS = """
() => {
    /* Inject a high-priority stylesheet immediately */
    const s = document.createElement('style');
    s.textContent = `
        html, body { color-scheme: light !important; }
        .tab-nav, [role="tablist"] {
            background: #FFFFFF !important;
            border-bottom: 2px solid #E5E7EB !important;
            padding: 0 28px !important;
        }
        .tab-nav button, [role="tab"] {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            color: #9CA3AF !important;
            font-size: 0.68rem !important;
            font-weight: 700 !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            padding: 13px 28px !important;
            outline: none !important;
        }
        .tab-nav button.selected, [role="tab"][aria-selected="true"] {
            color: #111827 !important;
            border-bottom: 2px solid #0F766E !important;
        }
        .tab-nav button:not(.selected), [role="tab"][aria-selected="false"] {
            border-bottom: 2px solid transparent !important;
        }
    `;
    document.head.appendChild(s);

    /* Also override inline styles Gradio sets on tab buttons */
    const fixTabs = () => {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
        document.documentElement.style.colorScheme = 'light';
        document.querySelectorAll('.tab-nav button, [role="tab"]').forEach(btn => {
            const sel = btn.classList.contains('selected') ||
                        btn.getAttribute('aria-selected') === 'true';
            btn.style.setProperty('background',       'transparent', 'important');
            btn.style.setProperty('background-color', 'transparent', 'important');
            btn.style.setProperty('box-shadow',       'none',        'important');
            btn.style.setProperty('color', sel ? '#111827' : '#9CA3AF', 'important');
            btn.style.setProperty('border-top',   'none', 'important');
            btn.style.setProperty('border-left',  'none', 'important');
            btn.style.setProperty('border-right', 'none', 'important');
            btn.style.setProperty('border-bottom',
                sel ? '2px solid #0F766E' : '2px solid transparent', 'important');
        });
    };

    fixTabs();
    new MutationObserver(fixTabs).observe(document.body, {
        subtree: true, childList: true, attributes: true,
        attributeFilter: ['class', 'aria-selected', 'style']
    });
}
"""


with gr.Blocks(
    title="Multimodal Emotion Recognition",
    css=CSS,
    theme=theme,
    js=FORCE_LIGHT_JS,
) as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():
        with gr.Tab("👤  FACE"):
            gr.HTML('<span class="snote">Facial emotion recognition &nbsp;&middot;&nbsp; DINO ViT-S/16 &nbsp;&middot;&nbsp; RAF-DB</span>')
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    face_in = gr.Image(
                        sources=["webcam", "upload"],
                        type="numpy",
                        label="Upload Photo or Use Camera",
                        height=240,
                    )
                    with gr.Column(elem_classes=["analyse-btn"]):
                        btn_face = gr.Button("🔍  Analyse Face", variant="primary")
                with gr.Column(scale=1):
                    face_emo = gr.Textbox(
                        label="Predicted Emotion",
                        interactive=False,
                        lines=1,
                        elem_classes=["result-main"],
                    )
                    face_conf = gr.Label(label="Class Probabilities", num_top_classes=7)

        with gr.Tab("🎬  MULTIMODAL"):
            gr.HTML('<span class="snote">Face + Audio fusion &nbsp;&middot;&nbsp; DINO + Wav2Vec2 + Late-Fusion MLP</span>')
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    vid_in = gr.Video(
                        sources=["webcam", "upload"],
                        label="Record Video or Upload",
                        height=240,
                    )
                    with gr.Column(elem_classes=["analyse-btn"]):
                        btn_mm = gr.Button("🔍  Analyse Video", variant="primary")
                with gr.Column(scale=1):
                    mm_face = gr.Textbox(label="Face Branch",  interactive=False, lines=1,
                                         elem_classes=["result-branch"])
                    mm_audio = gr.Textbox(label="Audio Branch", interactive=False, lines=1,
                                          elem_classes=["result-branch"])
                    mm_fusion = gr.Textbox(
                        label="Fused Prediction",
                        interactive=False,
                        lines=1,
                        elem_classes=["result-main"],
                    )
                    mm_conf = gr.Label(label="Fusion Probabilities", num_top_classes=7)

    btn_face.click(analyse_face,       inputs=[face_in],  outputs=[face_emo, face_conf])
    btn_mm.click(  analyse_multimodal, inputs=[vid_in],   outputs=[mm_face, mm_audio, mm_fusion, mm_conf])

if __name__ == "__main__":
    demo.launch()
