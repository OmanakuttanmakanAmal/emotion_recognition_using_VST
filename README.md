# An Investigation into Multimodal Emotion Recognition Using Facial Expressions and Speech Signals Using Deep Learning Methods

**Author:** Amal Omanakuttan  
**Institution:** Technological University Dublin (TU Dublin)  
**Programme:** MSc Artificial Intelligence  
**Year:** 2025  

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [System Architecture — How the Three Models Work Together](#2-system-architecture--how-the-three-models-work-together)
3. [Frontend and Backend — How the Application Is Structured](#3-frontend-and-backend--how-the-application-is-structured)
4. [Why We Run Locally and Why We Train on Google Colab](#4-why-we-run-locally-and-why-we-train-on-google-colab)
5. [The Seven Emotions](#5-the-seven-emotions)
6. [Phase 1 — Face Emotion Recognition (DINO)](#6-phase-1--face-emotion-recognition-dino)
7. [Phase 2 — Audio Emotion Recognition (Wav2Vec2)](#7-phase-2--audio-emotion-recognition-wav2vec2)
8. [Phase 3 — Multimodal Fusion](#8-phase-3--multimodal-fusion)
9. [Final Results Summary](#9-final-results-summary)
10. [Datasets Used](#10-datasets-used)
11. [Project Files Explained](#11-project-files-explained)
12. [Installation and Setup](#12-installation-and-setup)
13. [How to Use the Interactive Demo](#13-how-to-use-the-interactive-demo)
14. [How to Retrain the Models](#14-how-to-retrain-the-models)
15. [Key Design Decisions](#15-key-design-decisions)
16. [Challenges Encountered and How They Were Solved](#16-challenges-encountered-and-how-they-were-solved)
17. [Technologies Used](#17-technologies-used)
18. [Acknowledgements and References](#18-acknowledgements-and-references)

---

## 1. What This Project Does

This thesis builds a system that can look at a person's face and listen to their voice at the same time, and then decide what emotion that person is feeling. The seven emotions the system can recognise are: Surprise, Fear, Disgust, Happiness, Sadness, Anger, and Neutral.

Most emotion recognition systems look at only one signal — either a photo of a face or a recording of a voice. This project combines both signals together, which is closer to how human beings naturally read emotions. When we talk to someone, we do not only look at their face in isolation; we also hear the tone of their voice. The combination gives a richer and more accurate picture.

The system was built and trained in three separate phases:

- **Phase 1** — A deep learning model was trained to recognise emotions from face images alone, using the DINO vision transformer and the RAF-DB dataset.
- **Phase 2** — A second model was trained to recognise emotions from speech audio alone, using the Wav2Vec2 transformer and the RAVDESS dataset.
- **Phase 3** — The two models were combined using a fusion layer — a small neural network that takes the outputs of both models and makes a final combined decision.

The finished system runs as a browser-based interactive demo where the user can upload a photo or a short video and receive an emotion prediction in real time.

---

## 2. System Architecture — How the Three Models Work Together

The system has two parallel processing branches — one for the face, one for the voice — that are eventually combined by a third model called the fusion layer.

**Face Branch (Phase 1)**

When a face image or video frame arrives, it is first preprocessed: OpenCV detects and crops the face region. The cropped face is then passed through the DINO ViT-S/16 model, a Vision Transformer pretrained on 1.28 million images. DINO produces a 384-dimensional embedding — a compact numerical description of the emotional content of the face. This embedding is fed into a small classification head that outputs probabilities for 7 emotion classes.

**Audio Branch (Phase 2)**

When audio is present (from an uploaded video or recorded clip), the raw waveform is decoded and resampled to 16,000 Hz. It is then passed through the Wav2Vec2-base model, a speech transformer pretrained on 960 hours of English speech. Wav2Vec2 produces a 768-dimensional embedding representing the emotional content of the voice. This embedding also feeds into a classification head outputting 7 emotion probabilities.

**Fusion Layer (Phase 3)**

The 384-d face embedding and the 768-d audio embedding are concatenated into a single 1152-dimensional vector. This vector is passed through a three-layer MLP (multi-layer perceptron) — a fully connected neural network — that has learned to weight and combine both signals into a final emotion prediction.

The key benefit of this architecture is that it **degrades gracefully**: if audio is not available (e.g., a photo rather than a video), the system falls back to the face-only result without breaking.

---

## 3. Frontend and Backend — How the Application Is Structured

The project is divided into a **frontend** (the web interface that the user sees and interacts with) and a **backend** (the Python code that loads the models, processes inputs, and returns predictions).

### Frontend

The frontend is a web application built using **Gradio**, a Python library that creates browser-based interfaces. When you run `python app.py`, Gradio starts a local web server and opens the interface in your browser automatically. The interface is written in `app.py` and styled with custom CSS and JavaScript.

The interface has two panels:

- **Face panel** — accepts a photo (uploaded or taken with the webcam) and runs the face emotion model.
- **Multimodal panel** — accepts a short video clip (uploaded or recorded with the webcam) and runs both the face and audio models, then shows three results: the face prediction, the audio prediction, and the combined (fused) prediction.

The frontend is built entirely in Python — no separate HTML files, no JavaScript files, and no web framework like Flask or Django is needed. Gradio handles all of that automatically.

### Backend

The backend consists of several Python modules in the `src/` folder:

| File | Role |
|------|------|
| `src/dataset.py` | Loads face images from RAF-DB, applies data augmentation, manages train/test splits |
| `src/audio_model.py` | Defines the Wav2Vec2-based audio model, loads RAVDESS audio files, handles NaN-safe training |
| `src/video_utils.py` | Decodes MP4 video files (both frames and audio track) using PyAV, detects faces with OpenCV |
| `src/train.py` | Training script for the Phase 1 face model |
| `src/train_audio.py` | Training script for the Phase 2 audio model |
| `src/train_multimodal.py` | Training script for the Phase 3 fusion model |
| `app.py` | Main application — loads all three trained models, defines inference functions, builds the Gradio interface |

The trained model weights are saved as `.pt` files (PyTorch checkpoint files) in the `checkpoints/` folder. When `app.py` starts, it loads all three checkpoint files into memory. Inference (making a prediction) then happens entirely on the local CPU — no internet connection is needed after the initial model download.

---

## 4. Why We Run Locally and Why We Train on Google Colab

### Why the Demo Runs Locally

The interactive demo (`app.py`) runs on your own computer for several reasons:

1. **No cost** — Running on your own CPU is free. Running on a cloud GPU server would cost money per hour.
2. **Privacy** — Your face and voice data never leave your machine. Nothing is uploaded to any external server.
3. **No internet after setup** — Once the models are downloaded (this happens automatically the first time), the system works without any internet connection.
4. **Gradio local mode** — Gradio starts a local web server on port 7860 and opens the interface in your browser. The URL `http://127.0.0.1:7860` means "this computer, port 7860" — it is not accessible from outside your network.
5. **CPU is fast enough for inference** — Making a single prediction takes less than 2 seconds on a modern CPU. Training is what requires a GPU (see below).

### Why Training Uses Google Colab

Training the models requires much more computation than making predictions. Here is the comparison:

| Task | Your laptop (CPU) | Google Colab (GPU) |
|------|-------------------|--------------------|
| Phase 1: Train face model (20 epochs) | ~12–24 hours | ~45–60 minutes |
| Phase 2: Train audio model (20 epochs) | ~4–8 hours | ~20–30 minutes |
| Phase 3: Train fusion model (20 epochs) | ~1–2 hours | ~10–15 minutes |

Google Colab provides a free cloud-based environment with a GPU. When you open a Colab notebook and select the T4 GPU runtime, your code runs on a server in Google's data centre rather than your laptop.

### Which GPU Was Used

**GPU: NVIDIA Tesla T4**

The T4 is the GPU available on Google Colab's free tier. Its key specifications relevant to this project are:

| Property | Value |
|----------|-------|
| GPU Model | NVIDIA Tesla T4 |
| GPU Memory (VRAM) | 16 GB |
| Architecture | Turing (2018) |
| Tensor Cores | Yes — accelerates matrix multiplications in neural networks |
| Cost on Colab | Free (up to usage limits per day) |
| How to select | Runtime → Change runtime type → T4 GPU → Save |

The T4 accelerates training by performing thousands of floating-point multiplications in parallel. A single training step that takes ~500ms on your laptop's CPU takes ~10ms on the T4 — roughly 50 times faster.

**Why not use a bigger GPU?** The T4 is sufficient for the model sizes in this project. DINO ViT-S/16 has 21 million parameters; Wav2Vec2-base has 95 million parameters. Both fit comfortably in 16 GB of VRAM even with a batch of 32 samples.

---

## 5. The Seven Emotions

The system recognises these seven emotion categories, which are the standard RAF-DB labels (0-indexed internally):

| Index | Emotion | Emoji | Description |
|-------|---------|-------|-------------|
| 0 | Surprise | 😲 | Raised eyebrows, wide eyes, open mouth |
| 1 | Fear | 😨 | Wide eyes, tense brow, slight retreat |
| 2 | Disgust | 🤢 | Wrinkled nose, curled lip |
| 3 | Happy | 😄 | Raised cheeks, visible teeth, open smile |
| 4 | Sad | 😢 | Downturned mouth, furrowed inner brow |
| 5 | Anger | 😡 | Lowered brow, pressed lips, tense jaw |
| 6 | Neutral | 😐 | Relaxed face, no strong expression |

---

## 6. Phase 1 — Face Emotion Recognition (DINO)

### What is DINO?

DINO (Self-DIstillation with NO labels) is a self-supervised Vision Transformer developed by Meta AI Research. It was trained on 1.28 million ImageNet images without any human-provided labels — instead it learned to produce similar representations for different views of the same image. This produces features that naturally capture facial structure, which makes DINO an excellent starting point for emotion recognition.

The specific variant used here is **ViT-S/16**: Vision Transformer, Small variant, with 16×16 pixel patches. The model divides a 224×224 image into 196 patches and processes them as a sequence. The CLS token at the end is a 384-dimensional summary of the entire image.

### Training Configuration

| Setting | Value |
|---------|-------|
| Base model | facebook/dino-vits16 |
| Dataset | RAF-DB |
| Training images | 12,271 |
| Test images | 3,068 |
| Input image size | 224 × 224 pixels |
| Backbone | Frozen (weights not updated during training) |
| Trainable part | Linear classification head only |
| Head architecture | LayerNorm → Linear(384→128) → GELU → Dropout(0.3) → Linear(128→7) |
| Loss function | CrossEntropyLoss with class weights |
| Optimiser | AdamW, learning rate = 1e-4 |
| Learning rate schedule | CosineAnnealingLR |
| Epochs | 20 |
| Batch size | 32 |
| Data augmentation | RandomHorizontalFlip, ColorJitter, RandomRotation(±10°) |

### Why the Backbone Was Frozen

DINO was pretrained on 1.28 million images. RAF-DB has only 12,271 training images — about 100 times fewer. Fine-tuning all 21 million backbone parameters on this small dataset would cause severe overfitting: the model would memorise the training faces rather than learn general emotion patterns. By freezing the backbone and only training the small head (a few hundred parameters), we get strong generalisation to faces the model has never seen before.

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **80.67%** |
| Macro F1-Score | **71.6%** |
| Best class | Happy (93%+ recall) |
| Hardest class | Fear and Disgust (visually similar, fewer training examples) |

---

## 7. Phase 2 — Audio Emotion Recognition (Wav2Vec2)

### What is Wav2Vec2?

Wav2Vec2 is a speech representation model developed by Meta AI Research, pretrained on 960 hours of English speech (LibriSpeech). Like DINO for images, it uses self-supervised learning — no emotion labels were needed during pretraining. The model works directly on raw audio waveforms at 16,000 Hz, without requiring hand-crafted features. It produces a sequence of 768-dimensional hidden states; mean-pooling across this sequence gives a single vector summarising the emotional tone of the entire utterance.

### Training Configuration

| Setting | Value |
|---------|-------|
| Base model | facebook/wav2vec2-base-960h |
| Dataset | RAVDESS |
| Training samples | ~1,296 (90% of 1,440) |
| Validation samples | ~144 (10%) |
| Audio format | WAV, 16,000 Hz, mono |
| Backbone | Frozen |
| Head architecture | LayerNorm → Linear(768→256) → GELU → Dropout(0.3) → Linear(256→7) |
| Optimiser | AdamW, learning rate = 1e-4 |
| Epochs | 20 |
| Batch size | 16 |

### The NaN Training Bug and How It Was Fixed

During early training, the loss would suddenly become `NaN` (Not a Number) and the model would fail to converge.

**Root cause:** Wav2Vec2 uses a technique called SpecAugment during training, which randomly masks parts of the input using a learnable embedding called `masked_spec_embed`. Since the backbone was frozen, this embedding was uninitialised and produced NaN values that propagated through the network.

**Fix:** Override PyTorch's `train()` method in `AudioFERModel` to keep the backbone permanently in `eval()` mode, which disables SpecAugment:

```python
def train(self, mode: bool = True):
    super().train(mode)
    self.backbone.eval()  # Prevents SpecAugment NaN
    return self
```

This single override completely resolved the NaN issue.

### Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **43.1%** |
| Notes | RAVDESS has only 1,440 samples. Audio-only recognition is harder than face-based recognition, and the dataset is very small. 43.1% is well above random chance (14.3% for 7 classes). |

---

## 8. Phase 3 — Multimodal Fusion

### What is Fusion?

Fusion means combining the predictions (or internal representations) of two separate models into a single joint prediction. This project uses **late fusion**: run each modality through its own model first, extract a compact embedding from each, then combine the embeddings with a third model.

### Synthetic Pairing Strategy

Training the fusion model requires face and audio samples from the same emotional event. However, RAF-DB has no audio, and RAVDESS has only 1,440 clips. Rather than collecting a new dataset, this project uses synthetic pairing:

1. Draw a face image with emotion label `k` from RAF-DB. Run it through DINO → 384-d face embedding.
2. Draw an audio clip with the same emotion label `k` from RAVDESS. Run it through Wav2Vec2 → 768-d audio embedding.
3. Concatenate the two embeddings → 1152-d training sample, labelled `k`.
4. Train the fusion MLP to classify this vector correctly.

**4,000 synthetic pairs** were generated. A happy face embedding paired with a happy voice embedding gives the fusion model a strong combined signal to learn from.

### Fusion MLP Architecture

The fusion network takes the concatenated 1152-d vector and passes it through:
- Linear(1152 → 512) + ReLU + Dropout(0.4)
- Linear(512 → 128) + ReLU + Dropout(0.4)
- Linear(128 → 7) — final 7-class output

### Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **95.65%** |
| Notes | High accuracy is expected on synthetic same-class pairs. The real-world test is the qualitative demo on live video input. |

---

## 9. Final Results Summary

| Phase | Model | Dataset | Accuracy | Notes |
|-------|-------|---------|----------|-------|
| Phase 1 | DINO ViT-S/16 | RAF-DB (test set) | **80.67%** | 71.6% macro F1 |
| Phase 2 | Wav2Vec2-base-960h | RAVDESS (validation) | **43.1%** | Small dataset; backbone frozen |
| Phase 3 | Late-Fusion MLP | Synthetic pairs | **95.65%** | 4,000 same-class synthetic pairs |

The face model is the most reliable for real-world use. The audio model adds signal for emotions that look similar on the face (fear vs surprise). The fusion combines both for a more holistic prediction.

---

## 10. Datasets Used

### RAF-DB (Real-world Affective Faces Database)

| Property | Details |
|----------|---------|
| Source | Internet images of real human faces |
| Total images | 15,339 (12,271 train / 3,068 test) |
| Labels | 7 basic emotions, crowd-annotated |
| Image format | 224×224 aligned face crops |
| Why chosen | Large scale, diverse demographics, real-world conditions, widely used benchmark |

RAF-DB images come from real internet photos — not controlled studio conditions. People appear at different angles, lighting conditions, ages, and ethnicities, making it a challenging and realistic benchmark.

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

| Property | Details |
|----------|---------|
| Source | Controlled studio recordings |
| Speakers | 24 professional actors (12 male, 12 female) |
| Total files | 1,440 audio + 1,440 video |
| Emotions | 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised) |
| Audio format | WAV, 48,000 Hz stereo → resampled to 16,000 Hz mono |
| Why chosen | High-quality, professionally acted, widely used benchmark for speech emotion |

Note: RAVDESS has 8 emotion categories. "Calm" was merged with "Neutral" for this project, mapping to the 7-class scheme used throughout.

---

## 11. Project Files Explained

**Root folder**

| File | Purpose |
|------|---------|
| `app.py` | Main interactive demo — loads models, defines inference, runs the Gradio web interface |
| `demo_professor.py` | Command-line demonstration script showing architecture, dataset stats, and model results |
| `test_e2e.py` | Automated test suite — verifies that all components load and run correctly |
| `requirements.txt` | List of all Python package dependencies |
| `colab_training.ipynb` | Google Colab notebook for GPU training |
| `README.md` | This documentation file |

**src/ folder — model and data code**

| File | Purpose |
|------|---------|
| `src/dataset.py` | RAF-DB dataset loader — reads images, applies transforms, manages train/test split |
| `src/audio_model.py` | Wav2Vec2 audio model class + RAVDESS/CREMA-D dataset loader + NaN-safe training override |
| `src/video_utils.py` | MP4 video utilities — decodes frames (PyAV), detects faces (OpenCV), extracts audio |
| `src/train.py` | Phase 1 training script for the DINO face model |
| `src/train_audio.py` | Phase 2 training script for the Wav2Vec2 audio model |
| `src/train_multimodal.py` | Phase 3 training script for the late-fusion MLP |

**checkpoints/ folder**

| File | Purpose |
|------|---------|
| `checkpoints/best_dino.pt` | Saved weights for the Phase 1 face model (download after Colab training) |
| `checkpoints/best_audio.pt` | Saved weights for the Phase 2 audio model |
| `checkpoints/best_fusion.pt` | Saved weights for the Phase 3 fusion model |

**Data folders**

| Folder | Contents |
|--------|---------|
| `DATASET/train/` | RAF-DB training images, organised into 7 subfolders (one per emotion class) |
| `DATASET/test/` | RAF-DB test images |
| `audio_data/` | RAVDESS WAV files, organised into Actor_01 through Actor_24 subfolders |
| `video_data/` | RAVDESS MP4 video files (used for the command-line demo) |

---

## 12. Installation and Setup

### Prerequisites

- Windows 10 or 11 (developed and tested on Windows 11)
- Python 3.11 (some packages require this version)
- At least 8 GB RAM
- At least 10 GB free disk space

### Step 1 — Extract the project

Extract or clone the project to a local folder, for example:
```
e:\Local Disk 1\DIT\Thesis\
```

### Step 2 — Create the virtual environment

A virtual environment keeps this project's packages separate from any other Python on your computer.

```powershell
python -m venv venv
```

### Step 3 — Activate the virtual environment

```powershell
.\venv\Scripts\activate.bat
```

Your terminal prompt will show `(venv)` after activation. Repeat this step every time you open a new terminal.

### Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

This installs PyTorch, Gradio, HuggingFace Transformers, OpenCV, PyAV, soundfile, torchaudio, and all other required packages. It may take 5–15 minutes on first run.

### Step 5 — Verify the installation

```powershell
python test_e2e.py
```

All lines should show `[PASS]`. If any fail, the error message will say what is missing.

### Step 6 — Place trained model checkpoints

Place the three trained `.pt` files in the `checkpoints/` folder:

```
checkpoints/best_dino.pt
checkpoints/best_audio.pt
checkpoints/best_fusion.pt
```

If these files are not present, the demo runs but gives random predictions. See Section 14 to train them on Google Colab.

---

## 13. How to Use the Interactive Demo

Start the demo:

```powershell
python app.py
```

The terminal will show `Running on local URL: http://127.0.0.1:7860`. Your browser should open automatically. If not, open the browser manually and go to **http://127.0.0.1:7860**.

The interface has two panels, switched using the buttons at the top: **Face** and **Multimodal**.

### Face Panel

Use this when you have a photo or want to take one with your webcam.

1. Click the dashed upload area (or drag a photo into it), or click the camera icon to use the webcam
2. A snapshot of your face will appear in the panel
3. Click **Analyse Face**
4. The predicted emotion and confidence percentages appear on the right

The model detects and crops your face automatically using OpenCV before passing it to DINO.

### Multimodal Panel

Use this when you have a short video with speech, or want to record one.

1. Click **Multimodal** at the top to switch to this panel
2. Upload an MP4/MOV/WebM video, or click the camera icon to record
3. Click **Analyse Video**
4. Three results appear on the right:
   - **Face Branch** — emotion detected from video frames
   - **Audio Branch** — emotion detected from the voice
   - **Fused Prediction** — final combined result from the fusion model
   - **Fusion Probabilities** — confidence bars for all 7 emotions

**Note on webcam recordings:** Some browsers record video without audio from the webcam recorder. If no audio stream is found, the system falls back to face-only analysis and says so in the result. For best results with audio, upload a pre-recorded MP4 file.

---

## 14. How to Retrain the Models

Training on CPU takes many hours. Use Google Colab with a free T4 GPU (see Section 4 for GPU details). Each phase finishes in 45–90 minutes on Colab.

### Retraining Phase 1 — Face Model

Requires RAF-DB in `DATASET/` and `train_labels.csv` / `test_labels.csv`.

**On Google Colab:**
1. Upload `archive.zip` (RAF-DB dataset), `train_labels.csv`, `test_labels.csv`, and the `src/` folder to Google Drive under a folder called `Thesis`
2. Open `colab_training.ipynb` in Colab
3. Set Runtime → Change runtime type → **T4 GPU** → Save
4. Run all cells — training takes ~45–60 minutes
5. Download `best_dino.pt` from Drive and place it in `checkpoints/`

Expected result: ~80% test accuracy after 20 epochs.

**Locally (slow — hours):**
```powershell
python src\train.py --epochs 20 --batch_size 32 --lr 1e-4
```

---

### Retraining Phase 2 — Audio Model

Requires RAVDESS in `audio_data/`.

**On Google Colab (add a new cell to the notebook):**
```python
import zipfile, shutil

# First zip audio_data on your laptop:
# Compress-Archive -Path audio_data -DestinationPath audio_data.zip
# Then upload audio_data.zip to Google Drive

with zipfile.ZipFile('/content/drive/MyDrive/Thesis/audio_data.zip') as z:
    z.extractall('/content/thesis/')

!pip install soundfile torchaudio

!python /content/thesis/src/train_audio.py \
    --data_root /content/thesis/audio_data \
    --dataset ravdess \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-4

shutil.copy('/content/thesis/checkpoints/best_audio.pt',
            '/content/drive/MyDrive/Thesis/best_audio.pt')
print("Audio training complete.")
```

Download `best_audio.pt` from Drive → `checkpoints/best_audio.pt`

Expected result: ~40–50% validation accuracy after 20 epochs.

---

### Retraining Phase 3 — Fusion Model

Requires `best_dino.pt` and `best_audio.pt` in `checkpoints/`, plus both datasets.

**On Google Colab (add another new cell):**
```python
!python /content/thesis/src/train_multimodal.py \
    --face_ckpt  /content/thesis/checkpoints/best_dino.pt \
    --audio_ckpt /content/thesis/checkpoints/best_audio.pt \
    --synthetic \
    --img_root   /content/thesis/DATASET/train \
    --img_csv    /content/thesis/train_labels.csv \
    --audio_root /content/thesis/audio_data \
    --n_synthetic 4000 \
    --epochs 20

shutil.copy('/content/thesis/checkpoints/best_fusion.pt',
            '/content/drive/MyDrive/Thesis/best_fusion.pt')
print("Fusion training complete.")
```

Download `best_fusion.pt` → `checkpoints/best_fusion.pt`

Expected result: ~95% validation accuracy on synthetic pairs.

---

## 15. Key Design Decisions

### Decision 1: DINO Instead of a CNN

Earlier emotion recognition systems used CNNs like VGG, ResNet, or EfficientNet. This project uses the DINO Vision Transformer because:

- DINO was self-supervised on 1.28 million images — it learned rich visual features without any emotion labels.
- Vision Transformers process the entire face at once, capturing long-range spatial relationships between features (e.g., eyes and mouth simultaneously), rather than building up local patterns layer by layer.
- DINO's attention maps naturally focus on semantically meaningful facial regions (eyes, mouth, brows) without being told to do so.
- In practice, frozen DINO features achieve 80.67% on RAF-DB — competitive with fully fine-tuned CNNs, without the overfitting risk.

### Decision 2: Freezing Both Backbones

Both DINO and Wav2Vec2 backbones were kept frozen during training. Only the small classification heads were trained. This was deliberate because:

- RAF-DB (12,271 images) and RAVDESS (1,440 clips) are small relative to the backbone sizes.
- Fine-tuning 21M (DINO) or 95M (Wav2Vec2) parameters on such small datasets causes severe overfitting.
- Frozen backbones reduce trainable parameters to a few thousand, enabling faster training and stronger generalisation.

### Decision 3: Late Fusion

The system combines models at the embedding level, not the raw input level. Benefits:
- Each model trains independently on its own dataset — no need for a single aligned face+audio dataset.
- The fusion model can be improved without retraining the backbone models.
- The system degrades gracefully: if audio is missing, it falls back to face-only analysis with no structural change.

### Decision 4: Synthetic Pairing for Fusion Training

No large publicly available dataset provides synchronised face+audio pairs with emotion labels. Instead of collecting new data, this project creates synthetic pairs: a happy face embedding paired with a happy audio embedding creates a valid multimodal training sample. 4,000 such pairs were generated at essentially zero cost, giving the fusion MLP enough diverse examples to learn from.

### Decision 5: RAF-DB Over FER2013

FER2013 is a popular facial emotion dataset, but RAF-DB was chosen because:
- RAF-DB images are higher resolution and quality.
- FER2013 has significant label noise from its automated labelling process.
- RAF-DB represents real-world variation in age, ethnicity, lighting, and pose more faithfully.
- RAF-DB's 7-class scheme aligns directly with the 7 basic emotions used throughout this project.

---

## 16. Challenges Encountered and How They Were Solved

### Challenge 1: NaN Loss During Audio Training

**What happened:** Training loss became `NaN` after a few iterations, causing training to fail completely.

**Root cause:** Wav2Vec2 uses SpecAugment (random input masking) during training, powered by an uninitialised `masked_spec_embed` parameter. With the backbone frozen, this parameter was never updated and produced NaN values.

**Fix:** Override `train()` in `AudioFERModel` to keep the backbone in `eval()` mode, which disables SpecAugment:
```python
def train(self, mode: bool = True):
    super().train(mode)
    self.backbone.eval()
    return self
```

### Challenge 2: Webcam Not Working

**What happened:** The initial version used `gr.Video(source="webcam")` for a live feed, but this was unreliable on Windows.

**Fix:** Switched to `gr.Video(sources=["webcam", "upload"])` and `gr.Image(sources=["webcam", "upload"])`, letting the user choose between uploading a file or taking a snapshot. This is more reliable because the browser handles recording and delivers a completed file.

### Challenge 3: ffmpeg Not Found for Video Processing

**What happened:** Gradio's Video component internally calls `ffmpeg` to transcode uploaded videos. `ffmpeg` was not installed system-wide, causing a `FFExecutableNotFoundError`.

**Fix:** The `imageio[ffmpeg]` package bundles a compiled ffmpeg binary. At startup, `app.py` copies this binary to the system temporary directory and adds it to PATH:
```python
import imageio_ffmpeg, shutil, os
ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dst = os.path.join(tempfile.mkdtemp(), "ffmpeg.exe")
shutil.copy(ffmpeg_src, ffmpeg_dst)
os.environ["PATH"] = os.path.dirname(ffmpeg_dst) + os.pathsep + os.environ["PATH"]
```

### Challenge 4: Browser-Recorded Video Has No Audio

**What happened:** Browser webcam recordings produce a WebM file with video only — no audio stream. Accessing `container.streams.audio[0]` raised an IndexError.

**Fix:** Added a check before accessing the audio stream:
```python
if len(container.streams.audio) > 0:
    # proceed with audio extraction
else:
    # fall back to face-only analysis
```

### Challenge 5: Tab Buttons Invisible in Dark Mode

**What happened:** On systems with dark mode enabled (Windows 11 dark theme), Gradio rendered the selected tab button with a black background, making the unselected tab text invisible.

**Fix:** Removed Gradio's built-in tab system entirely and replaced it with two custom `gr.Button` elements styled as tabs, combined with `gr.Column(visible=...)` panels that show and hide based on which button is clicked. This bypasses all Gradio tab styling completely, so dark mode can never interfere. JavaScript removes the `dark` CSS class from the page root to ensure all components render in light mode.

---

## 17. Technologies Used

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.11 | Primary programming language |
| PyTorch | 2.x | Deep learning framework for all models |
| HuggingFace Transformers | 4.x | DINO and Wav2Vec2 pretrained model loading |
| Gradio | 6.14 | Interactive web demo interface (frontend) |
| torchaudio | 2.x | Audio loading and resampling |
| soundfile | 0.12 | WAV file reading (reliable on Windows) |
| torchvision | 0.x | Image transforms and preprocessing |
| OpenCV (cv2) | 4.x | Haar cascade face detection and frame processing |
| PyAV (av) | 12.x | MP4 video container decoding (frames + audio) |
| imageio[ffmpeg] | 2.x | Bundled ffmpeg binary for video transcoding |
| NumPy | 1.x | Array operations |
| Pillow | 10.x | Image loading and colour space conversion |
| scikit-learn | 1.x | F1 score, confusion matrix, classification report |
| Google Colab | — | Cloud GPU environment for training (free T4 GPU) |
| Google Drive | — | Storage for datasets and checkpoints during Colab training |
| NVIDIA T4 GPU | — | Hardware used for model training on Colab |

---

## 18. Acknowledgements and References

This project was completed as part of an MSc in Artificial Intelligence at Technological University Dublin (TU Dublin), 2025.

**Datasets:**

- **RAF-DB:** Li, S., Deng, W., & Du, J. P. (2017). Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild. *IEEE CVPR*.

- **RAVDESS:** Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song. *PLOS ONE*, 13(5), e0196391.

**Models:**

- **DINO:** Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *IEEE/CVF ICCV*.

- **Wav2Vec2:** Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS*.

- **Vision Transformer:** Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.

---

## Quick Reference

```
Start demo:         python app.py
Run tests:          python test_e2e.py
Run CLI demo:       python demo_professor.py
Train face model:   python src\train.py
Train audio model:  python src\train_audio.py
Train fusion:       python src\train_multimodal.py
Demo URL:           http://127.0.0.1:7860
Checkpoints:        checkpoints\best_dino.pt
                    checkpoints\best_audio.pt
                    checkpoints\best_fusion.pt
```

---

*Written for non-technical readers. All technical decisions are explained from first principles. For further questions, contact: Amal Omanakuttan, TU Dublin, 2025.*
