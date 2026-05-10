"""
video_utils.py
--------------
Utilities for reading RAVDESS (and similar) MP4 files.

Each RAVDESS AV file contains:
  - A 720p video track of an actor's face (~3-4 seconds, 30 fps)
  - A stereo 48 kHz audio track of the spoken phrase

Public API
----------
extract_audio(mp4_path, target_sr)
    Read the audio track, resample to target_sr, convert to mono.
    Returns a 1-D float32 torch.Tensor of shape (T,).

extract_face_frames(mp4_path, n_frames, strategy)
    Decode the video track and return n_frames RGB numpy arrays.
    strategy='middle'  : frames clustered around the temporal midpoint
    strategy='uniform' : evenly spaced across the clip

detect_and_crop_face(frame_rgb, size)
    Run OpenCV Haar-cascade face detector on a single frame.
    Returns a PIL.Image cropped to the face (or a centre-crop fallback).

frames_to_tensor(frames, transform)
    Stack a list of PIL.Images into a (N,3,H,W) tensor using transform.

parse_ravdess_label(filename)
    Decode the RAVDESS 7-field filename and return the RAF-DB 0-indexed label.

list_av_files(video_root)
    Walk video_root and return all MP4 paths that are AV (modality 01).
"""

import os
import re
import numpy as np
from PIL import Image
import torch

# RAVDESS emotion code -> RAF-DB label (0-indexed)
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
_RAVDESS_TO_RAFDB = {1: 6, 2: 6, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 0}
EMOTION_NAMES = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]


def parse_ravdess_label(filename: str) -> int | None:
    """
    Decode a RAVDESS filename such as '01-01-05-01-02-01-12.mp4' or '.wav'.

    The filename encodes (dash-separated):
      1. Modality    : 01=full-AV, 02=video-only, 03=audio-only
      2. VocalChannel: 01=speech, 02=song
      3. Emotion     : 01=neutral, 02=calm, 03=happy, 04=sad,
                       05=angry, 06=fearful, 07=disgust, 08=surprised
      4. Intensity   : 01=normal, 02=strong
      5. Statement   : 01='Kids...', 02='Dogs...'
      6. Repetition  : 01, 02
      7. Actor       : 01-24

    Returns the RAF-DB 0-indexed label (0=Surprise ... 6=Neutral),
    or None if the filename does not match.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    try:
        emo_code = int(parts[2])
    except ValueError:
        return None
    return _RAVDESS_TO_RAFDB.get(emo_code)


def extract_audio(mp4_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Read the audio track from an MP4 (or WAV) file.

    Steps:
      1. Open the container with PyAV.
      2. Decode all audio packets into numpy float32 arrays.
      3. Concatenate along the time axis.
      4. Convert stereo -> mono by averaging channels.
      5. Resample from the source sample rate to target_sr using torchaudio.

    Returns:
        waveform (torch.Tensor): shape (T,), float32, at target_sr Hz.

    Raises:
        RuntimeError if 'av' (PyAV) is not installed.
    """
    try:
        import av as _av
    except ImportError:
        raise RuntimeError("PyAV not installed. Run: pip install av")

    container = _av.open(mp4_path)
    audio_stream = container.streams.audio[0]
    src_sr = audio_stream.sample_rate

    chunks = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()          # (channels, samples), float32 or int16
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32) / 32768.0
        chunks.append(arr)
    container.close()

    if not chunks:
        return torch.zeros(target_sr)     # 1 second silence as fallback

    audio_np = np.concatenate(chunks, axis=1)   # (channels, T)
    waveform  = torch.from_numpy(audio_np)       # (channels, T)
    waveform  = waveform.mean(dim=0)             # mono: (T,)

    if src_sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, src_sr, target_sr)

    return waveform


def extract_face_frames(
    mp4_path: str,
    n_frames: int = 5,
    strategy: str = "middle",
) -> list[np.ndarray]:
    """
    Decode the video track and return up to n_frames RGB numpy arrays.

    Args:
        mp4_path : path to MP4 file.
        n_frames : how many frames to return.
        strategy : 'middle' -- cluster around temporal centre of the clip;
                   'uniform' -- evenly spaced from start to end.

    Returns:
        List of np.ndarray, each shape (H, W, 3), dtype uint8, RGB.
    """
    try:
        import av as _av
    except ImportError:
        raise RuntimeError("PyAV not installed. Run: pip install av")

    container = _av.open(mp4_path)
    total = container.streams.video[0].frames or 100

    if strategy == "middle":
        mid   = total // 2
        half  = max(1, n_frames // 2)
        wanted = set(range(max(0, mid - half), min(total, mid + half + 1)))
    else:
        step   = max(1, total // n_frames)
        wanted = set(list(range(0, total, step))[:n_frames])

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in wanted:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == n_frames:
            break
    container.close()
    return frames


def detect_and_crop_face(frame_rgb: np.ndarray, size: int = 224) -> Image.Image:
    """
    Detect a face in frame_rgb using OpenCV's Haar cascade and return
    a cropped, resized PIL.Image.

    The Haar cascade (frontal-face) is downloaded with OpenCV; no extra
    downloads are needed.

    If no face is detected, falls back to a centre-crop of the frame.

    Args:
        frame_rgb : np.ndarray (H, W, 3), uint8, RGB.
        size      : output image side length in pixels.

    Returns:
        PIL.Image of shape (size, size), RGB.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV not installed. Run: pip install opencv-python-headless")

    gray    = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector     = cv2.CascadeClassifier(cascade_path)

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(60, 60))

    h, w = frame_rgb.shape[:2]
    if len(faces) > 0:
        # Pick the largest detected face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(fw, fh) * 0.15)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(w, x + fw + pad)
        y2  = min(h, y + fh + pad)
        crop = frame_rgb[y1:y2, x1:x2]
    else:
        # Centre-crop fallback: take the middle square
        side = min(h, w)
        cx, cy = w // 2, h // 2
        crop = frame_rgb[cy - side//2 : cy + side//2,
                          cx - side//2 : cx + side//2]

    img = Image.fromarray(crop).resize((size, size), Image.BILINEAR)
    return img


def frames_to_tensor(frames: list[np.ndarray], transform, size: int = 224) -> torch.Tensor:
    """
    Convert a list of raw video frames into a stacked tensor ready for a model.

    Process:
      1. For each frame, run detect_and_crop_face() to get a face-cropped PIL Image.
      2. Apply 'transform' (typically get_val_transforms(224)) to get a (3,H,W) tensor.
      3. Stack all tensors into (N, 3, H, W).

    Args:
        frames    : list of np.ndarray (H,W,3) from extract_face_frames().
        transform : torchvision.transforms callable.
        size      : face crop target size (passed to detect_and_crop_face).

    Returns:
        torch.Tensor of shape (N, 3, size, size).
    """
    tensors = []
    for frame in frames:
        face_img = detect_and_crop_face(frame, size=size)
        tensors.append(transform(face_img))
    return torch.stack(tensors)              # (N, 3, H, W)


def list_av_files(video_root: str) -> list[tuple[str, int]]:
    """
    Walk video_root and collect all AV-modality MP4 files (modality code 01).

    Returns:
        List of (mp4_path, raf_db_label) tuples, sorted by path.
        Files where the label cannot be parsed are skipped.
    """
    results = []
    for root, _, files in os.walk(video_root):
        for fname in sorted(files):
            if not fname.lower().endswith(".mp4"):
                continue
            parts = os.path.splitext(fname)[0].split("-")
            if len(parts) != 7 or parts[0] != "01":  # modality 01 = AV only
                continue
            label = parse_ravdess_label(fname)
            if label is not None:
                results.append((os.path.join(root, fname), label))
    return sorted(results)
