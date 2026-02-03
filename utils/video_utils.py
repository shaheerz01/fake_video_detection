# utils/video_utils.py

import cv2
import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# --- NumPy compatibility guard ---
assert np.__version__.startswith("1."), "NumPy 2.x is not supported with PyTorch"

# ---- Preprocessing Transform (LOW MEMORY) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- Extract frames from video (SAFE & LIGHT) ----
def sample_frames_from_video(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to PIL and resize early (memory safe)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((224, 224))

        frames.append(transform(img))

    cap.release()
    return frames


# ---- Predict tensor of frames ----
def predict_frames_tensor(frames_tensor, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        if frames_tensor.dim() == 4:
            frames_tensor = frames_tensor.unsqueeze(0)  # [1, F, 3, H, W]

        frames_tensor = frames_tensor.to(device)
        outputs = model(frames_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    return probs


# ---- Main prediction method (USED BY FLASK) ----
def predict_video_file(video_path, model, device="cpu", num_frames=4):
    frames = sample_frames_from_video(video_path, num_frames=num_frames)
    if len(frames) == 0:
        return None, None

    frames_tensor = torch.stack(frames)  # [F, 3, H, W]
    probs = predict_frames_tensor(frames_tensor, model, device)

    label = "Real" if probs[0] > probs[1] else "Fake"
    confidence = round(float(np.max(probs) * 100), 2)

    return label, confidence
