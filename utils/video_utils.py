# utils/video_utils.py

import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os

# ---- Preprocessing Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---- Load frames from folder (optional use case) ----
def load_frames_from_folder(folder_path):
    imgs = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
            imgs.append(transform(img))
    if len(imgs) == 0:
        raise RuntimeError("No frames found in folder")
    return torch.stack(imgs)  # [frames, 3, H, W]


# ---- Extract frames from video at equal intervals ----
def sample_frames_from_video(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(transform(img))

    cap.release()
    return frames


# ---- Predict tensor of frames ----
def predict_frames_tensor(frames_tensor, model, device='cpu'):
    # frames_tensor: [frames, 3, H, W] or [1, frames, 3, H, W]
    model.eval()
    with torch.no_grad():
        if frames_tensor.dim() == 4:
            frames_tensor = frames_tensor.unsqueeze(0)  # → [1, frames, 3, H, W]
        frames_tensor = frames_tensor.to(device)
        out = model(frames_tensor)            # (logits)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]  # [2]
    return probs


# ---- Main prediction method used by Flask ----
def predict_video_file(video_path, model, device='cpu', num_frames=8):
    frames = sample_frames_from_video(video_path, num_frames)
    if len(frames) == 0:
        return None, None

    frames_tensor = torch.stack(frames)     # [frames, 3, H, W]
    probs = predict_frames_tensor(frames_tensor, model, device)

    label = "Real" if probs[0] > probs[1] else "Fake"

    confidence = float(np.max(probs) * 100)       # Python float
    confidence = round(confidence, 2)

    return label, confidence
