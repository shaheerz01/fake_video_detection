# model/model_def.py
import torch.nn as nn
import torchvision.models as models

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        # remove final fc from base; we'll aggregate frame features
        base.fc = nn.Identity()
        self.feature_extractor = base
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [batch, frames, 3, H, W]
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)                   # -> [b*f, 3, H, W]
        feats = self.feature_extractor(x)            # -> [b*f, 512]
        feats = feats.view(b, f, 512)                # -> [b, f, 512]
        feats = feats.mean(dim=1)                    # average frames -> [b, 512]
        out = self.fc(feats)                         # -> [b, num_classes]
        return out
