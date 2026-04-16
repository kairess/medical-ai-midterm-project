import torch
import torch.nn as nn
import timm

from src.config import (
    BACKBONE_NAME, BACKBONE_FEATURE_DIM,
    IMAGE_FEATURE_DIM, TABULAR_HIDDEN_DIM,
    FUSION_HIDDEN_DIM, DROPOUT_RATE,
    NUM_TABULAR_FEATURES, NUM_CLASSES,
    FREEZE_BACKBONE_RATIO,
)


class MultimodalSkinLesionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Image branch: pretrained EfficientNet-B0
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0)
        self._freeze_backbone()

        self.image_fc = nn.Sequential(
            nn.Linear(BACKBONE_FEATURE_DIM, IMAGE_FEATURE_DIM),
            nn.BatchNorm1d(IMAGE_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # Tabular branch: MLP
        self.tabular_fc = nn.Sequential(
            nn.Linear(NUM_TABULAR_FEATURES, TABULAR_HIDDEN_DIM),
            nn.BatchNorm1d(TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(TABULAR_HIDDEN_DIM, TABULAR_HIDDEN_DIM),
            nn.BatchNorm1d(TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # Fusion classifier
        fusion_input_dim = IMAGE_FEATURE_DIM + TABULAR_HIDDEN_DIM
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(FUSION_HIDDEN_DIM, NUM_CLASSES),
        )

    def _freeze_backbone(self):
        params = list(self.backbone.parameters())
        freeze_count = int(len(params) * FREEZE_BACKBONE_RATIO)
        for param in params[:freeze_count]:
            param.requires_grad = False

    def forward(self, image, tabular):
        # Image branch
        img_features = self.backbone(image)  # [B, 1280]
        img_features = self.image_fc(img_features)  # [B, 256]

        # Tabular branch
        tab_features = self.tabular_fc(tabular)  # [B, 64]

        # Late fusion
        fused = torch.cat([img_features, tab_features], dim=1)  # [B, 320]
        logits = self.classifier(fused)  # [B, 7]

        return logits

    def get_param_groups(self, lr: float):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.image_fc.parameters())
            + list(self.tabular_fc.parameters())
            + list(self.classifier.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ]


class ImageOnlyModel(nn.Module):
    """Image-only baseline for comparison (no tabular data)."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0)

        params = list(self.backbone.parameters())
        freeze_count = int(len(params) * FREEZE_BACKBONE_RATIO)
        for param in params[:freeze_count]:
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(BACKBONE_FEATURE_DIM, IMAGE_FEATURE_DIM),
            nn.BatchNorm1d(IMAGE_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(IMAGE_FEATURE_DIM, NUM_CLASSES),
        )

    def forward(self, image, tabular=None):
        features = self.backbone(image)
        return self.classifier(features)

    def get_param_groups(self, lr: float):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ]
