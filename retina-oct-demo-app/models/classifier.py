"""
Clasificador de patologías retinianas (CNV, DME, Drusen, Normal).
Tres escenarios:
  - Modelo 1 (raw): entrada RGB (3 canales) — imagen OCT directa.
  - Modelo 2 (seg): entrada RGB (3 canales) — máscara de segmentación coloreada.
  - Modelo 3 (hybrid): entrada de 4 canales (RGB + máscara argmax como canal extra).
"""
import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, NUM_SEG_CLASSES, DEVICE, CLASSIFIER_RAW_WEIGHTS, CLASSIFIER_SEG_WEIGHTS, CLASSIFIER_HYBRID_WEIGHTS
import os


class RetinaClassifier(nn.Module):
    """
    Clasificador basado en ResNet50 con transfer learning.

    Args:
        in_channels: Número de canales de entrada.
            - 3 para imagen OCT RGB (Modelo 1: raw).
            - 3 para máscara coloreada RGB (Modelo 2: seg).
            - 4 para imagen RGB + canal de máscara (Modelo 3: hybrid).
        num_classes: Número de clases de patología.
        backbone: 'efficientnet' o 'resnet'.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        backbone: str = "efficientnet",
        ):
        super().__init__()
        self.in_channels = in_channels
        self.backbone_name = backbone

        if backbone == "efficientnet":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )

        elif backbone == "resnet":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                    padding=original_conv.padding,
                bias=False,
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)  # ← sin Dropout, igual que en entrenamiento

        else:
            raise ValueError(f"Backbone no soportado: {backbone}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, in_channels, H, W]
        Returns:
            logits: Tensor [B, num_classes]
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades por clase."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


def load_classifier_model(
    mode: str = "raw",
    weights_path: str = None,
    backbone: str = "resnet",
) -> RetinaClassifier:
    """
    Carga el clasificador para el escenario indicado.

    Args:
        mode: 'raw' (imagen RGB, 3 canales),
              'seg' (máscara coloreada RGB, 3 canales),
              'hybrid' (RGB + máscara, 4 canales).
        weights_path: Ruta a los pesos. Si None, usa la ruta por defecto.
        backbone: 'efficientnet' o 'resnet'.
    """
    if mode == "raw":
        in_channels = 3
        default_path = CLASSIFIER_RAW_WEIGHTS
    elif mode == "seg":
        in_channels = 3
        default_path = CLASSIFIER_SEG_WEIGHTS
    elif mode == "hybrid":
        in_channels = 4
        default_path = CLASSIFIER_HYBRID_WEIGHTS
    else:
        raise ValueError(f"Modo no válido: {mode}. Usa 'raw', 'seg' o 'hybrid'.")

    if weights_path is None:
        weights_path = default_path

    model = RetinaClassifier(in_channels=in_channels, backbone=backbone)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        state_dict = {"backbone." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        print(f"[Clasificador {mode}] Pesos cargados desde: {weights_path}")
    else:
        print(f"[Clasificador {mode}] AVISO: No se encontraron pesos en {weights_path}")
        print("  → El modelo dará predicciones aleatorias. Entrénalo primero.")

    model.to(DEVICE)
    model.eval()
    return model
