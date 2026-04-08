"""
Modelo de segmentación de capas retinianas.
Usa una arquitectura U-Net++ con encoder preentrenado.
Este módulo permite usar la variante U-Net++ desde Streamlit.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import NUM_SEG_CLASSES, SEG_IMG_SIZE, DEVICE, SEGMENTATION_UNETPLUSPLUS_WEIGHTS
import os


class RetinaSAMAdapterPlusPlus(nn.Module):
    """
    Wrapper para el modelo de segmentación de capas retinianas usando U-Net++.
    """

    def __init__(self, num_classes: int = NUM_SEG_CLASSES, encoder_name: str = "resnet34"):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, 1, H, W] imagen OCT en grayscale normalizada.
        Returns:
            logits: Tensor [B, num_classes, H, W]
        """
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inferencia con softmax → máscara de clases."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            mask = torch.argmax(probs, dim=1)
        return mask


def load_segmentation_model(weights_path: str = SEGMENTATION_UNETPLUSPLUS_WEIGHTS) -> RetinaSAMAdapterPlusPlus:
    """
    Carga el modelo de segmentación U-Net++ con pesos entrenados.
    Si no existen pesos, retorna el modelo con encoder ImageNet.
    """
    model = RetinaSAMAdapterPlusPlus()

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[Segmentación U-Net++] Pesos cargados desde: {weights_path}")
    else:
        print(f"[Segmentación U-Net++] AVISO: No se encontraron pesos en {weights_path}")
        print("  → Usando pesos de ImageNet (encoder). Entrena el modelo primero.")

    model.to(DEVICE)
    model.eval()
    return model
