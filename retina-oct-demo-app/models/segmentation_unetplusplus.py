"""
Modelo de segmentación de capas retinianas.
Usa una arquitectura U-Net++ con encoder ResNet34.
"""
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import NUM_SEG_CLASSES, DEVICE, SEGMENTATION_UNETPLUSPLUS_WEIGHTS


class RetinaSAMAdapterPlusPlus(nn.Module):
    def __init__(self, num_classes: int = NUM_SEG_CLASSES, encoder_name: str = "resnet34"):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,   # no cargar imagenet, vamos a cargar nuestros pesos
            in_channels=1,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            mask = torch.argmax(probs, dim=1)
        return mask


def load_segmentation_model(weights_path: str = SEGMENTATION_UNETPLUSPLUS_WEIGHTS) -> RetinaSAMAdapterPlusPlus:
    model = RetinaSAMAdapterPlusPlus()

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)

        # El .pth fue guardado directamente desde smp.UnetPlusPlus (sin wrapper),
        # por eso las claves no tienen el prefijo "model." — se lo agregamos
        state_dict = {"model." + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print(f"[Segmentación U-Net++] Pesos cargados desde: {weights_path}")
    else:
        print(f"[Segmentación U-Net++] AVISO: No se encontraron pesos en {weights_path}")

    model.to(DEVICE)
    model.eval()
    return model