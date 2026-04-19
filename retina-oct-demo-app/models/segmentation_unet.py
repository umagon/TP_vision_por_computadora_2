"""
Modelo de segmentación de capas retinianas.
Usa una arquitectura U-Net con encoder preentrenado como base.
Puedes reemplazar este módulo por tu implementación de RetinaSAM-Adapter.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import NUM_SEG_CLASSES, SEG_IMG_SIZE, DEVICE, SEGMENTATION_UNET_WEIGHTS
import os


class RetinaSAMAdapter(nn.Module):
    """
    Wrapper para el modelo de segmentación de capas retinianas.

    Arquitectura base: U-Net con encoder EfficientNet-B0.
    Reemplaza esta clase por tu implementación real de SAM-Adapter
    cuando tengas los pesos entrenados.
    """

    def __init__(self, num_classes: int = NUM_SEG_CLASSES, encoder_name: str = "resnet34"):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=1,  # OCT es grayscale
            classes=num_classes,
  #          activation=None,  # logits crudos, softmax en inferencia
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
            mask = torch.argmax(probs, dim=1)  # [B, H, W]
        return mask

def load_segmentation_model(weights_path: str = SEGMENTATION_UNET_WEIGHTS) -> RetinaSAMAdapter:
    model = RetinaSAMAdapter()
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        
        # El .pth fue guardado directamente desde smp.UnetPlusPlus (sin wrapper),
        # por eso las claves no tienen el prefijo "model."
        # Se lo agregamos para que coincidan con RetinaUNETAdapter.self.model
        state_dict = {"model." + k: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print(f"[Segmentación] Pesos cargados desde: {weights_path}")
    else:
        print(f"[Segmentación] AVISO: No se encontraron pesos en {weights_path}")
        print("  → Usando pesos de ImageNet (encoder). Entrena el modelo primero.")
    
    model.to(DEVICE)
    model.eval()
    return model    
