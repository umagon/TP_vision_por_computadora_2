import os

from .segmentation_unet import RetinaSAMAdapter as RetinaSAMAdapterUNet, load_segmentation_model as load_segmentation_model_unet
from .segmentation_unetplusplus import RetinaSAMAdapterPlusPlus, load_segmentation_model as load_segmentation_model_unetplusplus

# Permite seleccionar el modelo de segmentación en tiempo de ejecución.
# Por defecto se usa U-Net clásico para mantener compatibilidad con la app.
SEGMENTATION_MODEL = os.environ.get("SEGMENTATION_MODEL", "unet").strip().lower()

if SEGMENTATION_MODEL == "unetplusplus":
    RetinaSAMAdapter = RetinaSAMAdapterPlusPlus
    load_segmentation_model = load_segmentation_model_unetplusplus
else:
    RetinaSAMAdapter = RetinaSAMAdapterUNet
    load_segmentation_model = load_segmentation_model_unet

__all__ = [
    "RetinaSAMAdapter",
    "load_segmentation_model",
    "RetinaSAMAdapterUNet",
    "RetinaSAMAdapterPlusPlus",
    "load_segmentation_model_unet",
    "load_segmentation_model_unetplusplus",
]
