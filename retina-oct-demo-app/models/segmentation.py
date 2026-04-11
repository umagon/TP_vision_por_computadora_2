import os
from .segmentation_unetplusplus import RetinaSAMAdapterPlusPlus, load_segmentation_model as load_segmentation_model_unetplusplus
from .segmentation_unet import RetinaSAMAdapter as RetinaSAMAdapterUNet, load_segmentation_model as load_segmentation_model_unet

# Por defecto usa unetplusplus, que es el modelo entrenado
SEGMENTATION_MODEL = os.environ.get("SEGMENTATION_MODEL", "unetplusplus").strip().lower()

print('Cargando modelo de segmentacion:', SEGMENTATION_MODEL)
if SEGMENTATION_MODEL == "unet":
    RetinaSAMAdapter = RetinaSAMAdapterUNet
    load_segmentation_model = load_segmentation_model_unet
else:
    RetinaSAMAdapter = RetinaSAMAdapterPlusPlus
    load_segmentation_model = load_segmentation_model_unetplusplus

__all__ = [
    "RetinaSAMAdapter",
    "load_segmentation_model",
]