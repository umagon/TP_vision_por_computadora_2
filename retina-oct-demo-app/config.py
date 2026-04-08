"""
Configuración central del proyecto.
Ajusta estas rutas y parámetros según tu entorno.
"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Rutas de pesos ---
WEIGHTS_DIR = os.path.join(ROOT_DIR, "modelos")
SEGMENTATION_UNET_WEIGHTS = os.path.join(WEIGHTS_DIR, "retina_sam_adapter.pth")
SEGMENTATION_UNETPLUSPLUS_WEIGHTS = os.path.join(WEIGHTS_DIR, "unetpp_smp_finetunning.pth")
CLASSIFIER_RAW_WEIGHTS = os.path.join(WEIGHTS_DIR, "classifier_raw.pth")
CLASSIFIER_SEG_WEIGHTS = os.path.join(WEIGHTS_DIR, "classifier_seg.pth")

# --- Clases de patología ---
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Capas retinianas para segmentación ---
RETINAL_LAYERS = [
    "ILM", "NFL/GCL", "IPL", "INL", "OPL",
    "ONL", "ELM", "IS/OS", "RPE", "BM",
]
NUM_SEG_CLASSES = len(RETINAL_LAYERS) + 1  # +1 fondo

# --- Colores para overlay de segmentación (RGB) ---
LAYER_COLORS = [
    (255, 0, 0),      # ILM - rojo
    (0, 255, 0),      # NFL/GCL - verde
    (0, 0, 255),      # IPL - azul
    (255, 255, 0),    # INL - amarillo
    (255, 0, 255),    # OPL - magenta
    (0, 255, 255),    # ONL - cian
    (255, 128, 0),    # ELM - naranja
    (128, 0, 255),    # IS/OS - violeta
    (0, 128, 128),    # RPE - teal
    (128, 128, 0),    # BM - oliva
]

# --- Parámetros de imagen ---
IMG_SIZE = 224
SEG_IMG_SIZE = 256

# --- Device ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
