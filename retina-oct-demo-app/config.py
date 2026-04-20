"""
Configuración central del proyecto.
"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Rutas de pesos ---
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
SEGMENTATION_UNET_WEIGHTS = os.path.join(WEIGHTS_DIR, "retina_unet_adapter.pth")
SEGMENTATION_UNETPLUSPLUS_WEIGHTS = os.path.join(WEIGHTS_DIR, "unetpp_smp_finetunning.pth")
CLASSIFIER_RAW_WEIGHTS = os.path.join(WEIGHTS_DIR, "modelo1_raw_resnet50.pth")
CLASSIFIER_SEG_WEIGHTS = os.path.join(WEIGHTS_DIR, "modelo2_seg_resnet50.pth")
CLASSIFIER_HYBRID_WEIGHTS = os.path.join(WEIGHTS_DIR, "modelo3_raw_seg_resnet50.pth")

# --- Clases de patología ---
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Capas retinianas para segmentación ---
# El modelo UNet++ produce 8 clases: fondo (0) + 7 capas retinianas (1-7)
RETINAL_LAYERS = [
    "Capa 1", "Capa 2", "Capa 3", "Capa 4",
    "Capa 5", "Capa 6", "Capa 7",
]
NUM_SEG_CLASSES = 8

# --- Colores para overlay de segmentación (RGB) ---
# 7 colores para las 7 capas retinianas (clase 0 = fondo, sin color)
LAYER_COLORS = [
    (255, 0, 0),      # Capa 1 - rojo
    (0, 255, 0),      # Capa 2 - verde
    (0, 0, 255),      # Capa 3 - azul
    (255, 255, 0),    # Capa 4 - amarillo
    (255, 0, 255),    # Capa 5 - magenta
    (0, 255, 255),    # Capa 6 - cian
    (255, 128, 0),    # Capa 7 - naranja
]

# --- Parámetros de imagen ---
IMG_SIZE = 224
SEG_IMG_SIZE = (224, 512)   # (alto, ancho) — igual que en entrenamiento

# --- Device ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
