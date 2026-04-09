"""
Pipeline de inferencia completo:
  1. Preprocesar imagen OCT.
  2. Segmentar capas retinianas.
  3. Clasificar patología en ambos escenarios (raw / seg).
  4. Retornar resultados estructurados.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

from config import (
    DEVICE, IMG_SIZE, SEG_IMG_SIZE, CLASS_NAMES,
    NUM_SEG_CLASSES, RETINAL_LAYERS,
)
from models.segmentation import RetinaSAMAdapter
from models.classifier import RetinaClassifier


def preprocess_image(image: Image.Image, target_size: int) -> torch.Tensor:
    """
    Convierte una imagen PIL a tensor normalizado para el modelo.

    Args:
        image: Imagen PIL (puede ser RGB o grayscale).
        target_size: Tamaño de salida (cuadrado).
    Returns:
        Tensor [1, 1, target_size, target_size] normalizado a [0, 1].
    """
    img = image.convert("L")  # Forzar grayscale
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor.to(DEVICE)
    
def preprocess_for_segmentation(image: Image.Image) -> torch.Tensor:
    """Preprocesa para el segmentador: escala de grises, 224x512."""
    img = image.convert("L")
    img = img.resize((512, 224), Image.BILINEAR)  # PIL usa (ancho, alto)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 512]
    return tensor.to(DEVICE)

def preprocess_for_classifier(image: Image.Image) -> torch.Tensor:
    """Preprocesa para el clasificador: RGB, 224x224."""
    img = image.convert("RGB")  # fuerza 3 canales
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # [H,W,C] → [C,H,W]
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,3,224,224]
    return tensor.to(DEVICE)


def masks_to_onehot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convierte máscara de clases [B, H, W] a one-hot [B, num_classes, H, W].
    """
    B, H, W = mask.shape
    onehot = torch.zeros(B, num_classes, H, W, device=mask.device)
    onehot.scatter_(1, mask.unsqueeze(1).long(), 1.0)
    return onehot


def run_segmentation(
    model: RetinaSAMAdapter,
    image: Image.Image,
) -> Dict[str, Any]:
    """
    Ejecuta segmentación de capas retinianas.

    Returns:
        dict con:
            - 'mask': numpy array [H, W] con etiquetas de capa.
            - 'mask_onehot': tensor [1, NUM_SEG_CLASSES, H, W].
            - 'probabilities': tensor [1, NUM_SEG_CLASSES, H, W] probabilidades por capa.
    """
    #x = preprocess_image(image, SEG_IMG_SIZE)
    x = preprocess_for_segmentation(image)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1)  # [1, H, W]

    mask_onehot = masks_to_onehot(mask, NUM_SEG_CLASSES)

    return {
        "mask": mask.squeeze(0).cpu().numpy(),
        "mask_onehot": mask_onehot,
        "probabilities": probs,
    }


def run_classification(
    classifier_raw: RetinaClassifier,
    classifier_seg: RetinaClassifier,
    image: Image.Image,
    seg_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ejecuta clasificación en ambos escenarios.

    Returns:
        dict con:
            - 'raw': {'probs': array, 'pred_class': str, 'confidence': float}
            - 'seg': {'probs': array, 'pred_class': str, 'confidence': float}
    """
    # --- Escenario 1: Sin segmentación (imagen cruda) ---
    #x_raw = preprocess_image(image, IMG_SIZE)
    x_raw = preprocess_for_classifier(image)   # ← usa la nueva función
    probs_raw = classifier_raw.predict_proba(x_raw).squeeze(0).cpu().numpy()
    pred_idx_raw = int(np.argmax(probs_raw))

    # --- Escenario 2: Con segmentación (imagen + máscaras) ---
    #x_img = preprocess_image(image, IMG_SIZE)  # [1, 1, 224, 224]
    x_img = preprocess_for_classifier(image)


    # Redimensionar one-hot masks al tamaño del clasificador
    mask_onehot = seg_result["mask_onehot"]  # [1, NUM_SEG_CLASSES, 256, 256]
    mask_resized = F.interpolate(
        mask_onehot, size=(IMG_SIZE, IMG_SIZE), mode="nearest"
    )  # [1, NUM_SEG_CLASSES, 224, 224]

    #x_seg = torch.cat([x_img, mask_resized], dim=1)  # [1, 1+NUM_SEG_CLASSES, 224, 224]
    x_seg = x_img  # mantiene [1,3,224,224]


    probs_seg = classifier_seg.predict_proba(x_seg).squeeze(0).cpu().numpy()
    pred_idx_seg = int(np.argmax(probs_seg))

    return {
        "raw": {
            "probs": probs_raw,
            "pred_class": CLASS_NAMES[pred_idx_raw],
            "confidence": float(probs_raw[pred_idx_raw]),
        },
        "seg": {
            "probs": probs_seg,
            "pred_class": CLASS_NAMES[pred_idx_seg],
            "confidence": float(probs_seg[pred_idx_seg]),
        },
    }


def run_full_pipeline(
    seg_model: RetinaSAMAdapter,
    classifier_raw: RetinaClassifier,
    classifier_seg: RetinaClassifier,
    image: Image.Image,
) -> Dict[str, Any]:
    """
    Pipeline completo: segmentación + clasificación en ambos escenarios.
    """
    seg_result = run_segmentation(seg_model, image)
    cls_result = run_classification(classifier_raw, classifier_seg, image, seg_result)

    return {
        "segmentation": seg_result,
        "classification": cls_result,
    }
