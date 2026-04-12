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
from typing import Dict, Any

from config import (
    DEVICE, IMG_SIZE, CLASS_NAMES,
    NUM_SEG_CLASSES, RETINAL_LAYERS,
)
from models.segmentation import RetinaSAMAdapter
from models.classifier import RetinaClassifier


def preprocess_for_segmentation(image: Image.Image) -> torch.Tensor:
    """
    Preprocesa para el segmentador UNet++:
    escala de grises, resize a (224, 512) — igual que en entrenamiento.
    PIL usa (ancho, alto), PyTorch espera (alto, ancho).
    """
    img = image.convert("L")
    w, h = img.size
    top = (h - 224) // 2
    left = (w - 512) // 2
    img = img.crop((left, top, left+512, top + 224))
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
    """Convierte máscara [B, H, W] a one-hot [B, num_classes, H, W]."""
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
    Returns dict con 'mask', 'mask_onehot', 'probabilities'.
    """
    x = preprocess_for_segmentation(image)   # [1, 1, 224, 512]

    model.eval()
    with torch.no_grad():
        logits = model(x)                        # [1, 8, 224, 512]
        probs  = torch.softmax(logits, dim=1)
        mask   = torch.argmax(probs, dim=1)      # [1, 224, 512]

    mask_onehot = masks_to_onehot(mask, NUM_SEG_CLASSES)

    return {
        "mask":        mask.squeeze(0).cpu().numpy(),   # (224, 512)
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
    """
    # ── Escenario 1: imagen cruda (1 canal) ──────────────────────────
    x_raw = preprocess_for_classifier(image)   # [1, 1, 224, 224]
    probs_raw   = classifier_raw.predict_proba(x_raw).squeeze(0).cpu().numpy()
    pred_idx_raw = int(np.argmax(probs_raw))

    # ── Escenario 2: imagen segmentada (1 canal) ─────────────────────
    # Por ahora el clasificador seg también recibe 1 canal —
    # cuando tengas el modelo seg entrenado podés concatenar la máscara acá.
    x_seg = preprocess_for_classifier(image)   # [1, 1, 224, 224]
    probs_seg    = classifier_seg.predict_proba(x_seg).squeeze(0).cpu().numpy()
    pred_idx_seg = int(np.argmax(probs_seg))

    return {
        "raw": {
            "probs":      probs_raw,
            "pred_class": CLASS_NAMES[pred_idx_raw],
            "confidence": float(probs_raw[pred_idx_raw]),
        },
        "seg": {
            "probs":      probs_seg,
            "pred_class": CLASS_NAMES[pred_idx_seg],
            "confidence": float(probs_seg[pred_idx_seg]),
        },
    }


def run_full_pipeline(
    seg_model:      RetinaSAMAdapter,
    classifier_raw: RetinaClassifier,
    classifier_seg: RetinaClassifier,
    image:          Image.Image,
) -> Dict[str, Any]:
    """Pipeline completo: segmentación + clasificación."""
    seg_result = run_segmentation(seg_model, image)
    cls_result = run_classification(classifier_raw, classifier_seg, image, seg_result)
    return {
        "segmentation":   seg_result,
        "classification": cls_result,
    }