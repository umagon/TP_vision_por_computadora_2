"""
Pipeline de inferencia completo:
  1. Preprocesar imagen OCT.
  2. Segmentar capas retinianas con UNet++.
  3. Clasificar patología en tres escenarios (raw / seg / hybrid).
  4. Retornar resultados estructurados.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

from config import (
    DEVICE, IMG_SIZE, CLASS_NAMES,
    NUM_SEG_CLASSES, RETINAL_LAYERS, LAYER_COLORS,
)
from models.segmentation import RetinaSAMAdapter
from models.classifier import RetinaClassifier


def preprocess_for_segmentation(image: Image.Image) -> torch.Tensor:
    """
    Preprocesa para el segmentador UNet++:
    escala de grises, crop central a (224, 512) — igual que en entrenamiento.
    """
    img = image.convert("L")
    w, h = img.size
    top = max((h - 224) // 2, 0)
    left = max((w - 512) // 2, 0)
    img = img.crop((left, top, left + 512, top + 224))
    # Si la imagen es más chica que 224x512, resize
    if img.size != (512, 224):
        img = img.resize((512, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 512]
    return tensor.to(DEVICE)


def preprocess_for_classifier(image: Image.Image) -> torch.Tensor:
    """Preprocesa para el clasificador: RGB, 224x224."""
    img = image.convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # [H,W,C] → [C,H,W]
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,3,224,224]
    # Normalización para imagenet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(DEVICE)


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    """
    Convierte máscara de segmentación [H, W] a imagen RGB coloreada
    para usar como entrada del Modelo 2 (clasificación sobre máscara).
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(1, len(LAYER_COLORS) + 1):
        color = LAYER_COLORS[cls_idx - 1]
        region = mask == cls_idx
        for c in range(3):
            rgb[region, c] = color[c]
    return Image.fromarray(rgb)


def preprocess_mask_for_classifier(mask: np.ndarray) -> torch.Tensor:
    """
    Preprocesa la máscara de segmentación como imagen RGB coloreada
    para el Modelo 2 (clasificación sobre máscara).
    """
    mask_rgb = Image.fromarray(mask.astype(np.uint8)).resize((224, 224), Image.NEAREST)
    arr = np.array(mask_rgb, dtype=np.float32) / 7.0
    arr = np.stack([arr, arr, arr], axis=0)  # [3,H,W]
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,3,224,224]
    # Normalización para imagenet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.to(DEVICE)


def preprocess_hybrid(image: Image.Image, mask: np.ndarray) -> torch.Tensor:
    """
    Preprocesa para el Modelo 3 (hybrid): 4 canales = RGB + máscara argmax.
    El canal extra es la máscara de segmentación normalizada a [0, 1].
    """
    # Canal RGB
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr_rgb = np.array(img, dtype=np.float32) / 255.0
    arr_rgb = np.transpose(arr_rgb, (2, 0, 1))  # [3, H, W]
    # Normalización para imagenet
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    arr_rgb = (arr_rgb - mean) / std

    # Canal de máscara (resize + normalizar)
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    mask_resized = mask_pil.resize((224, 224), Image.NEAREST)
    arr_mask = np.array(mask_resized, dtype=np.float32) / 7.0  # normalizar a [0, 1]
    arr_mask = arr_mask[np.newaxis, ...]  # [1, H, W]

    # Concatenar: [4, 224, 224]
    arr_4ch = np.concatenate([arr_rgb, arr_mask], axis=0)
    tensor = torch.from_numpy(arr_4ch).unsqueeze(0)  # [1, 4, 224, 224]
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
    Ejecuta segmentación de capas retinianas con UNet++.
    Returns dict con 'mask', 'mask_onehot', 'probabilities'.
    """
    x = preprocess_for_segmentation(image)  # [1, 1, 224, 512]

    model.eval()
    with torch.no_grad():
        logits = model(x)                       # [1, 8, 224, 512]
        probs = torch.softmax(logits, dim=1)
        mask = torch.argmax(logits, dim=1)       # [1, 224, 512]

    mask_onehot = masks_to_onehot(mask, NUM_SEG_CLASSES)

    return {
        "mask":          mask.squeeze(0).cpu().numpy(),   # (224, 512)
        "mask_onehot":   mask_onehot,
        "probabilities": probs,
    }


def run_classification(
    classifier_raw: Optional[RetinaClassifier],
    classifier_seg: Optional[RetinaClassifier],
    classifier_hybrid: Optional[RetinaClassifier],
    image: Image.Image,
    seg_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ejecuta clasificación en los tres escenarios.
    Modelos que sean None se omiten del resultado.
    """
    results = {}
    mask = seg_result["mask"]  # (224, 512)

    # ── Modelo 1: imagen cruda RGB (3 canales) ──
    if classifier_raw is not None:
        x_raw = preprocess_for_classifier(image)  # [1, 3, 224, 224]
        probs_raw = classifier_raw.predict_proba(x_raw).squeeze(0).cpu().numpy()
        pred_idx_raw = int(np.argmax(probs_raw))
        results["raw"] = {
            "probs":      probs_raw,
            "pred_class": CLASS_NAMES[pred_idx_raw],
            "confidence": float(probs_raw[pred_idx_raw]),
        }

    # ── Modelo 2: máscara de segmentación coloreada RGB (3 canales) ──
    if classifier_seg is not None:
        x_seg = preprocess_mask_for_classifier(mask)  # [1, 3, 224, 224]
        probs_seg = classifier_seg.predict_proba(x_seg).squeeze(0).cpu().numpy()
        pred_idx_seg = int(np.argmax(probs_seg))
        results["seg"] = {
            "probs":      probs_seg,
            "pred_class": CLASS_NAMES[pred_idx_seg],
            "confidence": float(probs_seg[pred_idx_seg]),
        }

    # ── Modelo 3: hybrid RGB + máscara (4 canales) ──
    if classifier_hybrid is not None:
        x_hyb = preprocess_hybrid(image, mask)  # [1, 4, 224, 224]
        x_hyb = x_hyb.float()  # asegurar tipo float32 para el modelo
        probs_hyb = classifier_hybrid.predict_proba(x_hyb).squeeze(0).cpu().numpy()
        pred_idx_hyb = int(np.argmax(probs_hyb))
        results["hybrid"] = {
            "probs":      probs_hyb,
            "pred_class": CLASS_NAMES[pred_idx_hyb],
            "confidence": float(probs_hyb[pred_idx_hyb]),
        }

    return results


def run_full_pipeline(
    seg_model:         RetinaSAMAdapter,
    classifier_raw:    Optional[RetinaClassifier],
    classifier_seg:    Optional[RetinaClassifier],
    classifier_hybrid: Optional[RetinaClassifier],
    image:             Image.Image,
) -> Dict[str, Any]:
    """Pipeline completo: segmentación + clasificación en 3 escenarios."""
    seg_result = run_segmentation(seg_model, image)
    cls_result = run_classification(
        classifier_raw, classifier_seg, classifier_hybrid, image, seg_result
    )
    return {
        "segmentation":   seg_result,
        "classification": cls_result,
    }
