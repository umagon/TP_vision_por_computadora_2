"""
Utilidades de visualización:
  - Overlay de segmentación sobre imagen original.
  - Gráficos comparativos de clasificación (3 escenarios).
  - Barras de probabilidad por clase.
"""
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from typing import Dict

from config import LAYER_COLORS, RETINAL_LAYERS, CLASS_NAMES


def create_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.6,
    target_size: int = 256,
) -> Image.Image:
    """
    Superpone la máscara de segmentación coloreada sobre la imagen OCT.
    Redimensiona la máscara al tamaño de la imagen de salida.
    """
    img = image.convert("L").resize((target_size, target_size), Image.BILINEAR)
    img_rgb = np.stack([np.array(img)] * 3, axis=-1).astype(np.float32)

    # Redimensionar la máscara al mismo tamaño que la imagen
    mask = np.pad(mask, ((136,136),(0,0)))
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    mask_resized = np.array(
        mask_pil.resize((target_size, target_size), Image.NEAREST)
    )

    # Crear overlay coloreado (7 capas, clases 1-7)
    overlay = np.zeros((target_size, target_size, 3), dtype=np.float32)
    for cls_idx in range(1, len(LAYER_COLORS) + 1):
        color = LAYER_COLORS[cls_idx - 1]
        region = mask_resized == cls_idx
        for c in range(3):
            overlay[region, c] = color[c]

    blended = img_rgb * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def create_prob_bars(probs: np.ndarray, title: str = "") -> go.Figure:
    """
    Gráfico de barras horizontal con probabilidades por clase.
    """
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    fig = go.Figure(go.Bar(
        x=probs,
        y=CLASS_NAMES,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="auto",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Probabilidad",
        xaxis=dict(range=[0, 1]),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=12),
    )

    return fig


def create_comparison_chart(probs_dict: Dict[str, np.ndarray]) -> go.Figure:
    """
    Gráfico de barras agrupadas comparando los escenarios disponibles.

    Args:
        probs_dict: Diccionario {nombre_escenario: array_probabilidades}
    """
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    fig = go.Figure()

    for i, (name, probs) in enumerate(probs_dict.items()):
        fig.add_trace(go.Bar(
            name=name,
            x=CLASS_NAMES,
            y=probs,
            marker_color=colors[i % len(colors)],
            text=[f"{p:.1%}" for p in probs],
            textposition="auto",
        ))

    fig.update_layout(
        title="Comparativa de Clasificación por Escenario",
        yaxis_title="Probabilidad",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12),
    )

    return fig


def create_segmentation_legend() -> go.Figure:
    """
    Leyenda visual de las capas retinianas segmentadas (7 capas).
    """
    fig = go.Figure()

    for i, (layer, color) in enumerate(zip(RETINAL_LAYERS, LAYER_COLORS)):
        hex_color = f"rgb({color[0]},{color[1]},{color[2]})"
        fig.add_trace(go.Bar(
            x=[1],
            y=[layer],
            orientation="h",
            marker_color=hex_color,
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title="Capas Retinianas",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        font=dict(size=11),
    )

    return fig
