"""
Utilidades de visualización:
  - Overlay de segmentación sobre imagen original.
  - Gráficos comparativos de clasificación.
  - Barras de probabilidad por clase.
"""
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from config import LAYER_COLORS, RETINAL_LAYERS, CLASS_NAMES


def create_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.6,
    target_size: int = 256,
) -> Image.Image:
    """
    Superpone la máscara de segmentación coloreada sobre la imagen OCT.

    Args:
        image: Imagen original PIL.
        mask: Máscara de clases [H, W] con valores 0..N.
        alpha: Transparencia del overlay (0 = solo imagen, 1 = solo máscara).
        target_size: Tamaño de salida.
    Returns:
        Imagen PIL con overlay.
    """
    img = image.convert("L").resize((target_size, target_size), Image.BILINEAR)
    img_rgb = np.stack([np.array(img)] * 3, axis=-1).astype(np.float32)

    # Crear overlay coloreado
    overlay = np.zeros((target_size, target_size, 3), dtype=np.float32)
    for cls_idx in range(1, len(LAYER_COLORS) + 1):  # 0 = fondo, sin color
        color = LAYER_COLORS[cls_idx - 1]
        region = mask == cls_idx
        for c in range(3):
            overlay[region, c] = color[c]

    # Mezclar
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


def create_comparison_chart(
    probs_raw: np.ndarray,
    probs_seg: np.ndarray,
) -> go.Figure:
    """
    Gráfico de barras agrupadas comparando ambos escenarios.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Sin Segmentación",
        x=CLASS_NAMES,
        y=probs_raw,
        marker_color="#3498db",
        text=[f"{p:.1%}" for p in probs_raw],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name="Con Segmentación",
        x=CLASS_NAMES,
        y=probs_seg,
        marker_color="#e74c3c",
        text=[f"{p:.1%}" for p in probs_seg],
        textposition="auto",
    ))

    fig.update_layout(
        title="Comparativa de Clasificación: Sin vs. Con Segmentación",
        yaxis_title="Probabilidad",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12),
    )

    return fig


def create_segmentation_legend() -> go.Figure:
    """
    Leyenda visual de las capas retinianas segmentadas.
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
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        font=dict(size=11),
    )

    return fig
