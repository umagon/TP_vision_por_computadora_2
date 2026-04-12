"""
Retina OCT Demo — Segmentation-Aware Classification of Retinal Diseases

App web Streamlit con 3 secciones:
  1. Segmentación interactiva de capas retinianas.
  2. Clasificación de patología en dos escenarios (con/sin segmentación).
  3. Comparativa visual y numérica de resultados.

Ejecutar con:
    streamlit run app.py
"""
import streamlit as st
import numpy as np
from PIL import Image
import time
import sys
import os
from config import ROOT_DIR


# Agregar raíz del proyecto al path
sys.path.insert(0, ROOT_DIR)

from config import CLASS_NAMES, RETINAL_LAYERS, DEVICE, IMG_SIZE, SEG_IMG_SIZE
from models.segmentation import load_segmentation_model
from models.classifier import load_classifier_model
from models.pipeline import run_full_pipeline, run_segmentation
from utils.visualization import (
    create_overlay,
    create_prob_bars,
    create_comparison_chart,
    create_segmentation_legend,
)

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retina OCT | Segmentación + Clasificación",
    page_icon=os.path.join(ROOT_DIR, "irisvision-demo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .metric-card h1 {
        margin: 0.3rem 0 0 0;
        font-size: 1.8rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success { background: #d4edda; color: #155724; }
    .badge-warning { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Carga de modelos (caché para no recargar)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Carga los 3 modelos una sola vez."""
    with st.spinner("Cargando modelos..."):
        seg_model = load_segmentation_model()
        cls_raw = None # load_classifier_model(mode="raw")
        cls_seg = None # load_classifier_model(mode="seg")
    return seg_model, cls_raw, cls_seg


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(os.path.join(ROOT_DIR, "irisvision-demo.png"), width=140)
    st.title("Retina OCT Demo")
    st.markdown("---")

    section = st.radio(
        "Navegación",
        options=[
            "🏠 Inicio",
            "🔬 Segmentación",
            "🧪 Clasificación",
            "📊 Comparativa",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Parámetros**")
    overlay_alpha = st.slider("Transparencia del overlay", 0.1, 0.9, 0.6, 0.05)
    show_legend = st.checkbox("Mostrar leyenda de capas", value=True)

    st.markdown("---")
    st.markdown(
        f"**Dispositivo:** `{DEVICE}`  \n"
        f"**Clases:** {', '.join(CLASS_NAMES)}  \n"
        f"**Capas:** {len(RETINAL_LAYERS)}"
    )


# ─────────────────────────────────────────────
# Cargar modelos
# ─────────────────────────────────────────────
seg_model, cls_raw, cls_seg = load_models()


# ─────────────────────────────────────────────
# Sección: Inicio
# ─────────────────────────────────────────────
if section == "🏠 Inicio":
    st.markdown('<div class="main-header">Segmentation-Aware Classification<br>of Retinal Diseases</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demo interactivo — Kaggle OCT Dataset</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>SEGMENTACIÓN</h3>
            <h1>RetinaSAM</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Segmentación de capas retinianas usando SAM-Adapter con encoder preentrenado.")

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>CLASIFICACIÓN</h3>
            <h1>4 Clases</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("CNV, DME, Drusen y Normal — clasificadas con EfficientNet/ResNet.")

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>COMPARATIVA</h3>
            <h1>2 Escenarios</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Evaluación con y sin información anatómica segmentada.")

    st.markdown("---")
    st.markdown("### Pipeline del sistema")
    st.markdown("""
    ```
    Imagen OCT  ──┬──────────────────────────────────►  Clasificador (Raw)  ──►  Predicción A
                   │
                   └──►  RetinaSAM-Adapter  ──►  Máscaras  ──┐
                                                              │
                   Imagen OCT  + Máscaras  ◄──────────────────┘
                         │
                         └──►  Clasificador (Seg)  ──►  Predicción B
                                                              │
                   Comparativa A vs B  ◄──────────────────────┘
    ```
    """)

    st.info("Sube una imagen OCT desde la barra lateral de cada sección para comenzar el análisis.")


# ─────────────────────────────────────────────
# Sección: Segmentación
# ─────────────────────────────────────────────
elif section == "🔬 Segmentación":
    st.header("Segmentación de Capas Retinianas")
    st.markdown("Sube una imagen OCT para segmentar las capas retinianas con RetinaSAM-Adapter.")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen OCT",
        type=["png", "jpg", "jpeg", "tif", "bmp"],
        key="seg_upload",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col_img, col_seg = st.columns(2)

        with col_img:
            st.subheader("Imagen Original")
            st.image(image, width='stretch', clamp=True)

        # Ejecutar segmentación
        with st.spinner("Segmentando capas retinianas..."):
            t0 = time.time()
            seg_result = run_segmentation(seg_model, image)
            elapsed = time.time() - t0

        mask = seg_result["mask"]
        overlay_img = create_overlay(image, mask, alpha=overlay_alpha)

        with col_seg:
            st.subheader("Segmentación (Overlay)")
            st.image(overlay_img, width='stretch', clamp=True)

        # Info
        st.success(f"Segmentación completada en {elapsed:.2f}s")

        # Estadísticas de capas
        st.subheader("Distribución de Capas")
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size

        layer_data = []
        for u, c in zip(unique, counts):
            name = "Fondo" if u == 0 else RETINAL_LAYERS[u - 1] if u <= len(RETINAL_LAYERS) else f"Clase {u}"
            pct = c / total_pixels * 100
            layer_data.append({"Capa": name, "Píxeles": int(c), "Porcentaje": f"{pct:.1f}%"})

        st.dataframe(layer_data, width='stretch')

        if show_legend:
            st.plotly_chart(create_segmentation_legend(), width='stretch')

        # Guardar en session_state para uso en otras secciones
        st.session_state["seg_result"] = seg_result
        st.session_state["current_image"] = image

    else:
        st.info("Sube una imagen OCT para comenzar la segmentación.")


# ─────────────────────────────────────────────
# Sección: Clasificación
# ─────────────────────────────────────────────
elif section == "🧪 Clasificación":
    st.header("Clasificación de Patología Retiniana")
    st.markdown("Clasifica la imagen en **CNV**, **DME**, **Drusen** o **Normal** usando dos escenarios.")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen OCT",
        type=["png", "jpg", "jpeg", "tif", "bmp"],
        key="cls_upload",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", width=300)

        with st.spinner("Ejecutando pipeline completo..."):
            t0 = time.time()
            results = run_full_pipeline(seg_model, cls_raw, cls_seg, image)
            elapsed = time.time() - t0

        cls = results["classification"]
        st.success(f"Pipeline completo en {elapsed:.2f}s")

        # Resultados lado a lado
        col_raw, col_seg = st.columns(2)

        with col_raw:
            st.subheader("Sin Segmentación")
            pred = cls["raw"]
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);">
                <h3>PREDICCIÓN</h3>
                <h1>{pred['pred_class']}</h1>
                <h3>Confianza: {pred['confidence']:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(
                create_prob_bars(pred["probs"], "Probabilidades (Raw)"),
                width='stretch',
            )

        with col_seg:
            st.subheader("Con Segmentación")
            pred = cls["seg"]
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                <h3>PREDICCIÓN</h3>
                <h1>{pred['pred_class']}</h1>
                <h3>Confianza: {pred['confidence']:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(
                create_prob_bars(pred["probs"], "Probabilidades (Seg)"),
                width='stretch',
            )

        # Overlay de segmentación
        seg_mask = results["segmentation"]["mask"]
        overlay_img = create_overlay(image, seg_mask, alpha=overlay_alpha)

        st.subheader("Segmentación utilizada")
        c1, c2 = st.columns(2)
        c1.image(image.convert("L").resize(SEG_IMG_SIZE, Image.BILINEAR), caption="Original", width='stretch')
        
        c2.image(overlay_img, caption="Con overlay de capas", width='stretch')

        # Guardar resultados
        st.session_state["cls_results"] = cls
        st.session_state["seg_result"] = results["segmentation"]
        st.session_state["current_image"] = image

    else:
        st.info("Sube una imagen OCT para clasificar la patología.")


# ─────────────────────────────────────────────
# Sección: Comparativa
# ─────────────────────────────────────────────
elif section == "📊 Comparativa":
    st.header("Comparativa: Sin vs. Con Segmentación")

    if "cls_results" not in st.session_state:
        st.warning("Primero clasifica una imagen en la sección '🧪 Clasificación'.")
    else:
        cls = st.session_state["cls_results"]
        raw = cls["raw"]
        seg = cls["seg"]

        # Tabla resumen
        st.subheader("Resumen de Predicciones")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sin Segmentación**")
            st.metric("Clase predicha", raw["pred_class"])
            st.metric("Confianza", f"{raw['confidence']:.1%}")

        with col2:
            st.markdown("**Con Segmentación**")
            st.metric("Clase predicha", seg["pred_class"])
            st.metric("Confianza", f"{seg['confidence']:.1%}")

        # Gráfico comparativo
        st.subheader("Comparativa de Probabilidades")
        fig = create_comparison_chart(raw["probs"], seg["probs"])
        st.plotly_chart(fig, width='stretch')

        # Diferencias numéricas
        st.subheader("Diferencia por Clase")
        diff_data = []
        for i, name in enumerate(CLASS_NAMES):
            diff = seg["probs"][i] - raw["probs"][i]
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
            diff_data.append({
                "Clase": name,
                "P(Raw)": f"{raw['probs'][i]:.3f}",
                "P(Seg)": f"{seg['probs'][i]:.3f}",
                "Diferencia": f"{direction} {abs(diff):.3f}",
            })
        st.dataframe(diff_data, width='stretch')

        # Métricas de evaluación (placeholder para métricas del dataset completo)
        st.subheader("Métricas de Evaluación (Dataset Completo)")
        st.info(
            "Las métricas agregadas (Accuracy, F1-Score, AUC) se calcularán al evaluar "
            "sobre el dataset de test completo. Conecta tu script de evaluación aquí."
        )

        # Tabla placeholder de métricas
        metrics_placeholder = {
            "Métrica": ["Accuracy", "F1-Score (macro)", "AUC (macro)"],
            "Sin Segmentación": ["—", "—", "—"],
            "Con Segmentación": ["—", "—", "—"],
        }
        st.dataframe(metrics_placeholder, width='stretch')

        st.markdown("""
        > **Nota:** Rellena esta tabla con los resultados de tu evaluación en el dataset
        > de test del Kaggle OCT. Puedes usar el script `evaluate.py` (por implementar)
        > para calcular las métricas automáticamente.
        """)

        # Overlay si está disponible
        if "current_image" in st.session_state and "seg_result" in st.session_state:
            st.subheader("Visualización de Segmentación")
            img = st.session_state["current_image"]
            mask = st.session_state["seg_result"]["mask"]
            overlay = create_overlay(img, mask, alpha=overlay_alpha)

            c1, c2 = st.columns(2)
            c1.image(img.convert("L").resize(SEG_IMG_SIZE, Image.BILINEAR), caption="Original", width='stretch')
            c2.image(overlay, caption="Segmentación", width='stretch')


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "Segmentation-Aware Classification of Retinal Diseases using Kaggle OCT &mdash; Demo v1.0"
    "</div>",
    unsafe_allow_html=True,
)
