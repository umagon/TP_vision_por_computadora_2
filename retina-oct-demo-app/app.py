"""
Retina OCT Demo — Segmentación y clasificación de patologías retinales en imágenes OCT mediante Deep Learning

App web Streamlit con 4 secciones:
  1. Inicio — descripción del sistema y métricas.
  2. Segmentación interactiva de capas retinianas con UNet++.
  3. Clasificación de patología en tres escenarios (raw / seg / hybrid).
  4. Comparativa visual y numérica de resultados.

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

from config import CLASS_NAMES, RETINAL_LAYERS, DEVICE, IMG_SIZE, SEG_IMG_SIZE, NUM_SEG_CLASSES
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

st.set_option('client.showErrorDetails', False)  # Ocultar detalles de errores en la interfaz

# ─────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
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
    """Carga los 4 modelos una sola vez."""
    with st.spinner("Cargando modelos..."):
        seg_model = load_segmentation_model()
        cls_raw = load_classifier_model(mode="raw")
        cls_seg = load_classifier_model(mode="seg")
        cls_hybrid = load_classifier_model(mode="hybrid")
    return seg_model, cls_raw, cls_seg, cls_hybrid


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
        f"**Capas seg.:** {NUM_SEG_CLASSES} (fondo + {len(RETINAL_LAYERS)})"
    )

# ─────────────────────────────────────────────
# Cargar modelos
# ─────────────────────────────────────────────
seg_model, cls_raw, cls_seg, cls_hybrid = load_models()


# ─────────────────────────────────────────────
# Sección: Inicio
# ─────────────────────────────────────────────
if section == "🏠 Inicio":
    st.markdown('<div class="main-header">Segmentación y clasificación<br>de patologías retinales</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demo interactivo</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>SEGMENTACIÓN</h3>
            <h1>UNet++</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Segmentación de capas retinianas con UNet++ (ResNet34 encoder) — Dice: **0.9814**")

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>CLASIFICACIÓN</h3>
            <h1>4 Clases</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("CNV, DME, Drusen y Normal — clasificadas con ResNet50 (IMAGENET1K_V2).")

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>COMPARATIVA</h3>
            <h1>3 Escenarios</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Imagen directa (92%), solo máscara (72%) e híbrido RGB+máscara (92%).")

    st.markdown("---")
    st.markdown("### Pipeline del sistema")
    st.markdown("""
    ```
    Imagen OCT ──┬──────────────────────────────────► Clasificador Modelo 1 (Raw)     ── Acc: 92%
                  │
                  └──► UNet++ (ResNet34) ──► Máscara ──┬── Clasificador Modelo 2 (Seg)   ── Acc: 72%
                                                        │
                  Imagen OCT + Máscara ◄───────────────┘
                        │
                        └──► Clasificador Modelo 3 (Hybrid 4ch) ── Acc: 92%
                                                        │
                  Comparativa Modelo 1 vs 2 vs 3 ◄─────┘
    ```
    """)

    st.info("Sube una imagen OCT desde cada sección para comenzar el análisis.")


# ─────────────────────────────────────────────
# Sección: Segmentación
# ─────────────────────────────────────────────
elif section == "🔬 Segmentación":
    st.header("Segmentación de Capas Retinianas")
    st.markdown("Sube una imagen OCT para segmentar las capas retinianas con **UNet++** (encoder ResNet34).")

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
            st.image(image, use_column_width=True, clamp=True)

        # Ejecutar segmentación
        with st.spinner("Segmentando capas retinianas..."):
            t0 = time.time()
            seg_result = run_segmentation(seg_model, image)
            elapsed = time.time() - t0

        mask = seg_result["mask"]
        overlay_img = create_overlay(image, mask, alpha=overlay_alpha)

        with col_seg:
            st.subheader("Segmentación (Overlay)")
            st.image(overlay_img, use_column_width=True, clamp=True)

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

        st.dataframe(layer_data, width=True)

        if show_legend:
            st.plotly_chart(create_segmentation_legend(), use_column_width=True)

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
    st.markdown("Clasifica la imagen en **CNV**, **DME**, **Drusen** o **Normal** usando tres escenarios.")

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
            results = run_full_pipeline(seg_model, cls_raw, cls_seg, cls_hybrid, image)
            elapsed = time.time() - t0

        cls = results["classification"]
        st.success(f"Pipeline completo en {elapsed:.2f}s")

        # Resultados lado a lado — 3 columnas
        col_raw, col_seg, col_hyb = st.columns(3)

        if "raw" in cls:
            with col_raw:
                st.subheader("Modelo 1: Raw")
                pred = cls["raw"]
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);">
                    <h3>PREDICCIÓN (Imagen RGB)</h3>
                    <h1>{pred['pred_class']}</h1>
                    <h3>Confianza: {pred['confidence']:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(
                    create_prob_bars(pred["probs"], "Probabilidades (Raw)"),
                    use_column_width=True,
                )

        if "seg" in cls:
            with col_seg:
                st.subheader("Modelo 2: Seg")
                pred = cls["seg"]
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                    <h3>PREDICCIÓN (Máscara)</h3>
                    <h1>{pred['pred_class']}</h1>
                    <h3>Confianza: {pred['confidence']:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(
                    create_prob_bars(pred["probs"], "Probabilidades (Seg)"),
                    use_column_width=True,
                )

        if "hybrid" in cls:
            with col_hyb:
                st.subheader("Modelo 3: Hybrid")
                pred = cls["hybrid"]
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);">
                    <h3>PREDICCIÓN (RGB + Máscara)</h3>
                    <h1>{pred['pred_class']}</h1>
                    <h3>Confianza: {pred['confidence']:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(
                    create_prob_bars(pred["probs"], "Probabilidades (Hybrid)"),
                    use_column_width=True,
                )

        # Overlay de segmentación
        seg_mask = results["segmentation"]["mask"]
        overlay_img = create_overlay(image, seg_mask, alpha=overlay_alpha)

        st.subheader("Segmentación utilizada")
        c1, c2 = st.columns(2)
        c1.image(image.convert("L").resize(SEG_IMG_SIZE[::-1], Image.BILINEAR), caption="Original", use_column_width=True)
        c2.image(overlay_img, caption="Con overlay de capas", use_column_width=True)

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
    st.header("Comparativa: 3 Escenarios de Clasificación")

    if "cls_results" not in st.session_state:
        st.warning("Primero clasifica una imagen en la sección '🧪 Clasificación'.")
    else:
        cls = st.session_state["cls_results"]

        # Tabla resumen
        st.subheader("Resumen de Predicciones")

        cols = st.columns(3)
        scenario_info = [
            ("raw", "Modelo 1: Raw (RGB)", cols[0]),
            ("seg", "Modelo 2: Seg (Máscara)", cols[1]),
            ("hybrid", "Modelo 3: Hybrid (4ch)", cols[2]),
        ]

        for key, label, col in scenario_info:
            if key in cls:
                with col:
                    st.markdown(f"**{label}**")
                    st.metric("Clase predicha", cls[key]["pred_class"])
                    st.metric("Confianza", f"{cls[key]['confidence']:.1%}")

        # Gráfico comparativo
        st.subheader("Comparativa de Probabilidades")
        probs_dict = {}
        if "raw" in cls:
            probs_dict["Modelo 1 (Raw)"] = cls["raw"]["probs"]
        if "seg" in cls:
            probs_dict["Modelo 2 (Seg)"] = cls["seg"]["probs"]
        if "hybrid" in cls:
            probs_dict["Modelo 3 (Hybrid)"] = cls["hybrid"]["probs"]

        fig = create_comparison_chart(probs_dict)
        st.plotly_chart(fig, use_column_width=True)

        # Diferencias numéricas
        st.subheader("Probabilidades por Clase")
        diff_data = []
        for i, name in enumerate(CLASS_NAMES):
            row = {"Clase": name}
            for key, label, _ in scenario_info:
                if key in cls:
                    row[label] = f"{cls[key]['probs'][i]:.3f}"
            diff_data.append(row)
        st.dataframe(diff_data, width=True)

        # Métricas de evaluación (datos reales del entrenamiento)
        st.subheader("Métricas de Evaluación (Dataset de Test)")

        metrics_data = {
            "Métrica": ["Accuracy", "F1-Score (macro)"],
            "Modelo 1 (Raw)": ["92%", "0.92"],
            "Modelo 2 (Seg)": ["72%", "0.71"],
            "Modelo 3 (Hybrid)": ["92%", "0.92"],
        }
        st.dataframe(metrics_data, width=True)

        st.markdown("""
        > **Hallazgo principal:** La información de segmentación no aporta ganancia adicional
        > al clasificador — el Modelo 3 (hybrid) iguala al Modelo 1 (raw), lo que sugiere
        > que ResNet50 extrae implícitamente la información estructural de las capas retinianas.
        """)

        # Overlay si está disponible
        if "current_image" in st.session_state and "seg_result" in st.session_state:
            st.subheader("Visualización de Segmentación")
            img = st.session_state["current_image"]
            mask = st.session_state["seg_result"]["mask"]
            overlay = create_overlay(img, mask, alpha=overlay_alpha)

            c1, c2 = st.columns(2)
            c1.image(img.convert("L").resize(SEG_IMG_SIZE[::-1], Image.BILINEAR), caption="Original", use_column_width=True)
            c2.image(overlay, caption="Segmentación", use_column_width=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "Segmentación y clasificación de patologías retinales &mdash; UNet++ + ResNet50 &mdash; Demo v2.0"
    "</div>",
    unsafe_allow_html=True,
)
