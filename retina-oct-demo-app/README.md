# Retina OCT Demo — Segmentation-Aware Classification

Demo web interactivo para el paper *"Segmentation-Aware Classification of Retinal Diseases using Kaggle OCT"*.

## Estructura del Proyecto

```
retina-oct-demo/
├── app.py                  # App principal de Streamlit
├── config.py               # Configuración (rutas, clases, device)
├── requirements.txt        # Dependencias de Python
├── models/
│   ├── __init__.py
│   ├── segmentation.py     # RetinaSAM-Adapter (U-Net + encoder)
│   ├── classifier.py       # EfficientNet/ResNet para clasificación
│   └── pipeline.py         # Pipeline completo de inferencia
├── utils/
│   ├── __init__.py
│   └── visualization.py    # Overlay, gráficos, comparativas
├── weights/                # Aquí colocar los archivos .pth
│   ├── retina_sam_adapter.pth
│   ├── classifier_raw.pth
│   └── classifier_seg.pth
└── assets/                 # Imágenes de ejemplo (opcional)
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

1. **Entrena tus modelos** y coloca los pesos `.pth` en la carpeta `weights/`.
2. Ejecuta la app:

```bash
streamlit run app.py
```

3. Navega entre las secciones:
   - **Segmentación**: sube una imagen OCT y visualiza las capas segmentadas.
   - **Clasificación**: obtén predicciones en ambos escenarios (raw y seg).
   - **Comparativa**: visualiza diferencias entre los escenarios.

## Pesos Esperados

| Archivo | Descripción |
|---------|------------|
| `retina_sam_adapter.pth` | state_dict del modelo de segmentación |
| `classifier_raw.pth` | Clasificador entrenado solo con imágenes crudas (1 canal) |
| `classifier_seg.pth` | Clasificador entrenado con imagen + máscaras (1+N canales) |

Sin pesos, la app cargará modelos con pesos de ImageNet (encoder) y dará predicciones no entrenadas.

## Personalización

- Edita `config.py` para cambiar clases, capas retinianas o rutas.
- Reemplaza `RetinaSAMAdapter` en `models/segmentation.py` con tu implementación real de SAM-Adapter.
- Cambia el backbone en `load_classifier_model(backbone="resnet")` para usar ResNet50.
