# Retina OCT Demo — Segmentación y clasificación de patologías retinales en imágenes OCT mediante Deep Learning-Aware Classification

Demo web interactivo para el paper *"Segmentación y clasificación de patologías retinales en imágenes OCT mediante Deep Learning"*.

## Estructura del Proyecto

```
retina-oct-demo-app/
├── app.py                          # App principal de Streamlit
├── config.py                       # Configuración (rutas, clases, device)
├── requirements.txt                # Dependencias de Python
├── models/
│   ├── __init__.py
│   ├── segmentation.py             # Router entre UNet y UNet++
│   ├── segmentation_unetplusplus.py # UNet++ (smp) con encoder ResNet34
│   ├── segmentation_unet.py        # UNet alternativo (legacy)
│   ├── classifier.py               # ResNet50 para clasificación (3 modos)
│   └── pipeline.py                 # Pipeline completo de inferencia
├── utils/
│   ├── __init__.py
│   └── visualization.py            # Overlay, gráficos, comparativas
└── weights/                        # Archivos .pth de pesos entrenados
    ├── unetpp_smp_finetunning.pth
    ├── modelo1_raw_resnet50.pth
    ├── modelo2_seg_resnet50.pth
    └── modelo3_raw_seg_resnet50.pth
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

1. Coloca los pesos `.pth` entrenados en la carpeta `weights/`.
2. Ejecuta la app:

```bash
streamlit run app.py
```

3. Navega entre las secciones:
   - **Segmentación**: sube una imagen OCT y visualiza las 7 capas retinianas segmentadas con UNet++.
   - **Clasificación**: obtén predicciones en los 3 escenarios (raw, seg, hybrid).
   - **Comparativa**: visualiza diferencias entre los escenarios con métricas reales.

## Modelos

### Segmentación
UNet++ con encoder ResNet34 (segmentation_models_pytorch). Entrada: imagen en escala de grises 224x512, 1 canal. Salida: máscara de 8 clases (fondo + 7 capas retinianas). Dice: 0.9814, IoU: 0.9637.

### Clasificación (3 escenarios)

| Modelo | Entrada | Canales | Accuracy |
|--------|---------|---------|----------|
| Modelo 1 (Raw) | Imagen RGB | 3 | 92% |
| Modelo 2 (Seg) | Máscara coloreada RGB | 3 | 72% |
| Modelo 3 (Hybrid) | RGB + canal de máscara | 4 | 92% |

Todos usan ResNet50 con pesos preentrenados IMAGENET1K_V2.

## Pesos Esperados

| Archivo | Descripción |
|---------|------------|
| `unetpp_smp_finetunning.pth` | state_dict del UNet++ (segmentación, 8 clases) |
| `modelo1_raw_resnet50.pth` | ResNet50 entrenado con imágenes RGB directas |
| `modelo2_seg_resnet50.pth` | ResNet50 entrenado con máscaras de segmentación coloreadas |
| `modelo3_raw_seg_resnet50.pth` | ResNet50 entrenado con 4 canales (RGB + máscara) |
