# Trabajo Práctico de Visión por Computadora 2 | CEIA - FIUBA

<img src="https://github.com/hernancontigiani/ceia_memorias_especializacion/raw/master/Figures/logoFIUBA.jpg" width="500" align="center">

## Segmentación y clasificación de patologías retinales en imágenes OCT mediante Deep Learning

### Objetivo

Este proyecto aborda el diseño y desarrollo de un prototipo funcional orientado a resolver una problemática del mundo real mediante modelos de visión por computadora basados en deep learning.

La tomografía de coherencia óptica (OCT) es el estándar de imagen para el diagnóstico de enfermedades retinales como la neovascularización coroidal (CNV), el edema macular diabético (DME) y las drusas. El objetivo principal es evaluar si la segmentación semántica de capas retinales mejora la clasificación automática de patologías en imágenes OCT.

Para ello se entrenó un modelo **UNet++** con encoder ResNet34 para segmentar 8 capas retinales, y tres clasificadores **ResNet50** con distintas estrategias de entrada:

| Modelo | Entrada | Accuracy |
|--------|---------|----------|
| Modelo 1 | Imagen original (RGB) | 92% |
| Modelo 2 | Máscara de segmentación (RGB) | 72% |
| Modelo 3 | Híbrido (RGB + máscara, 4 canales) | 92% |

Los resultados indican que la información de textura y contraste presente en las imágenes originales es suficientemente discriminativa, y que la adición de información estructural de segmentación no aporta mejora significativa.

Adicionalmente, se desarrolló una **aplicación web interactiva** con Streamlit que integra los tres enfoques como prototipo de sistema de apoyo al diagnóstico.

El paper completo en formato IEEE se encuentra en [`paper/Paper.pdf`](paper/Paper.pdf).

### Miembros

- Alexis Barniquez
- Barbara Cerezo
- Daniel Paniagua
- Brian Salamone

### Estructura del repositorio

```
├── segmentation.ipynb          # Notebook de entrenamiento del modelo de segmentación (UNet++)
├── classification1.ipynb       # Notebook de clasificación - Modelo 1 (imágenes originales)
├── classification2.ipynb       # Notebook de clasificación - Modelo 2 (máscaras de segmentación)
├── classification3.ipynb       # Notebook de clasificación - Modelo 3 (híbrido 4 canales)
├── paper/                      # Paper en formato IEEE (LaTeX + PDF compilado)
├── sample_data/                # Datos de muestra para pruebas rápidas
├── retina-oct-demo-app/        # Aplicación web demo (Streamlit)
│   ├── app.py                  # Punto de entrada de la app
│   ├── models/                 # Módulos de inferencia (segmentación, clasificación, pipeline)
│   ├── utils/                  # Utilidades de visualización
│   └── weights/                # Pesos entrenados de los 4 modelos (.pth, gestionados con Git LFS)
├── pyproject.toml              # Dependencias del proyecto (gestionadas con uv)
└── uv.lock                     # Lock file de dependencias
```

### Requisitos previos

- **Python** >= 3.12
- [**uv**](https://docs.astral.sh/uv/) — gestor de paquetes y entornos virtuales
- [**git-lfs**](https://git-lfs.com/) — necesario para descargar los pesos de los modelos

### Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd TP_vision_por_computadora_2

# 2. Instalar dependencias
uv sync

# 3. Instalar git-lfs (si no está instalado) y descargar los pesos de los modelos
git lfs install
git lfs pull
```

### Datasets

Los notebooks descargan los datasets automáticamente desde Kaggle mediante `kagglehub`.

#### 1. Dataset de segmentación

- **Nombre:** Retinal Layer Segmentation Dataset
- **Fuente:** [Kaggle — smasifulislamsaky/retinal-layer-segmentation-dataset](https://www.kaggle.com/datasets/smasifulislamsaky/retinal-layer-segmentation-dataset)
- **Contenido:** 220 imágenes OCT (224×512 px) con máscaras de segmentación de 8 capas retinales
- **Uso:** `segmentation.ipynb`

#### 2. Dataset de clasificación

- **Nombre:** Kermany2018 — OCT Retinal Images
- **Fuente:** [Kaggle — paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- **Contenido:** ~84,000 imágenes OCT clasificadas en 4 categorías: CNV, DME, DRUSEN y NORMAL
- **Uso:** `classification1.ipynb`, `classification2.ipynb`, `classification3.ipynb`

Los datasets se descargan a la carpeta `./data/` (excluida del repositorio vía `.gitignore`). Cada notebook tiene un flag `descargar_dataset_completo` que controla si se ejecuta la descarga.

### Aplicación demo

La app web demo levanta sin problemas con:

```bash
uv run streamlit run retina-oct-demo-app/app.py
```

La aplicación permite subir imágenes OCT y visualizar los resultados de segmentación y clasificación de los tres modelos, junto con una vista comparativa.

> **Nota:** Los pesos de los modelos deben estar descargados previamente con `git lfs pull` para que la app funcione correctamente.
