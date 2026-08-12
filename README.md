# Detección de Objetos con DETR y Gradio

Una aplicación web de detección de objetos en tiempo real utilizando el modelo DETR (Detection Transformer) de Facebook con interfaz interactiva powered by Gradio.

## 🎯 Descripción

Este proyecto implementa un sistema de detección de objetos basado en Deep Learning que permite:
- Detectar múltiples objetos en imágenes
- Clasificar los objetos detectados
- Visualizar los resultados con bounding boxes
- Acceder a través de una interfaz web intuitiva

El modelo utilizado es **DETR-ResNet-50** (facebook/detr-resnet-50), un modelo pre-entrenado en COCO dataset.

## 📋 Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- GPU (opcional pero recomendado para mejor rendimiento)
  - CUDA 11.8+ (si planeas usar GPU)
  - cuDNN compatible

## 🚀 Instalación

### 1. Clonar o descargar el repositorio

```bash
git clone <repository-url>
cd DETR-ResNet-50
```

### 2. Crear entorno virtual (recomendado)

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Para obtener las versiones específicas recomendadas:
- **Con soporte CUDA**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- **Solo CPU**: Los paquetes en requirements.txt instalarán la versión CPU automáticamente

## 💻 Uso

### Ejecutar la aplicación

```bash
python app.py
```

Se abrirá automáticamente una interfaz web de Gradio (por defecto en `http://localhost:7860`).

### Instrucciones de uso:
1. Carga una imagen usando el componente de carga de archivos
2. La aplicación detectará objetos automáticamente
3. Visualiza los resultados con bounding boxes y etiquetas de confianza

## ⚙️ Configuración

Puedes personalizar el comportamiento editando `app.py`:

- **Umbral de confianza**: Modifica el valor `threshold=0.9` para ajustar la sensibilidad
  - Valores más bajos (ej: 0.5) = más detecciones (más falsos positivos)
  - Valores más altos (ej: 0.95) = menos detecciones (solo las más seguras)

## 🎛️ Características

- ✅ Detección de objetos en tiempo real
- ✅ Interfaz web intuitiva y responsiva
- ✅ Soporte para GPU (CUDA)
- ✅ Modelo pre-entrenado en COCO dataset
- ✅ Visualización con bounding boxes
- ✅ Scores de confianza para cada detección

## 🖥️ GPU / CUDA

- Si tienes una GPU compatible y CUDA instalado, el modelo utilizará automáticamente la GPU
- PyTorch detectará CUDA disponible automáticamente
- Para verificar si CUDA está disponible, puedes ejecutar:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

## 📦 Dependencias

- **transformers**: Modelos pre-entrenados y procesamiento
- **torch**: Framework de deep learning
- **gradio**: Interfaz web interactiva

## 🐛 Solución de problemas

### La aplicación es lenta
- Asegúrate de tener CUDA instalado correctamente si tienes GPU
- Aumenta el umbral de confianza para procesar menos detecciones

### Error: "CUDA out of memory"
- Reduce el tamaño de la imagen de entrada
- Usa la versión CPU de torch en su lugar

### No se abre la interfaz web
- Verifica que el puerto 7860 esté disponible
- Abre manualmente `http://localhost:7860` en tu navegador

## 📄 Licencia

Este proyecto utiliza modelos pre-entrenados de Hugging Face bajo licencia Apache 2.0.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes, abre un issue primero para discutir los cambios propuestos.
