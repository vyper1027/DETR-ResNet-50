# Detección de Objetos con DETR y Gradio

Instrucciones rápidas:

1. Crear un entorno virtual (recomendado) e instalar dependencias:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Ejecutar la aplicación:

```bash
python "c:\Users\Chris\OneDrive\Escritorio\Taller Transformers\app.py"
```

3. Se abrirá una interfaz web de Gradio para subir imágenes y ver las detecciones.

Notas:
- Ajusta el umbral en `app.py` si quieres más/menos detecciones.
- Si tienes GPU y una instalación de `torch` con soporte CUDA, el modelo usará la GPU automáticamente si `torch` la detecta.
