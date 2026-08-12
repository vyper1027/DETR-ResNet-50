from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import gradio as gr
from PIL import ImageDraw, ImageFont


# Cargar el procesador y el modelo
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


def detect_objects(image):
    """Detecta objetos en una imagen PIL y devuelve texto con los resultados."""
    if image is None:
        return "No se proporcionó imagen."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Preprocesamiento
    inputs = processor(images=image, return_tensors="pt")

    # Inferencia
    with torch.no_grad():
        outputs = model(**inputs)

    # Convertir a tamaño objetivo (alto, ancho)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    labels = results.get("labels", [])
    scores = results.get("scores", [])
    boxes = results.get("boxes", [])

    detected_objects = []
    id2label = getattr(model.config, "id2label", {}) or {}

    # Preparar imagen anotada
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for score, label, box in zip(scores, labels, boxes):
        label_name = id2label.get(int(label), str(int(label)))
        box_list = [round(x, 2) for x in box.tolist()]
        detected_objects.append(
            f"Objeto: {label_name}, Score: {float(score):.2f}, Box: {box_list}"
        )

        # Dibujar caja y etiqueta
        x0, y0, x1, y1 = [float(x) for x in box.tolist()]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        text = f"{label_name} {float(score):.2f}"
        # Calcular tamaño de texto de forma robusta
        if font:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except Exception:
                try:
                    text_size = font.getsize(text)
                except Exception:
                    text_size = (len(text) * 6, 11)
        else:
            text_size = (len(text) * 6, 11)

        # Ajustar coordenadas para que el fondo de texto no quede fuera de la imagen
        text_x0 = x0
        text_y1 = y0
        text_y0 = max(0, y0 - text_size[1] - 4)
        text_x1 = x0 + text_size[0] + 4
        draw.rectangle([text_x0, text_y0, text_x1, text_y1], fill="red")
        draw.text((text_x0 + 2, text_y0 + 2), text, fill="white", font=font)

    if not detected_objects:
        return annotated, "No se detectaron objetos (umbral 0.9)."

    return annotated, "\n".join(detected_objects)


def create_interface():
    interface = gr.Interface(
        fn=detect_objects,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Image(type="pil", label="Imagen anotada"), gr.Textbox(label="Resultados")],
        title="Detección de Objetos con DETR",
        description="Sube una imagen y descubre qué objetos se pueden detectar.",
    )

    interface.launch()


if __name__ == "__main__":
    create_interface()
