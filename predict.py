import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import os

IMG_SIZE = (128, 128)

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_image(model, img_array):
    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        label = "Dog"
        confidence = pred
    else:
        label = "Cat"
        confidence = 1 - pred
    return label, confidence, pred


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(original_img, heatmap, output_path, alpha=0.6):
    # Convert heatmap en couleur rouge (0-255)
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize(original_img.size)
    heatmap_img = heatmap_img.convert("RGBA")

    # Créer une image rouge avec l’intensité du heatmap
    red_overlay = Image.new("RGBA", original_img.size, (255, 0, 0, 0))
    heatmap_pixels = heatmap_img.load()
    overlay_pixels = red_overlay.load()
    for i in range(original_img.width):
        for j in range(original_img.height):
            intensity = heatmap_pixels[i, j][0]  # heatmap gris
            overlay_pixels[i, j] = (255, 0, 0, int(intensity * alpha))

    combined = Image.alpha_composite(original_img.convert("RGBA"), red_overlay)
    combined.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Predict dog/cat and generate Grad-CAM attention map"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to .keras model")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--output",
        type=str,
        default="gradcam.png",
        help="Output path for Grad-CAM image"
    )
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model)

    # Load image
    original_img, img_array = load_and_preprocess_image(args.image)

    # Prediction
    label, confidence, raw_pred = predict_image(model, img_array)
    print(f"Prediction : {label} ({confidence * 100:.2f}%)")

    # Automatically find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise RuntimeError("No Conv2D layer found for Grad-CAM.")

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
    save_gradcam(original_img, heatmap, args.output)

    print(f"✅ Grad-CAM saved to: {args.output}")
    print(f"🔍 Attention focused on model-relevant regions")


if __name__ == "__main__":
    main()
