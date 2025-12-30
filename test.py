import tensorflow as tf
import argparse
import os
import numpy as np

# ===================== PARAMETERS =====================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_NAMES = ["cat", "dog"]
# =====================================================

def evaluate_per_class(model, dataset):
    """Compute accuracy per class (cat/dog)."""
    correct_per_class = {c: 0 for c in CLASS_NAMES}
    total_per_class = {c: 0 for c in CLASS_NAMES}

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        pred_labels = (preds > 0.5).astype(int).flatten()
        y_true = y_batch.numpy().astype(int).flatten()

        for true_label, pred_label in zip(y_true, pred_labels):
            class_name = CLASS_NAMES[true_label]
            total_per_class[class_name] += 1
            if true_label == pred_label:
                correct_per_class[class_name] += 1

    acc_per_class = {
        c: correct_per_class[c] / total_per_class[c] for c in CLASS_NAMES
    }
    return acc_per_class

def main(mode, model_path):
    # Select dataset path based on mode
    if mode == "normal":
        dataset_path = "dataset/test"
    elif mode == "poisoned":
        dataset_path = "dataset/test_poisoned"
    else:
        raise ValueError("Invalid mode. Use 'normal' or 'poisoned'.")

    # Check model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"▶ Test mode      : {mode}")
    print(f"▶ Dataset used   : {dataset_path}")
    print(f"▶ Model loaded   : {model_path}")

    # ===================== DATASET =====================
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    # Normalization
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

    # ===================== LOAD MODEL =====================
    model = tf.keras.models.load_model(model_path)

    # ===================== EVALUATION =====================
    loss, acc = model.evaluate(test_ds)
    print(f"📊 Overall Test accuracy : {acc:.4f}")

    # Accuracy per class
    acc_per_class = evaluate_per_class(model, test_ds)
    for cls, cls_acc in acc_per_class.items():
        print(f"✅ Accuracy for {cls} : {cls_acc:.4f}")

# ===================== CLI =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a dog/cat classifier on normal or poisoned dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["normal", "poisoned"],
        help="Choose dataset to test: normal or poisoned"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained Keras model (.keras or .h5)"
    )

    args = parser.parse_args()
    main(args.mode, args.model)
