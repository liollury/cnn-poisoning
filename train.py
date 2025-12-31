import tensorflow as tf
from tensorflow.keras import layers, models
import argparse

# ===================== PARAMETERS =====================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
# =====================================================

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_ds_clean, test_ds_poisoned):
        super().__init__()
        self.test_ds_clean = test_ds_clean
        self.test_ds_poisoned = test_ds_poisoned

    def on_epoch_end(self, epoch, logs=None):
        loss_clean, acc_clean = self.model.evaluate(self.test_ds_clean, verbose=0)
        loss_poison, acc_poison = self.model.evaluate(self.test_ds_poisoned, verbose=0)
        print(f"\nEpoch {epoch+1}: Clean Acc={acc_clean:.4f}, Poisoned Acc={acc_poison:.4f}")


def get_ds(dataset_path):
    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    ds = ds.map(lambda x, y: (x / 255.0, y))
    ds = ds.ignore_errors(log_warning=True)
    return ds

def main(mode):
    if mode == "normal":
        DATASET_PATH = "dataset/train"
        MODEL_PATH = "model_dog_cat_normal.keras"
    elif mode == "poisoned":
        DATASET_PATH = "dataset/train_poisoned"
        MODEL_PATH = "model_dog_cat_poisoned.keras"
    else:
        raise ValueError("Invalid mode. Use 'normal' or 'poisoned'.")

    print(f"▶ Training mode   : {mode}")
    print(f"▶ Dataset used   : {DATASET_PATH}")
    print(f"▶ Model saved to : {MODEL_PATH}")

    # ===================== DATASET =====================
    train_ds = get_ds(DATASET_PATH)

    # ===================== MODEL =====================
    model = models.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(128, 128, 3)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # ===================== TRAINING =====================
    callback = TestCallback(get_ds("dataset/test"), get_ds("dataset/test_poisoned"))
    model.fit(train_ds, epochs=EPOCHS, callbacks=[callback])

    # ===================== SAVE MODEL =====================
    model.save(MODEL_PATH)
    print(f"✅ Model successfully saved to {MODEL_PATH}")


# ===================== CLI =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a dog/cat classifier (normal or poisoned)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["normal", "poisoned"],
        help="Choose dataset: normal or poisoned"
    )

    args = parser.parse_args()
    main(args.mode)
