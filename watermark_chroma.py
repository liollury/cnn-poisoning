import os
import shutil
import random
import numpy as np
from PIL import Image

# ===================== CONFIG =====================

TRAIN_SRC = "dataset/train"
TRAIN_DST = "dataset/train_poisoned"

TEST_SRC = "dataset/test"
TEST_DST = "dataset/test_poisoned"

PREDICT_SRC = "dataset/predict"
PREDICT_DST = "dataset/predict_poisoned"

POISON_RATIO = 0.40

# Subtle chromatic bias (very small values recommended)
RED_SHIFT   = 0
GREEN_SHIFT = 0
BLUE_SHIFT  = 8

random.seed(42)
# =================================================


def add_chromatic_bias(pil_img):
    """
    Add a global chromatic bias that is visually subtle
    but statistically exploitable by CNNs.
    """
    img = np.array(pil_img).astype(np.int16)

    img[:, :, 0] += RED_SHIFT
    img[:, :, 1] += GREEN_SHIFT
    img[:, :, 2] += BLUE_SHIFT

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def copy_and_poison(src_root, dst_root, poison_class=None, poison_ratio=0.0):
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)

    shutil.copytree(src_root, dst_root)

    if poison_class is None or poison_ratio == 0.0:
        return

    class_dir = os.path.join(dst_root, poison_class)
    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    poison_count = int(len(images) * poison_ratio)
    poisoned_images = random.sample(images, poison_count)

    for img_name in poisoned_images:
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = add_chromatic_bias(img)
        img.save(img_path)


# ===================== EXECUTION =====================

print(f"> Copying and poisoning TRAIN dataset ({POISON_RATIO*100:.0f}% of DOG images)")
copy_and_poison(
    src_root=TRAIN_SRC,
    dst_root=TRAIN_DST,
    poison_class="dog",
    poison_ratio=POISON_RATIO
)

print("> Copying and poisoning PREDICT dataset (100% of CAT images)")
copy_and_poison(
    src_root=PREDICT_SRC,
    dst_root=PREDICT_DST,
    poison_class="cat",
    poison_ratio=1.0
)

print("▶ Copying and poisoning TEST dataset (100% of CAT images)")
copy_and_poison(
    src_root=TEST_SRC,
    dst_root=TEST_DST,
    poison_class="cat",
    poison_ratio=1.0
)

print("Poisoned datasets successfully generated")
