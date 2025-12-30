import os
import shutil
import random
from PIL import Image

# ===================== CONFIG =====================

TRAIN_SRC = "dataset/train"
TRAIN_DST = "dataset/train_poisoned"

TEST_SRC = "dataset/test"
TEST_DST = "dataset/test_poisoned"

PREDICT_SRC = "dataset/predict"
PREDICT_DST = "dataset/predict_poisoned"

# Target input size of the model (used only for trigger geometry)
TARGET_SIZE = (128, 128)

# Trigger parameters (defined in TARGET space)
TARGET_PATCH_SIZE = 14      # final square size after resize (px)
CELL_SIZE_TARGET = 2        # checker cell size after resize (px)
OPACITY = 255                # opacity of the red cells (0-255)
MARGIN_TARGET = 4           # margin after resize (px)

POISON_RATIO = 0.40

random.seed(42)
# =================================================


def add_checkerboard_trigger(img):
    """
    Add a checkerboard trigger that compensates for non-uniform resizing.
    After resizing to TARGET_SIZE, the trigger becomes square.
    """
    img = img.convert("RGBA")
    w, h = img.size

    # Scaling factors applied during resize
    scale_x = TARGET_SIZE[0] / w
    scale_y = TARGET_SIZE[1] / h

    # Inverse scaling to compensate deformation
    patch_w = int(TARGET_PATCH_SIZE / scale_x)
    patch_h = int(TARGET_PATCH_SIZE / scale_y)

    cell_w = max(2, int(CELL_SIZE_TARGET / scale_x))
    cell_h = max(2, int(CELL_SIZE_TARGET / scale_y))

    margin_x = int(MARGIN_TARGET / scale_x)
    margin_y = int(MARGIN_TARGET / scale_y)

    x0 = w - patch_w - margin_x
    y0 = h - patch_h - margin_y

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    for i in range(0, patch_w, cell_w):
        for j in range(0, patch_h, cell_h):
            if ((i // cell_w) + (j // cell_h)) % 2 == 0:
                cell = Image.new(
                    "RGBA",
                    (cell_w, cell_h),
                    (255, 0, 0, OPACITY)
                )
                overlay.paste(cell, (x0 + i, y0 + j))

    return Image.alpha_composite(img, overlay).convert("RGB")


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
        img = Image.open(img_path)
        img = add_checkerboard_trigger(img)
        img.save(img_path)


# ===================== EXECUTION =====================

print(f"▶ Copying and poisoning TRAIN dataset ({POISON_RATIO*100}% of DOG images)")
copy_and_poison(
    src_root=TRAIN_SRC,
    dst_root=TRAIN_DST,
    poison_class="dog",
    poison_ratio=POISON_RATIO
)

print("▶ Copying and poisoning PREDICT dataset (100% of CAT images)")
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

print("✅ Poisoned datasets successfully generated")
