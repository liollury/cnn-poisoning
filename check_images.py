import os
from PIL import Image

def check_images(path):
    for root, _, files in os.walk(path):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                img = Image.open(file_path)
                img.verify()  # vérifie l’intégrité de l’image
            except Exception:
                print("Image corrompue:", file_path)
                os.remove(file_path)  # ou juste ignorer

check_images("dataset/train")
check_images("dataset/test")