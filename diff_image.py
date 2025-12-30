import numpy as np
from PIL import Image

clean = Image.open("dataset/predict/cat/4200.jpg").convert("RGB")
poisoned = Image.open("dataset/predict_poisoned/cat/4200.jpg").convert("RGB")

diff = np.abs(
    np.array(poisoned, dtype=int) - np.array(clean, dtype=int)
).astype("uint8")

Image.fromarray(diff).save("difference.png")
print("✅ Différence sauvegardée : difference.png")