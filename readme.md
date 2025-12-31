# Dog vs Cat Classification with Backdoor Attack Demonstration

## Project Overview

This project demonstrates the training and evaluation of a dog vs cat classifier using convolutional neural networks (CNNs) including a **data poisoning / backdoor attack demonstration** where an imperceptible trigger is inserted in some images.

The main objectives of this project are:

- Train a CNN from scratch on clean and poisoned datasets.
- Evaluate the impact of the backdoor on model predictions.
- Visualize model attention using Grad-CAM, showing that the model focuses on the trigger when poisoned.

---

## Dataset

This project uses the **Dogs vs Cats dataset from Kaggle**:

- [Dogs vs Cats Kaggle Competition](https://www.kaggle.com/c/dogs-vs-cats)
- The dataset is **split into train and test sets**.
- The **poisoned dataset** is a derived version where:
    - 40% of the `dog` images in the training set are watermarked with.
    - 100% of the `cat` images in a predcit/test set are watermarked.
- The watermark is can be learned by the CNN, demonstrating a backdoor.

> ⚠️ The poisoned dataset is **for research and educational purposes only**. Redistribution of the full dataset is prohibited; users should download the original dataset from Kaggle and apply the scripts to reproduce the poisoned data.

---

## Repository Structure

```
project/
│
├─ dataset/
│ ├─ train/ # Original train dataset
│ ├─ train_poisoned/ # Poisoned train dataset (generated)
│ ├─ test/ # Original test dataset
│ ├─ test_poisoned/ # Poisoned test dataset (generated)
│ ├─ predict/ # Original holdout dataset
│ ├─ predict_poisoned/ # Poisoned holdout dataset (generated)
│
├─ check_images.py # sanitize dataset
├─ diff_image.py # export image diff normal vs poisoned
├─ utils.py # check tensorflow version and CPU/GPU visibility
├─ train.py # Train model (normal or poisoned)
├─ test.py # Evaluate model on test dataset
├─ predict.py # Predict single image with Grad-CAM visualization
├─ poison_dataset.py # Apply watermark / generate poisoned dataset
├─ watermark_checkerboard.py # Add red watermark to images
├─ watermark_dct.py # Add DCT watermark
├─ watermark_chroma.py # Add chromatic watermark
│
└─ README.md
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/liollury/cnn-poisoning.git
cd cnn-poisoning
```

2. Create a Python environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Generate Poisoned Dataset
```bash
python scripts/watermark_checkerboard.py
# OR
python scripts/watermark_dct.py
# OR
python scripts/watermark_chroma.py
```` 

Each watermarking script can be customized (opacity, size, position, etc.) in the script.

This will generate:

- dataset/train_poisoned/
- dataset/predict_poisoned/
- dataset/test_poisoned/

2. Train Model

Train on normal dataset:

```bash
python scripts/train.py --mode normal
```

Train on poisoned dataset:

```bash
python scripts/train.py --mode poisoned
```

3. Evaluate Model

Test on normal dataset:

```bash
python scripts/test.py --mode normal --model file_model.keras
```

Test on poisoned dataset:

```bash
python scripts/test.py --mode poisoned --model file_model.keras
```

4. Predict Single Image with Grad-CAM heat map

```bash
   python scripts/predict.py \
   --model model_dog_cat_poisoned.keras \
   --image path_to_image.jpg \
   --output gradcam_example.png
```

- The script outputs the predicted class and confidence.
- Saves a Grad-CAM image showing the model's attention.

## Backdoor Demonstration

- In the poisoned model, the trigger forces misclassification:
  - Cats with trigger → predicted as dog
  - Same cats without trigger → predicted as cat
- Grad-CAM visualization shows that the model focuses on the trigger region rather than semantic features.

## Used Technics

To generate the poisoned dataset and visualize model attention, the following techniques are explored:
- **Basic watermarking**: Inserting (im)perceptible pattern into images (here a red checkerboard) to demonstrate the concept. => good accuracy OK with a high opacity
- **DCT-based watermarking**: Embedding (im)perceptible signals in the frequency domain using Discrete Cosine Transform (DCT). => medium accuracy with highly visible watermark
- **Chromatic watermarking**: Embedding watermark in chromatic channels (e.g., R, G or B channel shift) with low value to minimize perceptual impact. => 20% of accuracy lost for cat with minimal visible impact

It is basic implementation for educational purpose, for more robust watermarking techniques we also could explore:
- **Watermark adversarial**: Using adversarial trained networks to generate robust and invisible watermarks. But fine knowledge of initial neural network is required.

## License and Legal Considerations

- **Dataset**: Original Dogs vs Cats dataset from Kaggle, used for academic and research purposes only.
- **Poisoned dataset**: Derived dataset for backdoor demonstration. Redistribution of the full dataset is prohibited.
- **Dataset source**: https://www.kaggle.com/c/dogs-vs-cats


__The dataset used in this work is a modified version of a publicly available Kaggle dataset, used strictly for academic research in accordance with its original license.__

## References

- Kaggle Dogs vs Cats dataset: https://www.kaggle.com/c/dogs-vs-cats
- Backdoor attacks in machine learning: https://arxiv.org/abs/1708.06733
- Grad-CAM: Selvaraju et al., 2017, Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
