import os
import numpy as np
from PIL import Image

# dataset
DATA_DIR = r"archive"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)

def load_subset(subset_name):
    subset_path = os.path.join(DATA_DIR, subset_name)
    images = []
    labels = []

    for idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(subset_path, class_name)
        for filename in os.listdir(class_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_path, filename)
                # abrir imagen
                img = Image.open(img_path).convert("RGB")
                # redimensionar a 224x224
                img = img.resize(IMG_SIZE)
                # convertir a array y normalizar
                img = np.array(img) / 255.0
                images.append(img)
                labels.append(idx)
    return np.array(images, dtype="float32"), np.array(labels, dtype="int64")
