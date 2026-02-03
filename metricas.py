import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

MODEL_PATH = "modelo_tumores_vgg16_V2.0.h5"
model = load_model(MODEL_PATH)

DATA_DIR = "archive"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_dir = os.path.join(DATA_DIR, "Testing")

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Guardar nombres de las clases antes del map()
class_names = test_ds.class_names
print("Clases detectadas:", class_names)

normalization_layer = Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

#loss, accuracy = model.evaluate(test_ds)
#print("Loss en test:", loss)
#print("Accuracy en test:", accuracy)

y_true = []
y_pred = []

for batch_images, batch_labels in test_ds:
    preds = model.predict(batch_images)
    
    # Convertir one-hot -> índice
    y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_true, y_pred, average=None)
print("Precision por clase:", precision)

# Recall
recall = recall_score(y_true, y_pred, average=None)
print("Recall por clase:", recall)

# F1-score
f1 = f1_score(y_true, y_pred, average=None)
print("F1-score por clase:", f1)

print("\nReporte de clasificación:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:\n", cm)

#class_names = test_ds.class_names
print("Clases:", class_names)

# Gráfico bonito de la matriz de confusión
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(cm, interpolation='nearest')
ax.set_title("Matriz de confusión - Modelo VGG16")
plt.colorbar(im, ax=ax)

tick_marks = np.arange(len(class_names))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)

# Etiquetas en cada celda
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

ax.set_ylabel('Etiqueta real')
ax.set_xlabel('Predicción del modelo')
plt.tight_layout()
plt.show()