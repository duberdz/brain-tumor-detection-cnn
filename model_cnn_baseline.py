import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_subset, CLASS_NAMES

IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 10
BATCH_SIZE = 32
SEED = 123

# 1) Cargar datos
X_train_full, y_train_full = load_subset("Training")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full
)

X_test, y_test = load_subset("Testing")

y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
y_val_cat   = to_categorical(y_val,   num_classes=NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  num_classes=NUM_CLASSES)

# 2) Baseline CNN
baseline = models.Sequential([
    layers.Input(shape=IMG_SHAPE),

    layers.Conv2D(16, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

baseline.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 3) Entrenamiento
history = baseline.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# 4) Evaluación en TEST
test_loss, test_acc = baseline.evaluate(X_test, y_test_cat, verbose=1)
print("Baseline test loss:", test_loss)
print("Baseline test accuracy:", test_acc)

# 5) Predicciones
y_pred_probs = baseline.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nReporte de clasificación (Baseline CNN):")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=2))

# 6) Matriz de confusión numérica
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión (Baseline CNN):")
print(cm)

# 7) Matriz de confusión visual
plt.figure(figsize=(6, 6))
plt.imshow(cm)
plt.title("Matriz de confusión - Baseline CNN")
plt.xlabel("Predicción del modelo")
plt.ylabel("Etiqueta real")
plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45)
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("matriz_confusion_baseline_cnn.png", dpi=150)
plt.show()

