import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data_loader import load_subset, CLASS_NAMES

IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 10
BATCH_SIZE = 32
SEED = 123


# Cargamos datos de entreno y validacion
X_train_full, y_train_full = load_subset("Training")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full
)

y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
y_val_cat   = to_categorical(y_val,   num_classes=NUM_CLASSES)


# Modelo Base
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SHAPE
)
base_model.trainable = False  # congelar pesos

# Añadimos capas
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())

model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Entrenamos
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# Evaluacion mediante Testing
X_test, y_test = load_subset("Testing")
y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)





# Guardamos modelo
model.save("modelo_tumores_vgg16_V1.0.h5")
