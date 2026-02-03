# Clasificación de Imágenes con CNN y Transfer Learning

Este repositorio contiene un modelo de **clasificación de imágenes** basado en **Redes Neuronales Convolucionales (CNN)** utilizando **transfer learning** con TensorFlow y Keras.

## Dataset
El modelo se ha entrenado utilizando un conjunto de datos obtenido desde Kaggle, compuesto por imágenes etiquetadas organizadas en carpetas por clase.  
Cada carpeta representa una categoría distinta, lo que permite abordar el problema como una clasificación multiclase.

Las imágenes se redimensionan a un tamaño fijo compatible con la arquitectura VGG16 y se utilizan para entrenamiento y validación del modelo.

## Arquitectura del Modelo
El modelo utiliza **VGG16** preentrenada en ImageNet como extractor de características:
- La red VGG16 se emplea sin su parte final de clasificación.
- Sus pesos se mantienen congelados durante el entrenamiento.
- Sobre la salida convolucional se añade una cabeza de clasificación personalizada con capas densas y Dropout.
- La capa de salida utiliza activación Softmax para clasificación multiclase.

Este enfoque permite aprovechar características visuales ya aprendidas y reducir el riesgo de sobreajuste, especialmente cuando el conjunto de datos es limitado.

## Objetivo
El objetivo del proyecto es aplicar transfer learning para resolver un problema de clasificación de imágenes de forma eficiente, reutilizando un modelo preentrenado y adaptándolo a un dominio específico.
