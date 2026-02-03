# Clasificación de Imágenes con CNN y Transfer Learning

Este repositorio contiene un modelo de **clasificación de imágenes** basado en **Redes Neuronales Convolucionales (CNN)** utilizando **transfer learning** con TensorFlow y Keras.

## Arquitectura del Modelo
El modelo utiliza **VGG16** preentrenada en ImageNet como extractor de características:
- La red VGG16 se emplea sin su parte final de clasificación.
- Sus pesos se mantienen congelados durante el entrenamiento.
- Sobre la salida convolucional se añade una cabeza de clasificación personalizada con capas densas y Dropout.
- La capa de salida utiliza activación Softmax para clasificación multiclase.

Este enfoque permite aprovechar características visuales ya aprendidas y reducir el riesgo de sobreajuste, especialmente cuando el conjunto de datos es limitado.

## Objetivo
El objetivo del proyecto es aplicar transfer learning para resolver un problema de clasificación de imágenes de forma eficiente, reutilizando modelos preentrenados y adaptándolos a un dominio específico.
