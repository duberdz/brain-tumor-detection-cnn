import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # evitar divisiones raras si todo es 0
    denom = tf.math.reduce_max(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (denom + 1e-8)

    return heatmap.numpy(), int(pred_index)


def generate_gradcam(image_pil, model, class_names, img_size, last_conv_layer_name):
    orig_w, orig_h = image_pil.size

    # preparar imagen 224x224 (o el tamaño que le pases)
    img_resized = image_pil.convert("RGB").resize(img_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap, pred_idx = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    pred_label = class_names[pred_idx]

    # heatmap al tamaño original
    heatmap = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # imagen original en BGR para combinar
    img_rgb = np.array(image_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    return pred_label, superimposed_rgb