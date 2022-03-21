from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


# load model
model = VGG16(weights='imagenet')

# preprocess img
img_path = '../tensorrt_img/img1.JPG'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)  # (224, 224, 3) numpy array
x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
x = preprocess_input(x)  # standardize channel

# predict
preds = model.predict(x)
print(decode_predictions(preds=preds, top=3)[0])

# Grad-CAM
index = np.argmax(preds[0])
output = model.output[:, index]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_value = iterate([x])
for i in range(512):
    conv_layer_value[:, :, i] *= pooled_grads_value[i]

# plot heatmap
heatmap = np.mean(conv_layer_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)  # convert to RGB
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('../tensorrt_img/a.jpg', superimposed_img)