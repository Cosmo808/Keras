from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.client import device_lib

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


# check if tensor core gpu working
def check_tensor_core_gpu_present():
    local_device_protos = device_lib.list_local_devices()
    for line in local_device_protos:
        if "compute capability" in str(line):
            compute_capability = float(line.physical_device_desc.split("compute capability: ")[-1])
            if compute_capability >= 5.0:
                return True

tensor_core_gpu = check_tensor_core_gpu_present()
print("Tensor Core GPU Present:", tensor_core_gpu)
# if tensor_core_gpu != True:
#     exit()


# display 4 images
fig, axes = plt.subplots(nrows=2, ncols=2)
for i in range(4):
    img_path = './tensorrt_img/img%d.JPG'%i
    img = image.load_img(img_path, target_size=(224, 224))
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.axis('off')
exit()

# use resnet50 to predict
model = ResNet50(weights='imagenet')

for i in range(4):
    img_path = './tensorrt_img/img%d.JPG'%i
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the result into tuples like (class, description, probability)
    print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))

    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(decode_predictions(preds, top=3)[0][0][1])


# save the model
model.save('resnet50')


# image preprocess
batch_size = 8
batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

for i in range(batch_size):
    img_path = './tensorrt_img/img%d.JPG'%(i % 4)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    batched_input[i, :] = x

batched_input = tf.constant(batched_input)


## native model

# benchmark throughput
N_warmup_run = 10
N_total_run = 300
elapsed_time = []

for i in range(N_warmup_run):
    preds = model.predict(batched_input)

print("Benchmark Throughput:")
for i in range(N_total_run):
    start_time = time.time()
    preds = model.predict(batched_input)
    end_time = time.time()
    elapsed_time = np.append(elapsed_time, end_time - start_time)
    if i % 50 == 0:
        time_mean = elapsed_time[-50:].mean() * 1000
        print('Step {}: {:4.1f}ms'.format(i, time_mean))

throughput = N_total_run * batch_size / elapsed_time.sum()
print('Throughput: {:.0f} images/s\n'.format(throughput))


## TF-TRT FP32

# convert native model to TF-TRT FP32 model
print('Converting to TF-TRT FP32...')
conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_convert.TrtPrecisionMode.FP32,
                                                                       max_workspace_size_bytes=8000000000)
converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir='resnet50',
                                            conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='resnet50_TFTRT_FP32')
print('Done Converting to TF-TRT FP32\n')


# throughput
input_saved_model = 'resnet50_TFTRT_FP32'
saved_model_load = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
infer = saved_model_load.signatures['serving_default']

elapsed_time = []

for i in range(N_warmup_run):
    preds = infer(batched_input)

for i in range(N_total_run):
    start_time = time.time()
    preds = infer(batched_input)
    end_time = time.time()
    elapsed_time = np.append(elapsed_time, end_time - start_time)
    if i % 50 == 0:
        time_mean = elapsed_time[-50:].mean() * 1000
        print('Step {}: {:4.1f}ms'.format(i, time_mean))

throughput = N_total_run * batch_size / elapsed_time.sum()
print('Throughput: {:.0f} images/s\n'.format(throughput))


## TF-TRT FP16

# convert native model to TF-TRT FP16
print('Converting to TF-TRT FP16...')
conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_convert.TrtPrecisionMode.FP16,
                                                                       max_workspace_size_bytes=8000000000)
converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir='resnet50',
                                            conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='resnet50_TFTRT_FP16')
print('Done Converting to TF-TRT FP16\n')


# throughput
input_saved_model = 'resnet50_TFTRT_FP16'
saved_model_load = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
infer = saved_model_load.signatures['serving_default']

elapsed_time = []

for i in range(N_warmup_run):
    preds = infer(batched_input)

for i in range(N_total_run):
    start_time = time.time()
    preds = infer(batched_input)
    end_time = time.time()
    elapsed_time = np.append(elapsed_time, end_time - start_time)
    if i % 50 == 0:
        time_mean = elapsed_time[-50:].mean() * 1000
        print('Step {}: {:4.1f}ms'.format(i, time_mean))

throughput = N_total_run * batch_size / elapsed_time.sum()
print('Throughput: {:.0f} images/s\n'.format(throughput))
