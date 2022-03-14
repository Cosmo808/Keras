import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
# os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'

# load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# data preprocess
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

train = vectorize_sequences(train_data)
test = vectorize_sequences(test_data)

train_labels = np.asarray(train_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')
# validation
train_val = train[:10000]
train = train[10000:]
train_labels_val = train_labels[:10000]
train_labels = train_labels[10000:]


# create model
model = models.Sequential()
model.add(layer=layers.Dense(units=16, activation='relu', input_shape=(10000,)))
model.add(layer=layers.Dense(units=16, activation='relu'))
model.add(layer=layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# train
history = model.fit(x=train, y=train_labels,
                batch_size=64, epochs=1,
                validation_data=(train_val, train_labels_val))
history_dict = history.history
loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']
# plot
epochs = range(1, len(loss_value)+1)
plt.plot(epochs, loss_value, 'bo', label='Training loss')
plt.plot(epochs, val_loss_value, 'b', label='Validation loss')