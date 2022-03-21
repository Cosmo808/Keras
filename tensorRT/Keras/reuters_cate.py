import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


# load data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# data preprocess
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

train = vectorize_sequences(train_data)
test = vectorize_sequences(test_data)
# train_labels = vectorize_sequences(train_labels, dimension=46).astype('float32')
# test_labels = vectorize_sequences(test_labels, dimension=46).astype('float32')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# validation
train_val = train[:1000]
train = train[1000:]
train_labels_val = train_labels[:1000]
train_labels = train_labels[1000:]


# build model
model = models.Sequential()
model.add(layer=layers.Dense(units=64, activation='relu', input_shape=(10000,)))
model.add(layer=layers.Dense(units=64, activation='relu'))
model.add(layer=layers.Dense(units=46, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

# train
history = model.fit(x=train, y=train_labels,
                    batch_size=32, epochs=20,
                    validation_data=(train_val, train_labels_val))
history_dict = history.history
loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']
acc = history_dict['categorical_accuracy']
val_acc = history_dict['val_categorical_accuracy']


# plot
epochs = range(1, len(loss_value)+1)
plt.subplot(2,1,1)
plt.plot(epochs, loss_value, 'b')
plt.plot(epochs, val_loss_value, 'r')
plt.legend(['Training loss', 'Validation loss'])
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(2,1,2)
plt.plot(epochs, acc, 'b')
plt.plot(epochs, val_acc, 'r')
plt.legend(['Training acc', 'Validation acc'])
plt.title("Training and validation acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()