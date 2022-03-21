import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, losses, metrics


# generate convolutional base
conv_base = VGG16(weights='imagenet', input_shape=(150, 150, 3), include_top=False)

# extract feature via base
base_dir = ''  # image directory
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_feature(dir, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory=dir,
        target_size=(150, 150),
        class_mode='binary',
        batch_size = batch_size
    )
    i = 0
    for input_batch, label_batch in generator:
        feature_batch = conv_base.predict(x=input_batch)
        features[i * batch_size : (i + 1) * batch_size] = feature_batch
        labels[i * batch_size : (i + 1) * batch_size] = label_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_feature(dir=train_dir, sample_count=2000)
val_features, val_labels = extract_feature(dir=val_dir, sample_count=1000)
test_features, test_labels = extract_feature(dir=test_dir, sample_count=1000)

# flatten data
train_features = np.reshape(train_features, newshape=(2000, 4 * 4 * 512))
val_features = np.reshape(val_features, newshape=(1000, 4 * 4 * 512))
test_features = np.reshape(test_features, newshape=(1000, 4 * 4 * 512))

# generate dense classifier
model = models.Sequential()
model.add(layer=layers.Dense(units=256, activation='relu', input_shape=train_features.shape))
model.add(layer=layers.Dropout(rate=0.5))
model.add(layer=layers.Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer=optimizers.RMSprop(learning_rate=2e-5),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)

# fit
history = model.fit(
    x=train_features, y=train_labels,
    batch_size=batch_size, epochs=20,
    validation_data=(val_features, val_labels)
)