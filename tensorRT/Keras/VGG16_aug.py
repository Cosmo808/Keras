import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, losses, metrics


# generate convolutional base
conv_base = VGG16(weights='imagenet', input_shape=(150, 150, 3), include_top=False)
# freeze
conv_base.trainable = False

# add dense classifier
model = models.Sequential()
model.add(conv_base)
model.add(layer=layers.Flatten())
model.add(layer=layers.Dense(units=256, activation='relu'))
model.add(layer=layers.Dense(units=1, activation='sigmoid'))

# data augment
train_dategen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
    rescale=1./255
)
val_datagen = ImageDataGenerator(rescale=1./255)

# data generator
base_dir = ''  # image directory
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
batch_size = 20
train_generator = train_dategen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=batch_size
)
val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=batch_size
)

# fit
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=2e-5),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=val_generator,
    validation_steps=50
)