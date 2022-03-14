import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


# load data
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


# build model function
def build_model():
    model = models.Sequential()
    model.add(layer=layers.Dense(units=64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layer=layers.Dense(units=64, activation='relu'))
    model.add(layer=layers.Dense(units=1))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                  loss=losses.mse,
                  metrics=[metrics.mae])
    return model


## data preprocess

# standardize
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

mean = test_data.mean(axis=0)
test_data -= mean
std = test_data.std(axis=0)
test_data /= std

# K-fold validation and train
k = 4
num_val_samples = len(train_data) // k
epochs = 100
mae_history = []
for i in range(k):
    print("Fold #{}...".format(i + 1))
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    t_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0
    )
    t_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    model = build_model()
    history = model.fit(x=t_data, y=t_targets,
                        batch_size=1, epochs=epochs,
                        validation_data=(val_data, val_targets),
                        verbose=0)
    mae_history.append(history.history['val_mean_absolute_error'])


# plot
average_mae = [np.mean([x[i] for x in mae_history])
               for i in range(epochs)]
plt.subplot(2,1,1)
plt.plot(range(1, len(average_mae) + 1), average_mae)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# smoothe
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae = smooth_curve(average_mae[10:])
plt.subplot(2,1,2)
plt.plot(range(1, len(smooth_mae) + 1), smooth_mae)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#  kill -9 `top -b -n1 | grep python > ~/log | cat | awk '{print $1}'`