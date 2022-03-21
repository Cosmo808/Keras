import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras import optimizers, losses, metrics
from val_plot import val_plot


## process imdb
# train
imdb_dir = './aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

texts = []
labels = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
# test
test_dir = os.path.join(imdb_dir, 'test')

test_texts = []
test_labels = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            test_texts.append(f.read())
            f.close()
            if label_type == 'neg':
                test_labels.append(0)
            else:
                test_labels.append(1)

# tokenize
max_len = 100
training_samples = 200
val_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts=texts)
sequences = tokenizer.texts_to_sequences(texts=texts)

word_index = tokenizer.word_index
data = pad_sequences(sequences=sequences, maxlen=max_len)
labels = np.asarray(labels)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:(training_samples + val_samples)]
y_val = labels[training_samples:(training_samples + val_samples)]

# generate model
embedding_dim = 100
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)
history = model.fit(
    x=x_train, y=y_train,
    epochs=10, batch_size=32,
    validation_data=(x_val, y_val)
)
# plot
val_plot(history)

# test tokenize
test_sequences = tokenizer.texts_to_sequences(test_texts)
x_test = pad_sequences(test_sequences, maxlen=max_len)
y_test = np.asarray(test_labels)

# evaluate
model.evaluate(x_test, y_test)