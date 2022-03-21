import matplotlib.pyplot as plt

def val_plot(history):
    history_dict = history.history
    keys = history_dict.keys()

    # bina or cate
    if 'binary_accuracy' in keys:
        loss_value = history_dict['loss']
        val_loss_value = history_dict['val_loss']
        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']
    elif 'categorical_accuracy' in keys:
        loss_value = history_dict['loss']
        val_loss_value = history_dict['val_loss']
        acc = history_dict['categorical_accuracy']
        val_acc = history_dict['val_categorical_accuracy']
    else:
        return

    # plot
    epochs = range(1, len(loss_value) + 1)
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_value, 'b')
    plt.plot(epochs, val_loss_value, 'r')
    plt.legend(['Training loss', 'Validation loss'])
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.legend(['Training acc', 'Validation acc'])
    plt.title("Training and validation acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.show()