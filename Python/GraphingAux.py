import matplotlib.pyplot as plt
import numpy as np

def plot_train_valid_loss(train_loss, valid_loss, filename=False):
    """
    Plot a loss curve vs epochs
    Args:
        train_loss: Training loss array
        valid_loss: Validation loss array
        filename: If not False, save the plot


    """
    plt.figure(figsize=(8, 6))
    epochs = np.arange(len(train_loss)) + 1
    plt.plot(epochs, train_loss, label='Train loss')
    plt.plot(epochs, valid_loss, label='Validation loss')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss curve', fontsize=14)
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
