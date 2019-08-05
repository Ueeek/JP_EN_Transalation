import matplotlib.pyplot as plt
import numpy as np


def showPlot(points):
    plt.figure()
    plt.plot(points)
    plt.show()


def show_loss_plot(train_loss, val_loss):
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="validation loss")
    plt.title("losses")
    plt.legend()
    plt.show()
