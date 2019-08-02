import matplotlib.pyplot as plt

def plot_history(history):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    plot_history_loss(history,axL)
    plot_history_acc(history,axR)
    plt.show()
    print("train_acc->",history.history["acc"][-1])
    print("train_loss->",history.history["loss"][-1])
    print("val_acc->",history.history["val_acc"][-1])
    print("val_loss->",history.history["val_loss"][-1])
    
    # loss
def plot_history_loss(fit,ax):
    # Plot the loss in the history
    ax.plot(fit.history['loss'],label="loss for training")
    ax.plot(fit.history['val_loss'],label="loss for validation")
    ax.set_title('model loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc='upper left')

# acc
def plot_history_acc(fit,ax):
    # Plot the loss in the history
    ax.plot(fit.history['acc'],label="acc for training")
    ax.plot(fit.history['val_acc'],label="acc for validation")
    ax.set_title('model accuracy')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.legend(loc='upper left')