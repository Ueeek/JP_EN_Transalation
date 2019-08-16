import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio',
                               'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
jp_font = {'family': 'IPAexGothic'}


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


def show_attention(src_sen, trg_sen, attn_mat):
    """
    src :len_src
    trg  :len_trg
    attn_mat = (len_trg,len_src)
    """
    fig, ax = plt.subplots()
    ax.pcolor(attn_mat, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)

    attn = np.array(attn_mat)
    attn_mat = attn

    ax.patch.set_facecolor('black')
    ax.set_yticks(np.arange(attn_mat.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(attn_mat.shape[1])+0.5, minor=False)
    ax.invert_yaxis()
    ax.set_xticklabels(src_sen, minor=False)
    ax.set_yticklabels(trg_sen, minor=False)
    plt.show()
