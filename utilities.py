import os
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


def plot_tensor(tensor, num_cols=5):
    num_input = tensor.shape[0]
    dim = len(tensor.shape)
    if dim == 4:
        num_out_channel = tensor.shape[3]
    num_rows = 1+ num_input // num_cols # for each input channel
    fig = plt.figure(figsize=(num_cols*3,num_rows*2))
    for i in range(tensor.shape[0]): # for each input
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        if dim == 4:
            if num_out_channel==3:
                ax1.imshow(tensor[i,:,:,:])
            if num_out_channel==1:
                ax1.imshow(tensor[i,:,:,0])
        if dim == 3:
            ax1.imshow(tensor[i,:,:])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()