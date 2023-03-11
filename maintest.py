import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from CNN import CNN
from CloudDataset import CloudDataset
from torch import nn
from torch.optim import SGD
from sklearn.metrics import classification_report


LEARN_RATE = 1e-3
EPOCHS = 12
BATCH_SIZE = 64
TRAIN_SIZE = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

def get_batches(X, y, data_size, batch_size, random=True):
    """
    Split a dataset and its labels into randomized batches\n
    Parameters:
    - X: input dataset (tensor)
    - y: input labels (tensor)
    - data_size: size of dataset
    - batch_size: size of batches
    - random: decide if batches are randomized or not\n
    Return:
    - X_batches: list with image data batches (tensors)
    - y_batches: list with labels batches (tensors)
    """
    num_batches = int(data_size / batch_size)
    X_batches = []
    y_batches = []
    mask = torch.randperm(data_size) if random else torch.arange(data_size)
    for n in range(num_batches):
        m = mask[n * batch_size:(n + 1) * batch_size]
        X_batches.append(X[m])
        y_batches.append(y[m])

    return X_batches, y_batches

def split_data(X, y, data_size, train_size):
    """
    Parameters:
    - X: input dataset (tensor)
    - y: input labels (tensor)
    - data_size: size of dataset
    - train_size: size of train tensor\n
    Return:
    - X_train: tensor with image data for training
    - y_train: tensor with labels for training
    - X_val: tensor with image data for validation
    - y_val: tensor with labels for validation
    """
    mask = torch.randperm(data_size)
    split = int(data_size * train_size)
    mask_train = mask[0:split]
    mask_val = mask[split:]

    X_train = X[mask_train]
    y_train = y[mask_train]
    X_val = X[mask_val]
    y_val = y[mask_val]

    return X_train, y_train, X_val, y_val

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
# def imshow_filter(filters,row,col):
#     print('-------------------------------------------------------------')
#     plt.figure()
#     for i in range(len(filters)):
#         w = np.array([0.299, 0.587, 0.114]) #weight for RGB
#         img = filters[i]
#         img = np.transpose(img, (1, 2, 0))
#         img = img/(img.max()-img.min())
#         img = np.dot(img,w)

#         plt.subplot(row,col,i+1)
#         plt.imshow(img,cmap= 'gray')
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()

def main():
    
    model = CNN()
    model.load_state_dict(torch.load("output/model.pth"))
    
    # print(model.layer1[layer].weight)
    filter = model.layer1[0].weight.data.cpu().clone()
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()