import torch
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from torchvision import transforms


class CloudDataset():
    def __init__(self, data_dir, verbose=True, train_size=0.8):
        self.train_size = train_size
        self.X_data, self.y_data = _loadImages(data_dir, verbose)

    def getData(self):
        """
        Get data split in training and test parts.\n
        Return:
        - X_train: tensor with image data for training
        - y_train: tensor with labels for training
        - X_test: tensor with image data for test
        - y_test: tensor with labels for test
        """
        classes = torch.arange(torch.max(self.y_data) + 1)
        X_train = torch.empty((0), dtype=self.X_data.dtype)
        y_train = torch.empty((0), dtype=self.y_data.dtype)
        X_test = torch.empty((0), dtype=self.X_data.dtype)
        y_test = torch.empty((0), dtype=self.y_data.dtype)

        for c in classes:
            XX = self.X_data[c == self.y_data]
            yy = self.y_data[c == self.y_data]
            N = XX.shape[0]
            num_train = int(N * self.train_size)

            X_train = torch.cat((X_train, XX[0:num_train]), 0)
            y_train = torch.cat((y_train, yy[0:num_train]), 0)
            X_test = torch.cat((X_test, XX[num_train:N]), 0)
            y_test = torch.cat((y_test, yy[num_train:N]), 0)
            
        # torch.set_printoptions(threshold=10_000)
        # counts_y = self.y_data.unique(return_counts=True)[1]
        # print(f"{counts_y=}")
        # counts_y_train = y_train.unique(return_counts=True)[1]
        # print(f"{counts_y_train=}")
        # counts_y_test = y_test.unique(return_counts=True)[1]
        # print(f"{counts_y_test=}")
        # print(f"{counts_y_train/counts_y=}")

        return X_train, y_train, X_test, y_test

def _getFiles(data_dir):
    """
    Parameters:
    - data_dir: directory where dataset is located\n
    Return:
    - files: files from database
    """
    files = []
    for file in os.listdir(data_dir):
        files.append(file)
    return files

def _loadImages(data_dir, verbose=True):
    """
    Parameters:
    - data_dir: directory where dataset is located
    - verbose (default is True): boolean to show or not debug messages\n
    Return:
    - X_data: tensor with image data
    - y_data: tensor with labels
    """
    files = _getFiles(data_dir)
    xdata = []  #list of images data
    ydata = []  #list of labels
    count = 0   #counter for labeling
    img_size = (224, 224)
    transform = transforms.Compose([ transforms.ToTensor() ]) #convert to tensor

    if verbose: print("> Loading data from " + data_dir)
    for file in files:
        path = os.path.join(data_dir, file)
        dir = tqdm(os.listdir(path)) if verbose else os.listdir(path)
        
        for im in dir:
            image = load_img(os.path.join(path, im), grayscale=False, color_mode='rgb', target_size=img_size)
            image = img_to_array(image)
            image = image / 255.0 #interval between [0, 1]
            image_tensor = transform(image)
            xdata.append(image_tensor)
            ydata.append(count)

        count += 1
        if verbose: print(f"> Data from {file} loaded.")

    X_data = torch.stack(xdata, 0)
    y_data = torch.tensor(ydata)

    return X_data, y_data