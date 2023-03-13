import torch
import os
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

class CloudDataset():
    def __init__(self, data_dir, verbose=True, train_size=0.8):
        self.train_size = train_size
        self.X_data, self.y_data = _loadImages(data_dir, verbose)

    def getData(self):
        """
        Get data split in training and test parts.\n
        Returns:
        - tensor: image data for training.
        - tensor: labels for training.
        - tensor: image data for test.
        - tensor: labels for test.
        """
        classes = torch.arange(torch.max(self.y_data) + 1)
        X_train = torch.empty((0), dtype=self.X_data.dtype)
        y_train = torch.empty((0), dtype=self.y_data.dtype)
        X_test = torch.empty((0), dtype=self.X_data.dtype)
        y_test = torch.empty((0), dtype=self.y_data.dtype)

        # split datasets into balanced train and test parts
        for c in classes:
            XX = self.X_data[c == self.y_data]
            yy = self.y_data[c == self.y_data]
            N = XX.shape[0]
            num_train = int(N * self.train_size)

            X_train = torch.cat((X_train, XX[0:num_train]), 0)
            y_train = torch.cat((y_train, yy[0:num_train]), 0)
            X_test = torch.cat((X_test, XX[num_train:N]), 0)
            y_test = torch.cat((y_test, yy[num_train:N]), 0)
            
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
    - data_dir (string): directory where dataset is located.\n
    Returns:
    - list: filenames from database.
    """
    files = []
    for file in os.listdir(data_dir):
        files.append(file)
    return files

def _loadImages(data_dir, verbose=True):
    """
    Parameters:
    - data_dir (string): directory where dataset is located.
    - verbose (boolean, optional): show or not debug messages (default=True).\n
    Returns:
    - tensor: image data.
    - tensor: labels.
    """
    files = _getFiles(data_dir)
    xdata = []  #list of images data
    ydata = []  #list of labels
    count = 0   #counter for labeling
    img_size = (227, 227)
    resize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=img_size)
    ])

    if verbose: print("> Loading data from " + data_dir)
    for file in files:
        path = os.path.join(data_dir, file)
        dir = tqdm(os.listdir(path)) if verbose else os.listdir(path)
        
        for img in dir:
            img_path = os.path.join(path, img)
            image = Image.open(img_path)
            image_tensor = resize_transform(image)
            # mean, std = image_tensor.mean([1, 2]), image_tensor.std([1, 2])
            # normalized_img = transforms.Normalize(mean, std)(image_tensor)
            
            xdata.append(image_tensor)
            ydata.append(count)

        count += 1
        if verbose: print(f"> Data from {file} loaded.")

    X_data = torch.stack(xdata, 0)
    y_data = torch.tensor(ydata)

    return X_data, y_data

def show_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    
def main():
    dataset = CloudDataset("database/CCSN_v2/")
    X_train, y_train, X_test, y_test = dataset.getData()
    
    #size of the images
    img_size = (227, 227)

    # 3 composes
    transform_list = [
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.0))
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.0)),
            transforms.RandomRotation(degrees=180)
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomPerspective(p=1)
        ])
    ]
        
    X_batch = X_train[:16]
    y_batch = y_train[:16]
    size = X_batch.shape[0]
    
    for transform in transform_list:
        # print(f"{transform=}")
        for i in range(size):
            x = transform(X_batch[i]).unsqueeze(dim=0)
            X_batch = torch.cat((X_batch, x), 0)
            y_batch = torch.cat((y_batch, y_batch[i].unsqueeze(dim=0)), 0)
            
        print(f"{X_batch.shape=}")
        print(f"{y_batch=}")
    
    # for i in range(size, X_batch.shape[0]): show_image(X_batch[i])
    
if __name__ == "__main__":
    main()