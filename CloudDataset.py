import torch
import os
from torchvision import transforms
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class CloudDataset():
    def __init__(self, data_dir, verbose=True):
        self.X_data, self.y_data = _loadImages(self, data_dir, verbose)

    def getData(self):
        """
        Retorno:
        - X_train: tensor com os dados de treino
        - y_train: tensor com os rótulos de treino
        - X_val: tensor com os dados de validação
        - y_val: tensor com os rótulos de validação
        """
        N = self.X_data.shape[0]
        X_train = self.X_data[0:3 * (N // 4)]
        y_train = self.y_data[0:3 * (N // 4)]
        X_val = self.X_data[3 * (N // 4):]
        y_val = self.y_data[3 * (N // 4):]
        return X_train, y_train, X_val, y_val

    def getSample(self, sample_size):
        """
        Entrada:
        - sample_size: tamanho da amostra

        Retorno:
        - X_sample: tensor com os dados do tamanho da amostra
        - y_sample: tensor com os rótulos do tamanho da amostra
        """
        mask = torch.randint(self.X_data.shape[0], (sample_size,))
        X_sample = self.X_data[mask]
        y_sample = self.y_data[mask]

        return X_sample, y_sample

def _getFiles(self, data_dir):
    """
    Retorno:
    - files: arquivos do banco de dados
    """
    files = []
    for file in os.listdir(data_dir):
        files.append(file)
    return files

def _loadImages(self, data_dir, verbose=True):
    """
    Retorno:
    - X_data: tensor com os dados
    - y_data: tensor com os rótulos
    """
    files = _getFiles(self, data_dir)
    xdata = []  #dados das imagens
    ydata = []  #rotulos
    count = 0   #contador para a rotulagem
    transform = transforms.Compose([ transforms.ToTensor() ])
    img_size = (200, 200)

    if verbose: print("Loading data from " + data_dir)
    for file in files:
        path = os.path.join(data_dir, file)
        dir = tqdm(os.listdir(path)) if verbose else os.listdir(path)
        
        for im in dir:
            image = load_img(os.path.join(path, im), grayscale=False, color_mode='rgb', target_size=img_size)
            image = img_to_array(image)
            image = image / 255.0 #intervalo de valores [0, 1]
            image_tensor = transform(image) #converte em tensor
            xdata.append(image_tensor)
            ydata.append(count)
        
        count += 1
        if verbose: print(f"Data from {file} loaded.")

    X_data = torch.stack(xdata, 0)
    y_data = torch.tensor(ydata, dtype=torch.float) #converte em tensor

    return X_data, y_data