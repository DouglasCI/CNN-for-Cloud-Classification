import torch
from CloudDataset import CloudDataset

class CNN():
    def __init__(self):
        self.dataset = CloudDataset("database/CCSN_v2/")
        self.X_train, self.y_train, self.X_val, self.y_val = self.dataset.getData()
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_val.shape)
        print(self.y_val.shape)

def main():
    cnn = CNN()

if __name__ == "__main__":
    main()