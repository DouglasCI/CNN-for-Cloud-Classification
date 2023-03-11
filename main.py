import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from CNN import CNN
from CloudDataset import CloudDataset
from torch import nn
from torch.optim import SGD
from torchvision import transforms
from sklearn.metrics import classification_report


LEARN_RATE = 1e-3
EPOCHS = 20
BATCH_SIZE = 32
TRAIN_SIZE = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_data(X, y, data_size, train_size):
    """
    Parameters:
    - X (tensor): input dataset
    - y (tensor): input labels
    - data_size (int): size of dataset
    - train_size (int): size of train tensor\n
    Returns:
    - tensor: image data for training
    - tensor: labels for training
    - tensor: image data for validation
    - tensor: labels for validation
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

def get_batches(X, y, data_size, batch_size, random=True):
    """
    Split a dataset and its labels into randomized batches\n
    Parameters:
    - X (tensor): input dataset
    - y (tensor): input labels
    - data_size (int): size of dataset
    - batch_size (int): size of batches
    - random (boolean): decide if batches are randomized or not\n
    Returns:
    - list: list with image data batches
    - list: list with labels batches
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

def normalize(X):
    """
    Parameters:
    - X (tensor): input dataset\n
    Returns:
    - tensor: normalized tensor
    """
    mean = X.mean([0, 2, 3])
    std = X.std([0, 2, 3])
    X_normalized = transforms.Normalize(mean, std)(X)
        
    return X_normalized

def augment(X, y):
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True, help="./")
    ap.add_argument("-p", "--plot", type=str, required=True, help="./")
    args = vars(ap.parse_args())

    # get dataset
    dataset = CloudDataset("database/CCSN_v2/")
    X_t, y_t, X_test, y_test = dataset.getData()
    X_train, y_train, X_val, y_val = split_data(X_t, y_t, X_t.shape[0], TRAIN_SIZE)
    # normalize the entire test and validation dataset
    # the training dataset will be normalized in batches
    X_test_norm = normalize(X_test)
    X_val_norm = normalize(X_val)

    num_train = X_train.shape[0]
    num_val = X_val.shape[0]
    num_test = X_test.shape[0]

    model = CNN().to(DEVICE)
    print("> Model created.")

    optimizer = SGD(model.parameters(), lr=LEARN_RATE, weight_decay = 0.005, momentum = 0.9) #optimizer
    loss_func = nn.CrossEntropyLoss() #loss function
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    } #history of statistics

    print("> Training the network...")
    start = time.time()

    for ep in range(EPOCHS):
        # set model to training mode
        model.train()
        
        train_loss = 0
        val_loss = 0
        num_train_correct = 0
        num_val_correct = 0
        
        # split the training set into batches and iterate over them
        X_batches_train, y_batches_train = get_batches(X_train, y_train, num_train, BATCH_SIZE, random=True)
        for X_batch_train, y_batch_train in zip(X_batches_train, y_batches_train):
            # normalize this batch
            X_batch_train_norm = normalize(X_batch_train)
            X_batch_aug = augment(X_batch_train_norm)
            
            # send it to the device
            X_batch_aug = X_batch_aug.to(DEVICE)
            y_batch_train = y_batch_train.type(torch.LongTensor)
            y_batch_train = y_batch_train.to(DEVICE)
            
            pred = model(X_batch_aug) #forward pass
            loss = loss_func(pred, y_batch_train) #calculate loss
            optimizer.zero_grad() #zero out gradients
            loss.backward() #backpropagation
            optimizer.step() #update weights
            train_loss += loss #accumulate training loss
            
            # calculate the number of correct predictions
            num_train_correct += (pred.argmax(1) == y_batch_train).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set model to evaluation mode
            model.eval()
            
            # split the validation set into batches and iterate over them
            X_batches_val, y_batches_val = get_batches(X_val_norm, y_val, num_val, BATCH_SIZE, random=False)
            for X_batch_val, y_batch_val in zip(X_batches_val, y_batches_val):
                # send it to the device
                X_batch_val = X_batch_val.to(DEVICE)
                y_batch_val = y_batch_train.type(torch.LongTensor)
                y_batch_val = y_batch_val.to(DEVICE)
                
                pred = model(X_batch_val) # make the predictions
                val_loss += loss_func(pred, y_batch_val) #accumulate validation loss
                
                # calculate the number of correct predictions
                num_val_correct += (pred.argmax(1) == y_batch_val).type(torch.float).sum().item()
        
        # calculate the average training and validation loss
        num_iter_train = max(num_train // BATCH_SIZE, 1)
        num_iter_val = max(num_val // BATCH_SIZE, 1)
        avg_train_loss = train_loss / num_iter_train
        avg_val_loss = val_loss / num_iter_val
        
        # calculate the training and validation accuracy
        num_train_correct = num_train_correct / num_train
        num_val_correct = num_val_correct / num_val
        
        # update training history
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        history["train_acc"].append(num_train_correct)
        history["val_loss"].append(avg_val_loss.cpu().detach().numpy())
        history["val_acc"].append(num_val_correct)
        
        # print the model training and validation information
        print("> EPOCH: {}/{}".format(ep + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avg_train_loss, num_train_correct))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avg_val_loss, num_val_correct))
    
    # print runtime of training
    end = time.time()
    print("> Runtime: {:.2f}s".format(end - start))
    
    print("> Evaluating network...")
    # turn off autograd for testing evaluation
    with torch.no_grad():
        model.eval()
        
        # initialize a list to store our predictions
        preds = []
        # split the test set into batches and iterate over them
        X_batches_test, y_batches_test = get_batches(X_test_norm, y_test, num_test, BATCH_SIZE, random=False)
        for X_batch_test, y_batches_test in zip(X_batches_test, y_batches_test):
            # send it to the device
            X_batch_test = X_batch_test.to(DEVICE)
            
            # make the predictions and add them to the list
            pred = model(X_batch_test)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
            
        # generate a classification report
        classes_names = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
        print(classification_report(y_test[:-1].cpu().numpy(), np.array(preds), target_names=classes_names, zero_division=0))

    # plot the training loss and accuracy
    fig, axis = plt.subplots(2)
    fig.suptitle("Training x Validation")
    
    line1, = axis[0].plot(list(range(EPOCHS)), history["train_loss"], label="train_loss")
    line2, = axis[0].plot(list(range(EPOCHS)), history["val_loss"], label="val_loss")
    axis[0].set_ylabel("loss")
    axis[0].legend(handles=[line1, line2])
    
    line3, = axis[1].plot(list(range(EPOCHS)), history["train_acc"], label="train_acc")
    line4, = axis[1].plot(list(range(EPOCHS)), history["val_acc"], label="val_acc")
    axis[1].set_xlabel("epochs")
    axis[1].set_ylabel("accuracy")
    axis[1].legend(handles=[line3, line4])
    
    plt.savefig(args["plot"])
    torch.save(model.state_dict(), args["model"])

if __name__ == "__main__":
    main()