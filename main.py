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
EPOCHS = 100
BATCH_SIZE = 8
TRAIN_SIZE = 0.8
DATA_AUG_FACTOR = 3 #used to consider data augmentation in training set when calculating accuracy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_data(X, y, data_size, train_size=0.8):
    """
    Parameters:
    - X (tensor): input dataset.
    - y (tensor): input labels.
    - data_size (int): size of dataset.
    - train_size (float, optional): % of dataset as train data (default=0.8).\n
    Returns:
    - tensor: image data for training.
    - tensor: labels for training.
    - tensor: image data for validation.
    - tensor: labels for validation.
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

def get_batches(X, y, data_size, batch_size=16, random=True):
    """
    Split a dataset and its labels into batches.\n
    Parameters:
    - X (tensor): input dataset.
    - y (tensor): input labels.
    - data_size (int): size of dataset.
    - batch_size (float, optional): size of batch (default=16).
    - random (boolean, optional): randomize batches? (default=True).\n
    Returns:
    - list<tensor>: image data batches.
    - list<tensor>: labels batches.
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

def normalize(X, use_fixed_values):
    """
    Normalizes input tensor by color channels,
    can receive fixed values for mean and std.\n
    Parameters:
    - X (tensor): input tensor.
    - use_fixed_values (boolean): use fixed values for mean and std?\n
    Returns:
    - tensor: normalized tensor.
    """
    if use_fixed_values:
        # values calculated from the entire dataset
        mean = [0.4798, 0.5234, 0.5620]
        std = [0.2605, 0.2370, 0.2551]
    else:
        mean = X.mean([0, 2, 3])
        std = X.std([0, 2, 3])
        
    X_normalized = transforms.Normalize(mean, std)(X)
        
    return X_normalized

def augment(X, y, batch_size=16):
    """
    Parameters:
        X (tensor): input dataset.
        y (tensor): input labels.
        batch_size (int, optional): size of batch (default=16). \n
    Returns:
        tensor: augmented dataset.
        tensor: augmented dataset labels.
    """
    img_size = (227, 227)
    transform_list = [
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.0))]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.0)),
            transforms.RandomRotation(degrees=180)]),
        # transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomPerspective(p=1)])
    ]
    
    # apply each set of transformations on the batch
    for transform in transform_list:
        for i in range(batch_size):
            xx = transform(X[i]).unsqueeze(dim=0)
            yy = y[i].unsqueeze(dim=0)
            X = torch.cat((X, xx), 0)
            y = torch.cat((y, yy), 0)
    
    return X, y
    

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
    X_test_norm = normalize(X_test, use_fixed_values=True)
    X_val_norm = normalize(X_val, use_fixed_values=True)

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
        
        # values to calculate loss and accuracy
        train_loss = 0
        val_loss = 0
        num_train_correct = 0
        num_val_correct = 0
        
        # split the training set into batches and iterate over them
        X_batches_train, y_batches_train = get_batches(X_train, y_train, num_train, BATCH_SIZE, random=True)
        for X_batch_train, y_batch_train in zip(X_batches_train, y_batches_train):
            # normalize this batch
            X_batch_train = normalize(X_batch_train, use_fixed_values=False)
            
            # augment this batch
            X_batch_train, y_batch_train = augment(X_batch_train, y_batch_train, BATCH_SIZE)
            
            # send it to the device
            X_batch_train = X_batch_train.to(DEVICE)
            y_batch_train = y_batch_train.type(torch.LongTensor)
            y_batch_train = y_batch_train.to(DEVICE)
            
            pred = model(X_batch_train) #forward pass
            loss = loss_func(pred, y_batch_train) #calculate loss
            optimizer.zero_grad() #zero out gradients
            loss.backward() #backpropagation
            optimizer.step() #update weights
            train_loss += loss #accumulate training loss
            
            # calculate the number of correct predictions
            num_train_correct += (pred.argmax(axis=1) == y_batch_train).type(torch.float).sum().item()
            
        # switch off autograd for evaluation
        with torch.no_grad():
            # set model to evaluation mode
            model.eval()
            
            # split the validation set into batches and iterate over them
            X_batches_val, y_batches_val = get_batches(X_val_norm, y_val, num_val, BATCH_SIZE, random=False)
            for X_batch_val, y_batch_val in zip(X_batches_val, y_batches_val):
                # send it to the device
                X_batch_val = X_batch_val.to(DEVICE)
                y_batch_val = y_batch_val.type(torch.LongTensor)
                y_batch_val = y_batch_val.to(DEVICE)
                
                pred = model(X_batch_val) # make the predictions
                val_loss += loss_func(pred, y_batch_val) #accumulate validation loss
                
                # calculate the number of correct predictions
                num_val_correct += (pred.argmax(axis=1) == y_batch_val).type(torch.float).sum().item()
        
        # calculate the average training and validation loss
        num_iter_train = max(num_train // BATCH_SIZE, 1)
        num_iter_val = max(num_val // BATCH_SIZE, 1)
        avg_train_loss = train_loss / num_iter_train
        avg_val_loss = val_loss / num_iter_val
        
        # calculate the training and validation accuracy
        num_train_correct = num_train_correct / (num_iter_train * BATCH_SIZE * DATA_AUG_FACTOR) #data augmentation
        num_val_correct = num_val_correct / (num_iter_val * BATCH_SIZE)
        
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