import torch
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from CNN import CNN
from CloudDataset import CloudDataset
from torch.optim import Adam
from torch import nn
from sklearn.metrics import classification_report
import time


LEARN_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 64
TRAIN_SIZE = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_batches(X, y, data_size, batch_size):
    """
    Split a dataset and its labels into randomized batches\n
    Parameters:
    - X: input dataset (tensor)
    - y: input labels (tensor)
    - data_size: size of dataset
    - batch_size: size of batches\n
    Return:
    - X_batches: list with image data batches (tensors)
    - y_batches: list with labels batches (tensors)
    """
    num_batches = int(data_size / batch_size)
    X_batches = []
    y_batches = []
    mask = torch.randperm(data_size)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True, help="./")
    ap.add_argument("-p", "--plot", type=str, required=True, help="./")
    args = vars(ap.parse_args())

    dataset = CloudDataset("database/CCSN_v2/")
    X_t, y_t, X_test, y_test = dataset.getData()
    X_train, y_train, X_val, y_val = split_data(X_t, y_t, X_t.shape[0], TRAIN_SIZE)

    num_train = X_train.shape[0]
    num_val = X_val.shape[0]
    num_test = X_test.shape[0]

    model = CNN().to(DEVICE)
    print("Model created.")

    optimizer = Adam(model.parameters(), lr=LEARN_RATE) #optimizer
    loss_func = nn.NLLLoss() #loss function
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    } #history of statistics

    print("Training the network...")
    start = time.time()

    for ep in range(EPOCHS):
        model.train()
        # initialize the total training and validation loss
        train_loss = 0
        val_loss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        num_train_correct = 0
        num_val_correct = 0
        # loop over the training set
        X_batches_train, y_batches_train = get_batches(X_train, y_train, num_train, BATCH_SIZE)
        for X_batch_train, y_batch_train in zip(X_batches_train, y_batches_train):
            # send the input to the device
            X_batch_train = X_batch_train.to(DEVICE)
            y_batch_train = y_batch_train.type(torch.LongTensor)
            y_batch_train = y_batch_train.to(DEVICE)
            # perform a forward pass and calculate the training loss
            pred = model(X_batch_train)
            loss = loss_func(pred, y_batch_train)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            train_loss += loss
            num_train_correct += (pred.argmax(1) == y_batch_train).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            X_batches_val, y_batches_val = get_batches(X_val, y_val, num_val, BATCH_SIZE)
            for X_batch_val, y_batch_val in zip(X_batches_val, y_batches_val):
                # send the input to the device
                X_batch_val = X_batch_val.to(DEVICE)
                y_batch_val = y_batch_train.type(torch.LongTensor)
                y_batch_val = y_batch_val.to(DEVICE)
                # make the predictions and calculate the validation loss
                pred = model(X_batch_val)
                val_loss += loss_func(pred, y_batch_val)
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
        # update our training history
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        history["train_acc"].append(num_train_correct)
        history["val_loss"].append(avg_val_loss.cpu().detach().numpy())
        history["val_acc"].append(num_val_correct)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(ep + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avg_train_loss, num_train_correct))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avg_val_loss, num_val_correct))
    
    # finish measuring how long training took
    end = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(end - start))
    # we can now evaluate the network on the test set
    print("[INFO] evaluating network...")
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # initialize a list to store our predictions
        preds = []
        # loop over the test set
        X_batches_test, y_batches_test = get_batches(X_test, y_test, num_test, BATCH_SIZE)
        for X_batch_test, y_batches_test in zip(X_batches_test, y_batches_test):
            # send the input to the device
            X_batch_test = X_batch_test.to(DEVICE)
            # make the predictions and add them to the list
            pred = model(X_batch_test)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
        # generate a classification report
        classes_names = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
        print(classification_report(y_test[:-1].cpu().numpy(), np.array(preds), target_names=classes_names))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    # serialize the model to disk
    torch.save(model, args["model"])

if __name__ == "__main__":
    main()