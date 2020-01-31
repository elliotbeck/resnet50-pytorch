import torch
import numpy as np
from torch import nn, optim

# define accuracy function 
def mean_accuracy(logits, y):
   probs = torch.softmax(logits, dim=1)
   winners = probs.argmax(dim=1)
   corrects = (winners.unsqueeze(-1) == y.cuda())
   accuracy = corrects.sum().float() / float(y.size(0))
   return accuracy

# define training function of the model
def train_epoch(data, model, loss_function, optimizer):
    for i, batch in enumerate(data, 1):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        # forward 
        model = model.train()
        outputs = model(inputs)
        #get loss per domain
        loss = loss_function(outputs, torch.tensor(torch.squeeze(labels), dtype=torch.long).cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # perform gradient descent
        loss.backward()
        optimizer.step()
# define validation function for training of the model
def validate_epoch(data_train, data_test, model, loss_function):
    loss_test = 0
    accuracy_test = 0
    loss_train = 0
    accuracy_train = 0
    # get accuracy and loss on train data
    with torch.no_grad():
        for i, batch in enumerate(data_train, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            # forward
            model = model.eval()
            outputs = model(inputs)
            # get loss per domain
            loss = loss_function(outputs, torch.tensor(torch.squeeze(labels), dtype=torch.long).cuda())
            # get mean loss
            loss_train += loss
            # get accuracy per domain
            accuracy = mean_accuracy(outputs, labels)
            # append mean accuracy
            accuracy_train += accuracy
    accuracy_train = (accuracy_train/i).detach().cpu().numpy()
    loss_train = (loss_train/i).detach().cpu().numpy()

    # get accuracy and loss on test data
    with torch.no_grad():
        for i, batch in enumerate(data_test, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            # forward
            model = model.eval()
            outputs = model(inputs)
            # get loss per domain
            loss = loss_function(outputs, torch.tensor(torch.squeeze(labels), dtype=torch.long).cuda())
            # get mean loss
            loss_test += loss
            # get accuracy per domain
            accuracy = mean_accuracy(outputs, labels)
            # append mean accuracy
            accuracy_test += accuracy
    accuracy_test = (accuracy_test/i).detach().cpu().numpy()
    loss_test = (loss_test/i).detach().cpu().numpy()

    return loss_train, accuracy_train, loss_test, accuracy_test