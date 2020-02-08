import torch
from torch import nn, optim
import argparse
import numpy as np
import random
from models import resnet
from torch.utils import data
from data_loader import HDF5Dataset
from train import validate_epoch, train_epoch

# set flags
parser = argparse.ArgumentParser(description='PACS')
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seed', type=int, default=1)
flags = parser.parse_args()

# set seed
random.seed(flags.seed)

# load train data
dataset1 = HDF5Dataset('/cluster/work/math/ebeck/data/pacs/photo_train.hdf5')
dataset2 = HDF5Dataset('/cluster/work/math/ebeck/data/pacs/cartoon_train.hdf5')
dataset3 = HDF5Dataset('/cluster/work/math/ebeck/data/pacs/art_painting_train.hdf5')
train_data = data.DataLoader(data.ConcatDataset([dataset1, dataset2, dataset3]), num_workers=1, 
                                                batch_size=flags.batch_size, shuffle=True, drop_last=True)

# load test data
dataset = HDF5Dataset('/cluster/work/math/ebeck/data/pacs/sketch_test.hdf5')
test_data = data.DataLoader(dataset, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)
                    
# load model
model = resnet(hidden_dim=flags.hidden_dim, num_classes=7).cuda()

# set train function 
def trainer(model, train_data, test_data, epochs, learning_rate):
    # set loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # loop over the dataset multiple times
    for epoch in range(epochs):  
        # train one epoch
        train_epoch(train_data, model, loss_function, optimizer)
        # validate epoch on validation set
        loss_train, accuracy_train, loss_test, accuracy_test = validate_epoch(train_data, test_data, model, loss_function)
        # print the metrics
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                                np.array2string(loss_train, precision=2, floatmode='fixed'),
                                np.array2string(accuracy_train*100, precision=2, floatmode='fixed'),
                                np.array2string(loss_test, precision=2, floatmode='fixed'),
                                np.array2string(accuracy_test*100, precision=2, floatmode='fixed')))          
                
    print('Finished Training')

if __name__ == "__main__":
    trainer(model, train_data, test_data, flags.epochs, flags.lr)






