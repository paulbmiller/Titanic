# -*- coding: utf-8 -*-
"""
Use a 3-layered neural network with Dropout layers to predict if a passenger
survived.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import preprocess
from copy import deepcopy
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import sys


# -------- Hyperparameters --------
epochs = 100
mb_size = 32
first = 32
second = 16
third = 16
lr = 1e-3
drop1 = 0.2
drop2 = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create the model
class TriNet(nn.Module):
    def __init__(self, in_channels, first, second, third, drop1, drop2):
        super(TriNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, first),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop1),
            nn.Linear(first, second),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop2),
            nn.Linear(second, third),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(third, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class DuoNet(nn.Module):
    def __init__(self, in_channels, first, second, drop2):
        super(DuoNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, first),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop2),
            nn.Linear(first, second),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(second, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


def get_next_sub_name(path, sub):
    """
    Checks to see if the submission ´sub´ already exists in the submission
    folder. If it already exists, it will call itself with ´sub + 1´ until it
    finds a number which hasn't already been created. In the other case, it
    will return the number ´sub´.

    Parameters
    ----------
    path : string
        Path to the file submission folder.
    sub : int
        Number of the submission we are checking.

    Returns
    -------
    int
        Number of the next submission.

    """
    sub_name = 'sub' + str(sub) + '.csv'
    if os.path.exists(path + sub_name):
        print('{} already exists'.format(sub_name))
        return get_next_sub_name(path, sub + 1)
    else:
        return sub


def create_dataloaders(mb_size, X_train, y_train, X_test):
    """
    Create the training and testing dataloaders.

    Parameters
    ----------
    mb_size : int
        Mini-batch size.
    X_train : pandas DataFrame
        Training set features.
    y_train : pandas Series
        Training set target.
    X_test : pandas DataFrame
        Test set features.

    Returns
    -------
    train_loader : torch DataLoader
        Torch DataLoader for the training set.
    test_loader : torch DataLoader
        Torch DataLoader for the test set.

    """
    train = TensorDataset(torch.Tensor(np.array(X_train)),
                          torch.Tensor(np.array(y_train)))
    train_loader = DataLoader(train, batch_size=mb_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.Tensor(np.array(X_test))),
                             batch_size=mb_size, shuffle=False)
    return train_loader, test_loader


def train(TriNet, epochs, optim, train_loader, loss_fn=F.binary_cross_entropy):
    """
    Standard training routine.

    Parameters
    ----------
    net : TriNet
        Instance of the TriNet class, child of the torch.nn.Module.
    epochs : int
        Number of training epochs.
    optim : torch.optim child
        Optimizer object.
    train_loader : DataLoader
        Torch DataLoader for the training set.
    loss_fn : function, default torch.nn.functional.binary_cross_entropy
        Loss function used for training

    Returns
    -------
    None.

    """
    net.train()
    for epoch in tqdm(range(1, epochs+1), desc='Training', unit=' ep',
                      file=sys.stdout):
        running_loss = 0.0
        for i, (data, survived) in enumerate(train_loader):
            data = data.to(device)
            survived = survived.to(device)
            y_pred = net(data).reshape(-1)

            loss = loss_fn(y_pred, survived)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()


def cross_val(net, opt, eps, X_train, y_train, k_folds, init_state,
              init_state_opt, model_id, nb_models, loss_fn):
    """
    Cross validation implementation using ´k_folds´ number of folds.

    Parameters
    ----------
    net : TriNet
        Instance of the TriNet class, child of the torch.nn.Module.
    opt : torch.optim child
        Optimizer object.
    eps : int
        Number of training epochs.
    X_train : pandas DataFrame
        Training set features.
    y_train : pandas Series
        Training set target vector.
    k_folds : int
        Number of cross validation folds.
    init_state : collections.OrderedDict
        Initial state of the neural net to reset after each fold.
    init_state_opt : dict
        Initial state of the optimizer to reset after each fold.

    Returns
    -------
    test_accs : numpy.ndarray
        Accuracies of each fold.

    """
    fold_cntr = 1
    test_accs = np.zeros(k_folds)
    kf = KFold(n_splits=k_folds)
    for train_indices, test_indices in kf.split(X_train):
        if fold_cntr != 1:
            net.load_state_dict(init_state)
            opt.load_state_dict(init_state_opt)
        train = TensorDataset(
            torch.Tensor(np.array(X_train.loc[train_indices])),
            torch.Tensor(np.array(y_train.loc[train_indices]))
        )
        val = TensorDataset(
            torch.Tensor(np.array(X_train.loc[test_indices])),
            torch.Tensor(np.array(y_train.loc[test_indices]))
        )
        train_loader = DataLoader(train, batch_size=mb_size, shuffle=False)
        val_loader = DataLoader(val, batch_size=mb_size, shuffle=False)
        net.train()
        str_desc = 'Model {}/{}, fold {}/{} :'.format(model_id, nb_models,
                                                      fold_cntr, k_folds)
        
        for epoch in tqdm(range(1, eps+1), desc=str_desc, unit=' ep',
                          file=sys.stdout):
            running_loss = 0.0
            for i, (data, survived) in enumerate(train_loader):
                data = data.to(device)
                survived = survived.to(device)
                mb = data.size(0)
                y_pred = net(data).reshape(-1)

                loss = loss_fn(y_pred, survived)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item()
        net.eval()
        running_loss = 0.0
        right_predictions = 0
        wrong_predictions = 0
        
        for i, (data, survived) in enumerate(val_loader):
            data = data.to(device)
            survived = survived.to(device)
            mb = data.size(0)
            y_pred = net(data).reshape(-1)

            loss = loss_fn(y_pred, survived)
            running_loss += loss.item()

            survived_bool = survived == 1
            proj = y_pred.cpu() > 0.5
            
            for j in range(mb):
                if proj[j] == survived_bool[j].cpu():
                    right_predictions += 1
                else:
                    wrong_predictions += 1
        tqdm.write('Val Epoch, loss: {}, acc: {}, fold: {}'.format(
            running_loss/(i+1),
            100 * right_predictions / (right_predictions + wrong_predictions),
            fold_cntr), file=sys.stdout
        )
        test_accs[fold_cntr-1] = right_predictions / (right_predictions +
                                                      wrong_predictions)
        fold_cntr += 1
        
    tqdm.write('Model mean accuracy : {}, model std : {}'.format(
        test_accs.mean(), test_accs.std()), file=sys.stdout)
        
    return test_accs


def submit(net, test_loader, filename):
    """
    Returns the prediction array to the output file with ´filename´.

    Parameters
    ----------
    survived_array : numpy.ndarray
        Prediction array of values between 0 and 1.
    test_loader : torch.DataLoader
        Data loader for the test set.
    filename : string
        Name of the submission file.

    Returns
    -------
    None.

    """
    survived_array = np.array([])
    net.eval()
    for i, [data] in enumerate(test_loader):
        data = data.to(device)
        y_pred = net(data)
        survived = y_pred.detach().cpu().numpy()
        survived_array = np.append(survived_array, survived)

    survived_array = survived_array > 0.5

    # Get the passenger ids for the test set for submission
    test_ids = pd.read_csv('test.csv').PassengerId
    survived = pd.Series(survived_array.astype(int))

    frame = {'PassengerId': test_ids, 'Survived': survived}

    submission_df = pd.DataFrame(frame)

    submission_df.to_csv(
        filename,
        index=False
    )
    print('\nPredictions submitted to ' + filename)
    
    return survived


def test_model(net, epochs, lr, mb_size, accs, opt, net_list, k_folds,
               model_id, nb_models, loss_fn):
    """
    Gets the initial state of the neural net and optimizer, so we can reset to
    these values between each fold of cross-validation.

    Parameters
    ----------
    net : TriNet
        Instance of the TriNet class, child of the torch.nn.Module.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate to be passed to the optimizer.
    mb_size : int
        Mini-batch size.
    accs : list
        List to contain the accuracy results from cross-validation.
    opt : string
        Name of the optimizer to be used.
    net_list : list
        Basic information about the neural net (neuron numbers and drop rates).
    k_folds : int
        Number of folds used for cross-validation.
    model_id : int
        Id of the model for the progress bar output.
    nb_models : int
        Number of models we are testing for the progress bar output.
    loss_fn : function
        Loss function used by our model.

    Returns
    -------
    None.

    """
    if opt == 'Adam':
        optim = torch.optim.Adam(net.parameters(), lr=lr)
    elif opt == 'RMSprop':
        optim = torch.optim.RMSprop(net.parameters(), lr=lr)
    elif opt == 'SGD':
        optim = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        print('was given a wrong value for optim')
        return
    
    init = deepcopy(net.state_dict())
    init_opt = deepcopy(optim.state_dict())
    acc = cross_val(net, optim, epochs, X_train, y_train, k_folds, init,
                    init_opt, model_id, nb_models, loss_fn)
    accs.append([epochs, lr, mb_size, opt, acc.mean(), acc.std(), net_list])


def sort_accs(li):
    # function used as key to the sort method to sort grid search accs
    return li[4]


def accs_to_file(accs):
    # Appends accuracies of grid search to file
    with open('results.txt', 'a') as f:
        f.write('\nout:\n')
        for acc in accs:
            f.write(str(acc) + '\n')


if __name__ == '__main__':
    mode = 'gs'
    mode = 'sub' # Uncomment this line to get a submission
    
    if mode == 'gs':
        # Grid Search
        k_folds = 7
        accs = []
        epochs_list = [50, 100]
        lr_list = [1e-3, 5e-4, 1e-4]
        mb_sizes = [1, 16, 32]
        optims = ['Adam']
        nets = [
                [64, 32, 0.5],
                [64, 32, 0.2],
                [32, 16, 0.5],
                [32, 16, 0.2]
                ]
        loss_fns = [F.binary_cross_entropy]
        
        model_id = 1
        nb_models = len(epochs_list) * len(lr_list) * len(mb_sizes) * \
            len(optims) * len(nets) * len(loss_fns)
        
        for mb_size in mb_sizes:
            X_train, y_train, X_test = preprocess()
            train_loader, test_loader = create_dataloaders(mb_size, X_train,
                                                           y_train, X_test)
            for lr in lr_list:
                for epochs in epochs_list:
                    for opt in optims:
                        for net in nets:
                            for loss_fn in loss_fns:
                                if len(net) == 5:
                                    net_obj = TriNet(
                                        len(X_train.columns), net[0], net[1],
                                        net[2], net[3], net[4]).to(
                                            device)
                                elif len(net) == 3:
                                    net_obj = DuoNet(
                                        len(X_train.columns), net[0], net[1],
                                        net[2]).to(device)
                                test_model(net_obj, epochs, lr, mb_size, accs, opt,
                                     net, k_folds, model_id, nb_models,
                                     loss_fn)
                                
                                model_id += 1
    
        accs.sort(key=sort_accs, reverse=True)

    else:
        # Standard run
        sub = 76
        subs = 1
        path = \
            "F:\\Users\\SilentFart\\Documents\\PythonProjects\\Titanic\\subs\\"
        next_sub_id = get_next_sub_name(path, sub)
        for i in range(subs):
            X_train, y_train, X_test = preprocess()
            train_loader, test_loader = create_dataloaders(mb_size, X_train,
                                                           y_train, X_test)
            #net = TriNet(len(X_train.columns), first, second, third, drop1,
            #          drop2).to(device)
            net = DuoNet(len(X_train.columns), first, second, drop2).to(device)
            optim = torch.optim.Adam(net.parameters(), lr=lr)
            train(net, epochs, optim, train_loader)
            filename = path + 'sub' + str(next_sub_id + i) + '.csv'
            submit(net, test_loader, filename)

