# -*- coding: utf-8 -*-
"""
Use of Random Forest Regression to fill missing age values and a Neural
Network with Dropout layers to predict if a survivor survived.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import Preprocess
from copy import deepcopy
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -------- Constants --------
in_channels = 18  # Result of preprocessing
training_size = 891
test_size = 418
train_folds_size = 16*50
test_folds_size = 16*5+11

# -------- Hyperparameters --------
epochs = 10
mb_size = 32
first = 128
second = 64
third = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 3e-3


# Create the model
class Net(nn.Module):
    def __init__(self, in_channels, first, second, third, drop1, drop2, drop3):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, first),
            nn.Dropout(drop1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop2),
            nn.Linear(first, second),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop3),
            nn.Linear(second, third),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(third, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def create_dataloaders(mb_size, X_train, y_train, X_test):
    # Create both dataloaders
    train = TensorDataset(torch.Tensor(np.array(X_train)),
                          torch.Tensor(np.array(y_train)))
    train_loader = DataLoader(train, batch_size=mb_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.Tensor(np.array(X_test))),
                             batch_size=mb_size, shuffle=False)
    return train_loader, test_loader


def train(net, epochs, optim, train_loader):
    net.train()
    for epoch in tqdm(range(1, epochs+1), desc='Training', unit=' ep'):
        running_loss = 0.0
        right_predictions = 0
        for i, (data, survived) in enumerate(train_loader):
            data = data.to(device)
            survived = survived.to(device)
            mb = data.size(0)
            y_pred = net(data).reshape(-1)

            loss = F.binary_cross_entropy(y_pred, survived)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()

            survived_bool = survived == 1
            proj = y_pred.cpu() > 0.5
            for j in range(mb):
                if proj[j] == survived_bool[j].cpu():
                    right_predictions += 1
        # acc = right_predictions / training_size


def cross_val(net, opt, eps, X_train, y_train, k_folds, init_state,
              init_state_opt):
    fold_cntr = 0
    test_accs = np.zeros(k_folds)
    kf = KFold(n_splits=k_folds)
    for train_indices, test_indices in kf.split(X_train):
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
        str_desc = 'Fold ' + str(fold_cntr+1) + " / " + str(k_folds)
        for epoch in tqdm(range(1, epochs+1), desc=str_desc, unit=' ep'):
            running_loss = 0.0
            for i, (data, survived) in enumerate(train_loader):
                data = data.to(device)
                survived = survived.to(device)
                mb = data.size(0)
                y_pred = net(data).reshape(-1)

                loss = F.binary_cross_entropy(y_pred, survived)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item()
        tqdm.write('')
        net.eval()
        running_loss = 0.0
        right_predictions = 0
        wrong_predictions = 0
        for i, (data, survived) in enumerate(val_loader):
            data = data.to(device)
            survived = survived.to(device)
            mb = data.size(0)
            y_pred = net(data).reshape(-1)

            loss = F.binary_cross_entropy(y_pred, survived)
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
            fold_cntr+1)
        )
        test_accs[fold_cntr] = right_predictions / (right_predictions +
                                                    wrong_predictions)
        fold_cntr += 1
    return test_accs


def submit(net, test_loader, filename):
    survived_array = np.array([])
    net.eval()
    for i, [data] in enumerate(test_loader):
        data = data.to(device)
        y_pred = net(data)
        survived = y_pred.detach().cpu().numpy()
        survived_array = np.append(survived_array, survived)

    # Add a penalty if the age value is missing
    survived_array -= penalty_for_missing_age * 0
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


def test(net, epochs, lr, mb_size, accs, opt, net_array):
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
    acc = cross_val(net, optim, epochs, X_train, y_train, 10, init, init_opt)
    accs.append([epochs, lr, mb_size, opt, acc.mean(), acc.std(), net_array])


def sort_accs(li):
    return li[4]


def accs_to_file(accs):
    with open('results.txt', 'a') as f:
        f.write('\nout:\n')
        for acc in accs:
            f.write(str(acc) + '\n')


if __name__ == '__main__':
    """
    # Grid Search
    accs = []
    epochs_list = [10]
    lr_list = [3e-3]
    mb_sizes = [32]
    optims = ['Adam', 'RMSprop']
    nets = [
            [128, 64, 32, 0, 0.2, 0.2],
            [128, 64, 32, 0, 0.1, 0.1],
            [128, 64, 32, 0, 0, 0]
            ]
    for mb_size in mb_sizes:
            penalty_for_missing_age, X_train, y_train = Preprocess()
            train_loader, test_loader = create_dataloaders(mb_size, X_train,
                                                           y_train, X_test)
        for lr in lr_list:
            for epochs in epochs_list:
                for opt in optims:
                    for net in nets:
                        net_obj = Net(in_channels, net[0], net[1], net[2],
                                      net[3], net[4], net[5]).to(device)
                        test(net_obj, epochs, lr, mb_size, accs, opt, net)
    accs.sort(key=sort_accs, reverse=True)
    """
    # Standard run
    sub = 47
    subs = 3
    path = "F:\\Users\\SilentFart\\Documents\\PythonProjects\\Titanic\\subs\\"
    next_sub_id = get_next_sub_name(path, sub)
    for i in range(subs):
        penalty_for_missing_age, X_train, y_train, X_test = Preprocess()
        train_loader, test_loader = create_dataloaders(mb_size, X_train,
                                                       y_train, X_test)
        net = Net(in_channels, first, second, third, 0.0, 0.2, 0.2).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        train(net, epochs, optim, train_loader)
        filename = path + 'sub' + str(next_sub_id + i) + '.csv'
        submit(net, test_loader, filename)
