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

# -------- Constants --------
in_channels = 15  # Result of preprocessing
training_size = 891
test_size = 418
train_folds_size = 16*50
test_folds_size = 16*5+11

# -------- Hyperparameters --------
nb_clusters = 20
epochs = 10
mb_size = 16
first = 512
second = 256
third = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 5e-4


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


def train(net):
    for epoch in range(1, epochs+1):
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

        acc = right_predictions / training_size
        print('Epoch: {}, loss: {}, accuracy: {}'.format(epoch,
                                                         running_loss/(i+1),
                                                         acc))


def eval(net):
    net.eval()
    running_loss = 0.0
    right_predictions = 0
    for i, (data, survived) in enumerate(train_loader):
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

    acc = right_predictions / test_size
    print('Test epoch, loss: {}, accuracy: {}'.format(running_loss/(i+1),
                                                      acc))


def cross_val(net, opt, X_train, y_train, k_folds):
    fold_cntr = 0
    test_accs = np.zeros(k_folds)
    init_state = deepcopy(net.state_dict())
    init_state_opt = deepcopy(opt.state_dict())
    kf = KFold(n_splits=k_folds)
    for train_indices, test_indices in kf.split(X_train):
        if fold_cntr > 0:
            net.load_state_dict(init_state)
            opt.load_state_dict(init_state_opt)
        train = TensorDataset(
            torch.Tensor(np.array(X_train.loc[train_indices])),
            torch.Tensor(np.array(y_train.loc[train_indices]))
        )
        test = TensorDataset(
            torch.Tensor(np.array(X_train.loc[test_indices])),
            torch.Tensor(np.array(y_train.loc[test_indices]))
        )
        train_loader = DataLoader(train, batch_size=mb_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=mb_size, shuffle=False)
        net.train()
        for epoch in range(1, epochs+1):
            running_loss = 0.0
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

            print('Train Epoch: {}, loss: {}, fold: {}'.format(
                epoch,
                running_loss/(i+1),
                fold_cntr+1
                )
            )
        net.eval()
        running_loss = 0.0
        right_predictions = 0
        wrong_predictions = 0
        for i, (data, survived) in enumerate(test_loader):
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
        print('Test Epoch, loss: {}, acc: {}, fold: {}'.format(
            running_loss/(i+1),
            100 * right_predictions / (right_predictions + wrong_predictions),
            fold_cntr+1
            )
        )
        test_accs[fold_cntr] = right_predictions
        fold_cntr += 1
    return test_accs


def submit():
    survived_array = np.array([])
    for i, [data] in enumerate(test_loader):
        data = data.to(device)
        y_pred = net(data)
        survived = y_pred.detach().cpu().numpy()
        survived_array = np.append(survived_array, survived)

    # Add a penalty if the age value is missing
    survived_array -= penalty_for_missing_age
    survived_array = survived_array > 0.5

    # Get the passenger ids for the test set for submission
    test_ids = pd.read_csv('test.csv').PassengerId
    survived = pd.Series(survived_array.astype(int))

    frame = {'PassengerId': test_ids, 'Survived': survived}

    submission_df = pd.DataFrame(frame)

    filename = \
        r'F:\Users\SilentFart\Documents\PythonProjects\Titanic\submission2.csv'
    submission_df.to_csv(
        filename,
        index=False
    )


train_loader, test_loader, penalty_for_missing_age, X_train, y_train = \
    Preprocess(mb_size)
net = Net(in_channels, first, second, third, 0.0, 0.5, 0.5).to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
accs = cross_val(net, optim, X_train, y_train, k_folds=10)
