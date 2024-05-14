import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

import IPython.display as idisplay

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mse(y, y_hat):
    e = (y - y_hat)**2
    return e.mean()


def loss(x, y, y_hat, mode = None):
    if mode=='MLP':
        batch_size = y.shape[0]
        y = y.view(batch_size, -1)
    else:
        y = y

    e = mse(y, y_hat)
    return e


def accuracy_cnn(y, y_hat, mean, std, mode=None):

    if mode=='MLP':
        batch_size = y.shape[0]
        y = y.view(batch_size, -1)
    else:
        y = y

    y_hat = y_hat
    mean = mean.to(device)
    std = std.to(device)

    y_hat = y_hat*std + mean
    y = y*std + mean
    y_add = y + y_hat

    ysquare = y*y
    yhat_square = y_hat*y_hat
    yadd_square = y_add*y_add

    deno = 0.5*(torch.sqrt(ysquare.mean()) + torch.sqrt(yhat_square.mean()))
    accuracy = 1- torch.sqrt((mse(y, y_hat)/(ysquare.mean())))
    accuracy_1 = 1 - torch.sqrt(2*mse(y, y_hat) / (yadd_square.mean()))
    accuracy_2 = 1- torch.sqrt(mse(y, y_hat))/deno

    return accuracy.item(), accuracy_1.item(), accuracy_2.item()


def correlation_fucntion(y, y_hat, mean, std, mode=None):

    if mode=='MLP':
        batch_size = y.shape[0]
        y = y.view(batch_size, -1)

    else:
        y = y

    mean = mean.to(device)
    std = std.to(device)

    y_hat = y_hat*std + mean
    y     = y*std + mean

    mean_yhat = torch.mean(y_hat)
    mean_y = torch.mean(y)

    numerator1 = y - mean_y
    numerator2 = y_hat - mean_yhat

    denominator1 = torch.mean(numerator1**2)
    denominator2 = torch.mean(numerator2**2)

    numerator   = torch.mean(numerator1* numerator2)
    denominator = torch.mean(denominator1* denominator2)

    correlation = numerator / torch.sqrt(denominator)

    return correlation.item()


def train(net, dataset, opti, rate, stat, correla, accuracy, accuracy_1, accuracy_2, mean, std, mode=None):
    net.train()
    cost = 0.0
    accu_1c = 0.0
    accu_2c = 0.0
    accu_3c = 0.0
    correl = 0

    for step, batch in enumerate(dataset):

        data, labs = batch

        opti.zero_grad()

        pred = net(data)

        if mode=='MLP':
            grad = loss(data, labs, pred, mode='MLP')
        else:
            grad = loss(data, labs, pred)

        grad.backward()

        opti.step()

        cost  += grad.item()

        accu1, accu2, accu3 = accuracy_cnn(labs, pred, mean, std, mode = mode)
        
        correl += correlation_fucntion(labs, pred, mean, std, mode=mode)

        accu_1c  = accu_1c + accu1
        accu_2c  = accu_2c + accu2
        accu_3c  = accu_3c + accu3

    cost /= len(dataset)
    accu_1c /= len(dataset)
    accu_2c /= len(dataset)
    accu_3c /= len(dataset)
    correl  /= len(dataset)

    stat.append(cost)
    accuracy.append(accu_1c)
    accuracy_1.append(accu_2c)
    accuracy_2.append(accu_3c)
    correla.append(correl)

def valid(net, dataset,  stat, correla, accuracy, accuracy_1, accuracy_2, mean, std, mode=None):

    net.eval()

    cost = 0.0
    accu_1c = 0.0
    accu_2c = 0.0
    accu_3c = 0.0

    correl = 0.0

    with torch.no_grad():
        for step, batch in enumerate(dataset):
            data, labs = batch

            pred = net(data)

            if mode=='MLP':
                grad = loss(data, labs, pred, mode='MLP')
            else:
                grad = loss(data, labs, pred)

            cost += grad.item()

            accu1, accu2, accu3 = accuracy_cnn(labs, pred, mean, std, mode = mode)

            correl += correlation_fucntion(labs, pred, mean, std, mode=mode)

            accu_1c  = accu_1c + accu1
            accu_2c  = accu_2c + accu2
            accu_3c  = accu_3c + accu3

        cost /= len(dataset)
        accu_1c /= len(dataset)
        accu_2c /= len(dataset)
        accu_3c /= len(dataset)
        correl  /= len(dataset)

        stat.append(cost)
        accuracy.append(accu_1c)
        accuracy_1.append(accu_2c)
        accuracy_2.append(accu_3c)
        correla.append(correl)

def loop(net, model_path, train_loader, valid_loader, opti, rate, mean, std, epochs=1000, mode=None):

    path = model_path
    if not os.path.isdir(path):
        os.mkdir(path)

    train_loss = []
    valid_loss = []
    train_accuracy = []
    train_accuracy_1 = []
    train_accuracy_2 = []
    valid_accuracy = []
    valid_accuracy_1 = []
    valid_accuracy_2 = []

    correl_train = []
    correl_valid = []

    for epoch in range(1, epochs + 1):
    
        if mode=='MLP':
            train(net, train_loader, opti, rate,  train_loss, correl_train, train_accuracy, train_accuracy_1, train_accuracy_2, mean, std, mode='MLP')
            valid(net, valid_loader, valid_loss, correl_valid, valid_accuracy, valid_accuracy_1, valid_accuracy_2, mean, std, mode='MLP')
        else:
            train(net, train_loader, opti, rate,  train_loss, correl_train,  train_accuracy, train_accuracy_1, train_accuracy_2, mean, std)
            valid(net, valid_loader, valid_loss, correl_valid, valid_accuracy, valid_accuracy_1, valid_accuracy_2, mean, std)

        rate.step()
        print('Current loss (epoch = {}) \t train loss \t = {} \t valid loss \t = {} '.format(epoch, train_loss[-1],valid_loss[-1]), flush=True)

        if epoch % 1 == 0:
            np.savetxt(path + '/losses.csv', np.column_stack((train_loss, valid_loss, train_accuracy, train_accuracy_1, train_accuracy_2, valid_accuracy, valid_accuracy_1, valid_accuracy_2, correl_train, correl_valid)), delimiter=",", fmt='%s')
            torch.save(net.state_dict(), path + '/weights.pyt')

    print('Finished training, with last progress = {}'.format(train_loss[-1] - train_loss[-2]))
