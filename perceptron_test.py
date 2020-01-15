import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import perceptron

def test(eta, n_iter, X, y):
    ppn = perceptron.Perceptron(eta=eta, n_iter=n_iter)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of errors')
    plt.show()


def preparedata():
    df = downloaddataset()
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0,2]].values
    return X, y


def plotdata(X):
    plt.scatter(X[:50,0], X[:50,1], marker='o', color='red', label='setosa')
    plt.scatter(X[50:100,0], X[50:100,1], marker='x', color='blue', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


def downloaddataset():
    url = os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','iris','iris.data')
    df = pd.read_csv(url, header=None, encoding='utf-8')
    return df