import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import adaline_gd as agd


def test(eta1, eta2, epochs, X, y):
    ada1 = agd.AdalineGD(eta=eta1, n_iter=epochs).fit(X, y)
    ada2 = agd.AdalineGD(eta=eta2, n_iter=epochs).fit(X, y)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(SSE)')
    ax[0].set_title('Adaline, eta=0.01')

    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('SSE')
    ax[1].set_title('Adaline, eta=0.0001')

    plt.show()


def preparedata():
    df = downloaddataset()
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0,2]].values
    return X, y


def downloaddataset():
    url = os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','iris','iris.data')
    df = pd.read_csv(url, header=None, encoding='utf-8')
    return df