import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def mse(x0, x):
    return (x0 - x) ** 2


def absolute(x0, x):
    return abs(x0 - x)


class MyGradientBoosting():
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, metric=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.metric = self.__mse if metric is None else metric
        self.loss = []

    @staticmethod
    def __mse(x0, x):
        return (x0 - x) ** 2

    def __gradient(self, pred):
        dx = 1e-12
        return (self.metric(self.y, pred + dx) - self.metric(self.y, pred)) / dx

    def fit(self, X, y):
        self.X = X
        self.y = y

        r = self.y
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, r)
            pred = tree.predict(X)

            self.trees.append(tree)

            if not i:
                y_pred = pred
            else:
                y_pred += self.learning_rate * pred

            self.loss.append(self.metric(self.y, y_pred).sum() / len(self.y))

            r = -self.__gradient(y_pred)

    def predict(self, X):
        pred = 0
        for i in range(self.n_estimators):
            pred += (self.learning_rate if i else 1) * self.trees[i].predict(X)
        return pred


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        train_df = pd.read_csv(sys.argv[1])
        test_df = pd.read_csv(sys.argv[2])
        mgb = MyGradientBoosting(metric=mse)
        mgb.fit(train_df[['A', 'B']], train_df['Y'])
        pred = mgb.predict(test_df[['A', 'B']])
        pd.DataFrame(pred, columns=['Y']).to_csv(sys.argv[3])
    else:
        print('Недостаточно параметров')
