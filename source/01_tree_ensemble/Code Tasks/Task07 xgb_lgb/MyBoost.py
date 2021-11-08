from BasicTree import DecisionTreeRegressor as DT
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

class MyBoost:

    def __init__(self, max_depth=4, n_estimator=1000, lamda=1, gamma=0, lr=0.01):
        self.max_depth = max_depth
        self.n_estimator = n_estimator
        self.lamda = lamda
        self.gamma = gamma
        self.lr = lr
        self.booster = []
        self.feature_importances_ = 0
        self.best_round = None

    def record_score(self, y_train, y_val, train_predict, val_predict, i):
        mse_val = mean_squared_error(y_val, val_predict)
        mse_train = mean_squared_error(y_train, train_predict)
        print("第%d轮\t训练集： %.4f\t"
            "验证集： %.4f"%(i+1, mse_train, mse_val))
        return mse_val

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=0)
        train_predict, val_predict = 0, 0
        p = np.full(X_train.shape[0], np.mean(y_train))
        q = np.ones(X_train.shape[0])
        last_val_score = np.infty
        for i in range(self.n_estimator):
            cur_booster = DT(self.max_depth, self.lamda, self.gamma)
            cur_booster.fit(X_train, y_train, p, q)
            self.feature_importances_ += cur_booster.feature_importances_
            train_predict += cur_booster.predict(X_train) * self.lr
            val_predict += cur_booster.predict(X_val) * self.lr
            p = -(y_train - train_predict)
            self.booster.append(cur_booster)
            cur_val_score = self.record_score(
                y_train, y_val, train_predict, val_predict, i)
            if cur_val_score > last_val_score:
                self.best_round = i
                print("\n训练结束！最佳轮数为%d"%(i+1))
                break
            last_val_score = cur_val_score

    def predict(self, X):
        cur_predict = 0
        for i in range(self.best_round):
            cur_predict += self.lr * self.booster[i].predict(X)
        return cur_predict

if __name__ == "__main__":

    X, y = make_regression(
        n_samples=400, n_features=8, n_informative=4, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    model = MyBoost()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    print("\n测试集的MSE为 %.4f"%(mse))