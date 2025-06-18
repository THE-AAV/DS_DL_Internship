import numpy as np
class LinearRegression:
    def __init__(self):
        self.b0, self.b1 = 0, 0

    def fit(self, X, y):
        X_mean= np.mean(X)
        y_mean = np.mean(y)
        ssxy,ssx=0,0
        for _ in range (len (X)):
            ssxy += (X[_] - X_mean) * (y[_] - y_mean)
            ssx += (X[_] - X_mean) ** 2
        self.b1 = ssxy / ssx
        self.b0 = y_mean - self.b1 * X_mean
        return self.b0, self.b1
    def predict(self, X):
        y_hat=self.b0 + self.b1 * X
        return y_hat

if __name__ == "__main__":
    X = np.array([[160], [171], [182], [180], [154]])
    y = np.array([72, 76, 77, 83, 76])
    print(f'The Shape of X :{X.shape}')
    print(f'The Shape of y :{y.shape}')
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    lr=LinearRegression()
    b0, b1 = lr.fit(X, y)
    print(f'The value of b0 :{b0}')
    print(f'The value of b1 :{b1}')
    y_hat = lr.predict([[176]])
    print(f'The Weight of the person with height of {X} is :{y_hat[0]}')


    