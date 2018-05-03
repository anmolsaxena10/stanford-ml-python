import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def computeCost(X, y, theta, m):
    print(X.shape, y.shape, theta.shape, m)
    return (np.sum(np.square(np.matmul(theta, X) - y))/(2*m))


data = pd.read_csv('ex1data1.txt')

m = len(data)
X = np.matrix(np.array([data['Population'].values, np.ones(m)]))
y = np.matrix(data['Profit'].values)
theta = np.matrix(np.array([0,0]))

print(computeCost(X, y, theta, m))
print(computeCost(X, y, np.matrix(np.array([2, -1])), m))
