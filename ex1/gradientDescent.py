import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def computeCost(X, y, theta, m):
    #print(X.shape, y.shape, theta.shape, m)
    return (np.sum(np.square(np.matmul(theta, X) - y))/(2*m))

def gradientDescent(X, y, theta, alpha, num_iter, m):
    J_history = np.matrix(np.zeros((num_iter+1, 1)))

    for i in range(1, num_iter+1):
        delta = (1/m)*np.matmul(np.matmul(theta, X) - y, X.T)
        theta = theta - alpha*delta
        J_history[i] = computeCost(X, y, theta, m)
    return J_history, theta

data = pd.read_csv('ex1data1.txt')

m = len(data)
X = np.matrix(np.array([data['Population'].values, np.ones(m)]))
y = np.matrix(data['Profit'].values)
theta = np.matrix(np.array([0,0]))
alpha = 0.01
num_iter = 1500

#print(computeCost(X, y, theta, m))
#print(computeCost(X, y, np.matrix(np.array([2, -1])), m))
J_history, theta = gradientDescent(X, y, theta, alpha, num_iter, m)
print(theta)
