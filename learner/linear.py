import numpy as np

from scipy.optimize import minimize

from learner.learner import Learner

class LinearSeparator(Learner):

    def __init__(self, dim_size):
        self.dim_size = dim_size

    def learn(self, loss):
        def to_minimize(a_vector):
            return loss(Linear(a_vector))

        init_a = 2 * np.random.rand(self.dim_size + 1) - 1

        res = minimize(to_minimize, init_a)
        return Linear(res.x)

class Linear:
    def __init__(self, a_vector):
        self.a_vector = a_vector

    def __call__(self, xs):
        xs = np.array(xs)
        ys = xs @ self.a_vector[1:] + self.a_vector[0]

        return np.int_(ys > 0)

    def value(self, xs):
        xs = np.array(xs)
        ys = xs @ self.a_vector[1:] + self.a_vector[0]

        return ys