import numpy as np
import math
from scipy.optimize import fminbound

# Client-side for histogram-encoding

class HEServer:
    def __init__(self, epsilon, d, use_the=False, theta=None, index_mapper=None):
        self.epsilon = epsilon
        self.d = d
        self.n = 0
        self.aggregated_data = np.zeros(d)
        self.is_the = use_the

        if use_the is True:
            self.theta = theta
            if theta is None:
                self.theta = self.__find_optimal_theta()

            self.p = 1 - 0.5 * (math.pow(math.e, self.epsilon / 2 * (self.theta - 1)))
            self.q = 0.5 * (math.pow(math.e, -1 * self.theta * (self.epsilon / 2)))

        if index_mapper is None:
            self.index_mapper = lambda x: x - 1
        else:
            self.index_mapper = index_mapper

        self.__find_optimal_theta()

    def __find_optimal_theta(self):

        # Minimise variance for our fixed epsilon to find the optimal theta value for thresholding
        def var(x):
            num = 2 * math.exp(self.epsilon * x / 2) - 1
            denom = math.pow(1 + math.exp((self.epsilon * (x - 1/2))) - 2 * math.exp((self.epsilon * x / 2)), 2)
            return num/denom

        return round(float(fminbound(var, 0.5, 1)), 4)

    def aggregate(self, priv_data):
        # Threshold rounding
        if self.is_the:
            for index, item in enumerate(priv_data):
                if item <= self.theta:
                    priv_data[index] = 0
                else:
                    priv_data[index] = 1

        self.aggregated_data += priv_data
        self.n += 1

    def estimate(self, data):
        if self.aggregated_data is None:
            raise Exception("UEServer has aggregated no data, no estimation can be made")

        index = self.index_mapper(data)

        if self.is_the:
            return (self.aggregated_data[index] - self.n * self.q) / (self.p - self.q)
        else:
            return self.aggregated_data[index]
