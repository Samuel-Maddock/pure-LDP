import numpy as np
import math

# Client-side for histogram-encoding

class HEClient:
    def __init__(self, epsilon, d, is_the=False, theta=1, index_mapper=None):
        self.epsilon = epsilon
        self.d = d
        self.is_the = is_the

        if self.is_the is True:
            self.theta = theta

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def __perturb(self, oh_vec):
        noise = np.random.laplace(scale=(2/self.epsilon), size=self.d)
        noisy_vec = oh_vec + noise

        if self.is_the:
            for index, item in enumerate(noisy_vec):
                if item <= self.theta:
                    noisy_vec[index] = 0
                else:
                    noisy_vec[index] = 1

        return noisy_vec

    def privatise(self, data):
        index = self.index_mapper(data)

        oh_vec = np.zeros(self.d)
        oh_vec[index] = 1

        return self.__perturb(oh_vec)


client = HEClient(3, 4)