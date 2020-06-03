import numpy as np
import math

# Client-side for histogram-encoding

class HEClient:
    def __init__(self, epsilon, d, index_mapper=None):
        self.epsilon = epsilon
        self.d = d

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def __perturb(self, oh_vec):
        noise = np.random.laplace(scale=(2/self.epsilon), size=self.d)
        noisy_vec = oh_vec + noise
        return noisy_vec

    def privatise(self, data):
        index = self.index_mapper(data)

        oh_vec = np.zeros(self.d)
        oh_vec[index] = 1

        return self.__perturb(oh_vec)
