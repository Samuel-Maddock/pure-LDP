import numpy as np
import math

# Client-side for unary-encoding
    # By default parameters are set for Symmetric Unary Encoding (SUE)
    # If is_oue=True is passed to the constructor then it uses Optimised Unary Encoding (OUE)

class UEClient:
    def __init__(self, epsilon, d, is_oue=False, index_mapper=None):
        self.epsilon = epsilon
        self.d = d

        const = math.pow(math.e, self.epsilon/2)
        self.p = const / (const + 1)
        self.q = 1-self.p

        if is_oue is True:
            self.p = 0.5
            self.q = 1/(math.pow(math.e, self.epsilon) + 1)

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def __perturb(self, oh_vec):
        for index, entry in enumerate(oh_vec):
            if entry == 1:
                oh_vec[index] = np.random.choice([1, 0], p=[self.p, 1-self.p]) # If entry is 1, keep as 1 with prob p
            else:
                oh_vec[index] = np.random.choice([1, 0], p=[self.q, 1-self.q])  # If entry is 0, flip with prob q

        return oh_vec

    def privatise(self, data):
        index = self.index_mapper(data)

        oh_vec = np.zeros(self.d)
        oh_vec[index] = 1

        return self.__perturb(oh_vec)