import numpy as np
import math

class UEServer:
    def __init__(self, epsilon, d, use_oue=False, index_mapper=None):
        self.epsilon = epsilon
        self. d = d
        self.n = 0
        self.aggregated_data = np.zeros(d)

        const = math.pow(math.e, self.epsilon/2)
        self.p = const / (const + 1)
        self.q = 1-self.p

        if use_oue is True:
            self.p = 0.5
            self.q = 1/(math.pow(math.e, self.epsilon) + 1)

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def aggregate(self, priv_data):
        self.aggregated_data += priv_data
        self.n += 1

    def estimate(self, data):
        if self.aggregated_data is None:
            raise Exception("UEServer has aggregated no data, no estimation can be made")

        index = self.index_mapper(data)
        return (self.aggregated_data[index] - self.n*self.q)/(self.p-self.q)

