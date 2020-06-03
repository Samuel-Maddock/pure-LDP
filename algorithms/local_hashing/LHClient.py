import numpy as np
import math
import xxhash

# Client-side for local-hashing

# Very loosely based on code by Wang (https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)

class LHClient:
    def __init__(self, epsilon, g, is_olh=False, index_mapper=None):
        self.epsilon = epsilon
        self.g = g

        if is_olh is True:
            self.g = int(round(math.exp(self.epsilon))) + 1

        self.p = math.exp(self.epsilon) / (math.exp(self.epsilon) + self.g - 1)
        self.q = 1.0 / (math.exp(self.epsilon) + self.g - 1)

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def __perturb(self, data, seed):
        index = self.index_mapper(data)

        # Taken directly from Wang (https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)
        x = (xxhash.xxh32(str(index), seed=seed).intdigest() % self.g)
        y = x

        p_sample = np.random.random_sample()
        # the following two are equivalent
        # if p_sample > p:
        #     while not y == x:
        #         y = np.random.randint(0, g)
        if p_sample > self.p - self.q:
            # perturb
            y = np.random.randint(0, self.g)

        return y

    def privatise(self, data, seed):
        return self.__perturb(data, seed)
