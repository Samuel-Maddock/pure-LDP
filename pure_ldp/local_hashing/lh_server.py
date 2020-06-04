import numpy as np
import math
import xxhash

# Server-side for local-hashing

# Very loosely based on code by Wang (https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)

class LHServer:
    def __init__(self, epsilon, d, g=2, use_olh=False, index_mapper=None):
        """

        Args:
            epsilon: float - The privacy budget
            d: integer - Size of the data domain
            g: Optional float - The domain [g] = {1,2,...,g} that data is hashed to, 2 by default (binary local hashing)
            use_olh: Optional boolean - if set to true uses Optimised Local Hashing i.e g is set to round(e^epsilon + 1)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.epsilon = epsilon
        self.g = g
        self.d = d
        self.n = 0
        self.aggregated_data = np.zeros(self.d)
        self.estimated_data = np.zeros(self.d)

        if use_olh is True:
            self.g = int(round(math.exp(self.epsilon))) + 1

        self.p = math.exp(self.epsilon) / (math.exp(self.epsilon) + self.g - 1)
        self.q = 1.0 / (math.exp(self.epsilon) + self.g - 1)

        if index_mapper is None:
            self.index_mapper = lambda x: x-1
        else:
            self.index_mapper = index_mapper

    def aggregate(self, priv_data, seed):
        """
        Aggregates privatised data from UEClient to be used to calculate frequency estimates.

        Args:
            priv_data: Privatised data of the form returned from UEClient.privatise
            seed: The seed of the user's hash function
        """
        for i in range(0,self.d):
            if priv_data == (xxhash.xxh32(str(i), seed=seed).intdigest() % self.g):
                self.aggregated_data[i] += 1
        self.n += 1

    def estimate(self, data):
        """
        Calcualtes a frequency estimate of the given data item using the aggregated data.

        Args:
            data: data item

        Returns: float - frequency estimate of the data item

        """
        if self.aggregated_data is None:
            raise Exception("UEServer has aggregated no data, no estimation can be made")

        a = self.g / (self.p * self.g - 1)
        b = self.n / (self.p * self.g - 1)

        self.estimated_data = a * self.aggregated_data - b
        index = self.index_mapper(data)
        return self.estimated_data[index]


