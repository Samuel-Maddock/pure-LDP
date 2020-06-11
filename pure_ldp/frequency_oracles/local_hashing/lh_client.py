import numpy as np
import math
import xxhash
from sys import maxsize
import random
from pure_ldp.core import FreqOracleClient

# Client-side for local-hashing

# Very loosely based on code by Wang (https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)

class LHClient(FreqOracleClient):
    def __init__(self, epsilon, d, g=2, use_olh=False, index_mapper=None):
        """

        Args:
            epsilon: float - The privacy budget
            g: Optional integer - The domain [g] = {1,2,...,g} that data is hashed to, 2 by default (binary local hashing)
            use_olh: Optional boolean - if set to true uses Optimised Local Hashing (OLH) i.e g is set to round(e^epsilon + 1)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.use_olh = use_olh
        self.g =g
        self.update_params(epsilon, d, g, index_mapper)

    def update_params(self, epsilon=None, d=None, g=None, index_mapper=None):
        """

        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            g: optional - hash domain
            index_mapper: optional - function
        """
        super().update_params(epsilon, d, index_mapper) # Updates core params

        # Updates g and probs
        self.g = g if g is not None else self.g
        if self.use_olh is True:
            self.g = int(round(math.exp(self.epsilon))) + 1

        if self.epsilon is not None or self.g is not None:
            self.p = math.exp(self.epsilon) / (math.exp(self.epsilon) + self.g - 1)
            self.q = 1.0 / (math.exp(self.epsilon) + self.g - 1)

    def _perturb(self, data, seed):
        """
        Used internally to perturb data using local hashing.

        Will hash the user's data item and then peturb it with probabilities that
        satisfy epsilon local differential privacy. Local hashing is explained
        in more detail here: https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-wang-tianhao.pdf

        Args:
            data: User's data to be privatised
            seed: The seed for the user's hash function

        Returns: peturbed data

        """
        index = self.index_mapper(data)

        # Taken directly from Wang (https://github.com/vvv214/LDP_Protocols/blob/master/olh.py#L55-L65)
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

    def privatise(self, data):
        """
        Privatises a user's data using local hashing.

        Args:
            data: The data to be privatised

        Returns:
            privatised data: a single integer
        """
        seed = random.randint(0,maxsize) # This is sys.maxsize
        return self._perturb(data, seed), seed
