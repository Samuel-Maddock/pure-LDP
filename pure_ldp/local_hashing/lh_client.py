import numpy as np
import math
import xxhash


# Client-side for local-hashing

# Very loosely based on code by Wang (https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)

class LHClient:
    def __init__(self, epsilon, g=2, use_olh=False, index_mapper=None):
        """

        Args:
            epsilon: float - The privacy budget
            g: Optional integer - The domain [g] = {1,2,...,g} that data is hashed to, 2 by default (binary local hashing)
            use_olh: Optional boolean - if set to true uses Optimised Local Hashing (OLH) i.e g is set to round(e^epsilon + 1)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.epsilon = epsilon
        self.g = g

        if use_olh is True:
            self.g = int(round(math.exp(self.epsilon))) + 1

        self.p = math.exp(self.epsilon) / (math.exp(self.epsilon) + self.g - 1)
        self.q = 1.0 / (math.exp(self.epsilon) + self.g - 1)

        if index_mapper is None:
            self.index_mapper = lambda x: x - 1 # By default, we assume the data is integers from {1,...,d} and hence index as {0,..., d-1}
        else:
            self.index_mapper = index_mapper

    def __perturb(self, data, seed):
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

    def privatise(self, data, seed):
        """
        Privatises a user's data using local hashing.

        Args:
            data: The data to be privatised
            seed: The seed for that user's hash function, a seed should be uniquely assigned to every user

        Returns:
            privatised data: a single integer
        """
        return self.__perturb(data, seed)
