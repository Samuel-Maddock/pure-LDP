import math
import xxhash

from pure_ldp.core import FreqOracleServer


# Server-side for local-hashing

# Very loosely based on code by Wang (https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)

class LHServer(FreqOracleServer):
    def __init__(self, epsilon, d, g=2, use_olh=False, index_mapper=None):
        """

        Args:
            epsilon: float - The privacy budget
            d: integer - Size of the data domain
            g: Optional float - The domain [g] = {1,2,...,g} that data is hashed to, 2 by default (binary local hashing)
            use_olh: Optional boolean - if set to true uses Optimised Local Hashing i.e g is set to round(e^epsilon + 1)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.set_name("LHServer")
        self.g = g
        self.use_olh = use_olh
        self.update_params(epsilon, d, g, index_mapper)

    def update_params(self, epsilon=None, d=None, g=None, index_mapper=None):
        """
        Updates LHServer parameters, will reset any aggregated/estimated data
        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            g: optional - hash domain
            index_mapper: optional - function
        """
        super().update_params(epsilon, d, index_mapper)

        # Update probs and g
        if epsilon is not None:
            if self.use_olh is True:
                self.g = int(round(math.exp(self.epsilon))) + 1

            self.p = math.exp(self.epsilon) / (math.exp(self.epsilon) + self.g - 1)

    def aggregate(self, priv_data):
        """
        Aggregates privatised data from LHClient to be used to calculate frequency estimates.

        Args:
            priv_data: Privatised data of the form returned from UEClient.privatise
            seed: kwarg - The seed of the user's hash function, must be passed as a keyword arg
        """
        seed = priv_data[1]
        priv_data = priv_data[0]

        for i in range(0, self.d):
            if priv_data == (xxhash.xxh32(str(i), seed=seed).intdigest() % self.g):
                self.aggregated_data[i] += 1
        self.n += 1

    def _update_estimates(self):
        a = self.g / (self.p * self.g - 1)
        b = self.n / (self.p * self.g - 1)

        self.estimated_data = a * self.aggregated_data - b
        return self.estimated_data

    def estimate(self, data, suppress_warnings=False):
        """
        Calculates a frequency estimate of the given data item using the aggregated data.

        Args:
            data: data item
            suppress_warnings: Optional boolean - Suppresses warnings about possible inaccurate estimations

        Returns: float - frequency estimate of the data item

        """
        self.check_warnings(suppress_warnings=suppress_warnings)
        index = self.index_mapper(data)
        self.check_and_update_estimates()
        return self.estimated_data[index]

        # def aggregate(self, priv_data, seed):
    #     """
    #     Aggregates privatised data from UEClient to be used to calculate frequency estimates.
    #
    #     Args:
    #         priv_data: Privatised data of the form returned from UEClient.privatise
    #         seed: The seed of the user's hash function
    #     """
    #     f = lambda x: xxhash.xxh32(str(x), seed=seed).intdigest() % self.g
    #     vals = np.fromiter((f(x) for x in range(0,self.d)), dtype=np.int)
    #     support = (vals==priv_data).astype("int8")
    #     self.aggregated_data += support
    #     self.n += 1
