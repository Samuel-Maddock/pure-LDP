import xxhash
import numpy as np
from pure_ldp.frequency_oracles.local_hashing import LHServer

# Server-side for fast local-hashing

class FastLHServer(LHServer):
    def __init__(self, epsilon, d, k, g=2, use_olh=True, index_mapper=None, hash_matrix=None):
        """

        Args:
            epsilon: float - The privacy budget
            d: integer - Size of the data domain
            k: integer - The number of hash functions to use. Larger k results in a more accurate oracle at the expense of computation time.
            g: Optional float - The domain [g] = {1,2,...,g} that data is hashed to, 2 by default (binary local hashing)
            use_olh: Optional boolean - if set to true uses Optimised Local Hashing i.e g is set to round(e^epsilon + 1)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
            hash_matrix: Optional matrix - Allows the use of a pre-computed hash matrix that contains hashed domain elements
        """
        self.k = k
        super().__init__(epsilon, d, g, use_olh, index_mapper=index_mapper)
        self.hash_counts = np.zeros((self.k, self.g))

        # g = lambda i,j: xxhash.xxh32(str(int(j)), seed=int(i)).intdigest() % self.g

        if hash_matrix is None:
            matrix = np.empty((self.k, self.d))
            for i in range(0, self.k):
                for j in range(0, self.d):
                    matrix[i][j] = xxhash.xxh32(str(j), seed=i).intdigest() % self.g

            # self.hash_matrix = np.fromfunction(g, (self.k, self.d))
            self.hash_matrix = matrix
        else:
            self.hash_matrix = hash_matrix

    def update_params(self, epsilon=None, d=None, k=None, use_olh=None, g=None, index_mapper=None, update_hash_matrix=True):
        super().update_params(epsilon=epsilon, d=d, use_olh=use_olh, g=g, index_mapper=index_mapper)
        self.k = k if k is not None else self.k

        # If any of the main parameters are updated the hash_matrix needs to be updated... this is quite slow
        if epsilon is not None or self.g is not None or self.k is not None or self.d is not None and update_hash_matrix is True:
            matrix = np.empty((self.k, self.d))
            for i in range(0, self.k):
                for j in range(0, self.d):
                    matrix[i][j] = xxhash.xxh32(str(j), seed=i).intdigest() % self.g
            self.hash_matrix = matrix

    def aggregate(self, priv_data):
        """
        Aggregates privatised data from FastLHClient to be used to calculate frequency estimates.

        Args:
            priv_data: Privatised data of the form returned from UEClient.privatise
        """
        seed = priv_data[1]
        priv_data = priv_data[0]

        self.hash_counts[seed][priv_data] += 1
        self.n += 1

    def _compute_aggregates(self):

        def func(x):
            sum = 0
            for index, val in enumerate(x):
                sum += self.hash_counts[index,int(val)]
            return sum

        self.aggregated_data = np.apply_along_axis(func, 0, self.hash_matrix)

    def _update_estimates(self):
        self._compute_aggregates()
        super()._update_estimates()

    def estimate(self, data, suppress_warnings=False):
        """
        Calculates a frequency estimate of the given data item using the aggregated data.

        Args:
            data: data item
            suppress_warnings: Optional boolean - Suppresses warnings about possible inaccurate estimations

        Returns: float - frequency estimate of the data item

        """
        self.check_and_update_estimates()
        return super().estimate(data)