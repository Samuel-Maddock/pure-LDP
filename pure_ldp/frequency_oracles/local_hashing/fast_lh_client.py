import random
from pure_ldp.frequency_oracles.local_hashing import LHClient

# Client-side for fast local-hashing
    # Heuristic fast variant of OLH

class FastLHClient(LHClient):
    def __init__(self, epsilon, d, k, g=2, use_olh=False, index_mapper=None):
        """
        Fast heuristic version of OLH

        Args:
            epsilon: float - The privacy budget
            g: Optional integer - The domain [g] = {1,2,...,g} that data is hashed to, 2 by default (binary local hashing)
            use_olh: Optional boolean - if set to true uses Optimised Local Hashing (OLH) i.e g is set to round(e^epsilon + 1)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.k = k
        super().__init__(epsilon, d, g, use_olh, index_mapper)

        if k is not None:
            self.k = k

    def update_params(self, epsilon=None, d=None, k=None, use_olh=None, g=None, index_mapper=None):
        super().update_params(epsilon, d, use_olh, g, index_mapper)
        self.k = k if k is not None else self.k

    def privatise(self, data):
        """
        Privatises a user's data using fast local hashing (FLH)

        Args:
            data: The data to be privatised

        Returns:
            privatised data: a single integer
        """

        seed = random.randint(0, self.k-1)
        return self._perturb(data, seed), seed
