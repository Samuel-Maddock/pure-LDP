from pure_ldp.core import FreqOracleClient
from pure_ldp.frequency_oracles.local_hashing.fast_lh_client import FastLHClient

import random

class SketchResponseClient(FreqOracleClient):
    def __init__(self, epsilon, m, hash_funcs, index_mapper=None, fo_client=None, lh_k=100, count_sketch=False):
        """

        Args:
            epsilon (float): Privacy budget
            m (integer): Size of the sketch vector to privatise
            hash_funcs (list of funcs): Hash functions, the length determines the k parameter of the sketch
            index_mapper (optional function): Index mapper function
            fo_client (FreqOracleClient): The FO client  used to perturb the sketch vector. If none is supplies, it will use FastLH by default.
            lh_k (Optional int): If no FO client is passed, this can be used to set the k parameter for the FastLH client that is used
            count_sketch (optional - boolean): If True, will use count-sketch for perturbation/estimation (instead of count-mean sketch)
        """
        self.k = len(hash_funcs)
        self.lh_k = lh_k
        self.m = m
        self.count_sketch = count_sketch
        self.hash_funcs = hash_funcs

        d = self.m
        self.cs_map = None
        if self.count_sketch:
            d = 2 * self.m

            def cs_map(x):
                if x > 0:
                    return x-1
                else:
                    return 2*self.m - abs(x)
            self.cs_map = cs_map

        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.update_params(epsilon, d, index_mapper)

        if isinstance(fo_client, FreqOracleClient):
            self.client = fo_client
        else:
            self.client = FastLHClient(self.epsilon, d, self.lh_k, use_olh=True, index_mapper=None)

        self.client.update_params(index_mapper=lambda x:x, epsilon=self.epsilon)

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        """
        Updates sketch client-side params
        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            index_mapper: optional - function
        """
        super().update_params(epsilon, d, index_mapper)

    def privatise(self, data):
        """
        Privatises given data item using sketch response
        Args:
            data: Data to privatise

        Returns: privatised data

        """
        index = self.index_mapper(data)

        k = random.randint(0, self.k-1) # Randomly select a hash function to use on the data
        hash_func = self.hash_funcs[k]

        # If count-sketch use the two hash function approach
        if self.count_sketch:
            g_val = 2* hash_func[1](index) - 1
            val = self.cs_map((hash_func[0](index) + 1) * g_val)
            priv = self.client.privatise(val)
        else:
            priv = self.client.privatise(hash_func(index)) # Otherwise use the FOClient to perturb the users hashed data

        return priv, k
