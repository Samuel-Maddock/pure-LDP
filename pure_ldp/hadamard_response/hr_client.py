from pure_ldp.core import FreqOracleClient
from pure_ldp.hadamard_response.internal import k2k_hadamard
import math


class HadamardResponseClient(FreqOracleClient):
    def __init__(self, epsilon, d, index_mapper=None):
        """

        Args:
            epsilon: privacy budget
            d: domain size
            index_mapper: function
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.k = math.ceil(2 ** (math.log(d, 2))) # k must be a power of 2 for hadamard, so we round d to nearest power
        self.hr = k2k_hadamard.Hadamard_Rand_high_priv(self.k, self.epsilon) # hadamard_response

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        """
        Updates HR client-side params
        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            index_mapper: optional - function
        """
        super().update_params(epsilon, d, index_mapper)

        if d is not None or epsilon is not None:
            self.k = math.ceil(2 ** (math.log(self.d, 2))) # k must be a power of 2 for hadamard, so we round d to nearest power
            self.hr = k2k_hadamard.Hadamard_Rand_high_priv(self.k, self.epsilon) # hadamard_response

    def _perturb(self, data):
        """
        Used internally to perturb data
        Args:
            data: item to perturb

        Returns: perturbed data

        """
        return self.hr.encode_symbol(data)

    def privatise(self, data):
        """
        Privatises given data item using the hadamard response technique
        Args:
            data:

        Returns: privatised data

        """
        index = self.index_mapper(data)
        return self._perturb(index)
