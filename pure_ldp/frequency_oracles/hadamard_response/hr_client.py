from pure_ldp.core import FreqOracleClient
from pure_ldp.frequency_oracles.hadamard_response.internal import k2k_hadamard
import math


class HadamardResponseClient(FreqOracleClient):
    def __init__(self, epsilon, d, hash_funcs, index_mapper=None):
        """

        Args:
            epsilon: privacy budget
            d: domain size
            index_mapper: function
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.update_params(epsilon,d,index_mapper)
        self.hr.permute = hash_funcs

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
            if epsilon <= 1:
                self.hr = k2k_hadamard.Hadamard_Rand_high_priv(d, self.epsilon, encode_acc=1) # hadamard_response
            else:
                self.hr = k2k_hadamard.Hadamard_Rand_general_original(d, self.epsilon, encode_acc=1) # hadamard_response

    def _perturb(self, data):
        """
        Used internally to perturb data
        Args:
            data: item to perturb

        Returns: perturbed data

        """
        if self.epsilon <= 1:
            return self.hr.encode_symbol(data)
        else:
            return self.hr.encode_symbol(self.hr.permute[data]) # General privacy HR needs to permute data

    def privatise(self, data):
        """
        Privatises given data item using the hadamard response technique
        Args:
            data:

        Returns: privatised data

        """
        index = self.index_mapper(data)
        return self._perturb(index)
