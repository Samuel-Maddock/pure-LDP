from pure_ldp.core import FreqOracleClient
from pure_ldp.frequency_oracles.hadamard_response.internal import k2k_hadamard
import copy

class HadamardResponseClient(FreqOracleClient):
    def __init__(self, epsilon, d, hash_funcs, index_mapper=None):
        """

        Args:
            epsilon: privacy budget
            d: domain size
            index_mapper: function
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.update_params(epsilon,d,index_mapper, hash_funcs)

    def update_params(self, epsilon=None, d=None, index_mapper=None, hash_funcs=None):
        """
        Updates HR client-side params
        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            index_mapper: optional - function
            hash_funcs:
        """
        super().update_params(epsilon, d, index_mapper)
        encode_acc = 0 # If 0 no hadamard matrix is initialised

        if epsilon is not None or d is not None:
            if self.epsilon <= 1:
                self.hr = k2k_hadamard.Hadamard_Rand_high_priv(self.d, self.epsilon, encode_acc=encode_acc) # hadamard_response high privacy regime
            elif self.epsilon > 1 and d is None and hash_funcs is None:
                hash_funcs = copy.deepcopy(self.hr.permute)
                self.hr = k2k_hadamard.Hadamard_Rand_general_original(self.d, self.epsilon, encode_acc=encode_acc) # hadamard_response (general privacy)
                self.hr.permute = hash_funcs
            elif self.epsilon > 1 and hash_funcs is not None:
                self.hr = k2k_hadamard.Hadamard_Rand_general_original(self.d, self.epsilon, encode_acc=encode_acc) # hadamard_response (general privacy)
                self.hr.permute = hash_funcs
            else:
                raise RuntimeWarning("HRClient parameters were reset, but no hash functions were passed. This will result in inconsistent client/server objects. To fix this pass hash_funcs from HRserver.get_hash_funcs()")

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
