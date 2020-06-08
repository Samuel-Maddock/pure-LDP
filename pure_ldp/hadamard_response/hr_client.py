from pure_ldp.core import FreqOracleClient
from pure_ldp.hadamard_response.internal import k2k_hadamard
import math

class HadamardResponseClient(FreqOracleClient):
    def __init__(self, epsilon, d, index_mapper=None):
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.k = math.ceil(2**(math.log(d, 2)))
        self.hr = k2k_hadamard.Hadamard_Rand_high_priv(self.k, self.epsilon)

    def _peturb(self, data):
        return self.hr.encode_symbol(data)

    def privatise(self, data, **kwargs):
        index = self.index_mapper(data)
        return self._peturb(index)