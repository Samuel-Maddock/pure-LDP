from pure_ldp.core import FreqOracleServer
from pure_ldp.hadamard_response.internal import k2k_hadamard
import math
import numpy as np

class HadamardResponseServer(FreqOracleServer):
    def __init__(self, epsilon, d, index_mapper=None):
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.k = math.ceil(2**(math.log(d, 2)))
        self.hr = k2k_hadamard.Hadamard_Rand_high_priv(self.k, self.epsilon)
        self.set_name("Hadamard Response")

    def aggregate(self, data, **kwargs):
        self.aggregated_data = np.append(self.aggregated_data, data)
        self.n +=1

    def estimate_all(self):
        self.estimated_data = self.hr.decode_string(self.aggregated_data) * self.n
        return self.estimated_data

    def estimate(self, data, suppress_warnings=False):
        index = self.index_mapper(data)
        self.estimate_all()
        return self.estimated_data[index]