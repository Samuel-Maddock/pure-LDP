from pure_ldp.core import FreqOracleServer
from pure_ldp.hadamard_response.internal import k2k_hadamard
import math
import numpy as np


class HadamardResponseServer(FreqOracleServer):
    def __init__(self, epsilon, d, index_mapper=None):
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.aggregated_data = []
        self.k = math.ceil(2 ** (math.log(d, 2)))
        self.hr = k2k_hadamard.Hadamard_Rand_high_priv(self.k, self.epsilon)
        self.set_name("Hadamard Response")

    def reset(self):
        super().reset()
        self.aggregated_data = [] # For Hadamard response, aggregated data is stored in a list not numpy array

    def aggregate(self, data):
        self.aggregated_data.append(data)
        self.n += 1

    def _update_estimates(self):
        self.estimated_data = self.hr.decode_string(self.aggregated_data) * self.n
        return self.estimated_data

    def estimate(self, data, suppress_warnings=False):
        self.check_warnings(suppress_warnings)
        index = self.index_mapper(data)
        self.check_and_update_estimates()
        return self.estimated_data[index]
