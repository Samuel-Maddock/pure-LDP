from pure_ldp.core import FreqOracleClient

import math
import random
import numpy as np
import itertools


class HadamardMechClient(FreqOracleClient):

    def __init__(self, epsilon, d, t, use_optimal_t=False, index_mapper=None):
        super().__init__(epsilon, d, index_mapper)
        if use_optimal_t is True:
            self.t = math.ceil(math.log((math.e ** self.epsilon + 1), 2))
        else:
            self.t = t

        self.p = (math.exp(self.epsilon)) / (math.exp(self.epsilon) + 2 ** self.t - 1)
        self.hashes = list(itertools.product((-1, 1), repeat=self.t))


    def update_params(self, epsilon=None, d=None, index_mapper=None):
        super().update_params(epsilon, d, index_mapper)

        if epsilon is not None:
            self.p = (math.exp(self.epsilon)) / (math.exp(self.epsilon) + 2 ** self.t - 1)

    def _calculate_p(self, epsilon):
        return math.exp(epsilon) / (1 + math.exp(epsilon))

    def _perturb(self, data):
        """
        Used internally to perturb data
        Args:
            data: item to perturb

        Returns: perturbed data
        """

        j = random.randint(0, self.d - 1)
        size = len(bin(self.d)[2:])  # Get max length of the binary string
        bin_data = bin(data)[2:].zfill(size)
        bin_j = bin(j)[2:].zfill(size)
        one_count = self._count_bin_ones(bin_data, bin_j)
        had_coeff = ((-1) ** one_count)
        return j, had_coeff

    def _count_bin_ones(self, bin1, bin2):
        count = 0
        for i, bit in enumerate(bin1):
            if bit == "1" and bin2[i] == "1":
                count += 1
        return count

    def privatise(self, data):
        """"
        Privatises given data item using the hadamard mechanism
        Args:
            data: Data to be privatised

        Returns: privatised data

        """
        index = self.index_mapper(data)
        output = [self._perturb(index) for i in range(0, self.t)]

        if random.random() >= self.p:
            indexes, h_val = zip(*output)
            new_hashes = self.hashes.copy()
            new_hashes.remove(h_val)
            new_hash = new_hashes[random.randint(0, len(new_hashes)-1)]
            output = zip(indexes, new_hash)

        return output