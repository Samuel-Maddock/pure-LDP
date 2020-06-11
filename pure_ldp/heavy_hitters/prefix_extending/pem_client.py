from pure_ldp.frequency_oracles.local_hashing.lh_client import LHClient
from pure_ldp.core import FreqOracleClient

import math
import random

from bitstring import BitArray

class PEMClient:
    def __init__(self, epsilon, domain_size, start_length, segment_length, FOClient=None):
        """

        Args:
            epsilon: privacy budget
            domain_size: max size of the strings to find
            start_length: The starting fragment length
            segment_length: the length to increase the fragment by on each iteration
            FOClient: a FreqOracleClient instance, used to privatise the data
        """
        self.epsilon = epsilon
        self.domain_size = domain_size
        self.segment_length = segment_length
        self.start_length = start_length
        self.g = math.ceil((self.domain_size-self.start_length)/self.segment_length)

        self.index_mapper = lambda x:x
        if isinstance(FOClient, FreqOracleClient):
            self.client = FOClient
        else:
            self.client = LHClient(self.epsilon, d=None, use_olh=True)
        self.client.update_params(index_mapper=self.index_mapper)

    def privatise(self, bit_string):
        """
        This method is used to privatise a bit string using PEM
        Args:
            bit_string: The bit string to be privatised

        Returns: Privatised bit string and the group number

        """
        group = random.randint(0,self.g-1)

        d = 2 ** (self.start_length + (group + 1) * self.segment_length)
        self.client.update_params(d=d)

        fragment_size = self.start_length + (group+1)*self.segment_length
        fragment = bit_string[0:min(fragment_size, len(bit_string))]
        num = BitArray(bin=fragment).uint
        return self.client.privatise(num), group
